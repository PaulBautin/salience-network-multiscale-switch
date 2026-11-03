"""Intracranial EEG: spectra, spectral parameterisation and projection onto the cortex.

Reads electroMICA recordings, computes Welch power spectra (also over the zero-gapped
clips of the MNI Open iEEG Atlas), fits the aperiodic exponent and per-band peak power
with specparam, and maps channel values onto the fsLR-32k surface through the contact
sensitivity maps (electroMICA leadfields) or Gaussian footprints around channel positions.
"""
import contextlib
import glob
import io
import logging
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from brainspace.mesh.array_operations import compute_point_area
from brainspace.mesh.mesh_creation import build_polydata
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# fsLR-32k vertices per hemisphere.
N_LH = 32492


# Preprocessing / Welch parameters of every iEEG analysis (Figure 2), passed to
# preprocess_and_compute_psd_ieeg and part of the channel-PSD cache key.
# Implementation of the band-pass filter, also part of the cache key: changing how
# preprocess_and_compute_psd_ieeg filters must change this, so cached spectra are recomputed.
PSD_FILTER = "butterworth, second-order sections, zero phase (sosfiltfilt)"
PSD_PARAMS = dict(fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)


# Canonical frequency bands (Hz), spanning the 0.5-80 Hz preprocessing band.
FREQ_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}


# specparam fit range (Hz) of every iEEG analysis (Figure 2). It stops below the 60 Hz
# line-noise notch of the MICA recordings and the transition band of the 80 Hz low-pass
# of PSD_PARAMS, and below the 50/60 Hz line frequencies and 80 Hz anti-aliasing filter
# of the MNI atlas recordings: specparam advises against fitting filtered-out frequencies.
SPEC_FIT_RANGE = (1.0, 45.0)


# Oscillatory peak-power readout bands (Hz) of Figure 2: the canonical bands
# clipped to the fit range, where peaks can be fitted (delta 1-4 Hz, gamma 30-45 Hz).
SPEC_BANDS = {name: (max(lo, SPEC_FIT_RANGE[0]), min(hi, SPEC_FIT_RANGE[1])) for name, (lo, hi) in FREQ_BANDS.items()}


def compute_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric (hat-function) per-vertex surface areas of a triangular mesh.

    Each vertex is assigned one third of the area of every triangle it belongs to
    -- the integral over the surface of the piecewise-linear hat function that is 1
    at the vertex and 0 at its neighbours. This matches the ``areasv`` routine used
    by electroMICA (``ComputeFeatureMaps``) to normalise channel sensitivities to a
    per-area density before thresholding; it delegates to brainspace's
    ``compute_point_area(..., area_as='one_third')``.

    Parameters
    ----------
    vertices
        Vertex coordinates, shape (n_vertices, 3), in millimetres.
    faces
        Triangle vertex indices, shape (n_faces, 3). MATLAB-style 1-based
        indexing is detected and converted to 0-based automatically.

    Returns
    -------
    np.ndarray
        Per-vertex area, shape (n_vertices,).
    """
    f = np.asarray(faces)
    if f.min() == 1:  # MATLAB 1-based faces
        f = f - 1
    surf = build_polydata(points=np.asarray(vertices, dtype=float), cells=f)
    return compute_point_area(surf, area_as='one_third')


def list_original_data_files(
    root: str,
    stage: str = 'W',
) -> list[str]:
    """
    List the iEEG recordings of one vigilance state of the original MICA dataset.

    Parameters
    ----------
    root : str
        Root directory of the BIDS iEEG recordings. Expected layout:
        ``<root>/sub-PX*/ses-01/*stage-<stage>.mat``
    stage : str
        Vigilance state: ``'W'`` (wakefulness), ``'N2'`` or ``'N3'``.

    Returns
    -------
    list[str]
        Sorted file paths, so that downstream channel order (and any
        cache keyed on this list) is deterministic.
    """
    return sorted(glob.glob(f"{root}/sub-PX*/ses-01/*stage-{stage}.mat"))


def compute_original_psd(
    root: str,
    stage: str = 'W',
    duration_s: float | None = None,
    **psd_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Welch PSD of every bipolar channel in the original iEEG MATLAB recordings.

    Each row of the returned table is one **bipolar channel** from one
    subject/session pair. In the electroMICA terminology, a *channel* is a
    differential recording between two physical *contacts*: ``ChannelName``
    stores the pair (e.g. ``"LCi1-LCi2"``), while ``ContactName1`` and
    ``ContactName2`` hold the individual contact identifiers that index into the
    electroMICA leadfield/sensitivity files.

    All recordings are truncated to the length of the shortest one (or to its first
    ``duration_s`` seconds, e.g. to match vigilance states) before preprocessing,
    so every channel's spectrum is estimated from the same number of samples. The shortest length is read from the MATLAB file headers first,
    then the files are streamed one at a time through
    :func:`preprocess_and_compute_psd_ieeg`, so only one subject's raw signal is
    held in memory. Every preprocessing step acts along the time axis
    independently per channel, so this matches processing all channels as one
    matrix. Truncation is by sample count, so all recordings must share one
    sampling rate.

    Parameters
    ----------
    root : str
        Root directory of the BIDS iEEG recordings (see
        :func:`list_original_data_files`).
    stage : str
        Vigilance state of the recordings: ``'W'``, ``'N2'`` or ``'N3'``.
    duration_s : float | None
        Analysed length in seconds (from the start of each
        recording); ``None`` uses the shortest recording.
    **psd_kwargs
        Forwarded to :func:`preprocess_and_compute_psd_ieeg`
        (``fmin``, ``fmax``, ``fs_target``, ``filter_order``, ``window_sec``,
        ``overlap_sec``).

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        - Channel table with columns:
            - Subject
            - Session
            - ChannelName    — bipolar pair label (e.g. ``"LCi1-LCi2"``)
            - SamplingRate
            - ContactName1   — first contact of the bipolar pair (upper element)
            - ContactName2   — second contact of the bipolar pair (lower element)
        - freq: frequencies in Hz, shape (n_frequencies,).
        - pxx: normalized PSD, shape (n_channels, n_frequencies), row-aligned
          with the channel table.

    Raises
    ------
    ValueError
        If no recording is found, a file's ``Data`` shape does not
        match its channel count, the recordings use more than one sampling
        rate, or ``duration_s`` exceeds the shortest recording.
    """
    pattern = re.compile(r"sub-(PX\d+)/ses-(\d+)")

    # Pass 1: header-only reads (channel names, sampling rate, Data shape).
    headers = []
    for filepath in list_original_data_files(root, stage):
        match = pattern.search(filepath)
        if match is None:
            logger.warning(f"Skipping {filepath}: path does not match sub-PX*/ses-*.")
            continue

        shapes = {name: shape for name, shape, _ in sio.whosmat(filepath)}
        missing = {"ChannelName", "Data", "SamplingRate"} - set(shapes)
        if missing:
            logger.warning(f"Skipping {os.path.basename(filepath)}: missing {sorted(missing)}.")
            continue

        meta = sio.loadmat(filepath, variable_names=["ChannelName", "SamplingRate"], simplify_cells=True)
        channel_names = [str(c) for c in np.atleast_1d(meta["ChannelName"])]
        shape = shapes["Data"]
        if len(shape) != 2 or len(channel_names) not in shape:
            raise ValueError(
                f"Channel count mismatch in {filepath}: "
                f"{shape} vs {len(channel_names)} names"
            )
        n_samples = shape[1] if shape[0] == len(channel_names) else shape[0]
        headers.append((filepath, *match.groups(), channel_names, float(meta["SamplingRate"]), n_samples))

    if not headers:
        raise ValueError(f"No stage-{stage} recordings found under {root}.")

    rates = sorted({h[4] for h in headers})
    if len(rates) > 1:
        raise ValueError(
            f"Recordings use several sampling rates {rates}; truncation to a common "
            f"sample count requires a single rate."
        )
    fs = rates[0]

    lengths = [h[5] for h in headers]
    min_len, max_len = min(lengths), max(lengths)
    if min_len != max_len:
        logger.warning(f"Variable channel lengths ({min_len} to {max_len} samples); truncating all to {min_len}.")
    if duration_s is not None:
        n_dur = int(round(duration_s * fs))
        if n_dur > min_len:
            raise ValueError(f"duration_s={duration_s} s ({n_dur} samples) exceeds the shortest "
                             f"stage-{stage} recording ({min_len} samples).")
        min_len = n_dur
        logger.info(f"stage-{stage}: analysing the first {duration_s:g} s ({n_dur} samples) of each recording.")

    # Pass 2: one subject at a time -> truncate -> preprocess -> PSD.
    rows, pxx_parts, freq = [], [], None
    for filepath, subject, session, channel_names, _, _ in headers:
        data = np.asarray(sio.loadmat(filepath, variable_names=["Data"])["Data"])
        if data.shape[0] != len(channel_names):  # enforce (n_channels, n_samples)
            data = data.T
        data = np.ascontiguousarray(data[:, :min_len], dtype=float)
        f, pxx = preprocess_and_compute_psd_ieeg(data, fs, **psd_kwargs)
        if freq is None:
            freq = f
        pxx_parts.append(pxx)
        rows.extend(
            {"Subject": subject, "Session": session, "ChannelName": name, "SamplingRate": fs}
            for name in channel_names
        )

    df = pd.DataFrame(rows)
    df[['ContactName1', 'ContactName2']] = df['ChannelName'].str.split('-', n=1, expand=True)
    return df, freq, np.vstack(pxx_parts)


def load_or_compute_channel_psd(
    raw_root: str,
    cache_path,
    stage: str = 'W',
    duration_s: float | None = None,
    **psd_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    :func:`compute_original_psd`, cached against its inputs.

    The cache holds the channel table, frequency axis and PSD (a few MB) with a key
    built from every recording's relative path, size and modification time plus
    ``stage``, ``duration_s``, ``psd_kwargs`` and the filter implementation
    (``PSD_FILTER``). Any change to the recordings, parameters or filter invalidates it, so the figures are never built from data that no
    longer matches the source.

    Parameters
    ----------
    raw_root : str
        Root of the original recordings (see :func:`list_original_data_files`).
    cache_path
        Pickle file of the cache.
    stage : str
        Vigilance state: ``'W'``, ``'N2'`` or ``'N3'``.
    duration_s : float | None
        Analysed length in seconds (see :func:`compute_original_psd`).
    **psd_kwargs
        Forwarded to :func:`preprocess_and_compute_psd_ieeg` (e.g. ``PSD_PARAMS``).

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        Channel table, frequencies and
        PSD, as :func:`compute_original_psd`.
    """
    cache_path = Path(cache_path)
    files = list_original_data_files(raw_root, stage)
    key = {
        "files": [(os.path.relpath(fp, raw_root), os.stat(fp).st_size, os.stat(fp).st_mtime_ns) for fp in files],
        "stage": stage,
        "duration_s": duration_s,
        "psd_params": psd_kwargs,
        "filter": PSD_FILTER,
    }
    if cache_path.exists():
        with open(cache_path, 'rb') as fh:
            cached = pickle.load(fh)
        if cached.get("key") == key:
            logger.info(f"Loading cached stage-{stage} channel PSD from {cache_path}")
            return cached["channels"], cached["freq"], cached["pxx"]
        logger.info(f"Channel PSD cache {cache_path} is stale (recordings or parameters changed); recomputing.")
    else:
        logger.info(f"Channel PSD cache not found; computing from {len(files)} stage-{stage} recordings in {raw_root}.")
    channels, freq, pxx = compute_original_psd(raw_root, stage=stage, duration_s=duration_s, **psd_kwargs)
    with open(cache_path, 'wb') as fh:
        pickle.dump({"key": key, "channels": channels, "freq": freq, "pxx": pxx}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Channel PSD saved to {cache_path}.")
    return channels, freq, pxx


def load_sensitivity_info(
    root_dir: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Load per-hemisphere **signed** contact sensitivity maps from electroMICA
    leadfield derivatives, plus the surface vertex areas needed to threshold them.

    electroMICA stores one leadfield ``.mat`` per hemisphere (hemi-L, hemi-R), each
    holding a ``ContactSensitivityMap`` of shape (n_contacts, 32492) -- the signed
    leadfield (potential per unit normal-oriented cortical dipole, Vm/A) on that
    hemisphere's fsLR-32k midthickness surface. The sign is physically meaningful:
    a bipolar channel's sensitivity is the *difference* of its two contacts' signed
    maps (``ComputeFeatureMaps``), so this function keeps the raw signed values and
    the two hemispheres separate. Rectification, thresholding and the bipolar
    difference are deferred to :func:`build_bipolar_sensitivity`, which combines
    contact pairs the way electroMICA does.

    Parameters
    ----------
    root_dir : str
        Root directory containing electroMICA derivatives.
        Expected layout:
        ``<root_dir>/sub-PX*/ses-01/model/
          *_leadfield_hemi-{L,R}_space-nativepro_surf-fsLR-32k_label-midthickness.mat``

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - df with one row per (Subject, Session, ContactName):
            - Subject, Session
            - ContactName — physical electrode identifier (upper-cased);
              distinct from bipolar ``ChannelName`` ("LCi1" vs "LCi1-LCi2").
            - Sens_L, Sens_R — signed sensitivity arrays of shape (32492,) on
              the LH and RH surfaces (zeros where that hemisphere's file lacks
              the contact).
        - areas: mapping ``(Subject, Session) -> {"L": areas_L, "R": areas_R}``
          of per-vertex surface areas (shape (32492,)) from each hemisphere's
          midthickness surface, used to normalise sensitivity to a per-area
          density before thresholding.
    """
    pattern = os.path.join(root_dir, "sub-PX*", "ses-01", "model", "*_leadfield_hemi-*_space-nativepro_surf-fsLR-32k_label-midthickness.mat")
    mat_files = glob.glob(pattern)

    pat_sub = re.compile(r"sub-(PX\d+)")
    pat_ses = re.compile(r"ses-(\d+)")
    pat_hemi = re.compile(r"hemi-(L|R)")

    # (subject, session, contact) -> {"Sens_L": arr, "Sens_R": arr}
    contacts: dict = {}
    areas: dict = {}

    for filepath in mat_files:
        match_sub = pat_sub.search(filepath)
        match_ses = pat_ses.search(filepath)
        match_hemi = pat_hemi.search(filepath)

        if match_sub is None or match_ses is None or match_hemi is None:
            logger.warning(f"Skipping leadfield {filepath}: cannot parse subject/session/hemisphere from its path.")
            continue

        subject, session, hemi = (
            match_sub.group(1),
            match_ses.group(1),
            match_hemi.group(1),
        )

        try:
            mat = sio.loadmat(filepath, simplify_cells=True)
        except (OSError, ValueError) as e:
            logger.warning(f"Skipping leadfield {os.path.basename(filepath)}: unreadable ({type(e).__name__}: {e}).")
            continue

        missing = {"ContactName", "ContactSensitivityMap"} - set(mat)
        if missing:
            logger.warning(f"Skipping leadfield {os.path.basename(filepath)}: missing {sorted(missing)}.")
            continue

        contact_names = [str(c).strip().upper() for c in mat["ContactName"]]
        sensitivity = np.asarray(mat["ContactSensitivityMap"], dtype=float)

        if sensitivity.ndim != 2:
            raise ValueError(
                f"Unexpected sensitivity shape in {filepath}: {sensitivity.shape}"
            )

        if sensitivity.shape[0] != len(contact_names):
            raise ValueError(
                f"Contact count mismatch in {filepath}: "
                f"{sensitivity.shape[0]} vs {len(contact_names)} names"
            )

        # Per-vertex surface areas for this subject/hemisphere (from the leadfield's
        # own midthickness mesh), used to normalise sensitivity to a density before
        # thresholding, exactly as electroMICA does.
        if {"Vertices", "Faces"}.issubset(mat):
            areas.setdefault((subject, session), {})[hemi] = compute_vertex_areas(
                mat["Vertices"], mat["Faces"]
            )
        else:
            logger.warning(
                f"Leadfield {os.path.basename(filepath)} lacks Vertices/Faces; "
                f"sub-{subject} ses-{session} hemi-{hemi} thresholds will fall back "
                f"to unit areas (plain sensitivity threshold)."
            )

        # Keep the raw SIGNED leadfield; the bipolar difference and thresholds are
        # applied later (see build_bipolar_sensitivity).
        col = f"Sens_{hemi}"
        for name, sens in zip(contact_names, sensitivity):
            contacts.setdefault((subject, session, name), {})[col] = sens

    if not contacts:
        return (
            pd.DataFrame(columns=["Subject", "Session", "ContactName", "Sens_L", "Sens_R"]),
            areas,
        )

    # One row per contact carrying both signed hemisphere maps (zeros where a
    # hemisphere's leadfield file did not include that contact).
    zeros = np.zeros(N_LH)
    rows = [
        {
            "Subject": subject,
            "Session": session,
            "ContactName": name,
            "Sens_L": maps.get("Sens_L", zeros),
            "Sens_R": maps.get("Sens_R", zeros),
        }
        for (subject, session, name), maps in contacts.items()
    ]
    df = pd.DataFrame.from_records(rows)

    return df, areas


def _threshold_channel_sensitivity(
    chan: np.ndarray,
    areas: np.ndarray,
    global_thresh: float,
    rel_thresh: float,
) -> np.ndarray:
    """Rectify a signed channel leadfield and zero sub-threshold vertices.

    Reproduces electroMICA's two thresholds, both applied to the per-area
    sensitivity *density* ``|chan| / area`` (Vm/A): a common absolute noise floor
    (``global_thresh``) and a channel-dependent relative floor equal to
    ``rel_thresh`` times the channel's "maximum" density, taken as its second-largest
    value as in electroMICA's ``ComputeFeatureMaps`` (``so[j, 1]`` of the descending
    sort). Retained values are returned as
    magnitudes (the projection weights); vertices below either floor are 0.
    NaN leadfield values are treated as zero sensitivity, so they neither
    propagate nor disable the channel's thresholds.

    Parameters
    ----------
    chan
        Signed channel leadfield, shape (n_channels, n_vertices).
    areas
        Per-vertex surface areas, shape (n_vertices,), broadcast over channels.
    global_thresh
        Absolute density floor (electroMICA ``GlobalTresh``, 0.001).
    rel_thresh
        Relative density floor fraction (electroMICA ``ChanTresh``, 0.05).

    Returns
    -------
    np.ndarray
        Non-negative thresholded sensitivity, shape of ``chan``.
    """
    mag = np.abs(np.nan_to_num(chan, nan=0.0))
    density = mag / (areas + 1e-12)
    # electroMICA ComputeFeatureMaps: so = sort(|sens| / area)[:, ::-1]; threshold at
    # so[j, 1] * ChanTresh, i.e. the second-largest density (partial sort here).
    second_max = np.partition(density, -2, axis=1)[:, -2]
    thresh_density = np.maximum(global_thresh, second_max * rel_thresh)[:, None]
    mag[density < thresh_density] = 0.0
    return mag


def build_bipolar_sensitivity(
    df_channels: pd.DataFrame,
    df_sensitivity: pd.DataFrame,
    areas: dict,
    *,
    global_thresh: float = 0.001,
    rel_thresh: float = 0.05,
) -> np.ndarray:
    """Assemble bipolar-channel surface sensitivities the electroMICA way.

    Each bipolar channel is paired with its two contacts' signed leadfields (looked
    up from ``df_sensitivity`` by ``ContactName``), and its sensitivity is the
    **signed difference** of those leadfields, computed per hemisphere, then
    thresholded (see :func:`_threshold_channel_sensitivity`) and rectified. The two
    hemispheres are finally folded onto a single fsLR-32k template by summing their
    magnitudes, ``|L1_LH - L2_LH| + |L1_RH - L2_RH|`` -- a deliberate coverage choice
    (contacts on either hemisphere inform the homologous template vertex), applied
    *after* the per-hemisphere signed difference so opposite-sign contacts do not
    cancel prematurely. This differs from electroMICA proper (which keeps hemispheres
    separate) only in that final fold. Channels are processed per (Subject, Session)
    so each mesh's per-vertex area vector is applied once (broadcast over the group's
    channels) rather than tiled per channel.

    Parameters
    ----------
    df_channels
        One row per bipolar channel, with columns ``Subject``,
        ``Session``, ``ContactName1``, ``ContactName2`` (contact names matching
        ``df_sensitivity``).
    df_sensitivity
        Per-contact signed maps from :func:`load_sensitivity_info`,
        with columns ``Subject``, ``Session``, ``ContactName``, ``Sens_L``,
        ``Sens_R`` (each shape (32492,)). Contacts absent here contribute zero;
        a warning reports, per (Subject, Session), how many contacts and
        channels this affects.
    areas
        ``(Subject, Session) -> {"L": areas_L, "R": areas_R}`` per-vertex
        surface areas from :func:`load_sensitivity_info`.
    global_thresh
        Absolute density noise floor (Vm/A).
    rel_thresh
        Channel-relative density floor fraction.

    Returns
    -------
    np.ndarray
        Non-negative bipolar sensitivity, shape (n_channels, 32492),
        row-aligned with ``df_channels``.
    """
    zeros = (np.zeros(N_LH), np.zeros(N_LH))
    lookup = {(r.Subject, r.Session, r.ContactName): (r.Sens_L, r.Sens_R)
              for r in df_sensitivity.itertuples(index=False)}

    df_channels = df_channels.reset_index(drop=True)
    out = np.zeros((len(df_channels), N_LH))
    for (subject, session), grp in df_channels.groupby(["Subject", "Session"], sort=False):
        area = areas.get((subject, session), {})

        names = pd.unique(grp[["ContactName1", "ContactName2"]].to_numpy().ravel())
        missing = [n for n in names if (subject, session, n) not in lookup]
        if missing:
            n_hit = int((grp["ContactName1"].isin(missing) | grp["ContactName2"].isin(missing)).sum())
            logger.warning(
                f"sub-{subject} ses-{session}: {len(missing)}/{len(names)} contacts have no "
                f"leadfield (treated as zero); {n_hit}/{len(grp)} channels affected."
            )

        # Signed contact leadfields, looked up once per group: (Sens_L, Sens_R) per channel.
        maps1 = [lookup.get((subject, session, n), zeros) for n in grp["ContactName1"]]
        maps2 = [lookup.get((subject, session, n), zeros) for n in grp["ContactName2"]]
        out[grp.index.to_numpy()] = sum(
            _threshold_channel_sensitivity(
                np.vstack([m[i] for m in maps1]) - np.vstack([m[i] for m in maps2]),
                area.get(hemi, np.ones(N_LH)), global_thresh, rel_thresh,
            )
            for i, hemi in enumerate(("L", "R"))
        )
    return out


def project_channel_measure(channel_vals: np.ndarray, sens: np.ndarray) -> np.ndarray:
    """
    Sensitivity-weighted projection of per-channel values onto the surface.

    Each surface vertex receives the sensitivity-weighted average of the per-channel
    values (the electroMICA smoothed feature map); vertices with no sensitivity
    (uncovered) are returned as NaN.

    Parameters
    ----------
    channel_vals
        Per-channel values, shape (n_channels,) for a scalar measure or
        (k, n_channels) for k values per channel (e.g. a spectrum).
    sens
        Bipolar sensitivity maps, shape (n_channels, n_vertices)
        (see :func:`build_bipolar_sensitivity`; NaN zeroed).

    Returns
    -------
    np.ndarray
        Shape (n_vertices,) or (k, n_vertices).
    """
    sens_sum = np.sum(sens, axis=0)
    surf = (channel_vals @ sens) / (sens_sum + 1e-12)
    surf[..., sens_sum == 0] = np.nan
    return surf


def mirror_to_cortex(template_vals: np.ndarray, df_surf: pd.DataFrame, network: str | None = None) -> np.ndarray:
    """
    Show a folded template-hemisphere map on both fsLR-32k hemispheres.

    The bipolar sensitivities fold both hemispheres onto one fsLR-32k template
    (:func:`build_bipolar_sensitivity`), so template vertex i stands for left vertex i
    and its homologue, right vertex ``N_LH + i``. The value is placed on both, kept on
    labelled cortex (or on ``network`` vertices when given) and NaN elsewhere.

    Parameters
    ----------
    template_vals
        Per-template-vertex values, shape (N_LH,).
    df_surf
        Surface table over both hemispheres with ``hemisphere`` and
        ``network`` columns (e.g. ``df_yeo_surf``).
    network
        Yeo network to keep; ``None`` keeps all labelled cortex.

    Returns
    -------
    np.ndarray
        Shape (2 * N_LH,), LH then RH.
    """
    full = np.tile(np.asarray(template_vals, dtype=float), 2)
    keep = df_surf['hemisphere'].notna().to_numpy()
    if network is not None:
        keep &= (df_surf['network'] == network).to_numpy()
    full[~keep] = np.nan
    return full


def preprocess_and_compute_psd_ieeg(
    data: np.ndarray,
    fs: float,
    fmin: float = 0.5,
    fmax: float = 80.0,
    fs_target: float = 200.0,
    filter_order: int = 4,
    window_sec: float = 2.0,
    overlap_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full iEEG preprocessing and PSD computation.

    Following the MNI Open iEEG Atlas procedure, the pipeline consists of:
    - Band-pass filtering (zero-phase Butterworth, as second-order sections)
    - Downsample to a target sampling rate
    - Demeaning
    - Welch PSD estimation
    - Frequency-range restriction
    - Power normalization

    Parameters
    ----------
    data
        iEEG data of shape (..., n_samples), where the last axis is time.
    fs
        Original sampling frequency in Hz.
    fmin
        Minimum frequency for band-pass filter and PSD in Hz.
    fmax
        Maximum frequency for band-pass filter and PSD in Hz.
    fs_target
        Target sampling frequency after downsampling in Hz.
    filter_order
        Order of the Butterworth band-pass filter (applied forward and backward).
    window_sec
        Length of each segment for Welch's method in seconds.
    overlap_sec
        Overlap between segments for Welch's method in seconds.

    Returns
    -------
    freq : np.ndarray
        Frequencies within [fmin, fmax] in Hz.
    pxx : np.ndarray
        Normalized PSD of shape (..., n_frequencies), summing to 1 along the last axis.
    """
    # Band-pass filter as cascaded second-order sections: in transfer-function (b, a) form a
    # 0.5 Hz corner at kHz sampling rates is ill-conditioned (at 2,000 Hz its gain at the
    # corner is -5 dB instead of -3 dB).
    sos = butter(filter_order, [fmin / (fs / 2), fmax / (fs / 2)], btype="band", output="sos")
    data = sosfiltfilt(sos, data, axis=-1)

    # Downsample
    if fs != fs_target:
        g = np.gcd(int(fs), int(fs_target))
        data = resample_poly(data, int(fs_target) // g, int(fs) // g, axis=-1)
        fs = fs_target

    # Demean
    data -= data.mean(axis=-1, keepdims=True)

    # Welch PSD
    freq, pxx = welch(
        data,
        fs=fs,
        window="hamming",
        nperseg=int(window_sec * fs),
        noverlap=int(overlap_sec * fs),
        axis=-1,
    )

    # Restrict frequency range
    mask = (freq >= fmin) & (freq <= fmax)
    pxx = pxx[..., mask]

    # Normalize power
    pxx /= np.sum(pxx, axis=-1, keepdims=True) + 1e-12

    return freq[mask], pxx  # type: ignore[return-value]


def _zero_gap_segments(x: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """``[start, stop)`` of the data between the zero gaps of one channel (see `welch_psd_segments`)."""
    zero = np.r_[False, x == 0, False].astype(int)
    edges = np.flatnonzero(np.diff(zero))
    starts, stops = edges[::2], edges[1::2]  # runs of exact zeros, [start, stop)
    gap = ((stops - starts) >= min_gap) | (starts == 0) | (stops == x.size)
    bounds = np.r_[0, np.column_stack((starts[gap], stops[gap])).ravel(), x.size].reshape(-1, 2)
    return [(int(a), int(b)) for a, b in bounds if b > a]


def welch_psd_segments(
    data: np.ndarray,
    fs: float,
    fmin: float = 0.5,
    fmax: float = 80.0,
    window_sec: float = 2.0,
    overlap_sec: float = 1.0,
    min_gap_s: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Welch PSD of recordings made of contiguous segments separated by zero gaps.

    The MNI Open iEEG Atlas distributes, per channel and vigilance state, artifact-free
    segments concatenated with 2-s buffers of zeros and zero-padded at the end (to 68 s
    in wakefulness). A Welch window spanning a buffer would mix a step discontinuity into
    the spectrum, so each channel is split at its gaps -- runs of exact zeros lasting at
    least ``min_gap_s``, or reaching either end of the recording (the padding) -- and the
    Welch periodogram (Hamming window, constant detrend, as in
    :func:`preprocess_and_compute_psd_ieeg`) is averaged over the windows that lie
    entirely within one segment. Segments shorter than one window are not used. The
    spectrum is then restricted to ``[fmin, fmax]`` and normalised to unit sum, as in
    :func:`preprocess_and_compute_psd_ieeg`. No filtering is applied: the recordings are
    expected at their analysis rate, already filtered.

    Parameters
    ----------
    data
        Recordings of shape (n_channels, n_samples) or (n_samples,).
    fs
        Sampling frequency in Hz.
    fmin, fmax
        Frequency range of the returned spectrum in Hz.
    window_sec
        Length of each Welch window in seconds.
    overlap_sec
        Overlap between Welch windows in seconds.
    min_gap_s
        Shortest run of zeros, in seconds, that separates two segments.

    Returns
    -------
    freq : np.ndarray
        Frequencies within [fmin, fmax] in Hz.
    pxx : np.ndarray
        Normalised PSD of shape (n_channels, n_frequencies), the mean of the
        within-segment window periodograms; NaN for a channel without a full window.
    n_windows : np.ndarray
        Number of Welch windows of each channel, shape (n_channels,).
    """
    data = np.atleast_2d(np.asarray(data, dtype=float))
    nperseg, noverlap = int(window_sec * fs), int(overlap_sec * fs)
    step = nperseg - noverlap
    freq = np.fft.rfftfreq(nperseg, 1.0 / fs)
    keep = (freq >= fmin) & (freq <= fmax)
    pxx = np.zeros((data.shape[0], int(keep.sum())))
    n_windows = np.zeros(data.shape[0], dtype=int)
    for ch, x in enumerate(data):
        for start, stop in _zero_gap_segments(x, int(round(min_gap_s * fs))):
            if stop - start < nperseg:
                continue
            n_win = (stop - start - nperseg) // step + 1
            _, p = welch(x[start:stop], fs=fs, window="hamming", nperseg=nperseg, noverlap=noverlap)
            pxx[ch] += n_win * p[keep]
            n_windows[ch] += n_win
    with np.errstate(invalid="ignore", divide="ignore"):
        pxx /= n_windows[:, None]
    pxx /= np.sum(pxx, axis=-1, keepdims=True) + 1e-12
    return freq[keep], pxx, n_windows


def compute_spectral_parameters(
    pxx: np.ndarray,
    freq: np.ndarray,
    bands: dict[str, tuple[float, float]] | None = None,
    fmin: float = SPEC_FIT_RANGE[0],
    fmax: float = SPEC_FIT_RANGE[1],
    aperiodic_mode: str = "knee",
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.05,
    peak_threshold: float = 2.0,
    nonnegative_knee: bool = True,
    return_model: bool = False,
) -> dict:
    """
    Parameterise power spectra into an aperiodic exponent and oscillatory band power.

    A single ``specparam`` fit (formerly FOOOF; Donoghue et al., 2020) decomposes each
    spectrum into an aperiodic component $L(f) = b - \\log_{10}(k + f^{\\chi})$ and a
    set of Gaussian oscillatory peaks, from which two non-redundant measures are
    returned: the aperiodic exponent $\\chi$ and, per requested band, the power of the
    strongest oscillatory peak *above the aperiodic fit*. Measured relative to the
    aperiodic fit, rather than integrated over the raw PSD, band power is not confounded
    by differences in the exponent (Donoghue et al., 2020). The settings
    follow the specparam 2.0 guidance: one model and one set of settings for every
    spectrum; a fit range (``SPEC_FIT_RANGE``, 1--45 Hz) that excludes the line-noise
    notch of the recordings and the transition band of the 80 Hz low-pass, since
    filtered-out frequencies should not be fitted; and, in ``'knee'`` mode, the knee
    bounded to $k \\ge 0$ (specparam's ``ap_bounds`` setting), so a spectrum without a
    knee in the fit range reduces to the fixed power law ($k = 0$) instead of taking a
    non-physical negative knee. The per-channel unit-sum PSD normalisation only shifts
    the aperiodic offset, leaving both measures unchanged.

    Parameters
    ----------
    pxx
        PSD array of shape (n_spectra, n_frequencies) or (n_frequencies,); the
        last axis is frequency. Power is supplied in linear units (``specparam``
        log-transforms internally).
    freq
        Frequency axis in Hz, shape (n_frequencies,).
    bands
        Mapping of band name to ``(fmin, fmax)`` in Hz for the oscillatory
        band-power readout (e.g. ``SPEC_BANDS``); every band must lie within the
        fit range. If ``None`` only the aperiodic parameters are returned.
    fmin
        Lower bound of the fitting range in Hz (the lowest bins are excluded
        because the high-pass makes them unreliable).
    fmax
        Upper bound of the fitting range in Hz.
    aperiodic_mode
        ``specparam`` aperiodic mode, ``'knee'`` or ``'fixed'``.
    peak_width_limits
        (min, max) Gaussian peak bandwidth in Hz; specparam
        recommends a lower limit of about twice the frequency resolution.
    max_n_peaks
        Maximum number of oscillatory peaks fitted per spectrum.
    min_peak_height
        Minimum peak height above the aperiodic fit (log10 power).
    peak_threshold
        Relative peak-detection threshold, in standard deviations of
        the flattened spectrum.
    nonnegative_knee
        In ``'knee'`` mode, bound the knee to $k \\ge 0$.
    return_model
        Also return the fitted ``SpectralGroupModel`` (for fit checks,
        e.g. :func:`plot_fit_quality`).

    Returns
    -------
    dict
        - ``'exponent'``: aperiodic exponent $\\chi$ (positive for $1/f$-like
          spectra), shape ``pxx.shape[:-1]``; NaN where the fit failed to converge.
        - ``'offset'``: aperiodic offset $b$ (log10 power), same shape.
        - ``'knee'``: aperiodic knee $k$, same shape; 0 in ``'fixed'`` mode, for
          which $L(f) = b - \\log_{10}(f^{\\chi})$ is the knee formula with $k = 0$.
          With ``'offset'`` and ``'exponent'`` it gives the fitted aperiodic
          component through :func:`aperiodic_component`.
        - ``'knee_freq'``: knee frequency $k^{1/\\chi}$ in Hz (specparam's
          ``compute_knee_frequency``; Gao et al., 2020), same shape; a value below
          ``fmin`` places the knee below the fit range. NaN in ``'fixed'`` mode.
        - ``'r_squared'``, ``'mae'``: goodness of fit ($R^2$) and mean absolute
          error (log10 power) of each fit, same shape.
        - ``'n_peaks'``: number of oscillatory peaks fitted per spectrum, same shape.
        - ``'band_power'``: ``{name: array}`` of oscillatory peak power for each
          requested band, same shape as ``'exponent'``; NaN where no peak was
          detected in the band (i.e. no oscillation) or the fit failed.
        - ``'report'``: specparam's methods report (code version, algorithm,
          settings, fit range), as text.
        - ``'model'``: the fitted ``SpectralGroupModel``, if ``return_model``.

    Raises
    ------
    ImportError
        If ``specparam`` is not installed.
    ValueError
        If a band extends outside the fit range ``[fmin, fmax]``.
    """
    try:
        from specparam import SpectralGroupModel
        from specparam.data.periodic import get_band_peak_group
        from specparam.params.aperiodic import compute_knee_frequency
        from specparam.reports.methods import methods_report_info
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "compute_spectral_parameters requires the `specparam` package "
            "(pip install specparam). It is a hard dependency of the iEEG spectral "
            "analysis."
        ) from exc

    outside = {name: rng for name, rng in (bands or {}).items() if rng[0] < fmin or rng[1] > fmax}
    if outside:
        raise ValueError(f"Bands {outside} extend outside the fit range [{fmin:g}, {fmax:g}] Hz; "
                         f"no peak can be fitted there (use bands clipped to the fit range, e.g. SPEC_BANDS).")

    out_shape = np.shape(pxx)[:-1]
    spectra = np.atleast_2d(pxx).astype(float)
    algorithm_settings = None
    if aperiodic_mode == "knee" and nonnegative_knee:
        # (offset, knee, exponent) bounds: only the knee is restricted, to k >= 0.
        algorithm_settings = {"ap_bounds": ((-np.inf, 0.0, -np.inf), (np.inf, np.inf, np.inf))}
    fg = SpectralGroupModel(
        peak_width_limits=list(peak_width_limits),
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        aperiodic_mode=aperiodic_mode,
        metrics=["error_mae", "gof_rsquared"],
        algorithm_settings=algorithm_settings,
        verbose=False,
    )
    fg.fit(freq, spectra, freq_range=[fmin, fmax])
    exponent = fg.get_params("aperiodic", "exponent").reshape(out_shape)
    offset = fg.get_params("aperiodic", "offset").reshape(out_shape)
    if aperiodic_mode == "knee":
        knee = fg.get_params("aperiodic", "knee").reshape(out_shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            knee_freq = np.asarray(compute_knee_frequency(knee, exponent), dtype=float)
    else:
        knee, knee_freq = np.zeros(out_shape), np.full(out_shape, np.nan)

    # Peak table rows are [CF, PW, BW, spectrum index]; count the peaks of each spectrum.
    peak_table = np.atleast_2d(fg.get_params("periodic"))
    n_peaks = np.zeros(spectra.shape[0], dtype=int)
    if peak_table.size:
        valid_rows = np.isfinite(peak_table[:, 0])
        n_peaks = np.bincount(peak_table[valid_rows, -1].astype(int), minlength=spectra.shape[0])

    band_power = {}
    for name, (lo, hi) in (bands or {}).items():
        # get_band_peak_group -> (n_spectra, 3) [centre freq, power, bandwidth] of the
        # strongest peak in the band; column 1 is the peak power above the aperiodic fit.
        peaks = np.atleast_2d(get_band_peak_group(fg, (lo, hi)))
        band_power[name] = peaks[:, 1].reshape(out_shape)

    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        methods_report_info(fg, concise=True)
    # The specparam report lists neither the fit modes nor the aperiodic bounds.
    report.write(f"Fit modes: aperiodic '{aperiodic_mode}', periodic 'gaussian'; aperiodic bounds: "
                 f"{'knee >= 0' if algorithm_settings else 'none'}; metrics: error_mae, gof_rsquared\n")

    out = {"exponent": exponent, "offset": offset, "knee": knee, "knee_freq": knee_freq.reshape(out_shape),
           "r_squared": np.asarray(fg.results.get_metrics("gof", "rsquared"), dtype=float).reshape(out_shape),
           "mae": np.asarray(fg.results.get_metrics("error", "mae"), dtype=float).reshape(out_shape),
           "n_peaks": n_peaks.reshape(out_shape), "band_power": band_power, "report": report.getvalue()}
    if return_model:
        out["model"] = fg
    return out


def aperiodic_component(
    freq: np.ndarray,
    offset: np.ndarray,
    knee: np.ndarray,
    exponent: np.ndarray,
) -> np.ndarray:
    """
    Fitted aperiodic component $L(f) = b - \\log_{10}(k + f^{\\chi})$ in log10 power.

    Evaluates the ``specparam`` aperiodic model from the parameters returned by
    :func:`compute_spectral_parameters`; subtracting it from the log10 power spectrum
    leaves the periodic (oscillatory) residual, i.e. the flattened spectrum.

    Parameters
    ----------
    freq
        Frequencies in Hz, shape (n_frequencies,).
    offset
        Aperiodic offset $b$, any shape ``S``.
    knee
        Aperiodic knee $k$ (0 for ``'fixed'`` mode), shape ``S``.
    exponent
        Aperiodic exponent $\\chi$, shape ``S``.

    Returns
    -------
    np.ndarray
        Aperiodic component, shape ``S + (n_frequencies,)``.
    """
    offset, knee, exponent = (np.asarray(a, dtype=float)[..., None] for a in (offset, knee, exponent))
    return offset - np.log10(knee + np.asarray(freq, dtype=float) ** exponent)


def fit_quality_summary(spec: dict, fmin: float = SPEC_FIT_RANGE[0], mask: np.ndarray | None = None) -> str:
    """One-line description of the fits returned by :func:`compute_spectral_parameters`.

    Reports, over the converged fits (optionally restricted by ``mask``): the median and
    interquartile range of $R^2$ and the number of fits with $R^2 < 0.9$ (descriptive,
    no exclusion), the median mean absolute error, the mean number of peaks per fit (a
    high value can indicate over-fitting), and the median knee frequency with the number
    of fits whose knee lies below the fit range (``< fmin``).

    Parameters
    ----------
    spec
        Output of :func:`compute_spectral_parameters`.
    fmin
        Lower bound of the fit range in Hz.
    mask
        Optional boolean selection of the spectra.

    Returns
    -------
    str
        The summary.
    """
    keep = np.isfinite(spec["exponent"]) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(spec["exponent"]))
    r2, mae, kf = spec["r_squared"][keep], spec["mae"][keep], spec["knee_freq"][keep]
    text = (f"{int(keep.sum())} fits; R^2 median {np.median(r2):.3f} (IQR {np.percentile(r2, 25):.3f}-"
            f"{np.percentile(r2, 75):.3f}), R^2 < 0.9 in {int((r2 < 0.9).sum())}; MAE median {np.median(mae):.3f}; "
            f"peaks per fit mean {np.mean(spec['n_peaks'][keep]):.2f}")
    if np.isfinite(kf).any():
        text += (f"; knee frequency median {np.nanmedian(kf):.1f} Hz (IQR {np.nanpercentile(kf, 25):.1f}-"
                 f"{np.nanpercentile(kf, 75):.1f}), below the fit range (< {fmin:g} Hz) in {int((kf < fmin).sum())}")
    return text


def hemisphere_slice(hemi: str) -> slice:
    """Rows of one hemisphere in an fsLR-32k array or table over both (left first).

    Parameters
    ----------
    hemi : {'LH', 'RH'}

    Returns
    -------
    rows : slice
        ``N_LH`` rows starting at 0 (LH) or ``N_LH`` (RH).
    """
    start = N_LH if hemi == 'RH' else 0
    return slice(start, start + N_LH)


def covered_vertices(sens: np.ndarray, mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Template vertices of `mask` that at least one channel covers and where `values` is finite.

    Parameters
    ----------
    sens : (n_channels, n_vertices) array
        Non-negative channel sensitivity maps (`build_bipolar_sensitivity`).
    mask : (n_vertices,) bool
        Candidate vertices, e.g. a network on the template hemisphere.
    values : (n_vertices,) array
        Map that must be defined at the vertex, e.g. the MPC gradient.

    Returns
    -------
    vertices : (n_covered,) int array
        Indices in increasing order.
    """
    return np.flatnonzero(np.asarray(mask, dtype=bool) & (np.sum(sens, axis=0) > 0) & np.isfinite(values))


def load_channel_sensitivity(df_channels: pd.DataFrame, root_dir: str) -> np.ndarray:
    """Bipolar sensitivity map of every channel, from the electroMICA leadfields.

    Loads the contact leadfields (`load_sensitivity_info`), matches the channels'
    contacts to them by upper-cased name (electroMICA's convention), and combines each
    pair into the channel's thresholded, hemisphere-folded sensitivity
    (`build_bipolar_sensitivity`). Channels whose contacts have no leadfield, or no
    suprathreshold sensitivity, carry zero weight; their number is logged.

    Parameters
    ----------
    df_channels : pd.DataFrame
        One row per bipolar channel with ``Subject``, ``Session``, ``ContactName1`` and
        ``ContactName2`` (e.g. from `compute_original_psd`).
    root_dir : str
        Root of the electroMICA derivatives (see `load_sensitivity_info`).

    Returns
    -------
    sens : (n_channels, N_LH) array
        Non-negative sensitivity, NaN set to 0, row-aligned with `df_channels`.
    """
    df_sensitivity, areas = load_sensitivity_info(root_dir=str(root_dir))
    logger.info(f"Sensitivity maps loaded: {df_sensitivity['Subject'].nunique()} subjects, "
                f"{len(df_sensitivity)} contacts")
    channels = df_channels.assign(ContactName1=df_channels['ContactName1'].str.upper(),
                                  ContactName2=df_channels['ContactName2'].str.upper())
    sens = np.nan_to_num(build_bipolar_sensitivity(channels, df_sensitivity, areas), nan=0.0)
    n_no_sens = int((sens.sum(axis=1) == 0).sum())
    if n_no_sens:
        logger.warning(f"{n_no_sens}/{len(sens)} channels have no suprathreshold sensitivity; "
                       f"they carry no weight in the surface projections.")
    return sens


def gaussian_channel_sensitivity(
    points_lh: np.ndarray,
    points_rh: np.ndarray,
    channel_vertex: np.ndarray,
    half_weight_mm: float = 4.0,
    rel_thresh: float = 0.05,
) -> np.ndarray:
    """Gaussian surface footprint of every channel, folded onto one fsLR-32k template.

    For channels located by a position only (no leadfield), the spatial sensitivity is
    modelled as a Gaussian of the Euclidean distance $d$ from the channel's vertex,
    $w(d) = \\exp(-\\ln 2 \\, d^2 / h^2)$, which halves at $d = h$
    (``half_weight_mm``). Distances are taken between the vertices of the channel's own
    hemisphere; weights below a fraction $r$ (``rel_thresh``) of the peak (1, at the
    channel's vertex) are set to zero, as the relative threshold of the electroMICA
    sensitivities (:func:`build_bipolar_sensitivity`), so each footprint ends at
    $d = h \\sqrt{\\log_2(1 / r)}$. A left vertex $i$ and a right
    vertex ``N_LH + i`` both map to template vertex $i$, the hemisphere fold of
    :func:`build_bipolar_sensitivity`, so the result is used in place of
    :func:`load_channel_sensitivity`.

    Parameters
    ----------
    points_lh, points_rh : (N_LH, 3) array
        Vertex coordinates (mm) of the left and right fsLR-32k surfaces, e.g. the
        midthickness.
    channel_vertex : (n_channels,) int array
        Vertex of each channel on both hemispheres joined (left first).
    half_weight_mm : float
        Distance $h$ (mm) at which the weight is one half.
    rel_thresh : float
        Weight below which a vertex is set to zero.

    Returns
    -------
    sens : (n_channels, N_LH) array
        Non-negative footprint of each channel on the template.

    Raises
    ------
    ValueError
        If a surface does not have ``N_LH`` vertices.
    """
    points = tuple(np.asarray(p, dtype=float) for p in (points_lh, points_rh))
    if any(p.shape != (N_LH, 3) for p in points):
        raise ValueError(f"Expected two fsLR-32k hemispheres of shape ({N_LH}, 3), got "
                         f"{[p.shape for p in points]}.")
    trees = tuple(cKDTree(p) for p in points)
    radius = half_weight_mm * np.sqrt(np.log2(1.0 / rel_thresh))
    sens = np.zeros((len(channel_vertex), N_LH))
    for i, v in enumerate(np.asarray(channel_vertex, dtype=int)):
        hemi, vertex = divmod(int(v), N_LH)
        idx = np.asarray(trees[hemi].query_ball_point(points[hemi][vertex], radius), dtype=int)
        d2 = np.sum((points[hemi][idx] - points[hemi][vertex]) ** 2, axis=1)
        sens[i, idx] = np.exp(-np.log(2.0) * d2 / half_weight_mm ** 2)
    sens[sens < rel_thresh] = 0.0
    return sens

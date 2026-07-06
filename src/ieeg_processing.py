import glob
import logging
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.integrate import simpson
from scipy.signal import butter, filtfilt, resample_poly, welch

from vtkmodules.vtkFiltersSources import vtkSphereSource

logger = logging.getLogger(__name__)

# fsLR-32k vertices per hemisphere.
N_LH = 32492


def compute_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric (hat-function) per-vertex surface areas of a triangular mesh.

    Each vertex is assigned one third of the area of every triangle it belongs to
    -- the integral over the surface of the piecewise-linear hat function that is 1
    at the vertex and 0 at its neighbours. This matches the ``areasv`` routine used
    by electroMICA (``ComputeFeatureMaps``) to normalise channel sensitivities to a
    per-area density before thresholding.

    Args:
        vertices: Vertex coordinates, shape (n_vertices, 3), in millimetres.
        faces: Triangle vertex indices, shape (n_faces, 3). MATLAB-style 1-based
            indexing is detected and converted to 0-based automatically.

    Returns:
        np.ndarray: Per-vertex area, shape (n_vertices,).
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    if f.min() == 1:  # MATLAB 1-based faces
        f = f - 1
    tri = v[f]  # (n_faces, 3, 3)
    tri_area = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    areas = np.zeros(v.shape[0])
    for k in range(3):
        np.add.at(areas, f[:, k], tri_area / 3.0)
    return areas


def load_original_data_files(
    root: str = '/host/verges/tank/data/BIDS_iEEG/original',
) -> pd.DataFrame:
    """
    Load original iEEG MATLAB files and return bipolar channel-level data.

    Each row corresponds to one **bipolar channel** from one subject/session
    pair. In the electroMICA terminology, a *channel* is a differential
    recording between two physical *contacts*: ``ChannelName`` stores the pair
    (e.g. ``"LCi1-LCi2"``), while ``ContactName1`` and ``ContactName2`` hold
    the individual contact identifiers that index into the electroMICA
    leadfield/sensitivity files.

    Args:
        root (str): Root directory of the BIDS iEEG dataset. Expected layout:
            ``<root>/sub-PX*/ses-01/*stage-W.mat``

    Returns:
        pd.DataFrame: Bipolar channel data with columns:
            - Subject
            - Session
            - ChannelName    — bipolar pair label (e.g. ``"LCi1-LCi2"``)
            - SamplingRate
            - Data           — raw signal array, shape (n_samples,)
            - ContactName1   — first contact of the bipolar pair (upper element)
            - ContactName2   — second contact of the bipolar pair (lower element)
    """
    pattern = re.compile(r"sub-(PX\d+)/ses-(\d+)")
    files = glob.glob(f"{root}/sub-PX*/ses-01/*stage-W.mat")

    rows = []

    for filepath in files:
        match = pattern.search(filepath)
        if match is None:
            continue

        subject, session = match.groups()

        mat = sio.loadmat(filepath, simplify_cells=True)
        required_keys = {"ChannelName", "Data", "SamplingRate"}
        if not required_keys.issubset(mat):
            continue

        channel_names = [str(c) for c in mat["ChannelName"]]
        fs = float(mat["SamplingRate"])
        data = np.asarray(mat["Data"])

        if data.ndim != 2:
            raise ValueError(f"Unexpected data shape in {filepath}: {data.shape}")

        # Enforce (n_channels, n_samples)
        if data.shape[0] != len(channel_names):
            if data.shape[1] == len(channel_names):
                data = data.T
            else:
                raise ValueError(
                    f"Channel count mismatch in {filepath}: "
                    f"{data.shape} vs {len(channel_names)} names"
                )

        for ch_name, ch_data in zip(channel_names, data):
            rows.append(
                {
                    "Subject": subject,
                    "Session": session,
                    "ChannelName": ch_name,
                    "SamplingRate": fs,
                    "Data": ch_data,
                }
            )
    df = pd.DataFrame(rows)
    df[['ContactName1', 'ContactName2']] = df['ChannelName'].str.split('-', n=1, expand=True)
    return df


def load_channel_info(root_dir: str = '/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA') -> pd.DataFrame:
    """
    Load channel information from electroMICA ChannelMap TSV files and the
    corresponding GIFTI surface maps.

    electroMICA distinguishes between *contacts* (physical electrodes, used in
    sensitivity/leadfield files) and *channels* (bipolar recordings, one per row
    here). ``ChannelNumber`` in the TSV indexes into the GIFTI channel map where
    each vertex stores the channel ID it belongs to.

    Surface vertex indices are returned in a **combined-hemisphere convention**:
    LH vertices are numbered 0–32491 and RH vertices 32492–64983, consistent
    with the 64984-vertex fsLR-32k whole-brain surface used elsewhere in this
    pipeline (contrast ``load_sensitivity_info`` /
    ``build_bipolar_sensitivity``, which fold the two hemispheres onto a single
    32k template after the per-hemisphere bipolar difference).

    Returns
    -------
    pd.DataFrame
        Columns:
        - Subject, Session, ChannelName, ChannelNumber
        - ChannelIndices_lh : list of int, LH vertex indices in [0, 32491]
        - ChannelIndices_rh : list of int, RH vertex indices in [32492, 64983]
    """
    # Constants for surface offsets (Conte69 / fs_LR 32k)
    N_VERTS_LH = 32492 

    tsv_pattern = os.path.join(root_dir, "sub-PX*", "ses-01", "feat", "*_ChannelMap.tsv")
    tsv_files = glob.glob(tsv_pattern)

    if not tsv_files:
        logger.warning('No ChannelMap TSV files found.')
        return pd.DataFrame(columns=["Subject", "Session", "ChannelName", "ChannelNumber",
                                     "ChannelIndices_lh", "ChannelIndices_rh"])

    pat_sub = re.compile(r"sub-(PX\d+)")
    pat_ses = re.compile(r"ses-(\d+)")
    
    all_records = []

    # Helper Function to Extract Indices
    def extract_indices(gii_files, channel_numbers, offset=0):
        """
        Loads GIFTI, extracts vertices for each channel, and adds offset.
        Returns a list of lists (one list of indices per channel).
        """
        # Handle missing files gracefully
        if not gii_files:
            return [[] for _ in channel_numbers]
        
        try:
            img = nib.load(gii_files[0])
            data = img.darrays[0].data
        except Exception as e:
            logger.debug('Error loading %s: %s', gii_files[0], e)
            return [[] for _ in channel_numbers]

        # Case A: 1D ROI Map (Value at vertex = Channel Number)
        if data.ndim == 1:
            # OPTIMIZATION: Instead of scanning the array N times (slow),
            # we group vertices by channel ID once using pandas.
            # Create a Series mapping VertexIndex -> ChannelNum
            # Only keep non-zero values
            mask = data > 0
            df_map = pd.DataFrame({
                'vertex': np.where(mask)[0] + offset,
                'channel': data[mask]
            })
            
            # Group by channel to get lists of vertices
            grouped = df_map.groupby('channel')['vertex'].apply(list).to_dict()
            
            # Map back to the requested channel_numbers list
            return [grouped.get(float(ch), []) for ch in channel_numbers]

        # Case B: 2D Matrix (Vertices x Channels)
        # Assuming column index corresponds to channel number (1-based)
        elif data.ndim == 2:
            indices_list = []
            for ch in channel_numbers:
                col_idx = int(ch) - 1
                if 0 <= col_idx < data.shape[1]:
                    # Find non-zero vertices and add offset
                    idxs = np.where(data[:, col_idx] > 0)[0] + offset
                    indices_list.append(idxs.tolist())
                else:
                    indices_list.append([])
            return indices_list
        
        return [[] for _ in channel_numbers]

    # Process Each TSV File
    for tsv_file in tsv_files:
        # Load Metadata
        try:
            df_meta = pd.read_csv(tsv_file, sep="\t")
        except Exception:
            continue

        # Basic Cleanup
        df_meta["ChannelName"] = df_meta["ChannelName"].astype(str).str.upper()
        if "ChannelNumber" not in df_meta.columns:
            logger.warning('Skipping %s: Missing ChannelNumber', tsv_file)
            continue

        # Extract Subject/Session info
        match_sub = pat_sub.search(tsv_file)
        match_ses = pat_ses.search(tsv_file)
        subject = match_sub.group(1) if match_sub else "Unknown"
        session = match_ses.group(1) if match_ses else "Unknown"
        
        df_meta["Subject"] = subject
        df_meta["Session"] = session

        deriv_root = os.path.join(root_dir, f"sub-{subject}", f"ses-{session}", "maps")
        
        files_lh = glob.glob(os.path.join(deriv_root, "*_hemi-L_*_surf-fsLR-32k_*.gii"))
        files_rh = glob.glob(os.path.join(deriv_root, "*_hemi-R_*_surf-fsLR-32k_*.gii"))

        # LH vertex indices stay in [0, N_VERTS_LH-1]; RH gets offset N_VERTS_LH
        # so that combined indexing covers the full 64984-vertex surface.
        df_meta["ChannelIndices_lh"] = extract_indices(files_lh, df_meta["ChannelNumber"], offset=0)
        df_meta["ChannelIndices_rh"] = extract_indices(files_rh, df_meta["ChannelNumber"], offset=N_VERTS_LH)

        all_records.append(df_meta)

    # Combine all records
    if not all_records:
        return pd.DataFrame(columns=["Subject", "Session", "ChannelName", "ChannelNumber",
                                     "ChannelIndices_lh", "ChannelIndices_rh"])
    df_all = pd.concat(all_records, ignore_index=True)
    
    # Reorder columns
    cols = ["Subject", "Session", "ChannelName", "ChannelNumber", "ChannelIndices_lh", "ChannelIndices_rh"]
    return df_all[cols]


def load_sensitivity_info(
    root_dir: str = '/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA',
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

    Args:
        root_dir (str): Root directory containing electroMICA derivatives.
            Expected layout:
            ``<root_dir>/sub-PX*/ses-01/model/
              *_leadfield_hemi-{L,R}_space-nativepro_surf-fsLR-32k_label-midthickness.mat``

    Returns:
        tuple[pd.DataFrame, dict]:
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
            continue

        subject, session, hemi = (
            match_sub.group(1),
            match_ses.group(1),
            match_hemi.group(1),
        )

        try:
            mat = sio.loadmat(filepath, simplify_cells=True)
        except (OSError, ValueError):
            continue

        required_keys = {"ContactName", "ContactSensitivityMap"}
        if not required_keys.issubset(mat):
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
    ``rel_thresh`` times the channel's second-largest density (a single-vertex-
    outlier-robust proxy for the maximum). Retained values are returned as
    magnitudes (the projection weights); vertices below either floor are 0.

    Args:
        chan: Signed channel leadfield, shape (n_channels, n_vertices).
        areas: Per-vertex surface areas, shape (n_channels, n_vertices).
        global_thresh: Absolute density floor (electroMICA ``GlobalTresh``, 0.001).
        rel_thresh: Relative density floor fraction (electroMICA ``ChanTresh``, 0.05).

    Returns:
        np.ndarray: Non-negative thresholded sensitivity, shape of ``chan``.
    """
    mag = np.abs(chan)
    density = mag / (areas + 1e-12)
    # electroMICA `so[:, 1]`: the channel's second-largest area-normalised density.
    second_max = np.sort(density, axis=1)[:, -2]
    thresh_density = np.maximum(global_thresh, second_max * rel_thresh)[:, None]
    mag[density < thresh_density] = 0.0
    return mag


def build_bipolar_sensitivity(
    df_channels: pd.DataFrame,
    areas: dict,
    *,
    global_thresh: float = 0.001,
    rel_thresh: float = 0.05,
) -> np.ndarray:
    """Assemble bipolar-channel surface sensitivities the electroMICA way.

    For each bipolar channel the sensitivity is the **signed difference of its two
    contacts' leadfields**, computed per hemisphere, then thresholded (see
    :func:`_threshold_channel_sensitivity`) and rectified. The two hemispheres are
    finally folded onto a single fsLR-32k template by summing their magnitudes,
    ``|L1_LH - L2_LH| + |L1_RH - L2_RH|`` -- a deliberate coverage choice (contacts
    on either hemisphere inform the homologous template vertex), applied *after* the
    per-hemisphere signed difference so opposite-sign contacts do not cancel
    prematurely. This differs from electroMICA proper (which keeps hemispheres
    separate) only in that final fold.

    Args:
        df_channels: One row per bipolar channel, with columns ``Subject``,
            ``Session`` and the four signed contact maps ``Sens1_L``, ``Sens1_R``,
            ``Sens2_L``, ``Sens2_R`` (each shape (32492,); missing entries treated
            as zero).
        areas: ``(Subject, Session) -> {"L": areas_L, "R": areas_R}`` per-vertex
            surface areas from :func:`load_sensitivity_info`.
        global_thresh: Absolute density noise floor (Vm/A).
        rel_thresh: Channel-relative density floor fraction.

    Returns:
        np.ndarray: Non-negative bipolar sensitivity, shape (n_channels, 32492),
        row-aligned with ``df_channels``.
    """
    zeros = np.zeros(N_LH)

    def _stack(col: str) -> np.ndarray:
        return np.vstack([v if isinstance(v, np.ndarray) else zeros
                          for v in df_channels[col]])

    def _areas(hemi: str) -> np.ndarray:
        # Fall back to unit areas if a subject's mesh was unavailable, so the
        # density thresholds degrade to a plain sensitivity threshold.
        return np.vstack([areas.get((s, ss), {}).get(hemi, np.ones(N_LH))
                          for s, ss in zip(df_channels["Subject"], df_channels["Session"])])

    chan_L = _stack("Sens1_L") - _stack("Sens2_L")
    chan_R = _stack("Sens1_R") - _stack("Sens2_R")
    sens_L = _threshold_channel_sensitivity(chan_L, _areas("L"), global_thresh, rel_thresh)
    sens_R = _threshold_channel_sensitivity(chan_R, _areas("R"), global_thresh, rel_thresh)
    return sens_L + sens_R


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
    - Band-pass filtering
    - Downsample to a target sampling rate
    - Demeaning
    - Welch PSD estimation
    - Frequency-range restriction
    - Power normalization

    Args:
        data: iEEG data of shape (..., n_samples), where the last axis is time.
        fs: Original sampling frequency in Hz.
        fmin: Minimum frequency for band-pass filter and PSD in Hz.
        fmax: Maximum frequency for band-pass filter and PSD in Hz.
        fs_target: Target sampling frequency after downsampling in Hz.
        filter_order: Order of the Butterworth band-pass filter.
        window_sec: Length of each segment for Welch's method in seconds.
        overlap_sec: Overlap between segments for Welch's method in seconds.

    Returns:
        freq (np.ndarray): Frequencies within [fmin, fmax] in Hz.
        pxx (np.ndarray): Normalized PSD of shape (..., n_frequencies), summing to 1 along the last axis.
    """
    # Band-pass filter
    b, a = butter(filter_order, [fmin / (fs / 2), fmax / (fs / 2)], btype="band")
    data = filtfilt(b, a, data, axis=-1)

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


def extract_band_power(pxx_raw: np.ndarray, freq: np.ndarray, band: tuple[float, float], relative: bool = True) -> np.ndarray:
    """
    Integrate PSD over a frequency band and return log10 power.

    Args:
        pxx_raw: PSD array of shape (..., n_frequencies).
        freq: Frequency axis in Hz, shape (n_frequencies,).
        band: (fmin, fmax) band limits in Hz.
        relative: If True, divide band power by total power before log.

    Returns:
        Log10 band power, shape (...,).
    """
    idx_band = (freq >= band[0]) & (freq <= band[1])
    bp = simpson(pxx_raw[..., idx_band], x=freq[idx_band], axis=-1)
    if relative:
        bp /= simpson(pxx_raw, x=freq, axis=-1) + 1e-12
    return np.log10(bp + 1e-12)


def compute_spectral_parameters(
    pxx: np.ndarray,
    freq: np.ndarray,
    bands: dict[str, tuple[float, float]] | None = None,
    fmin: float = 1.0,
    fmax: float = 80.0,
    aperiodic_mode: str = "knee",
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.05,
) -> dict:
    """
    Parameterise power spectra into an aperiodic exponent and oscillatory band power.

    A single ``specparam`` fit (formerly FOOOF; Donoghue et al., 2020) decomposes each
    spectrum into an aperiodic component $L(f) = b - \\log_{10}(k + f^{\\chi})$ and a
    set of Gaussian oscillatory peaks, from which two non-redundant measures are
    returned: the aperiodic exponent $\\chi$ and, per requested band, the power of the
    strongest oscillatory peak *above the aperiodic fit*. Deriving band power from the
    periodic component (rather than integrating the raw PSD) makes it orthogonal to the
    exponent, so the two measures do not re-encode the same 1/f change. Fitting in
    ``'knee'`` mode suits the broadband iEEG range (0.5--80 Hz), where the aperiodic
    component bends at low frequency, and the per-channel unit-sum PSD normalisation
    only shifts the aperiodic offset, leaving both measures unchanged.

    Args:
        pxx: PSD array of shape (n_spectra, n_frequencies) or (n_frequencies,); the
            last axis is frequency. Power is supplied in linear units (``specparam``
            log-transforms internally).
        freq: Frequency axis in Hz, shape (n_frequencies,).
        bands: Mapping of band name to ``(fmin, fmax)`` in Hz for the oscillatory
            band-power readout; if ``None`` only the exponent is returned.
        fmin: Lower bound of the fitting range in Hz (the lowest bins are excluded
            because filter roll-off makes them unreliable).
        fmax: Upper bound of the fitting range in Hz.
        aperiodic_mode: ``specparam`` aperiodic mode, ``'knee'`` or ``'fixed'``.
        peak_width_limits: (min, max) Gaussian peak bandwidth in Hz; the lower limit
            must exceed twice the frequency resolution.
        max_n_peaks: Maximum number of oscillatory peaks fitted per spectrum.
        min_peak_height: Minimum peak height above the aperiodic fit (log power).

    Returns:
        dict with:
            - ``'exponent'``: aperiodic exponent $\\chi$ (positive for $1/f$-like
              spectra), shape ``pxx.shape[:-1]``; NaN where the fit failed to converge.
            - ``'band_power'``: ``{name: array}`` of oscillatory peak power for each
              requested band, same shape as ``'exponent'``; NaN where no peak was
              detected in the band (i.e. no oscillation) or the fit failed.

    Raises:
        ImportError: If ``specparam`` is not installed.
    """
    try:
        from specparam import SpectralGroupModel
        from specparam.data.periodic import get_band_peak_group
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "compute_spectral_parameters requires the `specparam` package "
            "(pip install specparam). It is a hard dependency of the iEEG spectral "
            "analysis."
        ) from exc

    out_shape = np.shape(pxx)[:-1]
    spectra = np.atleast_2d(pxx).astype(float)
    fg = SpectralGroupModel(
        peak_width_limits=list(peak_width_limits),
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        aperiodic_mode=aperiodic_mode,
        verbose=False,
    )
    fg.fit(freq, spectra, freq_range=[fmin, fmax])
    exponent = fg.get_params("aperiodic", "exponent").reshape(out_shape)

    band_power = {}
    for name, (lo, hi) in (bands or {}).items():
        # get_band_peak_group -> (n_spectra, 3) [centre freq, power, bandwidth] of the
        # strongest peak in the band; column 1 is the peak power above the aperiodic fit.
        peaks = np.atleast_2d(get_band_peak_group(fg, (lo, hi)))
        band_power[name] = peaks[:, 1].reshape(out_shape)

    return {"exponent": exponent, "band_power": band_power}


def compute_gradient_quantiles(
    df_surf: pd.DataFrame,
    channel_indices: np.ndarray,
    gradient_col: str,
    quantiles: tuple[float, float] = (0.25, 0.75),
) -> np.ndarray:
    """Assign gradient quantile labels to channels.

    Marks the surface vertices covered by *channel_indices* as bottom-quantile
    (-1) or top-quantile (+1) based on their gradient value, writing the result
    into a ``'quantiles'`` column on *df_surf* in-place.  Returns the
    per-channel quantile label array.

    Args:
        df_surf: Surface DataFrame containing *gradient_col*.
        channel_indices: Integer vertex indices of each channel on the 32k surface.
        gradient_col: Name of the gradient column in *df_surf*.
        quantiles: (low, high) thresholds as fractions (default: 25th/75th percentile).

    Returns:
        np.ndarray: Quantile label per channel (-1, 0, or 1; NaN where unassigned).
    """
    low_q, high_q = np.nanquantile(df_surf[gradient_col], list(quantiles))
    channel_mask = np.zeros(len(df_surf), dtype=bool)
    channel_mask[channel_indices] = True
    df_surf.loc[channel_mask & (df_surf[gradient_col] <= low_q), "quantiles"] = -1
    df_surf.loc[channel_mask & (df_surf[gradient_col] >= high_q), "quantiles"] = 1
    return df_surf["quantiles"].iloc[channel_indices].values


def plot_surface_sphere(p, channel_position: list | np.ndarray, channel_color: np.ndarray, screenshot_path) -> None:
    # renderer index → extra Z rotation applied after the standard -90X/+90Z pair
    renderers = [(p.renderers[0][0], 0), (p.renderers[1][0], 180)]
    for renderer, extra_z in renderers:
        for pos, color in zip(channel_position, channel_color):
            sphere = vtkSphereSource()
            sphere.SetCenter(*pos)
            sphere.SetRadius(1.5)
            sphere.Update()
            actor = renderer.AddActor()
            actor.SetMapper(inputData=sphere.GetOutput())
            actor.GetProperty().SetColor(*color[:3])
            actor.GetProperty().SetOpacity(1.0)
            actor.RotateX(-90)
            actor.RotateZ(90)
            if extra_z:
                actor.RotateZ(extra_z)
    p.screenshot(screenshot_path, transparent_bg=True)
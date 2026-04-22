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
    pipeline (cf. ``load_sensitivity_info``, which folds both hemispheres into
    a single 32k space using bilateral summation).

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
    *,
    threshold: float = 0.001,
) -> pd.DataFrame:
    """
    Load and aggregate surface-based contact sensitivity maps from electroMICA
    leadfield derivatives.

    electroMICA stores one leadfield .mat file per hemisphere (hemi-L, hemi-R),
    each containing a ContactSensitivityMap of shape (n_contacts, 32492) for
    the fsLR-32k surface of that hemisphere. This function sums the LH and RH
    maps element-wise, yielding a single (32492,) bilateral sensitivity vector
    per contact.

    **Bilateral-sum design rationale**: fsLR-32k is a bilaterally symmetric
    template, so vertex index *i* on LH and vertex *i* on RH occupy
    approximately homologous cortical positions. Summing both hemispheres into
    the same 32k vertex space means that channels whose primary sensitivity lies
    on the "other" hemisphere still contribute to the surface projection,
    effectively doubling the number of channels that inform any given vertex.
    This is appropriate for group-level analyses where contacts are distributed
    across both hemispheres and the goal is maximal coverage.

    Contacts whose summed map is entirely zero after thresholding are excluded.

    Args:
        root_dir (str): Root directory containing electroMICA derivatives.
            Expected layout:
            ``<root_dir>/sub-PX*/ses-01/model/
              *_leadfield_hemi-{L,R}_space-nativepro_surf-fsLR-32k_label-midthickness.mat``
        threshold (float): Minimum absolute sensitivity value retained before
            hemisphere summation. Vertices below this value are set to zero.

    Returns:
        pd.DataFrame: One row per unique (Subject, Session, ContactName) with
            columns:
            - Subject
            - Session
            - ContactName  — physical electrode identifier (electroMICA
              ``ContactName``), upper-cased; distinct from bipolar
              ``ChannelName`` (e.g. "LCi1" vs "LCi1-LCi2").
            - ContactSensitivityMap — bilateral sensitivity array of shape
              (32492,), computed as the element-wise sum of the LH and RH
              fsLR-32k sensitivity maps for that contact.
    """
    pattern = os.path.join(root_dir, "sub-PX*", "ses-01", "model", "*_leadfield_hemi-*_space-nativepro_surf-fsLR-32k_label-midthickness.mat")
    mat_files = glob.glob(pattern)

    pat_sub = re.compile(r"sub-(PX\d+)")
    pat_ses = re.compile(r"ses-(\d+)")
    pat_hemi = re.compile(r"hemi-(L|R)")

    records = []

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
        sensitivity = np.asarray(mat["ContactSensitivityMap"])

        if sensitivity.ndim != 2:
            raise ValueError(
                f"Unexpected sensitivity shape in {filepath}: {sensitivity.shape}"
            )

        if sensitivity.shape[0] != len(contact_names):
            raise ValueError(
                f"Contact count mismatch in {filepath}: "
                f"{sensitivity.shape[0]} vs {len(contact_names)} names"
            )

        # Rectify, threshold, and drop contacts whose map is entirely zero
        sensitivity = np.abs(sensitivity)
        sensitivity[sensitivity < threshold] = 0.0
        active = sensitivity.any(axis=1)
        contact_names = np.asarray(contact_names)[active]
        sensitivity = sensitivity[active]

        for name, sens in zip(contact_names, sensitivity):
            records.append(
                {
                    "Subject": subject,
                    "Session": session,
                    "ContactName": name,
                    "Hemi": hemi,
                    "ContactSensitivityMap": sens,
                }
            )

    if not records:
        return pd.DataFrame(columns=["Subject", "Session", "ContactName", "ContactSensitivityMap"])
    df = pd.DataFrame.from_records(records)

    # Sum LH and RH sensitivity maps element-wise into a single (32492,) bilateral
    # vector per contact. Sort by Hemi first so the order is deterministic (L then R)
    # before stacking, which matters if any caller inspects per-hemisphere contributions.
    df = df.sort_values("Hemi")
    df = (df.groupby(["Subject", "Session", "ContactName"], as_index=False, sort=False)
          .agg(ContactSensitivityMap=("ContactSensitivityMap", lambda x: np.sum(np.stack(x.tolist()), axis=0))))

    return df


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


def compute_psd_vectorized(data: np.ndarray, fs: float, fmin: float = 0.5, fmax: float = 80.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute relative PSD for all channels simultaneously (no preprocessing).

    Unlike ``preprocess_and_compute_psd_ieeg``, this function skips filtering,
    downsampling, and demeaning — use it when the data are already preprocessed.

    Args:
        data: Array of shape (n_channels, n_times).
        fs: Sampling frequency in Hz.
        fmin: Lower frequency bound in Hz.
        fmax: Upper frequency bound in Hz.

    Returns:
        f_band: Frequencies within [fmin, fmax], shape (n_frequencies,).
        pxx_rel: PSD normalised by total power, shape (n_channels, n_frequencies).
    """
    f, pxx = welch(data, fs=fs, nperseg=int(2 * fs), noverlap=int(fs), window="hamming", axis=-1)
    mask = (f >= fmin) & (f <= fmax)
    pxx_rel = pxx[..., mask] / (np.sum(pxx, axis=-1, keepdims=True) + 1e-12)
    return f[mask], pxx_rel


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
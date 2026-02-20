import numpy as np
import pandas as pd
import scipy.io as sio
import glob
import re
import os
from scipy.signal import butter, filtfilt, resample_poly, welch
from scipy.integrate import simpson
import nibabel as nib
import matplotlib.pyplot as plt



def load_original_data_files(
    root: str = "/host/verges/tank/data/BIDS_iEEG/original",
):
    """
    Load original iEEG MATLAB files and return channel-level data.

    Each row in the returned DataFrame corresponds to one channel from one
    subject/session pair.

    Args:
        root (str): Root directory of the BIDS iEEG dataset.

    Returns:
        pd.DataFrame: Channel-level iEEG data with columns:
            - Subject
            - Session
            - ChannelName
            - SamplingRate
            - Data
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
    df['ContactName1'] = df['ChannelName'].str.split('-').str[0]
    df['ContactName2'] = df['ChannelName'].str.split('-').str[1]
    return df


def load_channel_info(root_dir="/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA"):
    """
    Load channel information from BIDS-iEEG channel TSV files and
    surface-based channel maps.

    Returns
    -------
    pd.DataFrame
        Columns include ChannelIndices_lh and ChannelIndices_rh with
        offsets applied for combined surface indexing (LH=0-32k, RH=32k+).
    """
    # Constants for surface offsets (Conte69 / fs_LR 32k)
    N_VERTS_LH = 32492 

    tsv_pattern = os.path.join(root_dir, "sub-PX*", "ses-01", "feat", "*_ChannelMap.tsv")
    tsv_files = glob.glob(tsv_pattern)

    if not tsv_files:
        print("No ChannelMap TSV files found.")
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
            print(f"Error loading {gii_files[0]}: {e}")
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
            print(f"Skipping {tsv_file}: Missing 'ChannelNumber'")
            continue

        # Extract Subject/Session info
        match_sub = pat_sub.search(tsv_file)
        match_ses = pat_ses.search(tsv_file)
        subject = match_sub.group(1) if match_sub else "Unknown"
        session = match_ses.group(1) if match_ses else "Unknown"
        
        df_meta["Subject"] = subject
        df_meta["Session"] = session

        # Define GIFTI paths
        # Using wildcard lookup to be safe against minor naming variations
        base_path = os.path.dirname(tsv_file).replace("feat", "maps")
        # Construct the prefix based on file structure assumptions
        # (You may need to adjust the path replacement logic if folder structure varies)
        deriv_root = os.path.join(root_dir, f"sub-{subject}", f"ses-{session}", "maps")
        
        files_lh = glob.glob(os.path.join(deriv_root, "*_hemi-L_*_surf-fsLR-32k_*.gii"))
        files_rh = glob.glob(os.path.join(deriv_root, "*_hemi-R_*_surf-fsLR-32k_*.gii"))

        # Extract Indices
        # FIX: LH gets offset 0. RH gets offset 32492.
        df_meta["ChannelIndices_lh"] = extract_indices(files_lh, df_meta["ChannelNumber"], offset=N_VERTS_LH)
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
    root_dir: str = "/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA",
    *,
    threshold: float = 0.001,
):
    """
    Load and aggregate surface-based contact sensitivity maps.

    Each row in the returned DataFrame corresponds to one unique
    (Subject, Session, ContactName) tuple, with sensitivity maps summed
    across hemispheres when applicable.

    Args:
        root_dir (str): Root directory containing electroMICA derivatives.
        threshold (float): Minimum absolute sensitivity value retained
            in the contact sensitivity maps.

    Returns:
        pd.DataFrame: Aggregated sensitivity information with columns:
            - Subject
            - Session
            - ContactName
            - ContactSensitivityMap
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

        # Rectify and threshold
        sensitivity = np.abs(sensitivity)
        sensitivity[sensitivity < threshold] = 0.0

        for name, sens in zip(contact_names, sensitivity):
            if not np.any(sens):
                continue

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

    # Aggregate across hemispheres
    df = (df.groupby(["Subject", "Session", "ContactName"], as_index=False)
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
):
    """
    Full iEEG preprocessing and PSD computation .

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
        data, fs = resample_poly(data, fs_target // g, fs // g, axis=-1), fs_target

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

    return freq[mask], pxx


def extract_band_power(pxx_raw, freq, band, relative=True):
    """
    Integrates power in a specific band. Returns Log-Power.
    """
    # Create mask
    idx_band = np.logical_and(freq >= band[0], freq <= band[1])
    
    # Integrate raw power (Area under the curve using Simpson's rule)
    bp = simpson(pxx_raw[..., idx_band], x=freq[idx_band], axis=-1)
    
    if relative:
        # Divide by total integral of the passed spectrum
        total_power = simpson(pxx_raw, x=freq, axis=-1)
        bp /= (total_power + 1e-12)

    # Return Log10 power (Standard for physiology)
    return np.log10(bp + 1e-12)


def compute_psd_vectorized(data, fs, fmin=0.5, fmax=80.0):
    """
    Vectorized PSD calculation for iEEG.
    data: (n_channels, n_times) array
    """
    # Compute PSD across the last axis for all channels simultaneously
    f, pxx = welch(data, fs=fs, nperseg=2 * fs, noverlap=1 * fs, window="hamming", axis=-1)
    
    # Slice frequency range
    mask = (f >= fmin) & (f <= fmax)
    f_band = f[mask]
    pxx_band = pxx[..., mask]

    # Normalize by Total Power (to preserve 1/f slope info relative to total energy)
    total_power = np.sum(pxx, axis=-1, keepdims=True)
    pxx_rel = pxx_band / (total_power + 1e-12)
    
    return f_band, pxx_rel
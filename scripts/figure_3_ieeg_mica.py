from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# The maps are only for 32k surfaces and fsaverage5. The 5k surfaces are not 
# able to properly depict the sensitivity, which varies quickly in space. 
# fsnative surfaces have triangles that vary widely in size (1000 times), 
# which leads to some numerical issues, I might see if I can fix the code in the future
#
# database 
# BIDS_ieeg: 30 sessions from 29 patients
# The iEEG Data is in host/verges/tank/data/BIDS_iEEG
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import scipy.io as sio
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.colors import Normalize
import numpy as np
from matplotlib.lines import Line2D
# import pycatch22 as catch22
from joblib import Parallel, delayed
import os
from brainspace import mesh
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations, mesh_elements
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace.null_models import SpinPermutations, spin_permutations, moran
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

from brainspace.mesh.array_operations import get_parcellation_centroids
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap
from scipy.signal import welch
from scipy.signal import hilbert, butter, filtfilt
import nibabel as nib
import glob

from brainspace.mesh.mesh_elements import get_points
from brainspace.plotting import plot_surf
from vtkmodules.vtkFiltersSources import vtkSphereSource
from matplotlib import cm
from matplotlib.colors import to_rgb
import re
import matplotlib as mp

from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore

import nibabel as nib
import scipy

from lapy import TriaMesh, Solver
import pickle

import numpy as np
import scipy.stats
from scipy.stats import pearsonr, zscore
from scipy.signal import welch
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from lapy import TriaMesh, Solver
import scipy.sparse.linalg as spla

from scipy.signal import butter, filtfilt, resample_poly, welch

from figure_1_t1map import compute_t1_gradient, load_yeo_atlas, load_t1_salience_profiles, convert_states_str2int, normalize_to_range
from figure_3_ieeg import smooth_lapy, preprocess_and_compute_psd_ieeg, extract_band_power


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


def frequency_band_analysis(df_channel, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf):
    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']
    
    # A. Setup Geometry
    surf_combined = load_conte69(join=True)
    n_vertices = surf_combined.GetPoints().shape[0]
    fs = df_channel['SamplingRate'].iloc[0]
    
    # B. Define Analysis Mask: SalVent network specifically within the RH
    mask = (df_yeo_surf['hemisphere'] == 'RH') & (df_yeo_surf['network'] == 'SalVentAttn')
    mask_indices = np.where(mask)[0]

    # C. Pre-calculate Moran Weights
    w = mesh_elements.get_ring_distance(surf_combined, n_ring=1, mask=mask.values)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=100, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)

    # 1. Find the length of each signal
    lengths = [len(sig) for sig in df_channel['Data']]
    min_len, max_len = min(lengths), max(lengths)
    
    if min_len != max_len:
        print(f"Warning: Variable lengths detected ({min_len} to {max_len} samples).")
        print(f"Truncating all channels to {min_len} samples for vectorization.")
    data_matrix = np.vstack([np.asarray(sig)[:min_len] for sig in df_channel['Data']])
    
    # 3. Compute PSD
    f, pxx_raw = preprocess_and_compute_psd_ieeg(data_matrix, fs)

    # E. Process Bands
    fig, axes = plt.subplots(1, len(band_order), figsize=(20, 4), sharex=True, sharey=True)
    band_maps = {}

    # Pre-process mapping indices (Run once, use many times)
    indices_list = [np.atleast_1d(idx) for idx in df_channel['ChannelIndices']]
    indices_list = [i[np.isfinite(i)] for i in indices_list]
    counts = np.array([idx.size for idx in indices_list])
    flat_idxs = np.concatenate(indices_list).astype(np.int64)

    for i, band in enumerate(band_order):
        # 1. Extract Power and zscore
        z = extract_band_power(pxx_raw, f, freq_bands[band], relative=False)
        
        # 2. Map to Surface (Vectorized)
        flat_vals = np.repeat(z, counts)
        
        sum_per_vertex = np.bincount(flat_idxs, weights=flat_vals, minlength=n_vertices)
        hits_per_vertex = np.bincount(flat_idxs, minlength=n_vertices)

        surf_map = np.zeros(n_vertices)
        valid_verts = hits_per_vertex > 0
        surf_map[valid_verts] = zscore(sum_per_vertex[valid_verts] / hits_per_vertex[valid_verts])

        # # 3. Smooth
        # smoothed = smooth_lapy(surf_map, surf_combined, sigma=10)
        # band_maps[band] = smoothed

        # 4. Correlation Analysis
        # Extract data only within the mask
        x_raw = surf_map[mask]
        y = df_yeo_surf.loc[mask, 't1_gradient1_SalVentAttn'].values

        # Filter: Only correlate vertices that had signal (non-zero smoothed)
        # AND are finite (no NaNs from bad electrodes)
        valid_data_mask = (x_raw != 0) & np.isfinite(x_raw) & np.isfinite(y)
        
        if np.sum(valid_data_mask) < 10:
            print(f"Skipping {band}: Not enough valid vertices in mask.")
            continue

        x_clean = x_raw[valid_data_mask]
        y_clean = y[valid_data_mask]

        # Z-score for statistics
        x_stats = scipy.stats.zscore(x_clean)
        y_stats = y_clean # Gradient is usually already normalized, or zscore it too

        salience_border = df_yeo_surf['salience_border'].values.astype(float)
        surf_map = np.zeros(n_vertices)
        idx = np.flatnonzero(mask)[valid_data_mask]
        surf_map[idx] = x_stats
        surf_map[df_yeo_surf['salience_border'].values] = np.nan
        surf32k_rh_infl.append_array(surf_map[32492:], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        layout = [['rh1', 'rh2']]
        view = [['lateral', 'medial']]
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(0, 0, 0, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, return_plotter=True)
        p.show()

        # Pearson
        r, _ = spearmanr(x_stats, y_stats)

        # Null Model
        if msr is not None:
            # Note: We must randomize the FULL 'y' vector that matches 'w_masked',
            # then subset to 'valid_data_mask' for the correlation check
            y_subset_full = y # The full vector matching the weight matrix
            
            r_null = []
            # Generate surrogates for the specific mask
            for y_surr_full in msr.randomize(y_subset_full):
                # Apply same valid mask filter
                r_null.append(spearmanr(x_stats, y_surr_full[valid_data_mask])[0])
            
            r_null = np.asarray(r_null)
            p_perm = np.mean(np.abs(r_null) >= np.abs(r))
        else:
            p_perm = np.nan

        # 5. Visualization
        # Plot Scatter
        slope, intercept = np.polyfit(x_stats, y_stats, 1)
        axes[i].scatter(x_stats, y_stats, s=10, alpha=0.3, c='gray', edgecolors='none')
        axes[i].set_ylim([-1, 1])
        axes[i].plot(x_stats, slope*x_stats + intercept, c=band_colors[i], lw=2.5)
        
        stats_text = f"$r={r:.2f}$\n$p_{{perm}}={p_perm:.3f}$"
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                     va='top', fontsize=11, fontweight='bold')
        axes[i].set_title(band.capitalize(), fontsize=14)
        
        # Optional: Plot Surface map (commented out to save rendering time in loop)
        # plot_hemispheres(...)

    plt.tight_layout()
    plt.show()
    return band_maps


def frequency_band_analysis_sensitivity(df_channel, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf):
    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']
    
    # A. Setup Geometry
    surf_combined = load_conte69(join=True)
    surf_lh, surf_rh = load_conte69(join=False)
    n_vertices = surf_combined.GetPoints().shape[0]
    fs = df_channel['SamplingRate'].iloc[0]
    
    # B. Define Analysis Mask: SalVent network specifically within the RH
    mask = ((df_yeo_surf['hemisphere'] == 'RH') & (df_yeo_surf['network'] == 'SalVentAttn')).values
    print(mask.shape)
    mask_indices = np.where(mask)[0]

    # C. Pre-calculate Moran Weights
    w = mesh_elements.get_ring_distance(surf_rh, n_ring=1, mask=mask[32492:])
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=100, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)
    print(msr)

    # 1. Find the length of each signal
    lengths = [len(sig) for sig in df_channel['Data']]
    min_len, max_len = min(lengths), max(lengths)
    
    if min_len != max_len:
        print(f"Warning: Variable lengths detected ({min_len} to {max_len} samples).")
        print(f"Truncating all channels to {min_len} samples for vectorization.")
    data_matrix = np.vstack([np.asarray(sig)[:min_len] for sig in df_channel['Data']])
    
    # 3. Compute PSD
    f, pxx_raw = preprocess_and_compute_psd_ieeg(data_matrix, fs)
    sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)
    surf_map = (pxx_raw.T @ sens) / (np.sum(sens, axis=0) + 1e-12)

    # Plot all PSDs colored by gradient value
    print(df_yeo_surf)
    fig, ax = plt.subplots(figsize=(6, 4))
    import matplotlib.ticker as ticker
    grad = df_yeo_surf['t1_gradient1_SalVentAttn'].values[32492:][mask[32492:]]
    surf_map_sal = surf_map[:, mask[32492:]].T
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-1, vmax=1)
    for i in range(surf_map_sal.shape[0]): 
        ax.loglog(f, surf_map_sal[i, :], color=custom_cmap(norm(grad[i])), alpha=0.1, rasterized=True)
    surf_map_top = np.mean(surf_map[:, (df_yeo_surf['quantile_idx'] == 1).values[32492:]],axis=1)
    ax.loglog(f, surf_map_top, color='red', alpha=0.8)
    surf_map_bottom = np.mean(surf_map[:, (df_yeo_surf['quantile_idx'] == -1).values[32492:]],axis=1)
    ax.loglog(f, surf_map_bottom, color='blue', alpha=0.8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized PSD')
    xticks = [0.5, 4, 8, 13, 30, 80]
    xtick_labels = ["0.5", "4", "8", "13", "30", "80"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    for x in xticks:
        ax.axvline(x=x, color="grey", linestyle="--", alpha=0.4)
    plt.savefig("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_mica_data.svg")

    # E. Process Bands
    fig, axes = plt.subplots(1, len(band_order), figsize=(20, 4), sharex=True, sharey=True)
    band_maps = {}

    for i, band in enumerate(band_order):
        # 1. Extract Power and zscore
        z = extract_band_power(pxx_raw, f, freq_bands[band], relative=False)
        print(np.vstack(df_channel['SensitivityMap_bip'].values).shape)


        sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)
        surf_map = z @ sens / (np.sum(sens, axis=0) + 1e-12)
        print(surf_map)

        #surf_map[df_yeo_surf['salience_border'].values] = np.nan
        # surf32k_rh_infl.append_array(surf_map, name="overlay2")
        # surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        # layout = [['rh1', 'rh2']]
        # view = [['lateral', 'medial']]
        # p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
        #     nan_color=(0, 0, 0, 1), cmap="Purples", transparent_bg=True, return_plotter=True)
        # p.show()

        # # 3. Smooth
        # smoothed = smooth_lapy(surf_map, surf_combined, sigma=10)
        # band_maps[band] = smoothed

        # 4. Correlation Analysis
        # Extract data only within the mask
        x_raw = surf_map[mask[32492:]]
        y = df_yeo_surf['t1_gradient1_SalVentAttn'].values[32492:][mask[32492:]]
        print(x_raw.shape, y.shape)


        # Filter: Only correlate vertices that had signal (non-zero smoothed)
        # AND are finite (no NaNs from bad electrodes)
        valid_data_mask = (x_raw != 0) & np.isfinite(x_raw) & np.isfinite(y)
        
        # if np.sum(valid_data_mask) < 10:
        #     print(f"Skipping {band}: Not enough valid vertices in mask.")
        #     continue

        x_clean = x_raw[valid_data_mask]
        y_clean = y[valid_data_mask]

        # Z-score for statistics
        x_stats = scipy.stats.zscore(x_clean)
        y_stats = y_clean # Gradient is usually already normalized, or zscore it too

        salience_border = df_yeo_surf['salience_border'].values.astype(float)
        surf_map = np.zeros(n_vertices)
        idx = np.flatnonzero(mask)[valid_data_mask]
        surf_map[idx] = x_stats
        surf_map[df_yeo_surf['salience_border'].values] = np.nan
        surf32k_rh_infl.append_array(surf_map[32492:], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        layout = [['rh1', 'rh2']]
        view = [['lateral', 'medial']]
        screenshot_path = f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_{band}_map.svg"
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(0, 0, 0, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)

        # Pearson
        r, _ = spearmanr(x_stats, y_stats)

        # Null Model
        if msr is not None:
            # Note: We must randomize the FULL 'y' vector that matches 'w_masked',
            # then subset to 'valid_data_mask' for the correlation check
            y_subset_full = y # The full vector matching the weight matrix
            
            r_null = []
            # Generate surrogates for the specific mask
            for y_surr_full in msr.randomize(y_subset_full):
                # Apply same valid mask filter
                r_null.append(spearmanr(x_stats, y_surr_full[valid_data_mask])[0])
            
            r_null = np.asarray(r_null)
            p_perm = np.mean(np.abs(r_null) >= np.abs(r))
        else:
            p_perm = np.nan

        # 5. Visualization
        # Plot Scatter
        slope, intercept = np.polyfit(x_stats, y_stats, 1)
        axes[i].scatter(x_stats, y_stats, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_ylim([-1, 1])
        axes[i].plot(x_stats, slope*x_stats + intercept, c=band_colors[i], lw=2.5)
        
        stats_text = f"$r={r:.2f}$\n$p_{{perm}}={p_perm:.3f}$"
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                     va='top', fontsize=11, fontweight='bold')
        axes[i].set_xlabel(band.capitalize())
        axes[0].set_ylabel('MPC gradient')
        
        # Optional: Plot Surface map (commented out to save rendering time in loop)
        # plot_hemispheres(...)

    plt.tight_layout()
    plt.savefig("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_mica_data_band.svg")
    return band_maps
        



def main():
    micapipe='/local_raid/data/pbautin/software/micapipe'
    # Load fsLR-32k inflated surface
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')

    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure2_df.tsv')

    # Load sensitivity for each contact information.
    df_sensitivity = load_sensitivity_info()

    # Load channel information
    cache_path = '/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/df_channel.pkl'
    if os.path.exists(cache_path):
        print(f"Loading cached channel info from {cache_path}...")
        with open(cache_path, 'rb') as f:
            df_channel_data = pickle.load(f)
    else:
        print("Cache not found. Loading and processing channel info...")
        df_channel_data = load_original_data_files()
        with open(cache_path, 'wb') as f:
                pickle.dump(df_channel_data, f)
                print(f"Channel info saved to {cache_path}.")


    # Align sensitivity maps by contact name
    df1 = df_channel_data.merge(df_sensitivity,left_on=['Subject', 'Session', 'ContactName1'],right_on=['Subject', 'Session', 'ContactName'],how='left').rename(columns={'ContactSensitivityMap': 'Sens1'})
    df2 = df1.merge(df_sensitivity,left_on=['Subject', 'Session', 'ContactName2'],right_on=['Subject', 'Session', 'ContactName'],how='left').rename(columns={'ContactSensitivityMap': 'Sens2'})
    df2['SensitivityMap_bip'] = df2['Sens1'] - df2['Sens2']
    df2['SensitivityMap_bip'] = df2['SensitivityMap_bip'].apply(lambda x: np.abs(x) if isinstance(x, np.ndarray) else np.zeros(32492))

    #plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, df2.loc[df2.ChannelName == 'RCA1-RCA2', 'SensitivityMap_bip'].values[0], cmap='viridis', color_bar=True)
    print(df2)
    frequency_band_analysis_sensitivity(df2, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf)
    print(df_channel)
    frequency_band_analysis(df_channel, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf)
    # #df_channel_info = df_channel_info[df_channel_info.Subject == 'PX001']
    # # Explode the lists and drop NaNs
    indices_rh = df_channel["ChannelIndices"].explode()
    indices_rh = indices_rh[~indices_rh.isna()].astype(int).to_numpy()
    plt_values = np.zeros(32492, dtype=int)
    plt_values[indices_rh] = 1

    # create df_electrodes with positions
    surf_lh = read_surface('/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA/sub-PX001/ses-01/surf/sub-PX001_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii')
    surf_rh = read_surface('/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA/sub-PX001/ses-01/surf/sub-PX001_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii')

    # Append to surface
    #salience_border = nib.load('/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA/sub-PX001/ses-01/maps/sub-PX001_ses-01_ChannelMap_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness.gii').darrays[0].data
    surf_rh.append_array(plt_values, name="overlay2")
    surfs = {'rh1': surf_rh, 'rh2': surf_rh}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
                nan_color=(220, 220, 220, 1), cmap="Purples", transparent_bg=True, return_plotter=True)
    p.show()
    # Add colored spheres
    for i, pos in enumerate(df_electrode_pos[['x', 'y', 'z']].values):
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[0][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        #actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)

    # Add colored spheres
    for i, pos in enumerate(df_electrode_pos[['x', 'y', 'z']].values):
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[1][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        #actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)
        actor.RotateZ(180)
    p.show()


if __name__ == "__main__":
    main()




# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Concatenate ieeg information from all sessions and subjects from electromica,
# derivatives, and map it to the surface using the provided contact sensitivity maps.
#
# The sensitivity maps are only for 32k surfaces and fsaverage5. The 5k surfaces 
# are not able to properly depict the sensitivity, which varies quickly in space.
# fsnative surfaces have triangles that vary widely in size (1000 times),
# which leads to some numerical issues.
#
# database
# BIDS_ieeg: 31 subjects
# The iEEG Data is in host/verges/tank/data/BIDS_iEEG
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3_ieeg_mica.py \
#   -ieeg_deriv /host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA \
#   -hemi RH
# Requires df_1a_{hemi}.tsv to exist (run figure_1a_t1map.py with matching -hemi first)
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24" 
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

#### imports
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl



import seaborn as sns

from brainspace.plotting import plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import  mesh_elements
from brainspace.datasets import load_conte69, load_gradient
from brainspace.null_models import moran
from scipy.stats import spearmanr


import re
import matplotlib as mp
import matplotlib.patches as patches
from scipy.stats import zscore
from scipy.ndimage import rotate

import logging

from src.atlas_load import load_yeo_atlas, convert_states_str2int
from src.ieeg_processing import load_sensitivity_info, load_original_data_files, preprocess_and_compute_psd_ieeg, extract_band_power
from src.plot_colors import yeo7_rgba, yeo7_rgb
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)


plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False


def get_parser() -> argparse.ArgumentParser:
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Process ieeg derivatives and surfaces.",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-ieeg_deriv",
        type=str,
        help="Absolute path to the ieeg derivatives folder (e.g., /data/mica/...)"
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-hemi",
        type=str,
        default="RH",
        choices=["both", "LH", "RH"],
        help="Hemisphere for analysis: 'both', 'LH', or 'RH' (default: RH)"
    )
    return parser


def frequency_band_analysis_sensitivity(df_channel: pd.DataFrame, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf: pd.DataFrame, project_root: Path, hemi: str = 'RH') -> None:
    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']
    N_LH = 32492
    hemi_offset = N_LH if hemi == 'RH' else 0

    # Setup Geometry
    surf_combined = load_conte69(join=True)
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    surf_hemi_infl = surf32k_rh_infl if hemi == 'RH' else surf32k_lh_infl
    n_vertices = surf_combined.GetPoints().shape[0]
    fs = df_channel['SamplingRate'].iloc[0]

    # Define Analysis Mask: SalVent network for the specified hemisphere
    if hemi in ('LH', 'RH'):
        mask = ((df_yeo_surf['hemisphere'] == hemi) & (df_yeo_surf['network'] == 'SalVentAttn')).values
    else:
        mask = (df_yeo_surf['network'] == 'SalVentAttn').values

    # Find top and bottom 25% of vertices in the SalVentAttn network based on the T1 gradient
    low_q, high_q = np.nanquantile(df_yeo_surf.loc[mask, "t1_gradient1_SalVentAttn"], [0.25, 0.75])
    df_yeo_surf.loc[mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] <= low_q), "quantiles"] = -1
    df_yeo_surf.loc[mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] >= high_q), "quantiles"] = 1

    # Pre-calculate Moran Weights
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=mask[hemi_offset:hemi_offset + N_LH])
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=100, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)

    # 1. Find the length of each signal
    lengths = [len(sig) for sig in df_channel['Data']]
    min_len, max_len = min(lengths), max(lengths)
    if min_len != max_len:
        logger.warning(f"Variable lengths detected ({min_len} to {max_len} samples). Truncating all to {min_len}.")
    data_matrix = np.vstack([np.asarray(sig)[:min_len] for sig in df_channel['Data']])

    # Compute PSD
    f, pxx_raw = preprocess_and_compute_psd_ieeg(data_matrix, fs)
    sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)
    surf_map = (pxx_raw.T @ sens) / (np.sum(sens, axis=0) + 1e-12)

    # Plot all PSDs colored by gradient value
    fig, ax = plt.subplots(figsize=(6, 4))
    grad = df_yeo_surf['t1_gradient1_SalVentAttn'].values[hemi_offset:hemi_offset + N_LH][mask[hemi_offset:hemi_offset + N_LH]]
    surf_map_sal = surf_map[:, mask[hemi_offset:hemi_offset + N_LH]].T
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-1, vmax=1)
    for i in range(surf_map_sal.shape[0]):
        ax.loglog(f, surf_map_sal[i, :], color=custom_cmap(norm(grad[i])), alpha=0.1, rasterized=True)
    surf_map_top = np.nanmean(surf_map[:, (df_yeo_surf['quantiles'] == 1).values[hemi_offset:hemi_offset + N_LH]], axis=1)
    ax.loglog(f, surf_map_top, color='red', alpha=0.8)
    surf_map_bottom = np.nanmean(surf_map[:, (df_yeo_surf['quantiles'] == -1).values[hemi_offset:hemi_offset + N_LH]], axis=1)
    ax.loglog(f, surf_map_bottom, color='blue', alpha=0.8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized PSD')
    xticks = [0.5, 4, 8, 13, 30, 80]
    xtick_labels = ["0.5", "4", "8", "13", "30", "80"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    for x in xticks:
        ax.axvline(x=x, color="grey", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_psd_{hemi}.svg", bbox_inches='tight')

    # Process Bands
    fig, axes = plt.subplots(1, len(band_order), figsize=(20, 4.5), sharex=True, sharey=True)
    band_maps = {}
    for i, band in enumerate(band_order):
        # Extract Power in Band for each channel
        z = extract_band_power(pxx_raw, f, freq_bands[band], relative=False)
        sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)
        surf_map = (z @ sens) / (np.sum(sens, axis=0) + 1e-12)
        surf_map[np.sum(sens, axis=0) == 0] = np.nan

        # Plot Surface Whole Brain Sensitivity Map
        surf_map[df_yeo_surf.hemisphere.isna()[hemi_offset:hemi_offset + N_LH]] = np.nan
        surf_hemi_infl.append_array(surf_map, name="overlay2")
        surfs = {'hemi1': surf_hemi_infl, 'hemi2': surf_hemi_infl}
        layout = [['hemi1', 'hemi2']]
        view = [['lateral', 'medial']]
        screenshot_path = project_root / f"results/figures/figure_3b_ieeg_mica_sensitivity_map_{hemi}.svg"
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="Purples", transparent_bg=True, screenshot=True, filename=screenshot_path)

        
        # Plot SalVentAttn network sensitivity on surface
        surf_map_sal = surf_map.copy()
        surf_map_sal[~mask[hemi_offset:hemi_offset + N_LH]] = np.nan
        surf_map_sal = surf_map_sal[hemi_offset:hemi_offset + N_LH]
        surf_hemi_infl.append_array(surf_map_sal, name="overlay2")
        surfs = {'hemi1': surf_hemi_infl, 'hemi2': surf_hemi_infl}
        layout = [['hemi1', 'hemi2']]
        view = [['lateral', 'medial']]
        screenshot_path = project_root / f"results/figures/figure_3b_ieeg_mica_sensitivity_map_{hemi}_salience.svg"
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="Purples", transparent_bg=True, screenshot=True, filename=screenshot_path)

        
        # Correlation Analysis
        x_raw = surf_map[mask[hemi_offset:hemi_offset + N_LH]]
        y = df_yeo_surf['t1_gradient1_SalVentAttn'].values[hemi_offset:hemi_offset + N_LH][mask[hemi_offset:hemi_offset + N_LH]]
        # Filter: Only correlate vertices that had signal
        valid_data_mask = (x_raw != 0) & np.isfinite(x_raw) & np.isfinite(y)
        # Z-score for statistics
        x_stats = zscore(x_raw[valid_data_mask])
        y_stats = zscore(y[valid_data_mask])

        # Plot
        surf_map = np.zeros(n_vertices)
        idx = np.flatnonzero(mask)[valid_data_mask]
        surf_map[idx] = x_stats
        surf_map[df_yeo_surf['salience_border'].isna()] = np.nan
        surf_hemi_infl.append_array(surf_map[hemi_offset:hemi_offset + N_LH], name="overlay2")
        surfs = {'hemi1': surf_hemi_infl, 'hemi2': surf_hemi_infl}
        layout = [['hemi1', 'hemi2']]
        view = [['lateral', 'medial']]
        screenshot_path = project_root / f"results/figures/figure_3b_ieeg_mica_{band}_map_{hemi}.svg"
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, share='both',
            nan_color=(0, 0, 0, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)

        # Pearson
        r, _ = spearmanr(x_stats, y_stats)
        r_null = []
        # Generate surrogates from full-mask y (size matches w geometry), then filter to valid vertices
        for y_surr in msr.randomize(y):
            r_null.append(spearmanr(x_stats, zscore(y_surr[valid_data_mask]))[0])

        r_null = np.asarray(r_null)
        p_perm = np.mean(np.abs(r_null) >= np.abs(r))
        logger.info(f"[Figure 3B] Band {band}: power vs MPC-gradient | Spearman r={r:.3f}, Moran permutation p={p_perm:.3e} (n_perm=100, n_vertices={valid_data_mask.sum()})")

        # Plot Scatter
        slope, intercept = np.polyfit(x_stats, y_stats, 1)
        axes[i].scatter(x_stats, y_stats, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_xlim([-3, 3])
        axes[i].set_ylim([-3, 3])
        axes[i].plot(x_stats, slope*x_stats + intercept, c=band_colors[i], lw=2.5)
        axes[i].text(0.05, 0.95, f"r = {r:.2f}\np = {p_perm:.2e}", transform=axes[i].transAxes, va="top")
        axes[i].set_xlabel(band.capitalize(), color=band_colors[i], fontsize=16)
        axes[i].set_aspect("equal")
        axes[0].set_ylabel('MPC gradient', fontsize=16)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_band_power_corr_{hemi}.svg")
    return band_maps


def salience_network_electrophysiological_similarity(
    df_channel: pd.DataFrame,
    surf32k_lh_infl,
    surf32k_rh_infl,
    df_yeo_surf: pd.DataFrame,
    project_root: Path,
    hemi: str = 'RH',
) -> None:
    N_LH = 32492
    hemi_offset = N_LH if hemi == 'RH' else 0
    surf_hemi_infl = surf32k_rh_infl if hemi == 'RH' else surf32k_lh_infl
    df_hemi = df_yeo_surf.iloc[hemi_offset:hemi_offset + N_LH].reset_index(drop=True)

    # Compute channel-level PSD
    lengths = [len(sig) for sig in df_channel['Data']]
    min_len = min(lengths)
    data_matrix = np.vstack([np.asarray(sig)[:min_len] for sig in df_channel['Data']])
    fs = df_channel['SamplingRate'].iloc[0]
    _, pxx_raw = preprocess_and_compute_psd_ieeg(data_matrix, fs)  # (n_channels, n_freqs)

    # Project channel PSD to surface vertices via sensitivity-weighted average
    sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)  # (n_channels, 32492)
    sens_sum = np.sum(sens, axis=0)  # (32492,)
    surf_psd = (pxx_raw.T @ sens) / (sens_sum + 1e-12)  # (n_freqs, 32492)
    covered = sens_sum > 0
    surf_psd[:, ~covered] = np.nan

    # (32492, n_freqs) vertex-level PSD
    surf_psd_v = surf_psd.T

    # SalVentAttn vertices and T1 gradient quantiles
    sal_mask = (df_hemi['network'] == 'SalVentAttn').values
    grad = df_hemi['t1_gradient1_SalVentAttn'].values
    grad_sal_finite = grad[sal_mask & np.isfinite(grad)]
    low_q, high_q = np.nanquantile(grad_sal_finite, [0.25, 0.75])
    top_mask = sal_mask & (grad >= high_q) & covered
    bot_mask = sal_mask & (grad <= low_q) & covered

    logger.info(f"[Figure 3B] SalVentAttn top-Q vertices: {top_mask.sum()}, bottom-Q: {bot_mask.sum()}")

    # Z-score each vertex's PSD across frequencies for Pearson correlation
    psd_mean = np.nanmean(surf_psd_v, axis=1, keepdims=True)
    psd_std = np.nanstd(surf_psd_v, axis=1, keepdims=True)
    surf_psd_z = np.where(covered[:, None], (surf_psd_v - psd_mean) / (psd_std + 1e-12), 0.0)

    n_freqs = surf_psd_z.shape[1]
    P_top = surf_psd_z[top_mask]  # (n_top, n_freqs)
    P_bot = surf_psd_z[bot_mask]  # (n_bot, n_freqs)

    # Mean absolute Pearson correlation of each vertex PSD with top/bottom SalVentAttn PSDs
    A_top = np.mean(np.abs(surf_psd_z @ P_top.T) / n_freqs, axis=1)  # (32492,)
    A_bot = np.mean(np.abs(surf_psd_z @ P_bot.T) / n_freqs, axis=1)  # (32492,)

    # ES defined for non-SalVentAttn, non-medial-wall vertices with coverage
    other_mask = covered & (df_hemi['network'] != 'SalVentAttn') & (df_hemi['network'] != 'medial_wall')
    logger.info(f"[Figure 3B] Other-network covered vertices: {other_mask.sum()}")
    es_map = np.full(N_LH, np.nan)
    es_map[other_mask] = zscore(A_top[other_mask] - A_bot[other_mask])

    # FC gradient at each surface vertex (same negation convention as figure_2)
    fc_raw = load_gradient("fc", join=True)
    fc_g1_hemi = -fc_raw[hemi_offset:hemi_offset + N_LH]
    es_valid = es_map[other_mask]
    fc_valid = zscore(fc_g1_hemi[other_mask], nan_policy='omit')

    r, p_val = spearmanr(es_valid, fc_valid, nan_policy='omit')
    logger.info(f"[Figure 3B] ES vs FC-gradient | Spearman r={r:.3f}, p={p_val:.3e} (n_vertices={other_mask.sum()})")

    # Network metadata
    network_color_map = {
        'Cont': yeo7_rgba[0], 'Default': yeo7_rgba[1], 'DorsAttn': yeo7_rgba[2],
        'Limbic': yeo7_rgba[3], 'SalVentAttn': yeo7_rgba[4], 'SomMot': yeo7_rgba[5], 'Vis': yeo7_rgba[6],
    }
    networks = df_hemi['network'].values[other_mask]
    df_es = pd.DataFrame({'ES': es_valid, 'fc_g1': fc_valid, 'network': networks})
    df_es['colors'] = [network_color_map[n] for n in networks]
    df_net = (df_es.groupby('network')
              .agg(ES=('ES', 'mean'), fc_g1=('fc_g1', 'mean'))
              .reset_index().sort_values('fc_g1'))
    df_net['colors'] = [network_color_map[n] for n in df_net['network']]


    # Plot A (scatter) + B (barplot)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(df_es['ES'], df_es['fc_g1'],
                    color=np.stack(df_es['colors'].to_numpy()), s=10, alpha=0.9, rasterized=True)
    sns.regplot(x='ES', y='fc_g1', data=df_es, scatter=False, color='black',
                line_kws={'linewidth': 1}, ax=axes[0])
    axes[0].text(0.05, 0.95, f"r = {r:.2f}\np = {p_val:.2e}", transform=axes[0].transAxes, va='top')
    axes[0].set_xlabel("ES$_{top}$ - ES$_{bottom}$", fontsize=16)
    axes[0].set_ylabel('FC gradient 1', fontsize=16)
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    axes[0].set_aspect('equal')
    axes[1].barh(df_net['network'], df_net['ES'],
                 color=df_net['colors'], edgecolor='black', alpha=0.8)
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set_xlabel("Mean ES$_{top}$ - ES$_{bottom}$", fontsize=16)
    axes[1].yaxis.set_label_position('right')
    axes[1].yaxis.tick_right()
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_es_scatter_{hemi}.svg")
    plt.close()

    # Connectivity matrix sorted by network (channel-level PSD Pearson correlation)
    # Assign each channel to its peak-sensitivity vertex's network, exclude medial_wall
    peak_idx = np.argmax(sens, axis=1)  # (n_channels,)
    channel_networks = df_hemi['network'].values[peak_idx]
    valid_mask = channel_networks != 'medial_wall'
    channel_networks_valid = channel_networks[valid_mask]

    pxx_z = zscore(pxx_raw[valid_mask], axis=0)
    A_psd = np.corrcoef(pxx_z)

    sort_idx = np.argsort(channel_networks_valid)
    A_sorted = A_psd[sort_idx][:, sort_idx]
    sorted_networks = channel_networks_valid[sort_idx]

    boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0] + 1
    boundaries = np.insert(boundaries, 0, 0)
    b_ext = np.append(boundaries, len(channel_networks_valid))

    mpc_fig = A_sorted.copy()
    mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    mpc_fig = rotate(mpc_fig, angle=-45, order=0, cval=np.nan)

    fig, ax = plt.subplots()
    for i, b in enumerate(boundaries):
        net_name = sorted_networks[b]
        color = network_color_map.get(net_name, yeo7_rgb[7])
        rect = patches.Rectangle(
            (len(channel_networks_valid) / 2 * np.sqrt(2), b * np.sqrt(2)),
            b_ext[i + 1] - b_ext[i], b_ext[i + 1] - b_ext[i],
            linewidth=2, edgecolor=color, facecolor='none', angle=45
        )
        ax.add_patch(rect)

    mpc_fig[mpc_fig > 1] = 1
    plt.imshow(mpc_fig, cmap='coolwarm', origin='upper')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_corr_{hemi}.svg")
    plt.close()

    # Plot ES surface map
    surf_hemi_infl.append_array(es_map, name='es_smooth')
    surfs = {'hemi1': surf_hemi_infl, 'hemi2': surf_hemi_infl}
    layout = [['hemi1', 'hemi2']]
    view = [['lateral', 'medial']]
    plot_surf(surfs, layout=layout, view=view, array_name='es_smooth',
              size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
              nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range='sym',
              transparent_bg=True, screenshot=True,
              filename=str(project_root / f"results/figures/figure_3b_ieeg_mica_es_map_{hemi}.svg"))


def main():
    # Setup Relative Paths
    parser = get_parser()
    args = parser.parse_args()
    ieeg_deriv = args.ieeg_deriv
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_3_ieeg_mica", project_root, args)
    logger.info(f"Dataset        : MICA iEEG (BIDS_iEEG, sub-PX*, ses-01, stage-W wakefulness)")
    logger.info(f"Sensitivity maps: electroMICA leadfield, fsLR-32k surface, bipolar derivation (|Sens1 - Sens2|)")
    logger.info(f"Preprocessing  : Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned")
    logger.info(f"PSD            : Welch method, Hamming window 2s, overlap 1s, normalized to unit sum")
    logger.info(f"Frequency bands: delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, gamma 30-80 Hz")
    logger.info(f"Null model     : Moran randomization (n_rep=100, procedure=singleton, random_state=0)")
    logger.info(f"Surface space  : fsLR-32k {args.hemi}, Schaefer-400, Yeo SalVentAttn network")

    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    # load atlases
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)

    ######### Part 1 -- T1 gradient (output of figure_1a_t1map.py)
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(f"Gradient dataframe not found at {path_df_1a}. Run figure_1a_t1map.py with -hemi {args.hemi} first.")
    logger.info(f"Loading gradient dataframe from {path_df_1a}")
    df_yeo_surf = pd.read_csv(path_df_1a)

    # Load sensitivity for each contact information.
    df_sensitivity = load_sensitivity_info(root_dir=ieeg_deriv)
    logger.info(f"Sensitivity maps loaded: {df_sensitivity['Subject'].nunique()} subjects, {len(df_sensitivity)} contacts")

    # Load channel information
    cache_path = project_root / 'data/dataframes/figure_3_channel_data_df.pkl'
    if cache_path.exists():
        logger.info(f"Loading cached channel info from {cache_path}...")
        with open(cache_path, 'rb') as f:
            df_channel_data = pickle.load(f)
    else:
        logger.info("Cache not found. Loading and processing channel info...")
        df_channel_data = load_original_data_files()
        with open(cache_path, 'wb') as f:
            pickle.dump(df_channel_data, f)
        logger.info(f"Channel info saved to {cache_path}.")
    logger.info(f"Channel data: {df_channel_data['Subject'].nunique()} subjects, {len(df_channel_data)} bipolar channels")

    # Align sensitivity maps by contact name
    df_channel_data[['ContactName1', 'ContactName2']] = df_channel_data[['ContactName1', 'ContactName2']].apply(lambda c: c.str.upper())
    df1 = df_channel_data.merge(df_sensitivity, left_on=['Subject', 'Session', 'ContactName1'], right_on=['Subject', 'Session', 'ContactName'], how='left').rename(columns={'ContactSensitivityMap': 'Sens1'})
    df2 = df1.merge(df_sensitivity, left_on=['Subject', 'Session', 'ContactName2'], right_on=['Subject', 'Session', 'ContactName'], how='left').rename(columns={'ContactSensitivityMap': 'Sens2'})
    df2['SensitivityMap_bip'] = df2['Sens1'] - df2['Sens2']
    df2['SensitivityMap_bip'] = df2['SensitivityMap_bip'].map(lambda x: np.abs(x) if isinstance(x, np.ndarray) else np.zeros(32492))

    # Perform frequency band analysis and correlate with T1 gradient in the SalVentAttn network
    # frequency_band_analysis_sensitivity(df2, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi)

    # Electrophysiological similarity: compare whole-brain spectral fingerprints to SalVentAttn gradient extremes
    salience_network_electrophysiological_similarity(df2, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi)


if __name__ == "__main__":
    main()




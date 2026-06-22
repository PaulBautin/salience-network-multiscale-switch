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
import matplotlib.ticker as ticker
from scipy.stats import zscore
from scipy.ndimage import rotate

import logging

from src.atlas_load import load_yeo_atlas, convert_states_str2int, compute_network_mask
from src.ieeg_processing import load_sensitivity_info, load_original_data_files, preprocess_and_compute_psd_ieeg, extract_band_power, compute_gradient_quantiles
from src.connectome_processing import empirical_p_twosided
from src.plot_colors import yeo7_rgb, yeo7_abbrev
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)


def _orient_fc_gradient(fc_vals: np.ndarray, networks: np.ndarray, label: str = "FC gradient") -> np.ndarray:
    """Orient an FC gradient so the default-mode network sits at the low end.

    The diffusion-map eigenvector polarity is arbitrary, so the gradient is
    oriented by anatomy rather than a hardcoded sign (mirrors
    ``figure_2_distance._load_fc_gradient``): it is flipped so the default-mode
    network occupies the low (default-mode) pole and the task-positive systems
    the high pole, matching the projection reading (high P = coupling to the
    task-positive end of the FC gradient). The chosen sign is logged so a change
    in the source gradient's polarity surfaces in the logs rather than silently
    inverting results.

    Parameters
    ----------
    fc_vals : np.ndarray
        Per-vertex FC gradient values (NaN allowed off-cortex).
    networks : np.ndarray
        Per-vertex Yeo network labels, aligned with ``fc_vals``.
    label : str
        Name used in the log message.

    Returns
    -------
    np.ndarray
        FC gradient with default-mode at the low end.
    """
    finite = np.isfinite(fc_vals)
    default_mean = np.nanmean(fc_vals[finite & (networks == "Default")])
    cortical_mean = np.nanmean(fc_vals[finite])
    if np.isfinite(default_mean) and default_mean > cortical_mean:
        logger.info(f"[{label}] flipped sign so the default-mode network sits at the low end.")
        return -fc_vals
    logger.info(f"[{label}] sign kept as loaded (default-mode already at the low end).")
    return fc_vals


def _pct_color_range(values: np.ndarray, lo: float = 5.0, hi: float = 100.0):
    """Percentile colour range for a surface map (``None`` if no finite values).

    Spreads skewed data across a sequential colormap by clipping the display range
    to the [lo, hi] percentiles of the finite values, so most of the dynamic range
    is used rather than bunching near one end.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmin == vmax:
        return None
    return (float(vmin), float(vmax))


def _plot_surf_safe(*args, **kwargs):
    """``plot_surf`` wrapper that degrades gracefully when VTK rendering fails.

    Mirrors ``figure_2_distance.save_brain_map``: some VTK builds raise
    ``AttributeError`` / ``RuntimeError`` while building the colorbar lookup table.
    In that case a warning is logged and the screenshot is skipped, rather than
    aborting the whole script (the analytic results and the matplotlib figures are
    unaffected).
    """
    try:
        return plot_surf(*args, **kwargs)
    except (AttributeError, RuntimeError) as e:
        fname = kwargs.get("filename", "<unknown>")
        logger.warning(f"plot_surf rendering failed for {fname} "
                       f"({type(e).__name__}: {e}); skipping screenshot.")
        return None


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
    optional.add_argument(
        "-network",
        type=str,
        default="SalVentAttn",
        choices=["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"],
        help="Yeo 7-network to use as the analysis target (default: SalVentAttn)"
    )
    return parser


def frequency_band_analysis_sensitivity(df_channel: pd.DataFrame, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf: pd.DataFrame, project_root: Path, hemi: str = 'RH', network: str = 'SalVentAttn', n_perm: int = 1000) -> None:
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

    # Define analysis mask: target network for the specified hemisphere
    if hemi in ('LH', 'RH'):
        mask = ((df_yeo_surf['hemisphere'] == hemi) & (df_yeo_surf['network'] == network)).values
    else:
        mask = (df_yeo_surf['network'] == network).values

    gradient_col = f't1_gradient1_{network}'
    compute_gradient_quantiles(df_yeo_surf, np.where(mask)[0], gradient_col)

    # Pre-calculate Moran Weights
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=mask[hemi_offset:hemi_offset + N_LH])
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_perm, procedure='singleton', tol=1e-6, random_state=0)
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

    # Plot all PSDs coloured by the MPC gradient, with the gradient-extreme means overlaid.
    fig, ax = plt.subplots(figsize=(6, 4))
    grad = df_yeo_surf[gradient_col].values[hemi_offset:hemi_offset + N_LH][mask[hemi_offset:hemi_offset + N_LH]]
    surf_map_sal = surf_map[:, mask[hemi_offset:hemi_offset + N_LH]].T
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-1, vmax=1)
    for i in range(surf_map_sal.shape[0]):
        ax.loglog(f, surf_map_sal[i, :], color=custom_cmap(norm(grad[i])), alpha=0.1, rasterized=True)
    surf_map_top = np.nanmean(surf_map[:, (df_yeo_surf['quantiles'] == 1).values[hemi_offset:hemi_offset + N_LH]], axis=1)
    ax.loglog(f, surf_map_top, color=custom_cmap(norm(1.0)), lw=2.5, alpha=0.9, label='top 25%')
    surf_map_bottom = np.nanmean(surf_map[:, (df_yeo_surf['quantiles'] == -1).values[hemi_offset:hemi_offset + N_LH]], axis=1)
    ax.loglog(f, surf_map_bottom, color=custom_cmap(norm(-1.0)), lw=2.5, alpha=0.9, label='bottom 25%')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized PSD')
    xticks = [0.5, 4, 8, 13, 30, 80]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    for x in xticks:
        ax.axvline(x=x, color="grey", linestyle="--", alpha=0.3, zorder=0)
    ax.legend(frameon=False, loc='lower left')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_psd_{hemi}.svg", bbox_inches='tight')

    # Process Bands (per-panel size matched to figure 2a / figure 3c so the shared
    # point-size fonts render at the same proportion across figures)
    fig, axes = plt.subplots(1, len(band_order), figsize=(3.0 * len(band_order), 3.2), sharex=True, sharey=True)
    band_maps = {}
    for i, band in enumerate(band_order):
        # Extract Power in Band for each channel
        z = extract_band_power(pxx_raw, f, freq_bands[band], relative=False)
        sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)
        surf_map = (z @ sens) / (np.sum(sens, axis=0) + 1e-12)
        surf_map[np.sum(sens, axis=0) == 0] = np.nan

        # Plot Surface Whole Brain Sensitivity Map. Spread the data across the Purples
        # map via a 5th-95th-percentile colour range so the (skewed) coverage values
        # use the full dynamic range rather than bunching in the light end.
        if i == 0:  # only plot the surface map for the first band to save time and space
            surf_map[df_yeo_surf.hemisphere.isna()[hemi_offset:hemi_offset + N_LH]] = np.nan
            surf_hemi_infl.append_array(surf_map, name="overlay2")
            surfs = {'hemi1': surf_hemi_infl, 'hemi2': surf_hemi_infl}
            layout = [['hemi1', 'hemi2']]
            view = [['lateral', 'medial']]
            cov_range = _pct_color_range(surf_map)
            screenshot_path = project_root / f"results/figures/figure_3b_ieeg_mica_sensitivity_map_{hemi}.svg"
            _plot_surf_safe(surfs, layout=layout, view=view, array_name="overlay2", size=(725, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap="Purples", color_range=cov_range, transparent_bg=True, screenshot=True, filename=screenshot_path)

            
            # Plot target network sensitivity on surface
            surf_map_sal = surf_map.copy()
            surf_map_sal[~mask[hemi_offset:hemi_offset + N_LH]] = np.nan
            surf_map_sal = surf_map_sal[hemi_offset:hemi_offset + N_LH]
            surf_hemi_infl.append_array(surf_map_sal, name="overlay2")
            surfs = {'hemi1': surf_hemi_infl, 'hemi2': surf_hemi_infl}
            layout = [['hemi1', 'hemi2']]
            view = [['lateral', 'medial']]
            cov_range_sal = _pct_color_range(surf_map_sal)
            screenshot_path = project_root / f"results/figures/figure_3b_ieeg_mica_sensitivity_map_{hemi}_salience.svg"
            _plot_surf_safe(surfs, layout=layout, view=view, array_name="overlay2", size=(725, 300), zoom=1.3, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap="Purples", color_range=cov_range_sal, transparent_bg=True, screenshot=True, filename=screenshot_path)

        
        # Correlation Analysis
        x_raw = surf_map[mask[hemi_offset:hemi_offset + N_LH]]
        y = df_yeo_surf[gradient_col].values[hemi_offset:hemi_offset + N_LH][mask[hemi_offset:hemi_offset + N_LH]]
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
        _plot_surf_safe(surfs, layout=layout, view=view, array_name="overlay2", size=(725, 300), zoom=1.4, share='both',
            nan_color=(0, 0, 0, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)

        # Spearman correlation + within-network Moran spatial null (add-one empirical p).
        r, _ = spearmanr(x_stats, y_stats)
        # Generate surrogates from full-mask y (size matches w geometry), then filter to valid vertices
        r_null = np.array([spearmanr(x_stats, zscore(y_surr[valid_data_mask]))[0]
                           for y_surr in msr.randomize(y)])
        p_perm = empirical_p_twosided(r_null, r)
        logger.info(f"[Figure 3B] Band {band}: power vs MPC-gradient | Spearman r={r:+.3f}, Moran permutation p={p_perm:.3e} (n_perm={n_perm}, n_vertices={valid_data_mask.sum()})")

        # Plot Scatter (figure_1b/2a idiom: bold stats box, square, despined)
        slope, intercept = np.polyfit(x_stats, y_stats, 1)
        axes[i].scatter(x_stats, y_stats, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_xlim([-3, 3])
        axes[i].set_ylim([-3, 3])
        axes[i].plot(x_stats, slope*x_stats + intercept, c=band_colors[i], lw=2.5)
        axes[i].text(0.05, 0.95, f"$r={r:+.2f}$\n$p={p_perm:.3f}$", transform=axes[i].transAxes, va="top", fontweight="bold", fontsize=12)
        axes[i].set_xlabel(band.capitalize(), color=band_colors[i], fontsize=16)
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[i].set_box_aspect(1)
    axes[0].set_ylabel('MPC gradient', fontsize=16)
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_band_power_corr_{hemi}.svg")
    return band_maps


# Canonical Yeo network -> integer index matching the rows of `yeo7_rgb`
# (the same convention `figure_3_ieeg_mni` uses via `convert_states_str2int`).
_NET_NAMES = np.array(["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"])
_NET_TO_INT = {name: int(i) for name, i in zip(*[_NET_NAMES, convert_states_str2int(_NET_NAMES)[0].astype(int)])}


def salience_network_electrophysiological_similarity(
    df_channel: pd.DataFrame,
    surf32k_lh_infl,
    surf32k_rh_infl,
    df_yeo_surf: pd.DataFrame,
    project_root: Path,
    hemi: str = 'RH',
    network: str = 'SalVentAttn',
    n_perm: int = 1000,
    min_valid: int = 10,
) -> None:
    """Electrophysiological-similarity projection (Figure 2a focus row, group level).

    Recasts the iEEG spectral similarity as a connectivity measure analogous to FC and
    runs the Figure 2 gradient-weighted projection on it. Each surface vertex carries a
    sensitivity-weighted PSD fingerprint; the electrophysiological-similarity (ES)
    connectivity between two vertices is the positive part of their PSD correlation. For
    each source-network (`network`) vertex i the projection score is the ES-weighted mean
    of the FC gradient across its non-network targets,

        P[i] = sum_j ES+_ij * g_FC[j] / sum_j ES+_ij ,

    (the weighted-mean projection of `connectome_processing.compute_projection_score`,
    evaluated here on the pre-sliced source x target block to avoid materialising the full
    32k x 32k similarity matrix). The group statistic is a single
    Spearman(g_MPC[i], P[i]) across source vertices, with significance from the within-
    network Moran spectral-randomisation null (per analysis hemisphere = one connected
    component) and the add-one empirical p. Renders the Figure-2a-style scatter and the
    ES projection brain map; the channel-level PSD correlation matrix is kept as a
    supplement (it is the ES connectivity measure itself).
    """
    N_LH = 32492
    hemi_offset = N_LH if hemi == 'RH' else 0
    surf_hemi_infl = surf32k_rh_infl if hemi == 'RH' else surf32k_lh_infl
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    df_hemi = df_yeo_surf.iloc[hemi_offset:hemi_offset + N_LH].reset_index(drop=True)
    net_labels = df_hemi['network'].values

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
    surf_psd_v = surf_psd.T  # (32492, n_freqs) vertex-level PSD

    # Z-score each vertex's PSD across frequencies; uncovered vertices -> 0 so their
    # similarity rows/columns vanish from the projection. The ES "connectivity" between
    # vertices i and j is then ES_ij = (z_i . z_j) / n_freqs (Pearson over frequencies).
    psd_mean = np.nanmean(surf_psd_v, axis=1, keepdims=True)
    psd_std = np.nanstd(surf_psd_v, axis=1, keepdims=True)
    surf_psd_z = np.where(covered[:, None], (surf_psd_v - psd_mean) / (psd_std + 1e-12), 0.0)
    n_freqs = surf_psd_z.shape[1]

    # Source = source-network vertices; targets = other cortical networks (FC-gradient
    # axis defined there). The FC gradient is oriented by anatomy (default-mode low).
    sal_mask = compute_network_mask(df_hemi, network, 'both')
    other_mask = (covered & (net_labels != network) & (net_labels != 'medial_wall')
                  & pd.notna(net_labels))
    grad = df_hemi[f't1_gradient1_{network}'].values.astype(float)

    fc_raw = load_gradient("fc", join=True)[hemi_offset:hemi_offset + N_LH]
    fc_g1_hemi = _orient_fc_gradient(fc_raw, net_labels, label="FC gradient")
    other_mask = other_mask & np.isfinite(fc_g1_hemi)
    logger.info(f"[Figure 3B] source {network} vertices: {sal_mask.sum()} "
                f"(covered+gradient: {(sal_mask & covered & np.isfinite(grad)).sum()}); "
                f"target vertices: {other_mask.sum()}")

    # ES-weighted projection of the FC gradient (source x target block only).
    z_src = surf_psd_z[sal_mask]                         # (n_sal, n_freqs)
    z_tgt = surf_psd_z[other_mask]                       # (n_tgt, n_freqs)
    W_block = (z_src @ z_tgt.T) / n_freqs                # (n_sal, n_tgt) PSD correlations
    W_pos = np.where(W_block > 0, W_block, 0.0)          # positive ES connections only
    g_tgt = zscore(fc_g1_hemi[other_mask])               # FC gradient at targets, SD units
    num = W_pos @ g_tgt
    den = W_pos.sum(axis=1)
    n_valid = (W_block > 0).sum(axis=1)
    P = np.where(n_valid >= min_valid, num / np.where(den > 0, den, np.nan), np.nan)

    # Group statistic: Spearman(MPC gradient, ES projection) over source vertices.
    g_sal = grad[sal_mask]
    finite = np.isfinite(g_sal) & np.isfinite(P)
    r_group, _ = spearmanr(g_sal[finite], P[finite])

    # Within-network Moran spatial null (single hemisphere -> one connected component).
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=sal_mask)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_perm, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)
    r_null = np.array([spearmanr(surr[finite], P[finite])[0]
                       for surr in msr.randomize(np.nan_to_num(g_sal))])
    p_moran = empirical_p_twosided(r_null, r_group)
    logger.info(f"[Figure 3B] ES projection vs {network} MPC-gradient | "
                f"Spearman r_group={r_group:+.3f}, Moran permutation p={p_moran:.3e} "
                f"(n_perm={n_perm}, n_src={int(finite.sum())})")

    # Per-source-vertex dominant target network (scatter colour) and per-target-network
    # mean projection (lollipop), both from the same positive-ES block W_pos.
    tgt_networks = net_labels[other_mask]
    tgt_net_list = [n for n in _NET_NAMES if (tgt_networks == n).any()]
    net_weight = np.column_stack([W_pos[:, tgt_networks == n].sum(axis=1) for n in tgt_net_list])
    has_weight = net_weight.sum(axis=1) > 0
    dominant_int = np.full(sal_mask.sum(), _NET_TO_INT[network])  # fallback: focus colour
    if has_weight.any():
        dom = np.argmax(net_weight[has_weight], axis=1)
        dominant_int[has_weight] = [_NET_TO_INT[tgt_net_list[d]] for d in dom]
    point_colors = yeo7_rgb[dominant_int]

    # Mean ES projection per target network: the projection restricted to that network's
    # targets, averaged over source vertices (reuses W_pos; only the target columns change).
    P_t_mean = {}
    for net in tgt_net_list:
        cols = (tgt_networks == net)
        den_t = W_pos[:, cols].sum(axis=1)
        num_t = W_pos[:, cols] @ g_tgt[cols]
        P_t = np.where(den_t > 0, num_t / np.where(den_t > 0, den_t, np.nan), np.nan)
        P_t_mean[net] = float(np.nanmean(P_t)) if np.isfinite(P_t).any() else np.nan

    # Figure 2a-style layout: scatter (MPC gradient vs ES projection) + per-target-network
    # lollipop, both target-network coloured.
    P_z = zscore(P, nan_policy='omit')
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(7.0, 3.2),
        gridspec_kw={'wspace': 0.45, 'width_ratios': [1.0, 1.0]},
    )
    ax.scatter(g_sal[finite], P_z[finite], s=15, alpha=0.75,
               c=point_colors[finite], edgecolor='none', rasterized=True)
    sns.regplot(x=g_sal[finite], y=P_z[finite], scatter=False, color='black',
                line_kws={'linewidth': 2.5}, ax=ax)
    ax.text(0.05, 0.95, f"$r={r_group:+.2f}$\n$p={p_moran:.3f}$",
            transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
    t = ax.set_title("Electrophysiological similarity – iEEG spectral fingerprint coupling",
                     loc='left', pad=15)
    t.set_in_layout(False)
    ax.set_xlabel("MPC gradient")
    ax.set_ylabel("ES projection")
    ax.set_ylim(-5, 5)
    ax.set_yticks([-4, 0, 4])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_box_aspect(1)
    sns.despine(ax=ax)

    # Horizontal lollipop: one stem per target network, length = mean ES projection,
    # coloured by target network, ordered by value.
    nets_sorted = sorted((n for n in tgt_net_list if np.isfinite(P_t_mean[n])),
                         key=lambda n: P_t_mean[n])
    for y, net in enumerate(nets_sorted):
        color = tuple(yeo7_rgb[_NET_TO_INT[net]])
        axl.hlines(y, 0, P_t_mean[net], colors=[color], lw=2.5)
        axl.scatter(P_t_mean[net], y, s=55, facecolors=[color], edgecolors=[color], zorder=3)
    axl.axvline(0, color='0.6', lw=1, zorder=0)
    axl.set_yticks(range(len(nets_sorted)))
    axl.set_yticklabels([yeo7_abbrev.get(n, n) for n in nets_sorted])
    for tick, net in zip(axl.get_yticklabels(), nets_sorted):
        tick.set_color(tuple(yeo7_rgb[_NET_TO_INT[net]]))
    axl.tick_params(axis='y', length=0)
    axl.set_xlabel("Mean ES projection")
    axl.set_box_aspect(1)
    axl.spines['right'].set_visible(False)
    axl.spines['top'].set_visible(False)
    axl.spines['left'].set_visible(False)

    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_es_scatter_{hemi}.svg",
                bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Channel-level PSD correlation matrix (the ES connectivity measure), network-sorted.
    peak_idx = np.argmax(sens, axis=1)  # (n_channels,)
    channel_networks = net_labels[peak_idx]
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
        color = yeo7_rgb[_NET_TO_INT.get(net_name, 7)]
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

    # ES projection brain map (z-scored P embedded on the analysis hemisphere).
    p_map = np.full(N_LH, np.nan)
    p_map[sal_mask] = P_z
    surf_hemi_infl.append_array(p_map, name='es_projection')
    surfs = {'hemi1': surf_hemi_infl, 'hemi2': surf_hemi_infl}
    layout = [['hemi1', 'hemi2']]
    view = [['lateral', 'medial']]
    _plot_surf_safe(surfs, layout=layout, view=view, array_name='es_projection',
              size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
              nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range='sym',
              transparent_bg=True, screenshot=True,
              filename=str(project_root / f"results/figures/figure_3b_ieeg_mica_es_map_{hemi}.svg"),
              cb__numberOfLabels=3, cb__labelTextProperty={'fontSize': 36, 'bold': False})


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
    logger.info(f"Statistic      : ES-weighted projection of the FC gradient (group level); band power vs MPC gradient")
    logger.info(f"Null model     : within-network Moran randomization (n_rep=1000, procedure=singleton, random_state=0), add-one empirical p")
    logger.info(f"Surface space  : fsLR-32k {args.hemi}, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Analysis network: {args.network}")

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
    df_yeo_surf = pd.read_csv(path_df_1a, sep="\t")

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

    # Per-band sensitivity-weighted power vs the within-network MPC gradient.
    frequency_band_analysis_sensitivity(df2, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi, network=args.network)

    # Electrophysiological-similarity projection: ES used as a connectivity measure (like
    # FC) to project the FC gradient and test it against the within-network MPC gradient.
    salience_network_electrophysiological_similarity(df2, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi, network=args.network)


if __name__ == "__main__":
    main()




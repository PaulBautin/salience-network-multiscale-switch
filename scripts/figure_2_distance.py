# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 2 - Gradient-driven connectivity fingerprints
#
# Tests whether the MPC (T1) gradient within the Salience/Ventral Attention network
# predicts how vertices connect to the rest of the brain. Network vertices are binned
# into K=10 equal-quantile decile bins by gradient rank; per-subject Spearman ρ
# between bin rank and mean connectivity to every other-network vertex quantifies the
# monotone trend. The group-average z-scored ρ map is then correlated with the
# whole-brain FC gradient (spin-test corrected).
#
# All connectivity matrices (SC, GD, MPC) are loaded at fsLR-5k resolution
# (9684 vertices) and all analysis runs at that vertex level.
#
# Figure 2A: For the SalVentAttn network, computes per-subject Spearman decile ρ
#            using three metrics (structural connectivity, geodesic distance, MPC)
#            and correlates the group ρ map with the whole-brain FC gradient.
# Figure 2B: Replicates the MPC fingerprint analysis for each of the 7 Yeo networks
#            and correlates each ρ map with the whole-brain FC gradient.
#
# Outputs:
#   results/figures/figure_2a_distance_metric.svg
#   results/figures/figure_2a_brain_{SC,GD,MPC}_rho.svg
#   results/figures/figure_2b_distance_network_{measure}.svg
#   results/figures/figure_2b_brain_{measure}_rho_{network}.svg
#   data/dataframes/df_2b_label_{hemisphere}.csv  (vertex-level cache)
#
# Requires figure_1a_t1map.py to have been run first (produces
#   data/dataframes/figure_1a_pni_to_mics_5k.csv).
#
# Example:
#   python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_2_distance.py -hemi LH
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24"
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

from functools import partial

from brainspace.plotting import plot_surf
import brainspace.plotting.surface_plotting as _bsp_sp
from brainspace.plotting.utils import _gen_grid as _orig_gen_grid
from brainspace.mesh.mesh_io import read_surface
from brainspace.null_models import SpinPermutations

from scipy.stats import spearmanr, zscore, rankdata

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker

from src.atlas_load import load_yeo_surf_5k, load_t1_salience_profiles, convert_states_str2int, compute_network_mask
from src.gradient_computation import compute_t1_gradient
from src.plot_colors import yeo7_rgba, yeo7_rgb, yeo7_abbrev
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)

# Matplotlib globals
plt.rcParams["font.size"] = 16
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False

N_LH_5K = 4842  # fsLR-5k left-hemisphere vertex count


def get_parser() -> argparse.ArgumentParser:
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute structural connectivity differences between MPC-gradient extremes in the salience network (Fig 2A) and across all Yeo networks (Fig 2B).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-hemi",
        type=str,
        default="both",
        choices=["both", "LH", "RH"],
        help="Hemisphere for analysis: 'both', 'LH', or 'RH' (default: both)"
    )
    return parser


def load_connectomes_5k(files: list, df_yeo_surf_5k: pd.DataFrame,
                        split_hemi: bool = True, log_transform: bool = False) -> np.ndarray:
    """
    Load fsLR-5k connectivity GIFTIs (shape 9684×9684) and return subject-averaged matrix.

    Each GIFTI contains one darray of shape (9684, 9684). Masks to cortical vertices,
    zeros inter-hemispheric edges when split_hemi=True, and optionally applies log1p.

    Parameters
    ----------
    files : list of str
        Paths to fsLR-5k connectivity GIFTI files, one per subject.
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame with a 'hemisphere' column to identify cortical vertices.
    split_hemi : bool, optional
        Zero out inter-hemispheric connections (default True).
    log_transform : bool, optional
        Apply log1p to the averaged matrix (default False).

    Returns
    -------
    A : np.ndarray, shape (n_cortex_5k, n_cortex_5k)
        Symmetric, subject-averaged connectivity matrix for cortical 5k vertices.
    """
    if not files:
        raise FileNotFoundError("No connectome files found.")

    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values  # (9684,)
    hemi = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values

    conn_stack = []
    for f in files:
        data = nib.load(f).darrays[0].data.astype(float)      # (9684, 9684)
        data = data[np.ix_(cortex_mask, cortex_mask)]          # cortex-only
        data = np.triu(data, 1) + data.T                       # micapipe stores upper triangle only
        data[data == 0] = np.nan                               # remove diagonal/absent edges; keep negative MPC correlations
        if split_hemi:
            same_hemi = hemi[:, None] == hemi[None, :]
            data[~same_hemi] = np.nan
        conn_stack.append(data)

    conn = np.stack(conn_stack, axis=0)
    nan_mask = np.mean(np.isnan(conn), axis=0) > 0.5
    mean_conn = np.nanmean(conn, axis=0)
    mean_conn[nan_mask] = np.nan
    A = np.nan_to_num(mean_conn, nan=0.0)
    A = np.triu(A, k=1)
    A = A + A.T
    if log_transform:
        A = np.log1p(A)
    return A


def fcn_group_bins(
    adj: np.ndarray, dist: np.ndarray, hemiid: np.ndarray, nbins: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Distance-dependent group-representative SC thresholding (Betzel et al. 2018).

    Generates a group-representative structural connectivity matrix by preserving
    within- and between-hemisphere connection-length distributions.

    Parameters
    ----------
    adj : np.ndarray, shape (n, n, n_sub)
        Per-subject SC matrices (binary or weighted).
    dist : np.ndarray, shape (n, n)
        Pairwise tract distance matrix (mean streamline length from tractography).
    hemiid : np.ndarray of bool, shape (n,)
        Hemisphere indicator: False = left hemisphere, True = right hemisphere.
    nbins : int
        Number of distance bins.

    Returns
    -------
    G : np.ndarray, shape (n, n)
        Symmetric binary group-consensus matrix (distance-dependent thresholding).
    Gc : np.ndarray, shape (n, n)
        Symmetric binary group-consensus matrix (consistency-based thresholding).

    References
    ----------
    Betzel, R. F., Griffa, A., Hagmann, P., & Miic, B. (2018).
    Distance-dependent consensus thresholds for generating group-representative
    structural brain networks. Network Neuroscience, 1–22.
    """
    assert adj.shape[0] == adj.shape[1], "adj must be square in its first two dims"
    if hemiid.ndim == 1:
        hemiid = hemiid[:, np.newaxis]

    n, nsub = adj.shape[0], adj.shape[-1]
    nonzero_dist = dist[np.nonzero(dist)]
    distbins = np.linspace(nonzero_dist.min(), nonzero_dist.max(), nbins + 1)
    distbins[-1] += 1

    C = np.sum(adj > 0, axis=2)
    W = np.sum(adj, axis=2) / np.where(C > 0, C, np.nan)
    W = np.nan_to_num(W, nan=0.0)

    inter_hemi_mask = np.dot(hemiid, ~hemiid.T)
    inter_hemi_mask = np.logical_or(inter_hemi_mask, inter_hemi_mask.T)

    Grp = np.zeros((n, n, 2))
    Gc_arr = np.zeros((n, n, 2))

    for j in range(2):
        inter_hemi = ~inter_hemi_mask if j else inter_hemi_mask
        m = dist * inter_hemi
        D = (adj > 0) * (dist * np.triu(inter_hemi))[..., np.newaxis]
        D = D[np.nonzero(D)]
        if len(D) == 0:
            continue
        tgt = len(D) / nsub

        G = np.zeros((n, n))
        for i_bin in range(nbins):
            mask = np.where(np.triu((m >= distbins[i_bin]) & (m < distbins[i_bin + 1]), 1))
            if len(mask[0]) == 0:
                continue
            n_D_bin = np.sum((D >= distbins[i_bin]) & (D < distbins[i_bin + 1]))
            frac = int(np.round(tgt * n_D_bin / len(D)))
            c = C[mask]
            idx = np.argsort(c)[::-1]
            G[mask[0][idx[:frac]], mask[1][idx[:frac]]] = 1
        Grp[:, :, j] = G

        I = np.where(np.triu(inter_hemi, 1))
        w = W[I]
        idx = np.argsort(w)[::-1]
        w_mat = np.zeros((n, n))
        nnz = int(G.sum())
        if nnz > 0:
            w_mat[I[0][idx[:nnz]], I[1][idx[:nnz]]] = 1
        Gc_arr[:, :, j] = w_mat

    G = np.sum(Grp, 2)
    G = G + G.T
    Gc = np.sum(Gc_arr, 2)
    Gc = Gc + Gc.T
    return G, Gc


def load_connectome_5k_dist_threshold(
    sc_files: list, dist_files: list, df_yeo_surf_5k: pd.DataFrame,
    nbins: int = 10, log_transform: bool = False
) -> np.ndarray:
    """
    Load fsLR-5k SC GIFTIs and apply distance-dependent group consensus thresholding.

    Implements the Betzel et al. (2018) distance-dependent consensus method, which
    preserves within- and between-hemisphere connection-length distributions when
    selecting group-representative edges across subjects.

    After binary thresholding, edges are weighted by the per-subject average SC.

    Parameters
    ----------
    sc_files : list of str
        Paths to fsLR-5k SC GIFTI files, one per subject.
    dist_files : list of str
        Paths to fsLR-5k edge-length GIFTI files, one per subject. Mean tract
        distance across all subjects is used as the reference distance matrix.
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame with 'hemisphere' column to identify cortical vertices.
    nbins : int, optional
        Number of distance bins for the Betzel thresholding (default: 10).
    log_transform : bool, optional
        Apply log1p to edge weights after thresholding (default: False).

    Returns
    -------
    A : np.ndarray, shape (n_cortex_5k, n_cortex_5k)
        Symmetric, distance-thresholded group-representative SC matrix for cortical
        5k vertices, weighted by per-subject average streamline count.
    """
    if not sc_files:
        raise FileNotFoundError("No SC files provided.")
    if not dist_files:
        raise FileNotFoundError("No distance files provided.")

    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemi = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values
    hemiid = (hemi == "RH")  # False = LH, True = RH

    sc_stack = []
    for f in sc_files:
        data = nib.load(f).darrays[0].data.astype(float)
        data = data[np.ix_(cortex_mask, cortex_mask)]
        data = np.triu(data, 1) + data.T  # micapipe stores upper triangle only
        data[data < 0] = 0.0
        sc_stack.append(data)
    adj = np.stack(sc_stack, axis=-1)  # (n_cortex, n_cortex, n_sub)

    dist_stack = []
    for f in dist_files:
        d = nib.load(f).darrays[0].data.astype(float)
        d = d[np.ix_(cortex_mask, cortex_mask)]
        d = np.triu(d, 1) + d.T  # micapipe stores upper triangle only
        d[d < 0] = 0.0
        dist_stack.append(d)
    dist = np.mean(np.stack(dist_stack, axis=0), axis=0)

    C = np.sum(adj > 0, axis=2)
    W = np.sum(adj, axis=2) / np.where(C > 0, C, np.nan)
    W = np.nan_to_num(W, nan=0.0)

    G, _ = fcn_group_bins(adj, dist, hemiid, nbins)

    A = G * W
    A = np.triu(A, k=1)
    A = A + A.T
    if log_transform:
        A = np.log1p(A)
    return A


def compute_pvals_spin(x: np.ndarray, y_surf_5k: np.ndarray,
                       df_yeo_surf_5k: pd.DataFrame,
                       spin_model: SpinPermutations, n_rand: int) -> np.ndarray:
    """
    Compute spin-permutation null Spearman correlations between 5k diff and rotated FC gradient.

    Rotates the 5k FC gradient `n_rand` times directly in fsLR-5k space, then correlates
    each rotation with `x`.

    Parameters
    ----------
    x : np.ndarray, shape (9684,)
        Connectivity-difference values at fsLR-5k (NaN outside the target region).
    y_surf_5k : np.ndarray, shape (9684,)
        Per-vertex 5k FC gradient values to rotate (NaN at medial wall).
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame (unused internally; kept for API consistency).
    spin_model : SpinPermutations
        Pre-fitted spin-permutation model (fitted on fsLR-5k spheres).
    n_rand : int
        Number of spin permutations.

    Returns
    -------
    r_spin : np.ndarray, shape (n_rand,)
    """
    medial_wall_mask = np.isnan(y_surf_5k)
    y_lh, y_rh = y_surf_5k[:4842], y_surf_5k[4842:]
    y_rotated = np.hstack(spin_model.randomize(y_lh, y_rh))
    r_spin = np.empty(n_rand)
    for j, perm in enumerate(y_rotated):
        perm = perm.copy()
        perm[medial_wall_mask] = np.nan
        mask = ~np.isnan(x) & ~np.isnan(perm)
        r_spin[j] = spearmanr(zscore(x[mask]), zscore(perm[mask]))[0]
    return r_spin


def compute_decile_slope_subjects(
    files: list,
    gradient_values_cortex: np.ndarray,
    network_mask_cortex: np.ndarray,
    other_idx_cortex: np.ndarray,
    df_yeo_surf_5k: pd.DataFrame,
    n_bins: int = 10,
    split_hemi: bool = False,
    log_transform: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-subject Spearman rank correlation of bin-mean connectivity ~ gradient decile rank.

    For each subject bins network vertices into n_bins equal-quantile gradient bins, computes
    mean connectivity from each bin to every other-network vertex, then summarises the monotone
    trend via Spearman ρ between bin rank (0..n_bins-1) and the bin means.

    Spearman ρ is rank-based and requires no weighting or distributional assumptions, making it
    appropriate for the right-skewed, zero-inflated SC values and Fisher-z MPC values that arise
    at fsLR-5k resolution. A minimum of 4 valid (non-NaN) bins is required before ρ is computed.

    Vertices where fewer than 50% of subjects contribute a valid ρ are set to NaN.
    Returns (mean_rho, se_rho), shape (n_other,), using 'slopes' as the internal variable name
    for API compatibility with downstream callers.
    """
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemi = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values
    n_other = int(other_idx_cortex.sum())

    bin_masks = compute_decile_bins(gradient_values_cortex, network_mask_cortex, n_bins=n_bins)
    x_rank = np.arange(n_bins, dtype=float)

    subject_slopes = []
    for f in files:
        data = nib.load(f).darrays[0].data.astype(float)        # (9684, 9684)
        data = data[np.ix_(cortex_mask, cortex_mask)]            # (n_cortex, n_cortex)
        data = np.triu(data, 1) + data.T
        data[data == 0] = np.nan
        if split_hemi:
            same_hemi = hemi[:, None] == hemi[None, :]
            data[~same_hemi] = np.nan
        if log_transform:
            data = np.log1p(np.maximum(data, 0))

        bin_means = np.full((n_bins, n_other), np.nan)
        for k, bm in enumerate(bin_masks):
            if bm.any():
                bin_means[k] = np.nanmean(data[np.ix_(bm, other_idx_cortex)], axis=0)

        slopes = np.full(n_other, np.nan)
        nan_cols = np.any(np.isnan(bin_means), axis=0)  # (n_other,)

        # Fast path: columns with no NaN — rank all at once, then vectorized Pearson
        clean = ~nan_cols
        if clean.any():
            ranks = rankdata(bin_means[:, clean], axis=0)       # (n_bins, n_clean)
            x_c = x_rank - x_rank.mean()                        # (n_bins,)
            y_c = ranks - ranks.mean(axis=0)                    # (n_bins, n_clean)
            denom = np.linalg.norm(x_c) * np.linalg.norm(y_c, axis=0)
            slopes[clean] = (x_c @ y_c) / np.where(denom > 0, denom, np.nan)

        # Slow path: columns with NaN — per-column with row masking
        for j in np.where(nan_cols)[0]:
            col = bin_means[:, j]
            valid = ~np.isnan(col)
            if valid.sum() >= 4:
                slopes[j] = spearmanr(x_rank[valid], col[valid])[0]

        subject_slopes.append(slopes)

    arr = np.stack(subject_slopes, axis=0)           # (n_subj, n_other)
    mean_slopes = np.nanmean(arr, axis=0)
    n_valid = (~np.isnan(arr)).sum(axis=0).astype(float)
    mean_slopes[n_valid < len(files) * 0.5] = np.nan  # require ≥50% subject coverage
    se_slopes = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    return mean_slopes, se_slopes


def compute_decile_bins(values: np.ndarray, mask: np.ndarray,
                        n_bins: int = 10) -> list:
    """Partition masked vertices into n_bins equal-quantile bins.

    Returns a list of n_bins boolean arrays (over the full vertex space) where
    each array selects vertices whose gradient value falls in that decile.
    Bins use half-open intervals [lo, hi) except the last which uses [lo, hi].
    """
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.nanquantile(values[mask], quantiles)
    bin_masks = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == n_bins - 1:
            b = mask & (values >= lo) & (values <= hi)
        else:
            b = mask & (values >= lo) & (values < hi)
        bin_masks.append(b)
    return bin_masks


def save_brain_map(surf_lh, surf_rh, values: np.ndarray, array_name: str, filename: Path,
                   hemisphere: str = "both") -> None:
    """Append `values` to fsLR-5k inflated surfaces and save a brain-map screenshot.

    Parameters
    ----------
    values : np.ndarray, shape (9684,)
        Per-vertex values for the full fsLR-5k surface (first N_LH_5K = LH, rest = RH).
    """
    surf_lh.append_array(values[:N_LH_5K], name=array_name)
    surf_rh.append_array(values[N_LH_5K:], name=array_name)
    surfs = {"rh1": surf_rh, "lh1": surf_lh}
    if hemisphere == "LH":
        layout, view = [["lh1", "lh1"]], [["lateral", "medial"]]
    elif hemisphere == "RH":
        layout, view = [["rh1", "rh1"]], [["lateral", "medial"]]
    else:
        layout, view = [["lh1", "rh1"]], [["lateral", "medial"]]
    _bsp_sp._gen_grid = partial(_orig_gen_grid, size_bar=0.20)
    try:
        plot_surf(
            surfs, layout=layout, view=view,
            array_name=array_name, size=(1200, 500), zoom=1.4, color_bar="bottom",
            share="both", nan_color=(220, 220, 220, 1), cmap="coolwarm",
            color_range=(-3, 3), transparent_bg=True, screenshot=True, filename=filename,
            cb__numberOfLabels=3,
            cb__labelTextProperty={'fontSize': 36, 'bold': False})
    finally:
        _bsp_sp._gen_grid = _orig_gen_grid


def _load_fc_gradient(project_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Load fsLR-5k FC gradient GIFTIs and attach fc_g1/fc_g1_network/network_int to a copy of df."""
    fc_lh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-5k_lh.shape.gii").darrays[0].data
    fc_rh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-5k_rh.shape.gii").darrays[0].data
    df = df.copy()
    df["fc_g1"] = -np.concatenate([fc_lh, fc_rh])
    df.loc[df["hemisphere"].isna(), "fc_g1"] = np.nan
    df["fc_g1_network"] = df.groupby("network")["fc_g1"].transform("mean")
    df["network_int"] = convert_states_str2int(df["network"].values)[0]
    return df


def _prepare_network_gradient(
    df: pd.DataFrame, network: str, df_pni: pd.DataFrame, hemisphere: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure T1 gradient exists for network; extract continuous gradient values.

    Modifies df in-place (adds gradient column if absent).
    Returns
    -------
    gradient_values_cortex : np.ndarray, shape (n_cortex,)
        Continuous gradient values for all cortex vertices; NaN outside the network.
    network_mask_cortex : np.ndarray of bool, shape (n_cortex,)
        True for network vertices with a valid (non-NaN) gradient value.
    other_idx : np.ndarray of bool, shape (9684,)
        True for non-target-network cortical vertices (full vertex space).
    """
    cortex_mask = df["hemisphere"].notna().values
    grad_col = f"t1_gradient1_{network}"
    if grad_col not in df.columns:
        net_mask_5k = compute_network_mask(df, network, hemisphere)
        t1_profiles = load_t1_salience_profiles(df_pni["path_t1_profile_5k"].tolist(), net_mask_5k)
        df.loc[net_mask_5k, grad_col] = compute_t1_gradient(t1_profiles)

    net_mask = cortex_mask & (df["network"] == network).values
    if hemisphere == "LH":
        net_mask = net_mask & (df["hemisphere"] == "LH").values
    elif hemisphere == "RH":
        net_mask = net_mask & (df["hemisphere"] == "RH").values

    grad_full = df[grad_col].values.astype(float)
    gradient_values_cortex = grad_full[cortex_mask]                         # (n_cortex,)
    network_mask_cortex = net_mask[cortex_mask] & ~np.isnan(gradient_values_cortex)

    if hemisphere == "both":
        other_idx = cortex_mask & (df["network"] != network).values
    else:
        other_idx = (cortex_mask & (df["network"] != network).values
                     & (df["hemisphere"] == hemisphere).values)
    return gradient_values_cortex, network_mask_cortex, other_idx


def _slope_and_correlate(
    df: pd.DataFrame, files: list, cfg: dict, rho_col: str,
    gradient_values_cortex: np.ndarray, network_mask_cortex: np.ndarray,
    other_idx: np.ndarray,
    spin_model: SpinPermutations, n_rand: int,
    group_threshold_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float, int]:
    """Per-subject Spearman decile ρ, stored in df, correlated with FC gradient.

    Streams subjects via compute_decile_slope_subjects.
    Modifies df in-place (stores z-scored mean ρ in rho_col, SE in rho_col+'_se').
    Returns (x_norm, y_norm, mask_label, spearman_r, spin_pval, ci_lo, ci_hi, n_subjects).
    """
    cortex_mask = df["hemisphere"].notna().values
    mean_slopes, se_slopes = compute_decile_slope_subjects(
        files, gradient_values_cortex, network_mask_cortex, other_idx[cortex_mask], df,
        split_hemi=cfg["split_hemi"], log_transform=cfg["log_transform"],
    )

    df.loc[other_idx, rho_col] = zscore(mean_slopes, nan_policy="omit")
    df.loc[other_idx, f"{rho_col}_se"] = se_slopes

    x = df[rho_col].values
    y = df["fc_g1"].values
    mask_label = ~np.isnan(x) & ~np.isnan(y)
    n_vert = int(mask_label.sum())
    x_norm, y_norm = zscore(x[mask_label]), zscore(y[mask_label])
    corr, _ = spearmanr(x_norm, y_norm)
    r_spin = compute_pvals_spin(x, y, df, spin_model, n_rand)
    pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))

    # Fisher-z 95 % CI for Spearman r
    z = np.arctanh(corr)
    se_z = 1.0 / np.sqrt(max(n_vert - 3, 1))
    ci_lo, ci_hi = float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z))

    n_subjects = len(files)
    return x_norm, y_norm, mask_label, corr, pv_spin, ci_lo, ci_hi, n_subjects


def struct_conn_metric_analysis(df_yeo_surf_5k: pd.DataFrame,
                                surf5k_lh_infl, surf5k_rh_infl,
                                df_pni: pd.DataFrame, project_root: Path,
                                spin_model: SpinPermutations, network: str = "SalVentAttn",
                                n_rand: int = 100, hemisphere: str = "both") -> None:
    """
    Figure 2A: correlate MPC-gradient-driven connectivity fingerprints with the FC gradient.

    Loads SC, GD, and MPC at fsLR-5k. Partitions T1-MPC gradient vertices within `network`
    into K=10 equal-quantile decile bins, computes per-subject Spearman ρ between bin rank
    and mean connectivity to all other-network vertices, then correlates the group-average
    z-scored ρ map with the whole-brain FC gradient (spin-test corrected).

    Parameters
    ----------
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame with 'mics', 'network', 'hemisphere' columns.
    surf5k_lh_infl, surf5k_rh_infl :
        Inflated fsLR-5k surfaces for brain-map screenshots.
    df_pni : pd.DataFrame
        Subject manifest with columns path_sc_5k, path_dist_5k, path_mpc_5k,
        path_t1_profile_5k.
    project_root : Path
        Repository root used to resolve output paths.
    spin_model : SpinPermutations
        Pre-fitted spin-permutation model (fitted on fsLR-5k spheres).
    network : str, optional
        Yeo network to use as the gradient anchor (default: 'SalVentAttn').
    n_rand : int, optional
        Number of spin permutations (default: 100).
    hemisphere : str, optional
        Hemisphere filter: 'both', 'LH', or 'RH' (default: 'both').
    """
    # Group-consensus SC threshold mask (Betzel 2018); used to exclude sparse edges in per-subject slopes.
    # A_sc_group is already cortex-indexed (n_cortex × n_cortex); > 0 gives the group-consensus mask.
    A_sc_group = load_connectome_5k_dist_threshold(
        df_pni["path_sc_5k"].tolist(), df_pni["path_sc_dist_5k"].tolist(),
        df_yeo_surf_5k, nbins=10, log_transform=True)
    sc_threshold = A_sc_group > 0   # (n_cortex, n_cortex)

    metric_configs_2a = {
        "SC":  {"files": df_pni["path_sc_5k"].tolist(),   "cfg": {"split_hemi": False, "log_transform": True},
                "sc_threshold": sc_threshold},
        "GD":  {"files": df_pni["path_dist_5k"].tolist(), "cfg": {"split_hemi": True,  "log_transform": False},
                "sc_threshold": None},
        "MPC": {"files": df_pni["path_mpc_5k"].tolist(),  "cfg": {"split_hemi": False, "log_transform": False},
                "sc_threshold": None},
    }

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)
    gradient_values_cortex, network_mask_cortex, other_idx = _prepare_network_gradient(
        df_yeo_surf_5k, network, df_pni, hemisphere)
    cortex_mask_2a = df_yeo_surf_5k["hemisphere"].notna().values

    fig, axes = plt.subplots(2, 3, figsize=(4 * 4, 10), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]}, sharey="row")

    for i, (name, mcfg) in enumerate(metric_configs_2a.items()):
        rho_col = f"{name}_rho"
        # For SC: build a (n_other,) mask — True where the other-network vertex has at least one
        # group-consensus edge. sc_threshold is cortex-indexed; other_cortex selects other-network columns.
        if mcfg["sc_threshold"] is not None:
            other_cortex = other_idx[cortex_mask_2a]
            gt_mask_other = mcfg["sc_threshold"][:, other_cortex].any(axis=0)  # (n_other,)
        else:
            gt_mask_other = None
        x_norm, y_norm, mask_label, corr, pv_spin, ci_lo, ci_hi, n_subjects = _slope_and_correlate(
            df_yeo_surf_5k, mcfg["files"], mcfg["cfg"], rho_col,
            gradient_values_cortex, network_mask_cortex, other_idx,
            spin_model, n_rand, group_threshold_mask=gt_mask_other)

        cortex_count = int(df_yeo_surf_5k["hemisphere"].notna().sum())
        n_nan_rho = int(df_yeo_surf_5k[rho_col].isna().sum())
        n_network_gray = cortex_count - int(other_idx.sum())
        n_sparsity_gray = n_nan_rho - n_network_gray
        logger.info(
            f"[{name}] brain map: {n_nan_rho} gray vertices = "
            f"{n_network_gray} network+medialwall + {n_sparsity_gray} connectivity-sparse"
        )

        save_brain_map(surf5k_lh_infl, surf5k_rh_infl,
                       df_yeo_surf_5k[rho_col].values,
                       array_name="overlay",
                       filename=project_root / f"results/figures/figure_2a_brain_{name}_rho.svg",
                       hemisphere=hemisphere)

        # Bar plot: mean Spearman ρ per network sorted by FC gradient
        df_plot = (df_yeo_surf_5k.loc[other_idx, ["network", "network_int", rho_col, "fc_g1_network"]]
                   .dropna(subset=[rho_col])
                   .sort_values("fc_g1_network"))
        df_plot = df_plot.copy()
        df_plot["network_abbrev"] = df_plot["network"].map(yeo7_abbrev)
        palette = {yeo7_abbrev.get(net, net): yeo7_rgba[int(net_idx)] for net, net_idx in
                   df_plot[["network", "network_int"]].drop_duplicates().itertuples(index=False)}
        sns.barplot(x=df_plot["network_abbrev"], y=rho_col, hue="network_abbrev",
                    data=df_plot, palette=palette, ax=axes[1, i], legend=False,
                    order=df_plot["network_abbrev"].unique())
        axes[1, i].axhline(0, color="black", linewidth=1, linestyle="--")
        axes[1, i].set_ylabel(f"{name} Spearman ρ (z)")
        axes[1, i].set_ylim(-1.5, 1.5)
        axes[1, i].set_aspect(1)
        axes[1, i].set(xlabel=None)
        axes[1, i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        logger.info(
            f"[Figure 2A] {name}: SalVentAttn Spearman decile ρ vs FC-G1 | "
            f"Spearman r={corr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], "
            f"spin-test p={pv_spin:.3e} (n_perm={n_rand}), n={n_subjects} subjects"
        )

        colors = [yeo7_rgb[int(k)] for k in df_yeo_surf_5k["network_int"].values[mask_label]]
        axes[0, i].scatter(x_norm, y_norm, s=5, alpha=0.7, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black",
                    line_kws={"linewidth": 1}, ax=axes[0, i])
        axes[0, i].text(0.05, 0.95,
                        f"r = {corr:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]\n"
                        f"p$_{{spin}}$ = {pv_spin:.3f}  n={n_subjects}",
                        transform=axes[0, i].transAxes, va="top", fontsize=12)
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-2.5, 2.5)
        axes[0, i].set_aspect("equal", adjustable="box")

    axes[0, 0].set_xlabel("SC Spearman ρ per gradient decile (z)")
    axes[0, 1].set_xlabel("GD Spearman ρ per gradient decile (z)")
    axes[0, 2].set_xlabel("MPC Spearman ρ per gradient decile (z)")
    axes[0, 0].set_ylabel("Principal FC gradient")
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_2a_distance_metric.svg")
    plt.close(fig)


_MEASURE_CONFIG = {
    "SC":  {"path_col": "path_sc_5k",   "split_hemi": False, "log_transform": True},
    "GD":  {"path_col": "path_dist_5k", "split_hemi": True,  "log_transform": False},
    "MPC": {"path_col": "path_mpc_5k",  "split_hemi": False, "log_transform": False},
}


def struct_conn_network_analysis(df_yeo_surf_5k: pd.DataFrame,
                                 surf5k_lh_infl, surf5k_rh_infl,
                                 df_pni: pd.DataFrame, project_root: Path,
                                 spin_model: SpinPermutations,
                                 networks: list[str] = ["SalVentAttn", "Limbic"],
                                 n_rand: int = 100, hemisphere: str = "both",
                                 measure: str = "SC") -> pd.DataFrame:
    """
    Figure 2B: replicate the connectivity-fingerprint/FC-gradient correlation for each Yeo network.

    All analysis at fsLR-5k. For each network, computes the T1-MPC gradient, partitions
    gradient vertices into K=10 equal-quantile decile bins, computes per-subject Spearman ρ
    between bin rank and mean connectivity to all other-network vertices, and correlates the
    group-average z-scored ρ map with the whole-brain FC gradient (spin-test corrected).

    Parameters
    ----------
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame.
    surf5k_lh_infl, surf5k_rh_infl :
        Inflated fsLR-5k surfaces.
    df_pni : pd.DataFrame
        Subject manifest with columns path_sc_5k, path_dist_5k, path_mpc_5k,
        path_t1_profile_5k.
    project_root : Path
        Repository root.
    spin_model : SpinPermutations
        Pre-fitted spin-permutation model (fitted on fsLR-5k spheres).
    networks : list of str, optional
        Yeo networks to analyse.
    n_rand : int, optional
        Number of spin permutations (default: 100).
    hemisphere : str, optional
        Hemisphere filter: 'both', 'LH', or 'RH' (default: 'both').
    measure : str, optional
        Connectivity measure to use: 'SC', 'GD', or 'MPC' (default: 'SC').

    Returns
    -------
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame with per-network connectivity-difference columns appended.
    """
    if measure not in _MEASURE_CONFIG:
        raise ValueError(f"measure must be one of {list(_MEASURE_CONFIG)}, got '{measure}'")
    cfg = _MEASURE_CONFIG[measure]
    files = df_pni[cfg["path_col"]].tolist()

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)

    n_col = int(np.ceil(len(networks) / 2))
    fig, axes = plt.subplots(2, n_col, figsize=(4 * n_col, 10), sharex=True, sharey=True, layout="constrained")
    axes = axes.flatten()

    for i, network in enumerate(networks):
        logger.info(f"Processing network: {network}")
        gradient_values_cortex, network_mask_cortex, other_idx = _prepare_network_gradient(
            df_yeo_surf_5k, network, df_pni, hemisphere)
        rho_col = f"{network}_{measure}_rho"
        x_norm, y_norm, mask_label, corr, pv_spin, ci_lo, ci_hi, n_subjects = _slope_and_correlate(
            df_yeo_surf_5k, files, cfg, rho_col,
            gradient_values_cortex, network_mask_cortex, other_idx,
            spin_model, n_rand)

        save_brain_map(surf5k_lh_infl, surf5k_rh_infl,
                       df_yeo_surf_5k[rho_col].values,
                       array_name="overlay2",
                       filename=project_root / f"results/figures/figure_2b_brain_{measure}_rho_{network}.svg",
                       hemisphere=hemisphere)

        logger.info(
            f"[Figure 2B] {network}: {measure} Spearman decile ρ vs FC-G1 | "
            f"Spearman r={corr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], "
            f"spin-test p={pv_spin:.3e} (n_perm={n_rand}), n={n_subjects} subjects"
        )

        colors = [yeo7_rgb[int(k)] for k in df_yeo_surf_5k["network_int"].values[mask_label]]
        axes[i].scatter(x_norm, y_norm, s=5, alpha=0.7, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black",
                    line_kws={"linewidth": 1}, ax=axes[i])
        axes[i].text(0.05, 0.95,
                     f"r = {corr:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]\n"
                     f"p$_{{spin}}$ = {pv_spin:.3f}  n={n_subjects}",
                     transform=axes[i].transAxes, va="top", fontsize=12)
        net_color = yeo7_rgb[int(df_yeo_surf_5k.loc[
            df_yeo_surf_5k["network"] == network, "network_int"].values[0])]
        net_abbrev = yeo7_abbrev.get(network, network)
        axes[i].set_title(net_abbrev, fontdict={"color": net_color})
        axes[i].set_xlabel(f"{measure} Spearman ρ per gradient decile (z)")
        if i % n_col == 0:
            axes[i].set_ylabel("Principal FC gradient")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect("equal", adjustable="box")

    axes[-1].set_axis_off()
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_2b_distance_network_{measure}.svg")
    return df_yeo_surf_5k


def main():
    parser = get_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_2_distance", project_root, args)
    logger.info("Surface space  : fsLR-5k, Yeo 7-network labels")
    logger.info("SC metric      : iFOD2 40M streamlines, SIFT2-weighted, log-transformed, fsLR-5k")
    logger.info("GD metric      : geodesic distance at fsLR-5k")
    logger.info("MPC metric     : microstructural profile covariance at fsLR-5k")
    logger.info("Null model     : spin permutation (SpinPermutations, n_rep=1000, random_state=42)")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    surf5k_lh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.L.inflated.surf.gii", itype="gii")
    surf5k_rh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.R.inflated.surf.gii", itype="gii")

    df_pni = pd.read_csv(project_root / "data/dataframes/figure_1a_pni_to_mics_5k.csv")
    n_rand = 1000
    spin_model_5k = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_5k_lh = read_surface(project_root / "data/surfaces/fsLR-5k.L.sphere.surf.gii", itype="gii")
    sphere_5k_rh = read_surface(project_root / "data/surfaces/fsLR-5k.R.sphere.surf.gii", itype="gii")
    spin_model_5k.fit(sphere_5k_lh, sphere_5k_rh)

    df_yeo_surf_5k = load_yeo_surf_5k(micapipe=project_root)

    path_df_1a_5k = project_root / f"data/dataframes/df_1a_{args.hemi}_fslr5k.tsv"
    path_df_1a_5k_both = project_root / "data/dataframes/df_1a_both_fslr5k.tsv"
    if not path_df_1a_5k.exists() and not path_df_1a_5k_both.exists():
        raise FileNotFoundError(
            f"fsLR-5k gradient dataframe not found at {path_df_1a_5k} or {path_df_1a_5k_both}. "
            f"Run figure_1a_t1map.py first."
        )
    found = path_df_1a_5k if path_df_1a_5k.exists() else path_df_1a_5k_both
    logger.info(f"fsLR-5k gradient dataframe found at {found}")

    # Figure 2A: SC / Dist / MPC metrics for SalVentAttn
    struct_conn_metric_analysis(df_yeo_surf_5k,
                                surf5k_lh_infl, surf5k_rh_infl,
                                df_pni, project_root, spin_model_5k,
                                network="SalVentAttn", n_rand=n_rand, hemisphere=args.hemi)

    # Figure 2B: replicate SC analysis per Yeo network
    networks = ["Limbic", "Default", "Cont", "SalVentAttn", "DorsAttn", "Vis", "SomMot"]
    df_yeo_surf_5k = struct_conn_network_analysis(
        df_yeo_surf_5k,
        surf5k_lh_infl, surf5k_rh_infl,
        df_pni, project_root, spin_model_5k,
        networks=networks, n_rand=n_rand, hemisphere=args.hemi, measure="MPC")

    df_yeo_surf_5k.to_csv(project_root / f"data/dataframes/df_2b_label_{args.hemi}.csv", index=False)


if __name__ == "__main__":
    main()

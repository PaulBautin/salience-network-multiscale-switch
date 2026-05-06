# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 2 - Structural connectivity at MPC gradient extremes
#
# Tests whether structural connectivity differs between vertices at the high vs. low
# ends of the MPC (T1) gradient computed in Figure 1a, within the Salience/Ventral
# Attention network and across all 7 Yeo networks.
#
# All connectivity matrices (SC, Dist, MPC) are loaded at native fsLR-5k resolution
# (9684 vertices) and all analysis runs at that vertex level.
#
# Figure 2A: For the SalVentAttn network, computes connectivity differences between
#            MPC-gradient-extreme vertices using three metrics (structural connectivity,
#            geodesic distance, MPC) and correlates those differences with the
#            whole brain FC gradient.
# Figure 2B: Replicates the SC-difference analysis for each of the 7 Yeo networks
#            and correlates the results with the whole brain FC gradient.
#
# Outputs:
#   results/figures/figure_2a_distance_metric.svg
#   results/figures/figure_2a_brain_{SC,Dist,MPC}_diff.svg
#   results/figures/figure_2b_distance_network.svg
#   results/figures/figure_2b_brain_SC_diff_{network}.svg
#   data/dataframes/df_2b_label_{hemisphere}.csv  (vertex-level cache)
#
# Requires figure_1a_t1map.py to have been run first (produces
#   data/dataframes/figure_1a_pni_to_mics.csv).
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

from brainspace.plotting import plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.null_models import SpinPermutations

from scipy.stats import spearmanr, zscore

import matplotlib.pyplot as plt
import matplotlib as mpl

from src.atlas_load import load_yeo_surf_5k, load_t1_salience_profiles, convert_states_str2int, compute_network_mask
from src.gradient_computation import compute_t1_gradient
from src.plot_colors import yeo7_rgba, yeo7_rgb
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)

# Matplotlib globals
plt.rcParams["font.size"] = 12
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
        data[data <= 0] = np.nan
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


def compute_top_bottom_diff(conn: np.ndarray, top_idx: np.ndarray, bottom_idx: np.ndarray,
                            other_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute z-scored top–bottom connectivity difference."""
    conn = conn.copy()
    conn[conn <= 0] = np.nan
    top = np.nanmean(conn[top_idx][:, other_idx], axis=0)
    bottom = np.nanmean(conn[bottom_idx][:, other_idx], axis=0)
    return zscore(top - bottom, nan_policy="omit"), top, bottom


def compute_quantile_mask(values: np.ndarray, mask: np.ndarray,
                          q: tuple[float, float] = (0.25, 0.75)) -> np.ndarray:
    """
    Label vertices as high (+1) or low (-1) gradient quantile extremes.

    Parameters
    ----------
    values : np.ndarray, shape (n_vertices,)
        Gradient values at the vertex level.
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean mask selecting vertices to include (e.g., a single network).
    q : tuple of float, optional
        Lower and upper quantile thresholds (default: (0.25, 0.75)).

    Returns
    -------
    out : np.ndarray of int, shape (n_vertices,)
        Array with values in {-1, 0, +1}.
    """
    low, high = np.nanquantile(values[mask], q)
    out = np.full(values.shape, 0)
    out[mask & (values <= low)] = -1
    out[mask & (values >= high)] = 1
    return out


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
    plot_surf(
        surfs, layout=layout, view=view,
        array_name=array_name, size=(1200, 500), zoom=1.4, color_bar="bottom",
        share="both", nan_color=(220, 220, 220, 1), cmap="coolwarm",
        color_range="sym", transparent_bg=True, screenshot=True, filename=filename,
    )


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


def _prepare_network_quantiles(
    df: pd.DataFrame, network: str, df_pni: pd.DataFrame, hemisphere: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure T1 gradient exists for network, compute quantile masks.

    Modifies df in-place (adds gradient and quantile_idx columns).
    Returns (top_idx, bottom_idx, other_idx) boolean arrays over all vertices.
    """
    cortex_mask = df["hemisphere"].notna().values
    grad_col = f"t1_gradient1_{network}"
    if grad_col not in df.columns:
        net_mask_5k = compute_network_mask(df, network, hemisphere)
        t1_profiles = load_t1_salience_profiles(df_pni["path_t1_profile_5k"].tolist(), net_mask_5k)
        df.loc[net_mask_5k, grad_col] = compute_t1_gradient(t1_profiles)

    net_mask = cortex_mask & (df["network"] == network).values
    if hemisphere == "both":
        net_mask_lh = net_mask & (df["hemisphere"] == "LH").values
        net_mask_rh = net_mask & (df["hemisphere"] == "RH").values
        df["quantile_idx"] = (
            compute_quantile_mask(df[grad_col].values, net_mask_lh) +
            compute_quantile_mask(df[grad_col].values, net_mask_rh)
        )
    elif hemisphere == "LH":
        net_mask_lh = net_mask & (df["hemisphere"] == "LH").values
        df["quantile_idx"] = compute_quantile_mask(df[grad_col].values, net_mask_lh)
    else:
        net_mask_rh = net_mask & (df["hemisphere"] == "RH").values
        df["quantile_idx"] = compute_quantile_mask(df[grad_col].values, net_mask_rh)
    df.loc[df["quantile_idx"] == 0, "quantile_idx"] = np.nan

    top_idx = cortex_mask & (df["quantile_idx"] == 1).values
    bottom_idx = cortex_mask & (df["quantile_idx"] == -1).values
    if hemisphere == "both":
        other_idx = cortex_mask & (df["network"] != network).values
    else:
        other_idx = cortex_mask & (df["network"] != network).values & (df["hemisphere"] == hemisphere).values
    return top_idx, bottom_idx, other_idx


def _diff_and_correlate(
    df: pd.DataFrame, A: np.ndarray, diff_col: str,
    top_idx: np.ndarray, bottom_idx: np.ndarray, other_idx: np.ndarray,
    spin_model: SpinPermutations, n_rand: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Compute top-bottom connectivity diff, store in df, correlate with FC gradient.

    Modifies df in-place (stores diff values in diff_col).
    Returns (x_norm, y_norm, mask_label, spearman_r, spin_pval).
    """
    cortex_mask = df["hemisphere"].notna().values
    diff, _, _ = compute_top_bottom_diff(A, top_idx[cortex_mask], bottom_idx[cortex_mask], other_idx[cortex_mask])
    df.loc[other_idx, diff_col] = diff

    x = df[diff_col].values
    y = df["fc_g1"].values
    mask_label = ~np.isnan(x) & ~np.isnan(y)
    x_norm, y_norm = zscore(x[mask_label]), zscore(y[mask_label])
    corr, _ = spearmanr(x_norm, y_norm)
    r_spin = compute_pvals_spin(x, y, df, spin_model, n_rand)
    pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))
    return x_norm, y_norm, mask_label, corr, pv_spin


def struct_conn_metric_analysis(df_yeo_surf_5k: pd.DataFrame,
                                surf5k_lh_infl, surf5k_rh_infl,
                                df_pni: pd.DataFrame, project_root: Path,
                                spin_model: SpinPermutations, network: str = "SalVentAttn",
                                n_rand: int = 100, hemisphere: str = "both") -> None:
    """
    Figure 2A: correlate MPC-gradient-driven connectivity fingerprints with the FC gradient.

    Loads SC, Dist, and MPC at fsLR-5k. Identifies top/bottom quantile vertices of the
    T1-MPC gradient within `network`, computes their mean connectivity difference to all
    other-network vertices, then correlates each difference vector with the whole-brain
    FC gradient (spin-test corrected).

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
    # Load fsLR-5k connectivity matrices
    A_sc   = load_connectomes_5k(df_pni["path_sc_5k"].to_list(),   df_yeo_surf_5k, split_hemi=False, log_transform=True)
    A_dist = load_connectomes_5k(df_pni["path_dist_5k"].to_list(), df_yeo_surf_5k, split_hemi=True, log_transform=False)
    A_mpc  = load_connectomes_5k(df_pni["path_mpc_5k"].to_list(),  df_yeo_surf_5k, split_hemi=False, log_transform=False)
    connectomes = {"SC": A_sc, "GD": A_dist, "MPC": A_mpc}

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)
    top_idx, bottom_idx, other_idx = _prepare_network_quantiles(df_yeo_surf_5k, network, df_pni, hemisphere)

    fig, axes = plt.subplots(2, 3, figsize=(4 * 4, 10), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]}, sharey="row")

    for i, (name, A) in enumerate(connectomes.items()):
        diff_col = f"{name}_diff"
        x_norm, y_norm, mask_label, corr, pv_spin = _diff_and_correlate(
            df_yeo_surf_5k, A, diff_col, top_idx, bottom_idx, other_idx, spin_model, n_rand)

        save_brain_map(surf5k_lh_infl, surf5k_rh_infl,
                       df_yeo_surf_5k[diff_col].values,
                       array_name="overlay",
                       filename=project_root / f"results/figures/figure_2a_brain_{name}_diff.svg",
                       hemisphere=hemisphere)

        # Bar plot: mean diff per network sorted by FC gradient
        df_plot = (df_yeo_surf_5k.loc[other_idx, ["network", "network_int", diff_col, "fc_g1_network"]]
                   .dropna(subset=[diff_col])
                   .sort_values("fc_g1_network"))
        palette = {net: yeo7_rgba[int(net_idx)] for net, net_idx in
                   df_plot[["network", "network_int"]].drop_duplicates().itertuples(index=False)}
        sns.barplot(x=df_plot["network"], y=diff_col, hue="network",
                    data=df_plot, palette=palette, ax=axes[1, i], legend=False)
        axes[1, i].axhline(0, color="black", linewidth=1)
        axes[1, i].set_ylabel("conn$_{top}$ - conn$_{bottom}$")
        axes[1, i].tick_params(axis="x", labelrotation=90)
        axes[1, i].set_ylim(-1.5, 1.5)
        axes[1, i].set_aspect(1)
        axes[1, i].set(xlabel=None)
        axes[1, i].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))

        logger.info(f"[Figure 2A] {name}: SalVentAttn top-bottom diff vs FC-G1 | Spearman r={corr:.3f}, spin-test p={pv_spin:.3e} (n_perm={n_rand})")

        colors = [yeo7_rgb[int(k)] for k in df_yeo_surf_5k["network_int"].values[mask_label]]
        axes[0, i].scatter(x_norm, y_norm, s=5, alpha=0.7, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black",
                    line_kws={"linewidth": 1}, ax=axes[0, i])
        axes[0, i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}",
                        transform=axes[0, i].transAxes, va="top")
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-3, 3)
        axes[0, i].set_aspect("equal", adjustable="box")

    axes[0, 0].set_xlabel("SC$_{top}$ - SC$_{bottom}$")
    axes[0, 1].set_xlabel("GD$_{top}$ - GD$_{bottom}$")
    axes[0, 2].set_xlabel("MPC$_{top}$ - MPC$_{bottom}$")
    axes[0, 0].set_ylabel("FC G1")
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_2a_distance_metric.svg")
    plt.close(fig)


def struct_conn_network_analysis(df_yeo_surf_5k: pd.DataFrame,
                                 surf5k_lh_infl, surf5k_rh_infl,
                                 df_pni: pd.DataFrame, project_root: Path,
                                 spin_model: SpinPermutations,
                                 networks: list[str] = ["SalVentAttn", "Limbic"],
                                 n_rand: int = 100, hemisphere: str = "both") -> pd.DataFrame:
    """
    Figure 2B: replicate the SC-fingerprint/FC-gradient correlation for each Yeo network.

    All analysis at fsLR-5k. For each network, computes the T1-MPC gradient, identifies
    top/bottom quantile vertices, computes their SC difference to all other-network
    vertices, and correlates with the whole-brain FC gradient (spin-test corrected).

    Parameters
    ----------
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame.
    surf5k_lh_infl, surf5k_rh_infl :
        Inflated fsLR-5k surfaces.
    df_pni : pd.DataFrame
        Subject manifest with columns path_sc_5k, path_t1_profile_5k.
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

    Returns
    -------
    df_yeo_surf_5k : pd.DataFrame
        5k surface DataFrame with per-network SC-difference columns appended.
    """
    A_sc_5k = load_connectomes_5k(df_pni["path_sc_5k"].tolist(), df_yeo_surf_5k, split_hemi=False, log_transform=True)

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)

    n_col = int(np.ceil(len(networks) / 2))
    fig, axes = plt.subplots(2, n_col, figsize=(4 * n_col, 10), sharex=True, sharey=True, layout="constrained")
    axes = axes.flatten()

    for i, network in enumerate(networks):
        logger.info(f"Processing network: {network}")
        top_idx, bottom_idx, other_idx = _prepare_network_quantiles(df_yeo_surf_5k, network, df_pni, hemisphere)
        diff_col = f"{network}_diff"
        x_norm, y_norm, mask_label, corr, pv_spin = _diff_and_correlate(
            df_yeo_surf_5k, A_sc_5k, diff_col, top_idx, bottom_idx, other_idx, spin_model, n_rand)

        save_brain_map(surf5k_lh_infl, surf5k_rh_infl,
                       df_yeo_surf_5k[diff_col].values,
                       array_name="overlay2",
                       filename=project_root / f"results/figures/figure_2b_brain_SC_diff_{network}.svg",
                       hemisphere=hemisphere)

        logger.info(f"[Figure 2B] {network}: SC top-bottom diff vs FC-G1 | Spearman r={corr:.3f}, spin-test p={pv_spin:.3e} (n_perm={n_rand})")

        colors = [yeo7_rgb[int(k)] for k in df_yeo_surf_5k["network_int"].values[mask_label]]
        axes[i].scatter(x_norm, y_norm, s=5, alpha=0.7, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black",
                    line_kws={"linewidth": 1}, ax=axes[i])
        axes[i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}",
                     transform=axes[i].transAxes, va="top")
        net_color = yeo7_rgb[int(df_yeo_surf_5k.loc[
            df_yeo_surf_5k["network"] == network, "network_int"].values[0])]
        axes[i].set_title(network, fontdict={"color": net_color})
        axes[i].set_xlabel("SC$_{top}$ - SC$_{bottom}$")
        if i % n_col == 0:
            axes[i].set_ylabel("FC G1")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect("equal", adjustable="box")

    axes[-1].set_axis_off()
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_2b_distance_network.svg")
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
    logger.info("Null model     : spin permutation (SpinPermutations, n_rep=100, random_state=42)")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    surf5k_lh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.L.inflated.surf.gii", itype="gii")
    surf5k_rh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.R.inflated.surf.gii", itype="gii")

    df_pni = pd.read_csv(project_root / "data/dataframes/figure_1a_pni_to_mics_5k.csv")
    n_rand = 100
    spin_model_5k = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_5k_lh = read_surface(project_root / "data/surfaces/fsLR-5k.L.sphere.surf.gii", itype="gii")
    sphere_5k_rh = read_surface(project_root / "data/surfaces/fsLR-5k.R.sphere.surf.gii", itype="gii")
    spin_model_5k.fit(sphere_5k_lh, sphere_5k_rh)

    df_yeo_surf_5k = load_yeo_surf_5k(micapipe=project_root)

    path_df_1a_5k = project_root / f"data/dataframes/df_1a_{args.hemi}_fslr5k.tsv"
    if not path_df_1a_5k.exists():
        raise FileNotFoundError(
            f"fsLR-5k gradient dataframe not found at {path_df_1a_5k}. "
            f"Run figure_1a_t1map.py with -hemi {args.hemi} first."
        )
    logger.info(f"fsLR-5k gradient dataframe found at {path_df_1a_5k}")

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
        networks=networks, n_rand=n_rand, hemisphere=args.hemi)

    df_yeo_surf_5k.to_csv(project_root / f"data/dataframes/df_2b_label_{args.hemi}.csv", index=False)


if __name__ == "__main__":
    main()

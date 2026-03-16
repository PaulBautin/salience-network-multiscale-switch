from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 2 – Structural connectivity at MPC gradient extremes
#
# Tests whether structural connectivity differs between parcels at the high vs. low
# ends of the MPC (T1) gradient computed in Figure 1a, within the Salience/Ventral
# Attention network and across all 7 Yeo networks.
#
# Figure 2A: For the SalVentAttn network, computes connectivity differences between
#            MPC-gradient-extreme parcels using three metrics (structural connectivity,
#            greedy-navigation path length, Euclidean distance) and correlates those
#            differences with the whole brain FC gradient.
# Figure 2B: Replicates the SC-difference analysis for each of the 7 Yeo networks
#            and correlates the results with the whole brain FC gradient.
#
# Outputs:
#   results/figures/figure_2a_distance_metric.svg
#   results/figures/figure_2a_brain_{SC,Nav,Dist}_diff.svg
#   results/figures/figure_2b_distance_network.svg
#   results/figures/figure_2b_brain_SC_diff_{network}.svg
#   data/dataframes/df_2b_label_{hemisphere}.csv  (parcel-level cache)
#
# Example:
#   python scripts/figure_2_distance.py \
#     -pni_deriv /data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


#### imports
import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import seaborn as sns

from brainspace.plotting import plot_surf, plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69, load_gradient
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.null_models import SpinPermutations

import bct.algorithms as bct_alg
import bct.utils as bct

from scipy.stats import spearmanr, zscore

import matplotlib.pyplot as plt
import matplotlib as mpl

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles, convert_states_str2int, load_bigbrain_gradients
from src.gradient_computation import compute_t1_gradient
from src.plot_colors import yeo7_rgba, yeo7_rgb
from src.logging_utils import setup_manuscript_logger

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Matplotlib globals
plt.rcParams["font.size"] = 12
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False


def get_parser():
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute structural connectivity differences between MPC-gradient extremes in the salience network (Fig 2A) and across all Yeo networks (Fig 2B).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument("-pni_deriv", type=str, required=True, help="Absolute path to the PNI derivatives folder (e.g., /data/mica/mica3/...)")
    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-hemi",
        type=str,
        default="both",
        choices=["both", "LH", "RH"],
        help="Hemisphere for analysis: 'both', 'LH', or 'RH' (default: both)"
    )
    return parser


def load_label_atlas(micapipe):
    """
    Load the Schaefer-400 parcellation lookup table and derive per-parcel metadata.

    Reads the LUT CSV and extracts the Yeo 7-network label, hemisphere (LH/RH),
    and an integer network code for each parcel.

    Parameters
    ----------
    micapipe : Path
        Root path of the micapipe repository (used to locate the LUT file).

    Returns
    -------
    df_label : pd.DataFrame
        One row per Schaefer-400 parcel with columns: label, network, hemisphere,
        network_int, and parcel coordinates.
    """
    df_label = pd.read_csv(micapipe / "data/parcellations/lut/lut_schaefer-400_mics.csv")
    # df_label_sub = pd.read_csv(micapipe + '/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    # df_label = pd.concat([df_label_sub, df_label])
    df_label["network"] = df_label["label"].str.extract(r"(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)")
    df_label["hemisphere"] = df_label["label"].str.extract(r"(LH|RH)")
    #df_label["network"] = df_label["network"].fillna("medial_wall")
    df_label["network_int"] = convert_states_str2int(df_label["network"].values)[0]
    return df_label


def load_connectomes(files, df_label, log_transform=False, split_hemi=True):
    """
    Load and average subject-level structural connectome GIFTI files.

    Reads each file, masks to cortical parcels (dropping 48 subcortical nodes),
    zeros out non-positive entries, and optionally zeros inter-hemispheric connections.
    Returns the subject-averaged matrix, optionally log1p-transformed.

    Parameters
    ----------
    files : list of str
        Paths to subject-level connectome GIFTI files (Schaefer-400 + subcortex).
    df_label : pd.DataFrame
        Parcel metadata from `load_label_atlas`; used to identify cortical parcels
        and hemisphere membership.
    log_transform : bool, optional
        Apply log1p to the averaged matrix (default False).
    split_hemi : bool, optional
        Zero out inter-hemispheric connections before averaging (default True).

    Returns
    -------
    A_400 : np.ndarray, shape (n_cortex, n_cortex)
        Symmetric, subject-averaged connectivity matrix for cortical parcels.
    """
    if not files:
        raise FileNotFoundError("No connectome files found.")

    n_subcortex = 48
    cortex_mask = df_label["hemisphere"].notna().to_numpy()
    valid_idx = np.concatenate((np.zeros(n_subcortex, dtype=bool), cortex_mask))
    df_label = df_label[cortex_mask]

    conn_stack = []

    for f in files:
        data = nib.load(f).darrays[0].data  # type: ignore
        data = data[np.ix_(valid_idx, valid_idx)]
        data[data <= 0] = np.nan
        # remove LH to RH connections if split_hemi is True
        if split_hemi:
            data_lh, data_rh, lh_idx, rh_idx = split_by_hemisphere(df_label, data)
            data = np.full((data.shape[0], data.shape[0]), np.nan)
            data[np.ix_(lh_idx, lh_idx)] = data_lh
            data[np.ix_(rh_idx, rh_idx)] = data_rh
        conn_stack.append(data)

    conn = np.stack(conn_stack, axis=0)

    nan_mask = np.mean(np.isnan(conn), axis=0) > 0.5
    mean_conn = np.nanmean(conn, axis=0)
    mean_conn[nan_mask] = np.nan
    A_400 = np.nan_to_num(mean_conn, nan=0.0)

    A_400 = np.triu(A_400, k=1)  # Extract the upper triangular part of the matrix
    A_400 = A_400 + A_400.T  # Reapply symmetry
    if log_transform:
        A_400 = np.log1p(A_400)  # Log-transform to normalize the data and reduce the dynamic range of connectivity values
    return A_400


def load_connectomes_euclidian(df_label):
    """
    Compute pairwise Euclidean distances between Schaefer-400 parcel centroids.

    Inter-hemispheric pairs are set to NaN so they are excluded from analyses
    that treat hemispheres independently.

    Parameters
    ----------
    df_label : pd.DataFrame
        Parcel metadata including 'coor.x', 'coor.y', 'coor.z', and 'hemisphere'.

    Returns
    -------
    A_400 : np.ndarray, shape (n_cortex, n_cortex)
        Symmetric Euclidean distance matrix; NaN for cross-hemisphere pairs.
    """
    exclude_idx = df_label[df_label["hemisphere"].isna()].index
    coords = df_label.drop(index=exclude_idx)[["coor.x", "coor.y", "coor.z"]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    A_400 = np.linalg.norm(diff, axis=-1)
    hemispheres = df_label[~df_label["hemisphere"].isna()].hemisphere.values
    hemi_equal = hemispheres[:, None] == hemispheres[None, :]
    A_400[~hemi_equal] = np.nan
    return A_400


def split_by_hemisphere(df_label, matrix):
    """
    Split a parcellation-level matrix into left- and right-hemisphere submatrices.

    Parameters
    ----------
    df_label : pd.DataFrame
        Parcel metadata with a 'hemisphere' column ('LH'/'RH'/NaN).
    matrix : np.ndarray, shape (n_cortex, n_cortex)
        Full connectivity or distance matrix.

    Returns
    -------
    lh_mat : np.ndarray
        Submatrix for left-hemisphere parcels.
    rh_mat : np.ndarray
        Submatrix for right-hemisphere parcels.
    lh_idx : np.ndarray of int
        Row/column indices of LH parcels in the full matrix.
    rh_idx : np.ndarray of int
        Row/column indices of RH parcels in the full matrix.
    """
    cortex_mask = df_label["hemisphere"].notna().to_numpy()
    hemi = df_label.loc[cortex_mask, "hemisphere"].to_numpy()

    lh_idx = np.where(hemi == "LH")[0]
    rh_idx = np.where(hemi == "RH")[0]

    return matrix[np.ix_(lh_idx, lh_idx)], matrix[np.ix_(rh_idx, rh_idx)], lh_idx, rh_idx


def compute_navigation(df_label, A_400, A_400_euclidean):
    """
    Compute greedy navigation separately for LH and RH
    and reassemble into a full matrix.
    """
    A_len = bct.other.weight_conversion(A_400, "lengths")

    A_lh, A_rh, lh_idx, rh_idx = split_by_hemisphere(df_label, A_len)
    D_lh, D_rh, _, _ = split_by_hemisphere(df_label, A_400_euclidean)

    _, _, PL_lh, _, _ = bct_alg.navigation_wu(A_lh, D_lh)
    _, _, PL_rh, _, _ = bct_alg.navigation_wu(A_rh, D_rh)

    n = A_400.shape[0]
    PL = np.zeros((n, n), dtype=float)

    PL[np.ix_(lh_idx, lh_idx)] = PL_lh
    PL[np.ix_(rh_idx, rh_idx)] = PL_rh

    PL = np.nan_to_num(PL, nan=0.0, posinf=0.0)
    PL = bct.other.weight_conversion(PL, "lengths")

    return PL


def compute_pvals_spin(x, df_yeo_surf, df_label, spin_model, n_rand):
    """
    Compute a spin-permutation null distribution of Spearman correlations with the FC gradient.

    Rotates the FC gradient on the sphere `n_rand` times and recomputes the Spearman
    correlation between each rotated gradient and `x` at the parcel level.

    Parameters
    ----------
    x : np.ndarray, shape (n_parcels,)
        The connectivity-difference values to correlate against the FC gradient.
    df_yeo_surf : pd.DataFrame
        Per-vertex surface DataFrame containing 'fc_g1' and 'mics' columns.
    df_label : pd.DataFrame
        Parcel-level metadata with a 'mics' column used for label-based reduction.
    spin_model : SpinPermutations
        Pre-fitted brainspace spin-permutation model.
    n_rand : int
        Number of spin permutations.

    Returns
    -------
    r_spin : np.ndarray, shape (n_rand,)
        Spearman r values under the spin null model.
    """
    y_surf = df_yeo_surf["fc_g1"].values
    y_lh, y_rh = y_surf[:32492], y_surf[32492:]
    y_rotated = np.hstack(spin_model.randomize(y_lh, y_rh))
    # Compute perm pval
    r_spin = np.empty(n_rand)
    for j, perm in enumerate(y_rotated):
        perm_label = reduce_by_labels(perm, df_yeo_surf["mics"].values, target_labels=df_label["mics"].values, red_op="mean")
        mask_label = ~np.isnan(x) & ~np.isnan(perm_label)
        x_norm = zscore(x[mask_label])
        perm_label = zscore(perm_label[mask_label])
        r_spin[j] = spearmanr(x_norm, perm_label)[0]
    return r_spin


def compute_top_bottom_diff(conn, top_idx, bottom_idx, other_idx):
    """Compute z-scored top–bottom connectivity difference."""
    conn = conn.copy()
    conn[conn <= 0] = np.nan

    top = np.nanmean(conn[top_idx][:, other_idx], axis=0)
    bottom = np.nanmean(conn[bottom_idx][:, other_idx], axis=0)

    return zscore(top - bottom, nan_policy="omit"), top, bottom


def compute_quantile_mask(values, mask, q=(0.25, 0.75)):
    """
    Label parcels as high (+1) or low (-1) gradient quantile extremes.

    Parcels within `mask` that fall at or below the q[0] quantile are labeled -1;
    those at or above q[1] are labeled +1; all others are set to 0.

    Parameters
    ----------
    values : np.ndarray, shape (n_parcels,)
        Gradient values (e.g., T1 MPC gradient) at the parcel level.
    mask : np.ndarray of bool, shape (n_parcels,)
        Boolean mask selecting the parcels to include (e.g., a single network).
    q : tuple of float, optional
        Lower and upper quantile thresholds (default: (0.25, 0.75)).

    Returns
    -------
    out : np.ndarray of int, shape (n_parcels,)
        Array with values in {-1, 0, +1}.
    """
    low, high = np.nanquantile(values[mask], q)
    out = np.full(values.shape, 0)
    out[mask & (values <= low)] = -1
    out[mask & (values >= high)] = 1
    return out


def save_brain_map(surf_lh, surf_rh, values, array_name, filename):
    """Append `values` to inflated surfaces and save a brain-map screenshot."""
    surf_lh.append_array(values[:32492], name=array_name)
    surf_rh.append_array(values[32492:], name=array_name)
    surfs = {"rh1": surf_rh, "lh1": surf_lh}
    plot_surf(
        surfs, layout=[["rh1", "lh1"]], view=[["medial", "lateral"]],
        array_name=array_name, size=(1200, 500), zoom=1.4, color_bar="bottom",
        share="both", nan_color=(220, 220, 220, 1), cmap="coolwarm",
        color_range="sym", transparent_bg=True, screenshot=True, filename=filename,
    )


def struct_conn_metric_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl,
                                df_pni, project_root, spin_model, network="SalVentAttn",
                                n_rand=100, hemisphere="both"):
    """
    Structural connectivity analysis linking FC gradients and different connectivity measures,
    T1-derived gradients, and connectivity-based differences.
    """
    # load connectomes
    A_sc = load_connectomes(df_pni["path_sc"].to_list(), df_label, log_transform=True, split_hemi=False)
    A_dist = load_connectomes(df_pni["path_dist"].to_list(), df_label, log_transform=False, split_hemi=False)
    A_dist = bct.other.weight_conversion(A_dist, "lengths")

    A_euc = load_connectomes_euclidian(df_label)
    A_nav = compute_navigation(df_label, A_sc, A_euc)

    connectomes = {
        "SC": A_sc,
        "Nav": A_nav,
        "Dist": A_dist,
    }

    # FC gradient 1 — load once; negate for df_label convention, keep raw for spin test
    _fc_raw = load_gradient("fc", join=True)
    fc_g1 = reduce_by_labels(-_fc_raw, df_yeo_surf["mics"].values, target_labels=df_label["mics"].values, red_op="mean")
    df_label = df_label.copy()
    df_label["fc_g1"] = fc_g1
    df_label.loc[df_label.hemisphere.isna(), "fc_g1"] = np.nan
    df_label["fc_g1_network"] = df_label.groupby("network")["fc_g1"].transform("mean")
    df_yeo_surf["fc_g1"] = _fc_raw
    df_yeo_surf.loc[df_yeo_surf.hemisphere.isna(), "fc_g1"] = np.nan

    # Find T1-derived gradient and quantiles within the network
    valid = df_label.hemisphere.notna().values
    grad_col = f"t1_gradient1_{network}"

    # Ensure parcel-level gradient exists
    if grad_col not in df_label.columns:
        if grad_col not in df_yeo_surf.columns:
            t1_salience_profiles = load_t1_salience_profiles(df_pni["path_t1_profile"].tolist(), df_yeo_surf, network=network, hemisphere=hemisphere)
            df_yeo_surf = compute_t1_gradient(df_yeo_surf, t1_salience_profiles, network=network, hemisphere=hemisphere)
        df_label[grad_col] = reduce_by_labels(
            df_yeo_surf[grad_col].values,
            df_yeo_surf["mics"].values,
            target_labels=df_label["mics"].values,
            red_op="mean",
        )

    # Quantiles computed only within the network, per hemisphere
    net_mask = valid & (df_label.network == network)
    if hemisphere == "both":
        net_mask_lh = net_mask & (df_label["hemisphere"] == "LH")
        net_mask_rh = net_mask & (df_label["hemisphere"] == "RH")
        df_label["quantile_idx"] = (
            compute_quantile_mask(df_label[grad_col].values, net_mask_lh)
            + compute_quantile_mask(df_label[grad_col].values, net_mask_rh)
        )
    elif hemisphere == "LH":
        net_mask_lh = net_mask & (df_label["hemisphere"] == "LH")
        df_label["quantile_idx"] = compute_quantile_mask(df_label[grad_col].values, net_mask_lh)
    else:  # RH
        net_mask_rh = net_mask & (df_label["hemisphere"] == "RH")
        df_label["quantile_idx"] = compute_quantile_mask(df_label[grad_col].values, net_mask_rh)
    df_label.loc[df_label["quantile_idx"] == 0, "quantile_idx"] = np.nan

    # plot hemispheres with quantiles
    df_yeo_surf = df_yeo_surf.merge(df_label[["mics", "quantile_idx"]], on="mics", how="left", validate="many_to_one")
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf["quantile_idx"].values)

    top_idx = valid & (df_label.quantile_idx == 1)
    bottom_idx = valid & (df_label.quantile_idx == -1)
    other_idx = valid & (df_label.network != network)

    # Prepare plot
    fig, axes = plt.subplots(2, 3, figsize=(4 * 4, 10), squeeze=False, gridspec_kw={"height_ratios": [2, 1]}, sharey="row")
    for i, (name, A) in enumerate(connectomes.items()):
        diff, top, bottom = compute_top_bottom_diff(A, top_idx[valid], bottom_idx[valid], other_idx[valid])
        df_label.loc[other_idx, f"{name}_diff"] = diff

        # Brain map
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"{name}_diff"]], on="mics", how="left", validate="many_to_one")
        save_brain_map(surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf[f"{name}_diff"].values,
                       array_name="overlay", filename=project_root / f"results/figures/figure_2a_brain_{name}_diff.svg")

        # Bar plot with average A_400_diff per metric aranged by FC G1 value
        df_plot = df_label.loc[other_idx, ["network", "network_int", f"{name}_diff", "fc_g1", "fc_g1_network"]].dropna(subset=[f"{name}_diff"]).sort_values("fc_g1_network")
        palette = {net: yeo7_rgba[int(net_idx)] for net, net_idx in (df_plot[["network", "network_int"]].drop_duplicates().itertuples(index=False))}
        sns.barplot(x=df_plot["network"], y=f"{name}_diff", hue="network", data=df_plot, palette=palette, ax=axes[1, i], legend=False)
        axes[1, i].axhline(0, color="black", linewidth=1)
        axes[1, i].set_ylabel("SC$_{top}$ - SC$_{bottom}$")
        axes[1, i].tick_params(axis="x", labelrotation=90)
        axes[1, i].set_ylim(-1.5, 1.5)
        # align barplot with scatter plot x-axis shape
        axes[1, i].set_aspect(1)
        axes[1, i].set(xlabel=None)
        axes[1, i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Correlation analysis
        x = df_label[f"{name}_diff"].values
        y = df_label["fc_g1"].values
        mask_label = ~np.isnan(x) & ~np.isnan(y)
        x_norm, y_norm = zscore(x[mask_label]), zscore(y[mask_label])
        corr, _ = spearmanr(x_norm, y_norm)
        r_spin = compute_pvals_spin(x, df_yeo_surf, df_label, spin_model, n_rand)
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))
        logging.info(f"[Figure 2A] {name}: SalVentAttn top-bottom diff vs FC-G1 | Spearman r={corr:.3f}, spin-test p={pv_spin:.3e} (n_perm={n_rand})")

        # Scatter
        colors = [yeo7_rgb[int(k)] for k in df_label["network_int"].values[mask_label]]
        axes[0, i].scatter(x_norm, y_norm, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[0, i])
        axes[0, i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[0, i].transAxes, va="top")
        axes[0, 0].set_xlabel("SC$_{top}$ - SC$_{bottom}$")
        axes[0, 1].set_xlabel("Nav$_{top}$ - Nav$_{bottom}$")
        axes[0, 2].set_xlabel("Dist$_{top}$ - Dist$_{bottom}$")
        axes[0, 0].set_ylabel("FC G1")
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-3, 3)
        axes[0, i].set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_2a_distance_metric.svg")
    plt.close(fig)


def struct_conn_network_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl,
                                 df_pni, project_root, spin_model, networks=["SalVentAttn", "Limbic"],
                                 n_rand=100, hemisphere="both"):
    """
    Structural connectivity analysis linking BigBrain gradients and SC,
    T1-derived gradients across networks, and connectivity-based differences.
    """
    # load connectomes
    A_400_sc = load_connectomes(df_pni["path_sc"].tolist(), df_label, log_transform=True)

    # bigbrain gradient 2
    bigbrain_g2 = reduce_by_labels(load_bigbrain_gradients(), df_yeo_surf["mics"].values, red_op="mean")
    df_label = df_label.copy()
    df_label["bigbrain_g2"] = bigbrain_g2
    df_label.loc[df_label.hemisphere.isna(), "bigbrain_g2"] = np.nan
    df_label["bigbrain_g2_network"] = df_label.groupby("network")["bigbrain_g2"].transform("mean")
    df_yeo_surf = df_yeo_surf.merge(df_label[["mics", "bigbrain_g2"]], on="mics", how="left", validate="many_to_one")

    # make subplot of the size of network
    n_col = int(np.ceil(len(networks) / 2))
    fig, axes = plt.subplots(2, n_col, figsize=(4 * n_col, 10), sharex=True, sharey=True, layout="constrained")
    axes = axes.flatten()
    valid_mask = df_label["hemisphere"].notna().values
    for i, network in enumerate(networks):
        logging.info(f"Processing network: {network}")
        grad_col = f"t1_gradient1_{network}"
        if grad_col not in df_label:
            if grad_col not in df_yeo_surf:
                t1_salience_profiles = load_t1_salience_profiles(df_pni["path_t1_profile"].tolist(), df_yeo_surf, network=network, hemisphere=hemisphere)
                df_yeo_surf = compute_t1_gradient(df_yeo_surf, t1_salience_profiles, network=network, hemisphere=hemisphere)
            else:
                logging.info(f"{grad_col} already exists in df_yeo_surf.")
            df_label[f"t1_gradient1_{network}"] = reduce_by_labels(df_yeo_surf[f"t1_gradient1_{network}"].values, df_yeo_surf["mics"].values, target_labels=df_label["mics"].values, red_op="mean")

        # Quantiles computed only within the network (and optionally filtered by hemisphere)
        net_mask = (df_label["network"] == network) & valid_mask
        if hemisphere == "both":
            net_mask_lh = net_mask & (df_label["hemisphere"] == "LH")
            net_mask_rh = net_mask & (df_label["hemisphere"] == "RH")
            df_label["quantile_idx"] = compute_quantile_mask(df_label[grad_col].values, net_mask_lh) + compute_quantile_mask(df_label[grad_col].values, net_mask_rh)
        elif hemisphere == "LH":
            net_mask_lh = net_mask & (df_label["hemisphere"] == "LH")
            df_label["quantile_idx"] = compute_quantile_mask(df_label[grad_col].values, net_mask_lh)
        else:  # RH
            net_mask_rh = net_mask & (df_label["hemisphere"] == "RH")
            df_label["quantile_idx"] = compute_quantile_mask(df_label[grad_col].values, net_mask_rh)
        df_label.loc[df_label["quantile_idx"] == 0, "quantile_idx"] = np.nan

        # low_q, high_q = np.nanquantile(df_label.loc[net_mask, f"t1_gradient1_{network}"], [0.25, 0.75])
        # df_label["quantile_idx"] = np.nan
        # df_label.loc[net_mask & (df_label[f"t1_gradient1_{network}"] <= low_q), "quantile_idx"] = -1
        # df_label.loc[net_mask & (df_label[f"t1_gradient1_{network}"] >= high_q), "quantile_idx"] = 1

        # Structural connectivity masks
        bottom_idx = valid_mask & (df_label["quantile_idx"] == -1)
        top_idx = valid_mask & (df_label["quantile_idx"] == 1)
        other_net = valid_mask & (df_label["network"] != network)
        diff, top, bottom = compute_top_bottom_diff(A_400_sc, top_idx[valid_mask], bottom_idx[valid_mask], other_net[valid_mask])

        df_label.loc[other_net, f"{network}_diff"] = diff
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"{network}_diff"]], on="mics", how="left", validate="many_to_one")
        save_brain_map(surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf[f"{network}_diff"].values,
                       array_name="overlay2", filename=project_root / f"results/figures/figure_2b_brain_SC_diff_{network}.svg")

        # Correlation analysis
        x = df_label[f"{network}_diff"].values
        y = df_label["bigbrain_g2"].values
        mask_label = ~np.isnan(x) & ~np.isnan(y)
        x_norm, y_norm = zscore(x[mask_label]), zscore(y[mask_label])
        corr, _ = spearmanr(x_norm, y_norm)
        r_spin = compute_pvals_spin(x, df_yeo_surf, df_label, spin_model, n_rand)
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))
        logging.info(f"[Figure 2B] {network}: SC top-bottom diff vs BigBrain-G2 | Spearman r={corr:.3f}, spin-test p={pv_spin:.3e} (n_perm={n_rand})")

        # Scatter
        colors = [yeo7_rgb[int(k)] for k in df_label["network_int"].values[mask_label]]
        axes[i].scatter(x_norm, y_norm, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[i])

        axes[i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[i].transAxes, va="top")
        axes[i].set_title(f"{network}", fontdict={"color": yeo7_rgb[int(df_label.loc[df_label.network == network, "network_int"].values[0])]})
        axes[i].set_xlabel("SC$_{top}$ - SC$_{bottom}$")

        if i % n_col == 0:
            axes[i].set_ylabel("BigBrain G2")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect("equal", adjustable="box")
    axes[-1].set_axis_off()
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_2b_distance_network.svg")
    return df_label


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Setup Paths dynamically
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_2_distance", project_root, args)
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"SC metric      : iFOD2 40M streamlines, SIFT2-weighted, log-transformed")
    logger.info(f"Distance metric: streamline edge lengths (converted to lengths via BCT)")
    logger.info(f"Navigation     : greedy navigation (BCT navigation_wu), computed per hemisphere")
    logger.info(f"Null model     : spin permutation (SpinPermutations, n_rep=100, random_state=42)")
    logging.info(f"Script path: {script_path}")
    logging.info(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / "data/surfaces/fsLR-32k.L.inflated.surf.gii", itype="gii")
    surf32k_rh_infl = read_surface(project_root / "data/surfaces/fsLR-32k.R.inflated.surf.gii", itype="gii")
    surf_32k = load_conte69(join=True)

    # shared setup: subject manifest, spin permutation model
    df_pni = pd.read_csv(project_root / "data/dataframes/figure_1a_pni_to_mics.csv")
    n_rand = 100
    spin_model = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True, with_normals=False, join=False)
    spin_model.fit(sphere_lh, sphere_rh)

    # load atlases
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)
    df_yeo_surf = pd.read_csv(project_root / f"data/dataframes/df_1a_{args.hemi}.tsv")
    df_label = load_label_atlas(micapipe=project_root)

    ######### Analysis
    # Part A -- SC, navigation, distance
    struct_conn_metric_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl,
                                df_pni, project_root, spin_model, network="SalVentAttn",
                                n_rand=n_rand, hemisphere=args.hemi)
    # Part B -- per network analysis
    network = ["Limbic", "Default", "Cont", "SalVentAttn", "DorsAttn", "Vis", "SomMot"]
    df_label = struct_conn_network_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl,
                                            df_pni, project_root, spin_model, networks=network,
                                            n_rand=n_rand, hemisphere=args.hemi)
    df_label.to_csv(project_root / f"data/dataframes/df_2b_label_{args.hemi}.csv", index=False)


if __name__ == "__main__":
    main()

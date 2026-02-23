from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Ooverlapping community detection algorithms (OCDAs) code
# conda activate env_ocda
# 
# Changes to make in github code:
# AttributeError: 'dict' object has no attribute 'iteritems': change .iteritems for .items
# AttributeError: module 'itertools' has no attribute 'izip_longest': change izip_longest for zip_longest
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_2_distance.py \
#   -pni_deriv /data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0
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
import os
import seaborn as sns

from brainspace.plotting import plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.null_models import SpinPermutations

import bct.algorithms as bct_alg
import bct.utils as bct

from scipy.stats import spearmanr, zscore
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles, convert_states_str2int, load_bigbrain_gradients
from src.gradient_computation import compute_t1_gradient
from src.plot_colors import yeo7_rgba, yeo7_rgb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Matplotlib globals
plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False


def get_parser():
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Process PNI derivatives and surfaces for T1 microstructural profiles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-pni_deriv", 
        type=str, 
        required=True,
        help="Absolute path to the PNI derivatives folder (e.g., /data/mica/mica3/...)"
    )
    return parser


def load_label_atlas(micapipe):
    df_label = pd.read_csv(micapipe / 'data/parcellations/lut/lut_schaefer-400_mics.csv')
    #df_label_sub = pd.read_csv(micapipe + '/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    #df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_label['network_int'] = convert_states_str2int(df_label['network'].values)[0]
    return df_label


def load_connectomes(files, df_label, log_transform=False):
    if not files:
        raise FileNotFoundError("No connectome files found.")

    n_subcortex = 48
    cortex_mask = df_label["hemisphere"].notna().to_numpy()
    valid_idx = np.concatenate((np.zeros(n_subcortex, dtype=bool), cortex_mask))

    conn_stack = []

    for f in files:
        data = nib.load(f).darrays[0].data  # type: ignore
        data = data[np.ix_(valid_idx, valid_idx)]
        data[data <= 0] = np.nan
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
    exclude_idx = df_label[df_label['hemisphere'].isna()].index
    coords = df_label.drop(index=exclude_idx)[["coor.x", "coor.y", "coor.z"]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    A_400 = np.linalg.norm(diff, axis=-1)
    hemispheres = df_label[~df_label['hemisphere'].isna()].hemisphere.values
    hemi_equal = hemispheres[:, None] == hemispheres[None, :]
    A_400[~hemi_equal] = np.nan
    return A_400


def plot_connectome(df_label, A_400):
    # --- Sort nodes by network ---
    print(df_label)
    node_networks = df_label[~df_label.hemisphere.isna()].network.values
    sort_idx = np.argsort(node_networks)
    A_sorted = A_400[sort_idx][:, sort_idx]
    sorted_networks = node_networks[sort_idx]
    sorted_quantiles = df_label[~df_label.hemisphere.isna()].quantile_idx.values[sort_idx]

    # --- Find network boundaries ---
    boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0]
    boundaries = np.insert(boundaries,0,0)

    mpc_fig = A_sorted.copy()
    mpc_fig = rotate(mpc_fig, angle=-45, order=0, cval=np.nan)

    mpc_fig_top = A_sorted.copy()
    mpc_fig_top[(np.tri(mpc_fig_top.shape[0], mpc_fig_top.shape[0]) == 1) | (sorted_quantiles != 1)] = np.nan
    mpc_fig_top = rotate(mpc_fig_top, angle=-45, order=0, cval=np.nan)

    mpc_fig_bottom = A_sorted.copy()
    mpc_fig_bottom[(np.tri(mpc_fig_bottom.shape[0], mpc_fig_bottom.shape[0]) == 1) | (sorted_quantiles != -1)] = np.nan
    mpc_fig_bottom = rotate(mpc_fig_bottom, angle=-45, order=0, cval=np.nan)


    fig, ax = plt.subplots()
    # Overlay borders
    b_ext = np.append(boundaries,400)
    for i, b in enumerate(boundaries):
        rect = patches.Rectangle((200 * np.sqrt(2), b * np.sqrt(2)), b_ext[i+1] - b_ext[i], b_ext[i+1] - b_ext[i], linewidth=2, edgecolor=yeo7_colors.colors[i], facecolor='none', angle=45)
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.imshow(mpc_fig, cmap='Purples', origin='upper')
    #plt.imshow(mpc_fig_top, cmap='Reds', origin='upper')
    #plt.imshow(mpc_fig_bottom, cmap='Blues', origin='upper')
    plt.axis('off')
    plt.title('Upper Triangle of MPC Rotated by 45°')
    plt.tight_layout()
    plt.show()


def split_by_hemisphere(df_label, matrix):
    cortex_mask = df_label["hemisphere"].notna().to_numpy()
    hemi = df_label.loc[cortex_mask, "hemisphere"].to_numpy()

    lh_idx = np.where(hemi == "LH")[0]
    rh_idx = np.where(hemi == "RH")[0]

    return (
        matrix[np.ix_(lh_idx, lh_idx)],
        matrix[np.ix_(rh_idx, rh_idx)],
        lh_idx,
        rh_idx,
    )


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
    PL = bct.other.weight_conversion(PL, 'lengths')

    return PL


def compute_pvals_spin(x, df_yeo_surf, df_label, spin_model, n_rand):
        y_surf = df_yeo_surf["bigbrain_g2"].values
        y_lh, y_rh = y_surf[:32492], y_surf[32492:]
        y_rotated = np.hstack(spin_model.randomize(y_lh, y_rh))
        # Compute perm pval
        r_spin = np.empty(n_rand)
        for j, perm in enumerate(y_rotated):
            perm_label = reduce_by_labels(perm, df_yeo_surf['mics'].values, target_labels=df_label['mics'].values, red_op='mean')
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
    low, high = np.nanquantile(values[mask], q)
    out = np.full(values.shape, np.nan)
    out[mask & (values <= low)] = -1
    out[mask & (values >= high)] = 1
    return out


def struct_conn_metric_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, network='SalVentAttn', n_rand=100):
    """
    Structural connectivity analysis linking BigBrain gradients and different connectivity measures,
    T1-derived gradients, and connectivity-based differences.
    """
    # load connectomes
    df_pni = pd.read_csv('/local_raid/data/pbautin/software/salience-network-multiscale-switch/data/dataframes/figure_1a_pni_to_mics.csv')
    A_sc = load_connectomes(df_pni["path_sc"].to_list(), df_label, log_transform=True)
    A_dist = load_connectomes(df_pni["path_dist"].to_list(), df_label, log_transform=False)
    A_dist = bct.other.weight_conversion(A_dist, "lengths")

    A_euc = load_connectomes_euclidian(df_label)
    A_nav = compute_navigation(df_label, A_sc, A_euc)

    connectomes = {
        "SC": A_sc,
        "Nav": A_nav,
        "Dist": A_dist,
    }
    
    # bigbrain gradient 2
    bigbrain_g2 = reduce_by_labels(load_bigbrain_gradients(), df_yeo_surf['mics'].values, target_labels=df_label['mics'].values, red_op='mean')
    df_label = df_label.copy()
    df_label["bigbrain_g2"] = bigbrain_g2
    df_label.loc[df_label.hemisphere.isna(), 'bigbrain_g2'] = np.nan
    df_label["bigbrain_g2_network"] = (df_label.groupby("network")["bigbrain_g2"].transform("mean"))
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'bigbrain_g2']], on='mics', how="left", validate="many_to_one")
    df_yeo_surf['bigbrain_g2'] = load_bigbrain_gradients()
    df_yeo_surf.loc[df_yeo_surf.hemisphere.isna(), 'bigbrain_g2'] = np.nan

    # Spin permutation null model
    spin_model = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    spin_model.fit(sphere_lh, sphere_rh)

    # Find T1-derived gradient and quantiles within the network
    valid = df_label.hemisphere.notna().values
    grad_col = f"t1_gradient1_{network}"

    if grad_col not in df_label:
        df_label[grad_col] = reduce_by_labels(
            df_yeo_surf[grad_col].values,
            df_yeo_surf["mics"].values,
            target_labels=df_label["mics"].values,
            red_op="mean",
        )

    net_mask = valid & (df_label.network == network)
    df_label["quantile_idx"] = compute_quantile_mask(df_label[grad_col].values, net_mask)

    top_idx = valid & (df_label.quantile_idx == 1)
    bottom_idx = valid & (df_label.quantile_idx == -1)
    other_idx = valid & (df_label.network != network)

    # Prepare plot
    fig, axes = plt.subplots(2, 3, figsize=(4 * 4, 10), squeeze=False, gridspec_kw={"height_ratios": [2, 1]}, sharey='row')
    for i, (name, A) in enumerate(connectomes.items()):
        diff, top, bottom = compute_top_bottom_diff(A, top_idx[valid], bottom_idx[valid], other_idx[valid])
        df_label.loc[other_idx, f"{name}_diff"] = diff

        # Brain map
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"{name}_diff"]], on="mics", how="left", validate="many_to_one")
        surf32k_lh_infl.append_array(df_yeo_surf[f"{name}_diff"].values[:32492], name="overlay")
        surf32k_rh_infl.append_array(df_yeo_surf[f"{name}_diff"].values[32492:], name="overlay")
        surfs = {'rh1': surf32k_rh_infl, 'lh1': surf32k_lh_infl}
        layout = [['rh1', 'lh1']]
        view = [['medial', 'lateral']]
        screenshot_path=f"/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2a_brain_{name}_diff.svg"
        plot_surf(surfs, layout=layout, view=view, array_name="overlay", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)

        
        # Bar plot with average A_400_diff per metric aranged by bigbrain value
        df_plot = (df_label.loc[other_idx, ["network", "network_int", f"{name}_diff", "bigbrain_g2", "bigbrain_g2_network"]].dropna(subset=[f"{name}_diff"]).sort_values("bigbrain_g2_network"))
        palette = {net: yeo7_rgba[int(net_idx)] for net, net_idx in (df_plot[["network", "network_int"]].drop_duplicates().itertuples(index=False))}
        sns.barplot(x=df_plot['network'], y=f"{name}_diff", hue='network', data=df_plot, palette=palette, ax=axes[1,i], legend=False)
        axes[1,i].axhline(0, color='black', linewidth=1)
        axes[1,i].set_ylabel("SC$_{top}$ - SC$_{bottom}$")
        axes[1,i].tick_params(axis='x', labelrotation=90) 
        axes[1,i].set_ylim(-1.5, 1.5)
        # align barplot with scatter plot x-axis shape
        axes[1,i].set_aspect(1)
        axes[1,i].set(xlabel=None) 
        axes[1,i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Correlation analysis
        x = df_label[f"{name}_diff"].values
        y = df_label["bigbrain_g2"].values
        mask_label = ~np.isnan(x) & ~np.isnan(y)
        x_norm, y_norm = zscore(x[mask_label]), zscore(y[mask_label])
        corr, pval = spearmanr(x_norm, y_norm)
        r_spin = compute_pvals_spin(x, df_yeo_surf, df_label, spin_model, n_rand)
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))


        # Scatter
        colors = [yeo7_rgb[int(k)] for k in df_label["network_int"].values[mask_label]]
        axes[0, i].scatter(x_norm, y_norm, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[0, i])
        axes[0, i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[0, i].transAxes, va="top")
        axes[0, 0].set_xlabel("SC$_{top}$ - SC$_{bottom}$")
        axes[0, 1].set_xlabel("Nav$_{top}$ - Nav$_{bottom}$")
        axes[0, 2].set_xlabel("Dist$_{top}$ - Dist$_{bottom}$")
        axes[0, 0].set_ylabel("BigBrain G2")
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-3, 3)
        axes[0, i].set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2a_distance_metric.svg")


def struct_conn_network_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks=['SalVentAttn','Limbic'], n_rand=100):
    """
    Structural connectivity analysis linking BigBrain gradients and SC,
    T1-derived gradients across networks, and connectivity-based differences.
    """
    # load connectomes
    df_pni = pd.read_csv('/local_raid/data/pbautin/software/salience-network-multiscale-switch/data/dataframes/figure_1a_pni_to_mics.csv')
    A_400_sc = load_connectomes(df_pni['path_sc'].tolist(), df_label, log_transform=True)
    
    # bigbrain gradient 2
    bigbrain_g2 = reduce_by_labels(load_bigbrain_gradients(), df_yeo_surf['mics'].values, red_op='mean')
    df_label = df_label.copy()
    df_label["bigbrain_g2"] = bigbrain_g2
    df_label.loc[df_label.hemisphere.isna(), 'bigbrain_g2'] = np.nan
    df_label["bigbrain_g2_network"] = (df_label.groupby("network")["bigbrain_g2"].transform("mean"))
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'bigbrain_g2']], on='mics', how="left", validate="many_to_one")

    # Spin permutation null model
    spin_model = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    spin_model.fit(sphere_lh, sphere_rh)

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
                t1_salience_profiles = load_t1_salience_profiles(df_pni['path_t1_profile'].tolist(), df_yeo_surf, network=network)
                df_yeo_surf = compute_t1_gradient(df_yeo_surf, t1_salience_profiles, network=network)
            else:
                logging.info(f"{grad_col} already exists in df_yeo_surf.")
            df_label[f"t1_gradient1_{network}"] = reduce_by_labels(df_yeo_surf[f"t1_gradient1_{network}"].values, df_yeo_surf['mics'].values, target_labels=df_label['mics'].values, red_op='mean')
        
        # Quantiles computed only within the network
        net_mask = (df_label["network"] == network) & valid_mask
        low_q, high_q = np.nanquantile(df_label.loc[net_mask, f"t1_gradient1_{network}"], [0.25, 0.75])
        df_label["quantile_idx"] = np.nan
        df_label.loc[net_mask & (df_label[f"t1_gradient1_{network}"] <= low_q), "quantile_idx"] = -1
        df_label.loc[net_mask & (df_label[f"t1_gradient1_{network}"] >= high_q), "quantile_idx"] = 1

        # Structural connectivity masks
        bottom_idx = valid_mask & (df_label["quantile_idx"] == -1)
        top_idx = valid_mask & (df_label["quantile_idx"] == 1)
        other_net = valid_mask & (df_label["network"] != network)
        diff, top, bottom = compute_top_bottom_diff(A_400_sc, top_idx[valid_mask], bottom_idx[valid_mask], other_net[valid_mask])

        df_label.loc[other_net, f"{network}_diff"] = diff
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"{network}_diff"]], on="mics", how="left", validate="many_to_one") 

        surf32k_rh_infl.append_array(df_yeo_surf[f"{network}_diff"].values[32492:], name="overlay2")
        surf32k_lh_infl.append_array(df_yeo_surf[f"{network}_diff"].values[:32492], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'lh1': surf32k_lh_infl}
        layout = [['rh1', 'lh1']]
        view = [['medial', 'lateral']]
        screenshot_path=f"/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2b_brain_SC_diff_{network}.svg"
        plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)
        
        # Correlation analysis
        x = df_label[f"{network}_diff"].values
        y = df_label["bigbrain_g2"].values
        mask_label = ~np.isnan(x) & ~np.isnan(y)
        x_norm, y_norm = zscore(x[mask_label]), zscore(y[mask_label])
        corr, pval = spearmanr(x_norm, y_norm)
        r_spin = compute_pvals_spin(x, df_yeo_surf, df_label, spin_model, n_rand)
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))

        # Scatter
        colors = [yeo7_rgb[int(k)] for k in df_label["network_int"].values[mask_label]]
        axes[i].scatter(x_norm, y_norm, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x_norm, y=y_norm, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[i])

        axes[i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[i].transAxes, va="top")
        axes[i].set_title(f"{network}", fontdict={"color": yeo7_rgb[int(df_label.loc[df_label.network == network, 'network_int'].values[0])]})
        axes[i].set_xlabel("SC$_{top}$ - SC$_{bottom}$")

        if i % n_col == 0:
            axes[i].set_ylabel("BigBrain G2")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect("equal", adjustable="box")
    axes[-1].set_axis_off()
    plt.tight_layout()
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2b_distance_network.svg")
    return df_label


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    # Setup Paths dynamically
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    pni_deriv = Path(args.pni_deriv)
    
    logging.info(f"Script path: {script_path}")
    logging.info(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    # load atlases
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)
    df_yeo_surf = pd.read_csv(project_root / 'data/dataframes/df_1a.tsv')
    path_df_2b = project_root / 'data/dataframes/df_2b_label.csv'
    if os.path.exists(path_df_2b):
        logging.info(f"Loading existing dataframe from {path_df_2b}")
        df_label = pd.read_csv(path_df_2b)
    else:
        df_label = load_label_atlas(micapipe=project_root)

    ######### Analysisis
    # Part A -- SC, navigation, distance
    # struct_conn_metric_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, network='SalVentAttn', n_rand=100)
    # Part B -- per network analysis
    network = ['Limbic', 'Default', 'Cont', 'SalVentAttn', 'DorsAttn', 'Vis', 'SomMot']
    df_label = struct_conn_network_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks=network, n_rand=100)
    df_label.to_csv(project_root / 'data/dataframes/df_2b_label.csv', index=False)
    


if __name__ == "__main__":
    main()



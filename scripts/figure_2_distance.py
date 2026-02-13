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
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations, mesh_elements
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh
from brainspace.gradient import GradientMaps, kernels
from brainspace.null_models import SpinPermutations, moran

import bct.algorithms as bct_alg
import bct.utils as bct

from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore
import os

from scipy.ndimage import rotate
import matplotlib.patches as patches

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles, convert_states_str2int
from src.gradient_computation import compute_t1_gradient
from src.plot_colors import yeo7_rgba, yeo7_rgb



plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False

def get_parser():
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Process PNI derivatives and surfaces.",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-pni_deriv", 
        type=str, 
        help="Absolute path to the PNI derivatives folder (e.g., /data/mica/...)"
    )
    return parser


def load_yeo_surf_5k(micapipe):
    #### load yeo atlas 7 network fslr5k
    atlas_yeo_lh_5k = nib.load(micapipe + '/parcellations/schaefer-400_fslr-5k_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh_5k = nib.load(micapipe + '/parcellations/schaefer-400_fslr-5k_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh_5k[atlas_yeo_rh_5k == 1800] = 2000
    yeo_surf_5k = np.concatenate((atlas_yeo_lh_5k, atlas_yeo_rh_5k), axis=0).astype(float)
    df_yeo_surf_5k = pd.DataFrame(data={'mics': yeo_surf_5k})

    df_label = pd.read_csv(micapipe + '/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label_sub = pd.read_csv(micapipe + '/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf_5k = df_yeo_surf_5k.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    return df_yeo_surf_5k


def load_label_atlas(micapipe):
    df_label = pd.read_csv(micapipe / 'data/parcellations/lut/lut_schaefer-400_mics.csv')
    #df_label_sub = pd.read_csv(micapipe + '/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    #df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_label['network_int'] = convert_states_str2int(df_label['network'].values)[0]
    return df_label


def load_connectomes(pni_deriv, df_label):
    files = glob.glob(f"{pni_deriv}/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii")
    if len(files) == 0:
        raise FileNotFoundError("No connectome files found.")

    # Average across subjects
    conn = np.stack([nib.load(f).darrays[0].data[48:,48:] for f in files], axis=0)
    A_400 = np.nanmean(conn, axis=0)

    # Enforce symmetry and log-transform
    A_400 = np.triu(A_400, k=1)
    A_400 = np.log(A_400 + A_400.T + 1.0)
    valid_idx = df_label.hemisphere.notna().values
    A_400 = A_400[np.ix_(valid_idx, valid_idx)]
    return A_400


def load_connectomes_dist(pni_deriv, df_label):
    files = glob.glob(f"{pni_deriv}/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_space-dwi_atlas-schaefer-400_desc-iFOD2-40M-SIFT2_full-edgeLengths.shape.gii")
    if len(files) == 0:
        raise FileNotFoundError("No connectome files found.")
    # Average across subjects
    conn = np.stack([nib.load(f).darrays[0].data[48:,48:] for f in files], axis=0)
    A_400 = np.nanmean(conn, axis=0)
    # Enforce symmetry
    A_400 = np.triu(A_400, k=1)
    A_400 = A_400 + A_400.T
    valid_idx = df_label.hemisphere.notna().values
    A_400 = A_400[np.ix_(valid_idx, valid_idx)]
    A_400 = bct.other.weight_conversion(A_400, 'lengths')
    A_400 = np.nan_to_num(A_400, nan=0.0, posinf=0.0, neginf=0.0)
    return A_400


def load_connectomes_euclidian(df_label):
    exclude_idx = df_label[df_label['hemisphere'].isna()].index
    coords = df_label.drop(index=exclude_idx)[["coor.x", "coor.y", "coor.z"]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def load_bigbrain_gradients():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    gradient_lh = nib.load(project_root / 'data/parcellations/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    gradient_rh = nib.load(project_root / 'data/parcellations/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    gradient = np.concatenate((gradient_lh, gradient_rh), axis=0)
    return gradient   


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


def connectome_per_hemi(df_label, connectome):
    # left hemisphere
    lh_mask = df_label[df_label.hemisphere.notna().values].hemisphere == 'LH'
    connectome_lh = connectome.copy()
    connectome_lh[~lh_mask, :] = 0.0
    connectome_lh[:, ~lh_mask] = 0.0
    # Right hemisphere
    rh_mask = df_label[df_label.hemisphere.notna().values].hemisphere == 'RH'
    connectome_rh = connectome.copy()
    connectome_rh[~rh_mask, :] = 0.0
    connectome_rh[:, ~rh_mask] = 0.0
    return connectome_lh, connectome_rh


def compute_navigation(df_label, A_400, A_400_euclidian):
    """ 
    Compute navigation for both hemispheres and concatenate
    """
    A_400_length = bct.other.weight_conversion(A_400, 'lengths')
    A_400_length_lh, A_400_length_rh = connectome_per_hemi(df_label, A_400_length)
    A_400_euclidian_lh, A_400_euclidian_rh = connectome_per_hemi(df_label, A_400_euclidian)
    _, _, PL_wei_lh, _, _ = bct_alg.navigation_wu(A_400_length_lh, A_400_euclidian_lh)
    _, _, PL_wei_rh, _, _ = bct_alg.navigation_wu(A_400_length_rh, A_400_euclidian_rh)
    PL_wei_lh = bct.other.weight_conversion(PL_wei_lh, 'lengths')
    PL_wei_rh = bct.other.weight_conversion(PL_wei_rh, 'lengths')
    PL_wei = PL_wei_lh + PL_wei_rh
    PL_wei = np.nan_to_num(PL_wei, nan=0.0, posinf=0.0, neginf=0.0)
    return PL_wei


def struct_conn_metric_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, network='SalVentAttn', n_rand=100):
    """
    Structural connectivity analysis linking BigBrain gradients and different connectivity measures,
    T1-derived gradients, and connectivity-based differences.
    """
    # load connectomes
    A_400_sc = load_connectomes(pni_deriv, df_label)
    A_400_dist = load_connectomes_dist(pni_deriv, df_label)
    A_400_euclidian = load_connectomes_euclidian(df_label)
    A_400_nav = compute_navigation(df_label, A_400_sc, A_400_euclidian)
    
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
    n_lh = sphere_lh.n_points

    # Prepare plot
    fig, axes = plt.subplots(2, 3, figsize=(4 * 4, 10), squeeze=False, gridspec_kw={"height_ratios": [2, 1]}, sharey='row')
    valid_mask = df_label["hemisphere"].notna().values
    for i, conn in enumerate([A_400_sc, A_400_nav, A_400_dist]):
        path = pni_deriv + '/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_atlas-schaefer-400_desc-intensity_profiles.shape.gii'
        t1_salience_profiles = load_t1_salience_profiles(path, df_label, network=network)
        df_label = compute_t1_gradient(df_label, t1_salience_profiles, network=network)

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

        # Compute connectivity difference between bottom and top
        A_bottom = np.nanmean(conn[bottom_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_top = np.nanmean(conn[top_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_diff = zscore(A_top - A_bottom, nan_policy="omit")

        df_label.loc[other_net, f"A_400_diff{network}{i}"] = A_diff.astype(float)
        df_label.loc[other_net, f"A_400_top{network}{i}"] = A_top.astype(float)
        df_label.loc[other_net, f"A_400_bottom{network}{i}"] = A_bottom.astype(float)
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"A_400_diff{network}{i}", f"A_400_top{network}{i}", f"A_400_bottom{network}{i}"]], on="mics", how="left", validate="many_to_one") 

        surf32k_rh_infl.append_array(df_yeo_surf[f"A_400_diff{network}{i}"].values[32492:], name="overlay2")
        surf32k_lh_infl.append_array(df_yeo_surf[f"A_400_diff{network}{i}"].values[:32492], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'lh1': surf32k_lh_infl}
        layout = [['rh1', 'lh1']]
        view = [['medial', 'lateral']]
        screenshot_path=f"/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2a_brain_dist{i}.svg"
        plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)

        
        # Bar plot with average A_400_diff per metric aranged by bigbrain value
        df_plot = (df_label.loc[other_net, ["network", "network_int", f"A_400_diff{network}{i}", "bigbrain_g2", "bigbrain_g2_network"]].dropna(subset=[f"A_400_diff{network}{i}"]).sort_values("bigbrain_g2_network"))
        palette = {net: yeo7_rgba[int(net_idx)] for net, net_idx in (df_plot[["network", "network_int"]].drop_duplicates().itertuples(index=False))}
        sns.barplot(x=df_plot['network'], y=f"A_400_diff{network}{i}", hue='network', data=df_plot, palette=palette, ax=axes[1,i], legend=False)
        axes[1,i].axhline(0, color='black', linewidth=1)
        axes[1,i].set_ylabel("SC$_{top}$ - SC$_{bottom}$")
        axes[1,i].tick_params(axis='x', labelrotation=90) 
        axes[1,i].set_ylim(-1.5, 1.5)
        # align barplot with scatter plot x-axis shape
        axes[1,i].set_aspect(1)
        axes[1,i].set(xlabel=None) 
        axes[1,i].yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

        # Correlation analysis
        x = df_label[f"A_400_diff{network}{i}"].values
        y = zscore(df_label["bigbrain_g2"].values, nan_policy="omit")

        # Surface maps
        x_surf = df_yeo_surf[f"A_400_diff{network}{i}"].values
        y_surf = zscore(df_yeo_surf["bigbrain_g2"].values, nan_policy="omit")
        corr, pval = spearmanr(x_surf, y_surf, nan_policy="omit")
        # Split hemispheres
        x_lh, x_rh = x_surf[:n_lh], x_surf[n_lh:]
        # Generate rotated surrogate maps
        x_rotated = np.hstack(spin_model.randomize(x_lh, x_rh))
        # Compute perm pval
        r_spin = np.empty(n_rand)
        mask = ~np.isnan(y_surf)
        for j, perm in enumerate(x_rotated):
            mask_rot = mask & ~np.isnan(perm)
            r_spin[j] = spearmanr(perm[mask_rot], y_surf[mask_rot], nan_policy="omit")[0]
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))

        # Scatter
        colors = [yeo7_rgb[int(k)] for k in df_label["network_int"]]
        axes[0, i].scatter(x, y, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[0, i])
        axes[0, i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[0, i].transAxes, va="top")
        axes[0, 0].set_xlabel("SC$_{top}$ - SC$_{bottom}$")
        axes[0, 1].set_xlabel("Nav$_{top}$ - Nav$_{bottom}$")
        axes[0, 2].set_xlabel("Dist$_{top}$ - Dist$_{bottom}$")
        axes[0, 0].set_ylabel("BigBrain G2")
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-3, 3)
        axes[0, i].set_aspect("equal", adjustable="box")
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2a_distance_metric.svg")


def struct_conn_network_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks=['SalVentAttn','Limbic'], n_rand=100):
    """
    Structural connectivity analysis linking BigBrain gradients and SC,
    T1-derived gradients across networks, and connectivity-based differences.
    """
    # load connectomes
    A_400_sc = load_connectomes(pni_deriv, df_label)
    
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
    n_lh = sphere_lh.n_points

    # make subplot of the size of network
    n_col = int(np.ceil(len(networks) / 2))
    fig, axes = plt.subplots(2, n_col, figsize=(4 * n_col, 10), sharex=True, sharey=True, layout="constrained")
    axes = axes.flatten()
    valid_mask = df_label["hemisphere"].notna().values
    for i, network in enumerate(networks):
        print(f"Processing network: {network}")
        path = pni_deriv + '/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_atlas-schaefer-400_desc-intensity_profiles.shape.gii'
        t1_salience_profiles = load_t1_salience_profiles(path, df_label, network=network)
        df_label = compute_t1_gradient(df_label, t1_salience_profiles, network=network)
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"t1_gradient1_{network}"]], on="mics", how="left", validate="many_to_one")
        
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

        # Compute connectivity difference between bottom and top
        A_bottom = np.nanmean(A_400_sc[bottom_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_top = np.nanmean(A_400_sc[top_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_diff = zscore(A_top - A_bottom, nan_policy="omit")

        df_label.loc[other_net, f"A_400_diff{network}{i}"] = A_diff
        df_label.loc[other_net, f"A_400_top{network}{i}"] = A_top
        df_label.loc[other_net, f"A_400_bottom{network}{i}"] = A_bottom
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"A_400_diff{network}{i}"]], on="mics", how="left", validate="many_to_one") 

        surf32k_rh_infl.append_array(df_yeo_surf[f"A_400_diff{network}{i}"].values[32492:], name="overlay2")
        surf32k_lh_infl.append_array(df_yeo_surf[f"A_400_diff{network}{i}"].values[:32492], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'lh1': surf32k_lh_infl}
        layout = [['rh1', 'lh1']]
        view = [['medial', 'lateral']]
        screenshot_path=f"/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2b_brain_dist_{network}.svg"
        plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)
        
        # Correlation analysis
        x = df_label[f"A_400_diff{network}{i}"].values
        y = zscore(df_label["bigbrain_g2"].values, nan_policy="omit")

        # Surface maps
        x_surf = df_yeo_surf[f"A_400_diff{network}{i}"].values
        y_surf = zscore(df_yeo_surf["bigbrain_g2"].values, nan_policy="omit")
        corr, pval = spearmanr(x_surf, y_surf, nan_policy="omit")
        # Split hemispheres
        x_lh, x_rh = x_surf[:n_lh], x_surf[n_lh:]
        # Generate rotated surrogate maps
        x_rotated = np.hstack(spin_model.randomize(x_lh, x_rh))
        # Compute perm pval
        r_spin = np.empty(n_rand)
        mask = ~np.isnan(y_surf)
        for j, perm in enumerate(x_rotated):
            mask_rot = mask & ~np.isnan(perm)
            r_spin[j] = spearmanr(perm[mask_rot], y_surf[mask_rot], nan_policy="omit")[0]
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))

        # Scatter
        colors = [yeo7_rgb[int(k)] for k in df_label["network_int"]]
        axes[i].scatter(x, y, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[i])

        axes[i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[i].transAxes, va="top")
        axes[i].set_title(f"{network}", fontdict={"color": yeo7_rgb[int(df_label.loc[df_label.network == network, 'network_int'].values[0])]})
        axes[i].set_xlabel("SC$_{top}$ - SC$_{bottom}$")

        if i % n_col == 0:
            axes[i].set_ylabel("BigBrain G2")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect("equal", adjustable="box")
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_2b_distance_network.svg")


def main():
    # Setup Relative Paths
    parser = get_parser()
    args = parser.parse_args()
    pni_deriv=args.pni_deriv
    script_path = Path(__file__).resolve()
    print(f"Script path: {script_path}")
    project_root = script_path.parent.parent
    print(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    # load atlases
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)
    df_label = load_label_atlas(micapipe=project_root)

    ######### Analysisis
    # Part A -- SC, navigation, distance
    struct_conn_metric_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, network='SalVentAttn', n_rand=1000)
    # Part B -- per network analysis
    network = ['Limbic', 'Default', 'Cont', 'SalVentAttn', 'DorsAttn', 'Vis', 'SomMot']
    struct_conn_network_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks=network, n_rand=1000)


if __name__ == "__main__":
    main()



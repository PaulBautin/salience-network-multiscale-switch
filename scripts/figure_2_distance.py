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
from argparse import Namespace
import glob
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
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

from lapy import Solver, TriaMesh, heat
from lapy import diffgeo
import matplotlib as mp

import bct.algorithms as bct_alg
import bct.utils as bct

from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore
import os
from figure_1_t1map import compute_t1_gradient, load_yeo_atlas, load_t1_salience_profiles, convert_states_str2int


from scipy.ndimage import rotate
import matplotlib.patches as patches
import matplotlib as mpl

## Yeo 2011, 7 network colors
yeo7_rgb = np.array([
    [255, 180, 80],    # Frontoparietal (brighter orange)
    [230, 90, 100],    # Default Mode (brighter red)
    [0, 170, 50],      # Dorsal Attention (more vivid green)
    [225, 225, 180],   # Limbic (lighter yellow-green)
    [210, 100, 255],   # Ventral Attention (lighter purple)
    [100, 160, 220],   # Somatomotor (lighter blue)
    [170, 70, 200],     # Visual (brighter violet)
    [0, 0, 0],     # Visual black
], dtype=float) / 255  # Normalize to 0–1
# Optional alpha channel for transparency
alpha = np.ones((8, 1))  # All fully opaque
yeo7_rgba = np.hstack((yeo7_rgb, alpha))
yeo7_colors = mp.colors.ListedColormap(yeo7_rgba)

new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


def autofix(W, copy=True):
    '''
    Fix a bunch of common problems. More specifically, remove Inf and NaN,
    ensure exact binariness and symmetry (i.e. remove floating point
    instability), and zero diagonal.


    Parameters
    ----------
    W : np.ndarray
        weighted connectivity matrix
    copy : bool
        if True, returns a copy of the matrix. Otherwise, modifies the matrix
        in place. Default value=True.

    Returns
    -------
    W : np.ndarray
        connectivity matrix with fixes applied
    '''
    if copy:
        W = W.copy()

    # zero diagonal
    np.fill_diagonal(W, 0)

    # remove np.inf and np.nan
    W[np.where(np.isinf(W))] = 0
    W[np.where(np.isnan(W))] = 0

    # ensure exact binarity
    u = np.unique(W)
    if np.all(np.logical_or(np.abs(u) < 1e-8, np.abs(u - 1) < 1e-8)):
        W = np.around(W, decimals=5)

    # ensure exact symmetry
    if np.allclose(W, W.T):
        W = np.around(W, decimals=5)

    return W

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
    df_label = pd.read_csv(micapipe + '/parcellations/lut/lut_schaefer-400_mics.csv')
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

    # Enforce symmetry and log-transform
    A_400 = np.triu(A_400, k=1)
    A_400 = A_400 + A_400.T
    valid_idx = df_label.hemisphere.notna().values
    A_400 = A_400[np.ix_(valid_idx, valid_idx)]
    return A_400


def load_connectomes_euclidian(df_label):
    exclude_idx = df_label[df_label['hemisphere'].isna()].index
    coords = df_label.drop(index=exclude_idx)[["coor.x", "coor.y", "coor.z"]].to_numpy(dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def load_bigbrain_gradients():
    gradient_lh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    gradient_rh = nib.load('/local_raid/data/pbautin/software/BigBrainWarp/spaces/tpl-fs_LR/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
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


def struct_conn_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks='Limbic'):
    """
    Structural connectivity analysis linking BigBrain gradients,
    T1-derived gradients, and navigation-based connectivity differences.
    """
    A_400 = load_connectomes(pni_deriv, df_label)
    # plot_connectome(df_label, A_400)
    A_400_dist = load_connectomes_dist(pni_deriv, df_label)
    # plot_connectome(df_label, A_400_dist)
    A_400_euclidian = load_connectomes_euclidian(df_label)
    print(df_label)
    lh_mask = df_label[df_label.hemisphere.notna().values].hemisphere == 'LH'
    A_400_euclidian_lh = A_400_euclidian.copy()
    A_400_euclidian_lh[~lh_mask, :] = 0.0
    A_400_euclidian_lh[:, ~lh_mask] = 0.0
    A_400_length_lh = bct.other.weight_conversion(A_400, 'lengths')
    A_400_length_lh[~lh_mask, :] = 0.0
    A_400_length_lh[:, ~lh_mask] = 0.0
    _, _, PL_wei_lh, _, _ = bct_alg.navigation_wu(A_400_length_lh, A_400_euclidian_lh)
    PL_wei_lh = bct.other.weight_conversion(PL_wei_lh, 'lengths')

    rh_mask = df_label[df_label.hemisphere.notna().values].hemisphere == 'RH'
    A_400_euclidian_rh = A_400_euclidian.copy()
    A_400_euclidian_rh[~rh_mask, :] = 0.0
    A_400_euclidian_rh[:, ~rh_mask] = 0.0
    A_400_length_rh = bct.other.weight_conversion(A_400, 'lengths')
    A_400_length_rh[~rh_mask, :] = 0.0
    A_400_length_rh[:, ~rh_mask] = 0.0
    _, _, PL_wei_rh, _, _ = bct_alg.navigation_wu(A_400_length_rh, A_400_euclidian_rh)
    PL_wei_rh = bct.other.weight_conversion(PL_wei_rh, 'lengths')
    PL_wei = PL_wei_lh + PL_wei_rh
    PL_wei = np.nan_to_num(PL_wei, nan=0.0, posinf=0.0, neginf=0.0)

    A_400_dist = bct.other.weight_conversion(A_400_euclidian, 'lengths')
    # plot_connectome(df_label, PL_wei)
    
    # bigbrain gradient 2
    bigbrain_g2 = zscore(reduce_by_labels(load_bigbrain_gradients(), df_yeo_surf['mics'].values, red_op='mean'))
    df_label = df_label.copy()
    df_label["bigbrain_g2"] = bigbrain_g2
    df_label["bigbrain_g2_network"] = (df_label.groupby("network")["bigbrain_g2"].transform("mean"))
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'bigbrain_g2']], on='mics', validate="many_to_one", how='left')


    ## Spin permutation null model
    n_rand = 1000
    spin_model = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    spin_model.fit(sphere_lh, sphere_rh)
    n_lh = sphere_lh.n_points

    # make subplot of the size of network
    fig, axes = plt.subplots(2, 3, figsize=(4 * 4, 10), squeeze=False, gridspec_kw={"height_ratios": [2, 1]}, sharey='row')
    valid_mask = df_label["hemisphere"].notna().values
    for i, A_400 in enumerate([A_400, PL_wei, A_400_dist]):
        print(A_400.shape)
        network = networks[0]
        path = f"{pni_deriv}/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_atlas-schaefer-400_desc-intensity_profiles.shape.gii"
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

        A_bottom = np.nanmean(A_400[bottom_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_top = np.nanmean(A_400[top_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_diff = zscore(A_top - A_bottom, nan_policy="omit")

        df_label.loc[other_net, f"A_400_diff{network}{i}"] = A_diff.astype(float)
        df_label.loc[other_net, f"A_400_top{network}{i}"] = A_top.astype(float)
        df_label.loc[other_net, f"A_400_bottom{network}{i}"] = A_bottom.astype(float)
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"A_400_diff{network}{i}", f"A_400_top{network}{i}", f"A_400_bottom{network}{i}"]], on="mics", how="left", validate="many_to_one",)
        screenshot_path=f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure2_dist{i}_brain.svg"
        # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf[f"A_400_diff{network}{i}"].values, size=(400, 400), zoom=1.3, color_bar='bottom', share='both',
        #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym', layout_style="grid", screenshot=True, filename=screenshot_path) 
        

        surf32k_rh_infl.append_array(df_yeo_surf[f"A_400_diff{network}{i}"].values[32492:], name="overlay2")
        surf32k_lh_infl.append_array(df_yeo_surf[f"A_400_diff{network}{i}"].values[:32492], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'lh1': surf32k_lh_infl}
        layout = [['rh1', 'lh1']]
        view = [['medial', 'lateral']]
        plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)
        #p.show()
        print(df_yeo_surf)
        
        # Bar plot with average A_400_diff per network aranged by bigbrain valu
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
        import matplotlib.ticker as ticker
        axes[1,i].yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Correlation analysis
        x = zscore(df_plot[f"A_400_diff{network}{i}"].values)
        y = df_plot["bigbrain_g2"].values
        corr, pval = spearmanr(x, y, nan_policy="omit")

        # Surface maps
        x_surf = zscore(df_yeo_surf[f"A_400_diff{network}{i}"].values)
        y_surf = df_yeo_surf["bigbrain_g2"].values
        # Split hemispheres
        x_lh, x_rh = x_surf[:n_lh], x_surf[n_lh:]
        y_lh, y_rh = y_surf[:n_lh], y_surf[n_lh:]
        # Generate rotated surrogate maps
        x_rotated = np.hstack(spin_model.randomize(x_lh, x_rh))
        # Compute perm pval
        r_spin = np.empty(n_rand)
        for j, perm in enumerate(x_rotated):
            r_spin[j] = spearmanr(perm, x_surf, nan_policy="omit")[0]
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))

        # # Plot null dist
        # ins = axes[0, i].inset_axes(bounds=[0.05,0.75,0.2,0.2])
        # ins.hist(r_spin, bins=25, density=True, alpha=0.5, color=(.8, .8, .8))
        # ins.axvline(corr, lw=2, ls='--', color='k')
        # ins.get_xaxis().set_visible(False)
        # ins.get_yaxis().set_visible(False)
        # # axs[k].set_xlabel(f'Correlation with {fn}')
        # # if k == 0:
        # # axs[k].set_ylabel('Density')
        

        # Scatter + regression
        colors = [yeo7_rgb[int(k)] for k in df_plot["network_int"]]
        axes[0, i].scatter(x, y, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[0, i])

        axes[0, i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[0, i].transAxes, va="top")

        if i == 0: 
            axes[0, i].set_ylabel("BigBrain G2")
        axes[0, i].set_xlim(-3, 3)
        axes[0, i].set_ylim(-3, 3)
        axes[0, i].set_aspect("equal", 'box')
        axes[0, i].set_xlabel("SC$_{top}$ - SC$_{bottom}$")
        if i == 1:
            axes[0, i].set_xlabel("Nav$_{top}$ - Nav$_{bottom}$")
        if i == 2:
            axes[0, i].set_xlabel("Dist$_{top}$ - Dist$_{bottom}$")

    plt.tight_layout()
    #plt.show()
    plt.savefig("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure2_distance_conn.svg")



def struct_conn_analysis_network(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks='Limbic'):
    """
    Structural connectivity analysis linking BigBrain gradients,
    T1-derived gradients, and navigation-based connectivity differences.
    """
    A_400 = load_connectomes(pni_deriv, df_label)
    # plot_connectome(df_label, A_400)
    # A_400_dist = load_connectomes_dist(pni_deriv, df_label)
    # # plot_connectome(df_label, A_400_dist)
    # A_400_euclidian = load_connectomes_euclidian(df_label)
    # print(df_label)
    # lh_mask = df_label[df_label.hemisphere.notna().values].hemisphere == 'LH'
    # A_400_euclidian_lh = A_400_euclidian.copy()
    # A_400_euclidian_lh[~lh_mask, :] = 0.0
    # A_400_euclidian_lh[:, ~lh_mask] = 0.0
    # A_400_length_lh = bct.other.weight_conversion(A_400, 'lengths')
    # A_400_length_lh[~lh_mask, :] = 0.0
    # A_400_length_lh[:, ~lh_mask] = 0.0
    # _, _, PL_wei_lh, _, _ = bct_alg.navigation_wu(A_400_length_lh, A_400_euclidian_lh)
    # PL_wei_lh = bct.other.weight_conversion(PL_wei_lh, 'lengths')

    # rh_mask = df_label[df_label.hemisphere.notna().values].hemisphere == 'RH'
    # A_400_euclidian_rh = A_400_euclidian.copy()
    # A_400_euclidian_rh[~rh_mask, :] = 0.0
    # A_400_euclidian_rh[:, ~rh_mask] = 0.0
    # A_400_length_rh = bct.other.weight_conversion(A_400, 'lengths')
    # A_400_length_rh[~rh_mask, :] = 0.0
    # A_400_length_rh[:, ~rh_mask] = 0.0
    # _, _, PL_wei_rh, _, _ = bct_alg.navigation_wu(A_400_length_rh, A_400_euclidian_rh)
    # PL_wei_rh = bct.other.weight_conversion(PL_wei_rh, 'lengths')
    # PL_wei = PL_wei_lh + PL_wei_rh
    # PL_wei = np.nan_to_num(PL_wei, nan=0.0, posinf=0.0, neginf=0.0)
    # plt.imshow(PL_wei, cmap='viridis')
    # plt.show()

    # A_400_dist = bct.other.weight_conversion(A_400_euclidian, 'lengths')
    # # plot_connectome(df_label, PL_wei)
    
    # bigbrain gradient 2
    bigbrain_g2 = zscore(reduce_by_labels(load_bigbrain_gradients(), df_yeo_surf['mics'].values, red_op='mean'))
    df_label = df_label.copy()
    df_label["bigbrain_g2"] = bigbrain_g2
    df_label["bigbrain_g2_network"] = (df_label.groupby("network")["bigbrain_g2"].transform("mean"))
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'bigbrain_g2']], on='mics', validate="many_to_one", how='left')


    ## Spin permutation null model
    n_rand = 1000
    spin_model = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    spin_model.fit(sphere_lh, sphere_rh)
    n_lh = sphere_lh.n_points

    # make subplot of the size of network
    n_col = np.ceil(len(networks) / 2).astype(int)
    fig, axes = plt.subplots(2, n_col, figsize=(4 * np.ceil(len(networks) / 2).astype(int), 10), sharex=True, sharey=True, layout="constrained")
    valid_mask = df_label["hemisphere"].notna().values
    for i, network in enumerate(networks):
        axes = axes.flatten()
        path = f"{pni_deriv}/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_atlas-schaefer-400_desc-intensity_profiles.shape.gii"
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

        A_bottom = np.nanmean(A_400[bottom_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_top = np.nanmean(A_400[top_idx[valid_mask]][:, other_net[valid_mask]], axis=0)
        A_diff = zscore(A_top - A_bottom, nan_policy="omit")

        df_label.loc[other_net, f"A_400_diff{network}{i}"] = A_diff.astype(float)
        df_label.loc[other_net, f"A_400_top{network}{i}"] = A_top.astype(float)
        df_label.loc[other_net, f"A_400_bottom{network}{i}"] = A_bottom.astype(float)
        df_yeo_surf = df_yeo_surf.merge(df_label[["mics", f"A_400_diff{network}{i}", f"A_400_top{network}{i}", f"A_400_bottom{network}{i}"]], on="mics", how="left", validate="many_to_one",)
        screenshot_path=f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure2_dist{i}_brain.svg"
        plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf[f"A_400_diff{network}{i}"].values, size=(500, 400), zoom=1.4, color_bar='right', share='both',
                nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym', layout_style="grid", screenshot=True, filename=screenshot_path) 
        # print(df_yeo_surf)
        
        # Bar plot with average A_400_diff per network aranged by bigbrain valu
        df_plot = (df_label.loc[other_net, ["network", "network_int", f"A_400_diff{network}{i}", "bigbrain_g2", "bigbrain_g2_network"]].dropna(subset=[f"A_400_diff{network}{i}"]).sort_values("bigbrain_g2_network"))

        # Correlation analysis
        x = zscore(df_plot[f"A_400_diff{network}{i}"].values, nan_policy="omit")
        y = df_plot["bigbrain_g2"].values
        corr, pval = spearmanr(x, y, nan_policy="omit")
        print(f"Correlation {network}: r={corr:.2f}, p={pval:.2e}")

        # Surface maps
        x_surf = zscore(df_yeo_surf[f"A_400_diff{network}{i}"].values, nan_policy="omit")
        y_surf = df_yeo_surf["bigbrain_g2"].values
        # Split hemispheres
        x_lh, x_rh = x_surf[:n_lh], x_surf[n_lh:]
        # Generate rotated surrogate maps
        x_rotated = np.hstack(spin_model.randomize(x_lh, x_rh))
        # Compute perm pval
        r_spin = np.empty(n_rand)
        for j, perm in enumerate(x_rotated):
            r_spin[j] = spearmanr(perm, x_surf, nan_policy="omit")[0]
        pv_spin = np.mean(np.abs(r_spin) >= np.abs(corr))

        # # Plot null dist
        # ins = axes[0, i].inset_axes(bounds=[0.05,0.75,0.2,0.2])
        # ins.hist(r_spin, bins=25, density=True, alpha=0.5, color=(.8, .8, .8))
        # ins.axvline(corr, lw=2, ls='--', color='k')
        # ins.get_xaxis().set_visible(False)
        # ins.get_yaxis().set_visible(False)
        # # axs[k].set_xlabel(f'Correlation with {fn}')
        # # if k == 0:
        # # axs[k].set_ylabel('Density')
        

        # Scatter + regression
        colors = [yeo7_rgb[int(k)] for k in df_plot["network_int"]]
        axes[i].scatter(x, y, s=10, alpha=0.9, c=colors, rasterized=True)
        sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[i])

        axes[i].text(0.05, 0.95, f"r = {corr:.2f}\np = {pv_spin:.2e}", transform=axes[i].transAxes, va="top")
        axes[i].set_title(f"{network}", fontdict={"color": yeo7_rgb[int(df_label.loc[df_label.network == network, 'network_int'].values[0])]})
        axes[i].set_xlabel("SC$_{top}$ - SC$_{bottom}$")

        if i == 0 or i == 4: 
            axes[0].set_ylabel("BigBrain G2")
        axes[i].set_xlim(-3, 3)
        axes[i].set_ylim(-3, 3)
        axes[i].set_aspect("equal", 'box')

    plt.tight_layout()
    #plt.show()
    plt.savefig("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure2_distance_net.svg")


def graph_theory_analysis(df_label, pni_deriv, network='SalVentAttn'):
    A_400 = load_connectomes(pni_deriv)
    # metrics
    degree = bct_alg.degrees_und(A_400)
    clustering = bct_alg.clustering_coef_wu(A_400)
    betweenness = bct_alg.betweenness_wei(A_400)


def main():
    #### Define paths
    micapipe='/local_raid/data/pbautin/software/micapipe'
    pni_deriv = '/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0'

    #### load surfaces
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf5k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
    surf5k_lh_sphere = read_surface(micapipe + '/surfaces/fsLR-5k.L.sphere.surf.gii', itype='gii')
    surf5k_rh_sphere = read_surface(micapipe + '/surfaces/fsLR-5k.R.sphere.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    df_yeo_surf = load_yeo_atlas(micapipe, surf_32k)
    df_label = load_label_atlas(micapipe)

    network = 'SalVentAttn'
    path = pni_deriv + '/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_atlas-schaefer-400_desc-intensity_profiles.shape.gii'
    t1_salience_profiles = load_t1_salience_profiles(path, df_label, network='SalVentAttn')
    df_label = compute_t1_gradient(df_label, t1_salience_profiles, network='SalVentAttn')
    # Compute quantile thresholds
    df_label["quantile_idx"] = np.nan
    low_thresh = np.nanquantile(df_label['t1_gradient1_' + network].values, 0.25)
    high_thresh = np.nanquantile(df_label['t1_gradient1_' + network].values, 1 - 0.25)
    df_label.loc[df_label['t1_gradient1_' + network] <= low_thresh, 'quantile_idx'] = -1
    df_label.loc[df_label['t1_gradient1_' + network] >= high_thresh, 'quantile_idx'] = 1

    ######### Part 1
    # show salience network uniqueness from graph theory metrics
    #graph_theory_analysis(df_label, pni_deriv, network)

    ######### Part 2
    network = ['Limbic', 'Default', 'Cont','SalVentAttn', 'DorsAttn', 'Vis', 'SomMot']
    #network = ['SalVentAttn', 'SomMot']
    #struct_conn_analysis(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks=['SalVentAttn'])
    struct_conn_analysis_network(df_label, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, pni_deriv, networks=network)







    df_label.loc[~df_label.hemisphere.isna(), 'A_bottom'] = np.mean(A_400[df_label[~df_label.hemisphere.isna()].quantile_idx == -1, :], axis=0)
    df_label.loc[~df_label.hemisphere.isna(), 'A_top'] = np.mean(A_400[df_label[~df_label.hemisphere.isna()].quantile_idx == 1, :], axis=0)
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'A_bottom', 'A_top']], on='mics', validate="many_to_one", how='left')
    A_diff = df_yeo_surf['A_top'].values - df_yeo_surf['A_bottom'].values
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['A_bottom'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='Blues', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['A_top'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='Reds', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=A_diff, size=(500, 400), zoom=1.4, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym', layout_style="grid")
    

    # # Z-score for normalization
    # df_corr = pd.DataFrame({
    #     'network': df_yeo_surf['network'],
    #     'corr_diff': zscore(A_diff, nan_policy='omit')
    # })

    # # Compute mean correlation difference per network
    # df_corr_mean = (
    #     df_corr
    #     .dropna(subset=['corr_diff'])
    #     .groupby('network')['corr_diff']
    #     .mean()
    # )
   
    # # Create SpinPermutations model (1000 rotations)
    # spin_model = SpinPermutations(n_rep=100, random_state=42)
    # spin_model.fit(sphere32k_lh, sphere32k_rh)

    # # Split data into hemispheres
    # n_lh = surf32k_lh_infl.n_points
    # # Generate rotated surrogate maps
    # corr_spins = np.hstack(spin_model.randomize(A_diff[:n_lh], A_diff[n_lh:]))  # shape: (1000, n_vertices)
    # print(corr_spins.shape)

    # # Compute mean per network for each permutation
    # df_corr_spin = pd.DataFrame({
    #     'network': df_yeo_surf['network']
    # })

    # spin_means = []
    # for perm in corr_spins:
    #     df_corr_spin['corr_diff'] = perm
    #     spin_means.append(
    #         df_corr_spin.dropna().groupby('network')['corr_diff'].mean().reindex(df_corr_mean.index)
    #     )
    # spin_means = np.vstack(spin_means)
    # # Compute null mean and 95% CI across spins
    # spin_mean = np.nanmean(spin_means, axis=0)
    # spin_ci = np.nanstd(spin_means, axis=0) * 1.96  # 95% CI

    # # ---------------------------------------------------------------------
    # # Plot: bars = null model (spin), scatter = real data
    # # ---------------------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(6, 4))
    # x = np.arange(len(df_corr_mean))

    # # Bars for null distribution mean ± 95% CI
    # ax.bar(x, spin_mean, yerr=spin_ci, color='lightgrey', edgecolor='black', alpha=0.8, capsize=3, label='Spin null mean ± 95% CI')
    # #ax.errorbar(x, spin_mean, yerr=spin_ci)

    # # Scatter points for empirical correlation difference
    # ax.scatter(x, df_corr_mean.values, color=yeo7_rgba[:-1], alpha=0.8, s=100, zorder=5, label='Empirical mean')

    # # Aesthetics
    # ax.axhline(0, color='black', linewidth=1)
    # ax.set_xticks(x)
    # ax.set_xticklabels(df_corr_mean.index, rotation=90, ha='center', fontsize=10)
    # ax.set_ylabel('Mean correlation difference (Top – Bottom)', fontsize=12)
    # ax.set_title('Empirical vs. Spin null network correlation differences', fontsize=13, pad=12)
    # #ax.legend(frameon=False, fontsize=10)
    # plt.tight_layout()
    # plt.show()


    ### Structural connectivity
    A_400 = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/func/desc-me_task-rest_bold/surf/sub-PNC*_ses-a1_atlas-schaefer-400_desc-FC.shape.gii')
    A_400 = np.average(np.array([nib.load(f).darrays[0].data for f in A_400[:]]), axis=0)
    #A_400 = np.log(np.triu(A_400,1) + A_400.T + 1)
    A_400 = A_400[49:, 49:]
    A_400 = np.arctanh(A_400)
    A_400[~np.isfinite(A_400)] = 0
    A_400 = np.triu(A_400,1)+A_400.T

    from scipy.ndimage import rotate
    # --- Sort nodes by network ---
    node_networks = np.delete(df_label['network'].values[49:], 200, axis=0)
    print(node_networks)
    sort_idx = np.argsort(node_networks)
    print(node_networks[sort_idx])
    A_sorted = A_400[sort_idx][:, sort_idx]
    sorted_networks = node_networks[sort_idx]
    import matplotlib.patches as patches

    # --- Find network boundaries ---
    boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0] + 1
    boundaries = np.insert(boundaries,0,0)

    mpc_fig = A_sorted.copy()
    mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    mpc_fig = rotate(mpc_fig, angle=-45, order=0, cval=np.nan)
    fig, ax = plt.subplots()
    # Overlay borders
    b_ext = np.append(boundaries,400)
    for i, b in enumerate(boundaries):
        rect = patches.Rectangle((200 * np.sqrt(2), b * np.sqrt(2)), b_ext[i+1] - b_ext[i], b_ext[i+1] - b_ext[i], linewidth=2, edgecolor=yeo7_colors.colors[i], facecolor='none', angle=45)
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.imshow(mpc_fig, cmap='coolwarm', vmin=-1, vmax=1, origin='upper')
    plt.axis('off')
    plt.title('Upper Triangle of MPC Rotated by 45°')
    plt.tight_layout()
    plt.show()

    print(A_400.shape)
    df_label.loc[~df_label.hemisphere.isna(), 'A_bottom'] = np.mean(A_400[df_label[~df_label.hemisphere.isna()].quantile_idx == -1, :], axis=0)
    df_label.loc[~df_label.hemisphere.isna(), 'A_top'] = np.mean(A_400[df_label[~df_label.hemisphere.isna()].quantile_idx == 1, :], axis=0)
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'A_bottom', 'A_top']], on='mics', validate="many_to_one", how='left')
    A_diff = df_yeo_surf['A_top'].values - df_yeo_surf['A_bottom'].values
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['A_bottom'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='Blues', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['A_top'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='Reds', transparent_bg=True)
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=A_diff, size=(500, 400), zoom=1.4, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym', layout_style="grid")
    

    # Z-score for normalization
    df_corr = pd.DataFrame({
        'network': df_yeo_surf['network'],
        'corr_diff': zscore(A_diff, nan_policy='omit')
    })

    # Compute mean correlation difference per network
    df_corr_mean = (
        df_corr
        .dropna(subset=['corr_diff'])
        .groupby('network')['corr_diff']
        .mean()
    )
   
    # Create SpinPermutations model (1000 rotations)
    spin_model = SpinPermutations(n_rep=100, random_state=42)
    spin_model.fit(sphere32k_lh, sphere32k_rh)

    # Split data into hemispheres
    n_lh = surf32k_lh_infl.n_points
    # Generate rotated surrogate maps
    corr_spins = np.hstack(spin_model.randomize(A_diff[:n_lh], A_diff[n_lh:]))  # shape: (1000, n_vertices)
    print(corr_spins.shape)

    # Compute mean per network for each permutation
    df_corr_spin = pd.DataFrame({
        'network': df_yeo_surf['network']
    })

    spin_means = []
    for perm in corr_spins:
        df_corr_spin['corr_diff'] = perm
        spin_means.append(
            df_corr_spin.dropna().groupby('network')['corr_diff'].mean().reindex(df_corr_mean.index)
        )
    spin_means = np.vstack(spin_means)
    # Compute null mean and 95% CI across spins
    spin_mean = np.nanmean(spin_means, axis=0)
    spin_ci = np.nanstd(spin_means, axis=0) * 1.96  # 95% CI

    # ---------------------------------------------------------------------
    # Plot: bars = null model (spin), scatter = real data
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df_corr_mean))

    # Bars for null distribution mean ± 95% CI
    ax.bar(x, spin_mean, yerr=spin_ci, color='lightgrey', edgecolor='black', alpha=0.8, capsize=3, label='Spin null mean ± 95% CI')
    #ax.errorbar(x, spin_mean, yerr=spin_ci)

    # Scatter points for empirical correlation difference
    ax.scatter(x, df_corr_mean.values, color=yeo7_rgba[:-1], alpha=0.8, s=100, zorder=5, label='Empirical mean')

    # Aesthetics
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df_corr_mean.index, rotation=90, ha='center', fontsize=10)
    ax.set_ylabel('Mean correlation difference (Top – Bottom)', fontsize=12)
    ax.set_title('Empirical vs. Spin null network correlation differences', fontsize=13, pad=12)
    #ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()


    ## effective connectivity
    # df_ec = pd.read_csv('/local_raid/data/pbautin/data/labels_Schaefer2018_400Parcels_7Networks_order.txt', header=None, names=['label'])
    # df_ec['network'] = df_ec['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    # df_ec['hemisphere'] = df_ec['label'].str.extract(r'(LH|RH)')
    # ec = np.loadtxt('/local_raid/data/pbautin/data/EC_Schaefer2018_400Parcels_7Networks_order.txt')#[:, df_ec.network == 'Default']
    # df_ec['ec'] = ec.tolist()
    # df_ec_sorted = df_ec.sort_values(by=['hemisphere', 'network'], ascending=False, inplace=True)
    # df_ec_sorted = df_ec.sort_values(by=['hemisphere', 'network'], ascending=False)
    # print(df_ec.index)
    
    # # Define symmetric bounds for zero-centering
    # from matplotlib.colors import TwoSlopeNorm
    # norm = TwoSlopeNorm(vmin=np.nanmin(ec)-0.1, vcenter=0, vmax=np.nanmax(ec)+0.1)

    # # Plot heat map
    # plt.figure(figsize=(18, 7))
    # plt.imshow(np.vstack(df_ec['ec'].values)[df_ec.index.values[:, None], df_ec.index.values][:200,:], cmap='coolwarm', norm=norm, aspect='equal')
    # boundaries = np.where(df_ec_sorted['network'].values[:-1] != df_ec_sorted['network'].values[1:])[0] + 1
    # print(boundaries)
    # for b in boundaries:
    #     if b <= 200:
    #         plt.axhline(b - 0.5, color='black', linewidth=0.7)
    #     plt.axvline(b - 0.5, color='black', linewidth=0.7)
    # plt.colorbar(label='Effective Connectivity')
    # plt.yticks(ticks=[np.where(df_ec_sorted.loc[df_ec_sorted.hemisphere == 'RH', 'network'] == n)[0].mean() for n in df_ec_sorted.network.unique()], labels=df_ec_sorted.network.unique())
    # plt.tight_layout()
    # plt.show()

    # df_label = df_label.merge(df_ec[['label', 'ec']], on='label', how='left')
    # df_label.loc[df_label['network'].eq('SalVentAttn'), 'mean_ec'] = df_label.loc[df_label['network'].eq('SalVentAttn'), 'ec'].apply(np.mean)
    # df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'mean_ec']], on='mics', validate="many_to_one", how='left')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['mean_ec'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #     nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')


    # df_label.loc[~df_label.ec.isna(), 'salience_ec'] = np.mean(np.vstack(df_label.loc[df_label.network == 'SalVentAttn', 'ec'].values), axis=0)
    # df_label.loc[~df_label.ec.isna(), 'bottom_ec'] = np.mean(np.vstack(df_label.loc[df_label.quantile_idx == -1, 'ec'].values), axis=0)
    # df_label.loc[~df_label.ec.isna(), 'top_ec'] = np.mean(np.vstack(df_label.loc[df_label.quantile_idx == 1, 'ec'].values), axis=0)
    # df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'salience_ec', 'bottom_ec', 'top_ec', 'mean_ec']], on='mics', validate="many_to_one", how='left')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['mean_ec'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['salience_ec'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['bottom_ec'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['top_ec'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['top_ec'].values - df_yeo_surf['bottom_ec'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')



    ##########################################################################
    ####################### ANALYSIS #########################################
    ######### Part 1 -- T1 map

    # ### Load the data from PNI dataset 5k
    # t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-5k_desc-intensity_profiles.shape.gii'))
    # print("number of files/subjects: {}".format(len(t1_files[:])))
    # # t1 profiles (n_subject, n_features, n_vertices)
    # t1_profiles = np.stack([nib.load(f).darrays[0].data for f in t1_files[:]])
    # t1_salience_profiles = t1_profiles[:, :, df_yeo_surf_5k['network'].eq('SalVentAttn').to_numpy()]
    # t1_salience_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_salience_profiles[:, :, :], np.nanmean(t1_profiles, axis=2))]
    # gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    # gm_t1.fit(t1_salience_mpc, sparsity=0.9)
    # t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    # print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    # df_yeo_surf_5k.loc[df_yeo_surf_5k['network'].eq('SalVentAttn'), 't1_gradient1_SalVentAttn'] = normalize_to_range(t1_gradients[:, 0], -1, 1)
    # plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    
    # # Compute quantile thresholds
    # low_thresh = np.nanquantile(df_yeo_surf_5k['t1_gradient1_SalVentAttn'].values, 0.25)
    # high_thresh = np.nanquantile(df_yeo_surf_5k['t1_gradient1_SalVentAttn'].values, 1 - 0.25)
    # # Identify vertex indices for extremes
    # df_yeo_surf_5k.loc[np.where(df_yeo_surf_5k['t1_gradient1_SalVentAttn'].values <= low_thresh)[0], 'quantile_idx'] = -1
    # df_yeo_surf_5k.loc[np.where(df_yeo_surf_5k['t1_gradient1_SalVentAttn'].values >= high_thresh)[0], 'quantile_idx'] = 1
    # plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['quantile_idx'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    
    # df_yeo_surf_5k.to_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure2_5k_df.tsv', index=False)
    


    ### Structural connectivity
    df_yeo_surf_5k = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure2_5k_df.tsv')
    A_vertex = glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/dwi/connectomes/sub-PNC*_ses-a1_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii')
    A_vertex = np.average(np.array([nib.load(f).darrays[0].data for f in A_vertex[:10]]), axis=0)
    A_vertex = np.log(np.triu(A_vertex,1) + A_vertex.T + 1)
    A_vertex = A_vertex[df_yeo_surf_5k['network'].values != 'medial_wall',:][:,df_yeo_surf_5k['network'].values != 'medial_wall']

    # from scipy.ndimage import rotate
    # # --- Sort nodes by network ---
    # node_networks = df_yeo_surf_5k['network'].values
    # print(node_networks)
    # sort_idx = np.argsort(node_networks)
    # print(node_networks[sort_idx])
    # A_sorted = A_vertex[sort_idx][:, sort_idx]
    # sorted_networks = node_networks[sort_idx]
    # import matplotlib.patches as patches

    # # --- Find network boundaries ---
    # boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0] + 1
    # boundaries = np.insert(boundaries,0,0)

    # mpc_fig = A_sorted.copy()
    # mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    # mpc_fig = rotate(mpc_fig, angle=-45, order=0, cval=np.nan)
    # fig, ax = plt.subplots()
    # # Overlay borders
    # b_ext = np.append(boundaries,node_networks.shape[0])
    # for i, b in enumerate(boundaries):
    #     rect = patches.Rectangle((node_networks.shape[0] / 2  * np.sqrt(2), b * np.sqrt(2)), b_ext[i+1] - b_ext[i], b_ext[i+1] - b_ext[i], linewidth=2, edgecolor=yeo7_colors.colors[i], facecolor='none', angle=45)
    #     # Add the patch to the Axes
    #     ax.add_patch(rect)

    # mpc_fig[mpc_fig > 1] = 1
    # plt.imshow(mpc_fig, cmap='Blues', origin='upper')
    # plt.axis('off')
    # plt.title('Upper Triangle of MPC Rotated by 45°')
    # plt.tight_layout()
    # plt.show()

    df_yeo_surf_5k.loc[~df_yeo_surf_5k.hemisphere.isna(), 'A_bottom'] = np.mean(A_vertex[df_yeo_surf_5k[~df_yeo_surf_5k.hemisphere.isna()].quantile_idx == -1, :], axis=0)
    df_yeo_surf_5k.loc[~df_yeo_surf_5k.hemisphere.isna(), 'A_top'] = np.mean(A_vertex[df_yeo_surf_5k[~df_yeo_surf_5k.hemisphere.isna()].quantile_idx == 1, :], axis=0)
    #print(df_yeo_surf_5k)
    #df_yeo_surf_5k = df_yeo_surf_5k.merge(df_yeo_surf_5k[['mics', 'A_bottom', 'A_top']], on='mics', validate="many_to_one", how='left')
    A_salience = df_yeo_surf_5k['A_top'].values - df_yeo_surf_5k['A_bottom'].values
    # A_salience_bottom = np.average(A_vertex[df_yeo_surf_5k.quantile_idx == -1, :], axis=0)
    # A_salience_top = np.average(A_vertex[df_yeo_surf_5k.quantile_idx == 1, :], axis=0)
    # A_salience = A_salience_top - A_salience_bottom
    
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=A_salience, size=(500, 400), zoom=1.4, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym', layout_style="grid")
    
    # Z-score for normalization
    df_corr = pd.DataFrame({
        'network': df_yeo_surf_5k['network'],
        'corr_diff': zscore(A_salience, nan_policy='omit')
    })

    # Compute mean correlation difference per network
    df_corr_mean = (
        df_corr
        .dropna(subset=['corr_diff'])
        .groupby('network')['corr_diff']
        .mean()
    )
   
    # Create SpinPermutations model (1000 rotations)
    spin_model = SpinPermutations(n_rep=100, random_state=42)
    spin_model.fit(surf5k_lh_sphere, surf5k_rh_sphere)

    # Split data into hemispheres
    n_lh = surf5k_lh_infl.n_points
    # Generate rotated surrogate maps
    corr_spins = np.hstack(spin_model.randomize(A_salience[:n_lh], A_salience[n_lh:]))  # shape: (1000, n_vertices)
    print(corr_spins.shape)

    # Compute mean per network for each permutation
    df_corr_spin = pd.DataFrame({
        'network': df_yeo_surf_5k['network']
    })

    spin_means = []
    for perm in corr_spins:
        df_corr_spin['corr_diff'] = perm
        spin_means.append(
            df_corr_spin.dropna().groupby('network')['corr_diff'].mean().reindex(df_corr_mean.index)
        )
    spin_means = np.vstack(spin_means)
    # Compute null mean and 95% CI across spins
    spin_mean = np.nanmean(spin_means, axis=0)
    spin_ci = np.nanstd(spin_means, axis=0) * 1.96  # 95% CI

    # ---------------------------------------------------------------------
    # Plot: bars = null model (spin), scatter = real data
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df_corr_mean))

    # Bars for null distribution mean ± 95% CI
    ax.bar(x, spin_mean, yerr=spin_ci, color='lightgrey', edgecolor='black', alpha=0.8, capsize=3, label='Spin null mean ± 95% CI')
    #ax.errorbar(x, spin_mean, yerr=spin_ci)

    # Scatter points for empirical correlation difference
    ax.scatter(x, df_corr_mean.values, color=yeo7_rgba[:-1], alpha=0.8, s=100, zorder=5, label='Empirical mean')

    # Aesthetics
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df_corr_mean.index, rotation=90, ha='center', fontsize=10)
    ax.set_ylabel('Mean correlation difference (Top – Bottom)', fontsize=12)
    ax.set_title('Empirical vs. Spin null network correlation differences', fontsize=13, pad=12)
    #ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()
    
    
    
    # ### Load the data from PNI dataset 32k
    # t1_files = sorted(glob.glob('/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'))
    # print("number of files/subjects: {}".format(len(t1_files[:])))
    # # t1 profiles (n_subject, n_features, n_vertices)
    # t1_profiles = np.stack([nib.load(f).darrays[0].data for f in t1_files[:]])
    # t1_salience_profiles = t1_profiles[:, :, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    # t1_salience_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_salience_profiles[:, :, :], np.nanmean(t1_profiles, axis=2))]
    # gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    # gm_t1.fit(t1_salience_mpc, sparsity=0.9)
    # t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    # print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient1_SalVentAttn'] = normalize_to_range(t1_gradients[:, 0], -1, 1)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    
    # # Compute quantile thresholds
    # low_thresh = np.nanquantile(df_yeo_surf['t1_gradient1_SalVentAttn'].values, 0.25)
    # high_thresh = np.nanquantile(df_yeo_surf['t1_gradient1_SalVentAttn'].values, 1 - 0.25)
    # # Identify vertex indices for extremes
    # df_yeo_surf.loc[np.where(df_yeo_surf['t1_gradient1_SalVentAttn'].values <= low_thresh)[0], 'quantile_idx'] = -1
    # df_yeo_surf.loc[np.where(df_yeo_surf['t1_gradient1_SalVentAttn'].values >= high_thresh)[0], 'quantile_idx'] = 1
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['quantile_idx'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym')
    
    # df_yeo_surf.to_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure2_df.tsv', index=False)

    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure2_df.tsv')

    ### Geodesic distance
    df_yeo_surf.loc[:, 'bottom'] = df_yeo_surf.quantile_idx == -1
    df_yeo_surf.loc[:, 'top'] = df_yeo_surf.quantile_idx == 1
    l = 0.12

    # right hemisphere
    surf_32k_rh = mesh.mesh_operations.mask_points(surf_32k, np.array(df_yeo_surf.hemisphere == 'RH'))
    tria_32k_rh = TriaMesh(mesh.mesh_elements.get_points(surf_32k_rh), mesh.mesh_elements.get_cells(surf_32k_rh))

    heat_f_rh_bottom = heat.diffusion(tria_32k_rh, df_yeo_surf.loc[df_yeo_surf.hemisphere == 'RH', 'bottom'].values, m=2)
    geodesic_rh_bottom = np.exp(-l * diffgeo.compute_geodesic_f(tria_32k_rh, heat_f_rh_bottom))
    heat_f_rh_top = heat.diffusion(tria_32k_rh, df_yeo_surf.loc[df_yeo_surf.hemisphere == 'RH', 'top'].values, m=2)
    geodesic_rh_top = np.exp(-l * diffgeo.compute_geodesic_f(tria_32k_rh, heat_f_rh_top))
    df_yeo_surf.loc[df_yeo_surf.hemisphere == 'RH', 'geodesic'] = geodesic_rh_top - geodesic_rh_bottom

    # left hemisphere
    surf_32k_lh = mesh.mesh_operations.mask_points(surf_32k, np.array(df_yeo_surf.hemisphere == 'LH'))
    tria_32k_lh = TriaMesh(mesh.mesh_elements.get_points(surf_32k_lh), mesh.mesh_elements.get_cells(surf_32k_lh))

    heat_f_lh_bottom = heat.diffusion(tria_32k_lh, df_yeo_surf.loc[df_yeo_surf.hemisphere == 'LH', 'bottom'].values, m=2)
    geodesic_lh_bottom = np.exp(-l * diffgeo.compute_geodesic_f(tria_32k_lh, heat_f_lh_bottom))
    heat_f_lh_top = heat.diffusion(tria_32k_lh, df_yeo_surf.loc[df_yeo_surf.hemisphere == 'LH', 'top'].values, m=2)
    geodesic_lh_top = np.exp(-l * diffgeo.compute_geodesic_f(tria_32k_lh, heat_f_lh_top))
    df_yeo_surf.loc[df_yeo_surf.hemisphere == 'LH', 'geodesic'] = geodesic_lh_top - geodesic_lh_bottom


    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['geodesic'].values, size=(500, 400), zoom=1.4, color_bar='right', share='both',
            nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, color_range='sym', layout_style="grid")
    
    # Z-score for normalization
    df_corr = pd.DataFrame({
        'network': df_yeo_surf['network'],
        'corr_diff': zscore(df_yeo_surf['geodesic'].values, nan_policy='omit')
    })

    # Compute mean correlation difference per network
    df_corr_mean = (
        df_corr
        .dropna(subset=['corr_diff'])
        .groupby('network')['corr_diff']
        .mean()
    )
   
    # Create SpinPermutations model (1000 rotations)
    spin_model = SpinPermutations(n_rep=100, random_state=42)
    spin_model.fit(sphere32k_lh, sphere32k_rh)

    # Split data into hemispheres
    n_lh = surf32k_lh_infl.n_points
    # Generate rotated surrogate maps
    corr_spins = np.hstack(spin_model.randomize(df_yeo_surf['geodesic'].values[:n_lh], df_yeo_surf['geodesic'].values[n_lh:]))  # shape: (1000, n_vertices)
    print(corr_spins.shape)

    # Compute mean per network for each permutation
    df_corr_spin = pd.DataFrame({
        'network': df_yeo_surf['network']
    })

    spin_means = []
    for perm in corr_spins:
        df_corr_spin['corr_diff'] = perm
        spin_means.append(
            df_corr_spin.dropna().groupby('network')['corr_diff'].mean().reindex(df_corr_mean.index)
        )
    spin_means = np.vstack(spin_means)
    # Compute null mean and 95% CI across spins
    spin_mean = np.nanmean(spin_means, axis=0)
    spin_ci = np.nanstd(spin_means, axis=0) * 1.96  # 95% CI

    # ---------------------------------------------------------------------
    # Plot: bars = null model (spin), scatter = real data
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(df_corr_mean))

    # Bars for null distribution mean ± 95% CI
    ax.bar(x, spin_mean, yerr=spin_ci, color='lightgrey', edgecolor='black', alpha=0.8, capsize=3, label='Spin null mean ± 95% CI')
    #ax.errorbar(x, spin_mean, yerr=spin_ci)

    # Scatter points for empirical correlation difference
    ax.scatter(x, df_corr_mean.values, color=yeo7_rgba[:-1], alpha=0.8, s=100, zorder=5, label='Empirical mean')

    # Aesthetics
    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df_corr_mean.index, rotation=90, ha='center', fontsize=10)
    ax.set_ylabel('Mean correlation difference (Top – Bottom)', fontsize=12)
    ax.set_title('Empirical vs. Spin null network correlation differences', fontsize=13, pad=12)
    #ax.legend(frameon=False, fontsize=10)
    plt.tight_layout()
    plt.show()
    




if __name__ == "__main__":
    main()



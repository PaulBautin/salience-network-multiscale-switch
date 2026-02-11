from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Local microstructural heterogeneity of the salience network
#
# example:
# python figure_1_t1map.py
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

#### imports
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations, mesh_operations
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace.mesh import mesh_elements

from brainspace.null_models import SpinPermutations, moran
from scipy.stats import spearmanr


from brainspace.gradient import GradientMaps, kernels
import scipy
from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore
from joblib import Parallel, delayed

from sklearn.linear_model import LinearRegression
from functools import partial
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA


def surf_type_isolation(surf_type_test, i):
    # Work on a copy of the input array to avoid modifying the original
    surf_type_copy = surf_type_test.copy()
    surf_type_copy[surf_type_copy != i] = np.nan
    return surf_type_copy


def load_mpc(File):
     """Loads and process a MPC"""
     mpc = nib.load(File).darrays[0].data
     mpc = np.triu(mpc,1)+mpc.T
     mpc[~np.isfinite(mpc)] = np.finfo(float).eps
     mpc[mpc==0] = np.finfo(float).eps
     return(mpc)


def normalize_to_range(data, target_min, target_max):
    """
    Normalizes a NumPy array or list of numerical data to a specified target range.

    Args:
        data (np.array or list): The input numerical data.
        target_min (float): The desired minimum value of the normalized range.
        target_max (float): The desired maximum value of the normalized range.

    Returns:
        np.array: The normalized data within the target range.
    """
    data = np.array(data) # Ensure data is a NumPy array for min/max operations
    
    original_min = np.nanmin(data)
    original_max = np.nanmax(data)

    if original_min == original_max: # Handle cases where all values are the same
        return np.full_like(data, (target_min + target_max) / 2)

    # Normalize to 0-1 range first
    normalized_0_1 = (data - original_min) / (original_max - original_min)

    # Scale to the target range
    scaled_data = target_min + (normalized_0_1 * (target_max - target_min))
    return scaled_data


def load_t1_salience_profiles(path, df_yeo_surf, network='SalVentAttn'):
    ## t1 profiles (n_subject, n_features, n_vertices)
    t1_files = glob.glob(path)
    print("number of files/subjects: {}".format(len(t1_files)))
    t1_salience_profiles = np.stack([nib.load(f).darrays[0].data[:, df_yeo_surf['network'].eq(network).to_numpy()] for f in t1_files[:]])
    return t1_salience_profiles


def compute_t1_gradient(df_yeo_surf, t1_salience_profiles, network='SalVentAttn'):
    print(t1_salience_profiles.shape)
    #t1_salience_profiles = t1_profiles[:, :, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    t1_salience_mpc = [partial_corr_with_covariate(subj_data, covar=t1_mean_profile) for subj_data, t1_mean_profile in zip(t1_salience_profiles[:, :, :], np.nanmean(t1_salience_profiles, axis=2))]
    gm_t1 = GradientMaps(n_components=10, random_state=None, approach='dm', kernel='normalized_angle', alignment='procrustes')
    gm_t1.fit(t1_salience_mpc, sparsity=0.9)
    t1_gradients = np.mean(np.asarray(gm_t1.aligned_), axis=0)
    print("gradient lambdas: {}".format(np.mean(np.asarray(gm_t1.lambdas_), axis=0)))
    df_yeo_surf.loc[df_yeo_surf['network'].eq(network), 't1_gradient1_' + network] = t1_gradients[:, 0]
    # df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient2_salience'] = t1_gradients[:, 1]
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient2_salience'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True)
    # df_yeo_surf.to_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_df.tsv', index=False)
    return df_yeo_surf


def plot_gradient_profiles(df_yeo_surf, t1_salience_profiles):
    t1_gradient = df_yeo_surf['t1_gradient1_salience'].dropna().values
    print(df_yeo_surf)
    #    Plot
    n_plot = 30
    step = len(t1_gradient) // n_plot
    sorted_gradient_indx = np.argsort(t1_gradient)[::step]
    sorted_gradient = t1_gradient[sorted_gradient_indx]
    plt.figure(figsize=(8,8))
    plt.imshow(t1_salience_profiles[0,: , np.argsort(t1_gradient)].T, aspect='auto')
    plt.show()
    norm = mpl.colors.Normalize(vmin=np.min(t1_gradient), vmax=np.max(t1_gradient))
    cmap = mpl.colormaps.get_cmap('coolwarm')
    colors = [cmap(norm(g)) for g in sorted_gradient]
    plt.figure(figsize=(6, 10))
    profile = t1_salience_profiles[0,::]
    for idx, color in zip(sorted_gradient_indx, colors):
        plt.plot(profile[:, idx] / 1000, np.arange(profile.shape[0]), color=color, alpha=0.8, lw=3)
    plt.xlabel("Cortical Depth (0 = WM, 1 = Pial)")
    plt.ylabel("T1 Map Intensity")
    plt.title("Cortical Depth Profiles Colored by Gradient (Pial on Top)")
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.tight_layout()
    plt.show()


def context_analysis(df_yeo_surf, surf_32k, modalities, n_rep=10):
    ## Correlation analyses
    x = zscore(df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 't1_gradient1_salience'].values)
    # Moran spatial autocorrelation model
    w = mesh_elements.get_ring_distance(surf_32k, n_ring=1, mask=df_yeo_surf['network'].eq('SalVentAttn').values)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_rep, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)

    # Plot 
    fig, axes = plt.subplots(len(modalities), 1, figsize=(4, 4 * len(modalities)), sharex=True, sharey=True)
    for ax, label in zip(axes, modalities):
        y = df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), label].values
        rand = msr.randomize(y)
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 25, "alpha": 0.7}, line_kws={"color": "black"})
        r_obs, p = spearmanr(x, y, nan_policy='omit')
        r_rand = np.asarray([spearmanr(x, d)[0] for d in rand])
        pv_rand = np.mean(np.abs(r_rand) >= np.abs(r_obs))
        ax.set_title(f"$r={r_obs:.2f}, p={pv_rand:.2e}$", fontsize=12)
    plt.tight_layout()
    plt.show()

    r_vals, labels = [], []
    for label in modalities:
        if label in df_yeo_surf.columns:
            y = df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), label].values
            if len(y) > 1 and not np.all(np.isnan(y)):
                r, _ = spearmanr(x, y, nan_policy='omit')
                r_vals.append(r)
                labels.append(label)

    # Convert to numpy
    r_vals = np.array(r_vals)
    print(r_vals)

    if r_vals.size == 0:
        raise ValueError("No valid correlations could be computed. Check your modality columns.")

    # Half-circle polar coordinates
    N = len(r_vals)
    theta = np.linspace(-np.pi /2 + np.pi/N*0.8, np.pi /2, N, endpoint=False)
    radii = np.abs(r_vals)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 5))
    bars = ax.bar(theta, radii, width=np.pi/N*0.8, align="center", alpha=0.8)

    # Color by sign
    for bar, r in zip(bars, r_vals):
        bar.set_facecolor("tab:red" if r < 0 else "tab:blue")
    #plt.grid(False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()


def cortical_type_analysis(df_yeo_surf):
    # Define type labels
    type_labels = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
    label_map = dict(zip(range(1, 8), type_labels))
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    # Prepare spin permutations
    n_rand = 100
    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sp.fit(sphere32k_lh, points_rh=sphere32k_rh)

    # Compute and store results
    all_data = {}
    real_data = {}

    df_yeo_surf.loc[df_yeo_surf.surf_type.isna(), 'surf_type'] = 7  # Replace NaNs with dummy label
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    state[np.isnan(state)] = np.where(state_name == 'medial_wall')[0][0]  # Replace NaNs with dummy label

    for net_idx, net_name in enumerate(state_name):
        mask = (state == net_idx)
        mask_lh, mask_rh = mask[:32492], mask[32492:]

        # Empirical
        expected_types = np.arange(1, 8)  # Cortical types 1 to 7
        comp = df_yeo_surf.surf_type.values[mask] * mask[mask]
        observed_types, counts = np.unique(comp, return_counts=True)
        counts_dict = dict(zip(observed_types, counts))
        full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
        percentages = (full_counts / len(comp)) * 100
        real_data[net_name] = dict(zip(expected_types, percentages))

        # comp = surf_type[mask] * mask[mask]
        # u, c = np.unique(comp, return_counts=True)
        # perc = (c / len(comp)) * 100
        # real_data[net_name] = dict(zip(u, perc))

        # Null distribution
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        comp_dict = {val: [] for val in df_yeo_surf.surf_type.unique()}
        for n in range(n_rand):
            comp = df_yeo_surf.surf_type.values[net_rot[n]] * net_rot[n][net_rot[n]]
            u, c = np.unique(comp, return_counts=True)
            counts_dict = dict(zip(u, c))
            full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
            perc = (full_counts / len(comp)) * 100
            for val in comp_dict:
                comp_dict[val].append(dict(zip(expected_types, perc)).get(val, 0))
        df = pd.DataFrame(comp_dict)
        df.rename(columns={k: label_map.get(k, k) for k in df.columns}, inplace=True)
        all_data[net_name] = df

    # --- Plotting ---
    # Setup: Salience in full column
    n_total = len(all_data)
    n_cols = 4
    sal_idx = np.where(state_name == "SalVentAttn")[0][0]
    other_names = [n for i, n in enumerate(state_name)
                if i != sal_idx and n != "medial_wall"]
    n_rows = int(np.ceil(len(other_names) / (n_cols - 1)))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.6)

    # Plot Salience in full column
    ax_sal = fig.add_subplot(gs[:, 0])  # full height first column
    df = all_data["SalVentAttn"]
    sns.barplot(data=df, ax=ax_sal, color='lightgrey')
    rdict = {label_map.get(k, k): v for k, v in real_data["SalVentAttn"].items()}
    sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax_sal)
    ax_sal.set_title("SalVentAttn")
    ax_sal.set_ylim(0, 60)
    ax_sal.tick_params(axis='x', labelrotation=90)

    # Plot other networks
    for i, net_name in enumerate(other_names):
        row, col = divmod(i, n_cols - 1)
        ax = fig.add_subplot(gs[row, col + 1])
        df = all_data[net_name]
        sns.barplot(data=df, ax=ax, color='lightgrey')
        rdict = {label_map.get(k, k): v for k, v in real_data[net_name].items()}
        sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax)
        ax.set_title(net_name)
        ax.set_ylim(0, 60)
        ax.tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    plt.show()

    # # Convert real_data (dict of percentages) to a DataFrame
    # del real_data['medial_wall']
    # networks = list(real_data.keys())
    # cortical_types = np.arange(1, 8)

    # # Create DataFrame with rows = networks, columns = cortical types
    # real_df = pd.DataFrame.from_dict(real_data, orient='index')
    # real_df.columns = [label_map[t] for t in cortical_types]

    # # Compute pairwise KS test statistics
    # n_nets = len(networks)
    # ks_matrix = np.zeros((n_nets, n_nets))

    # for i in range(n_nets):
    #     for j in range(n_nets):
    #         ks_stat, _ = ks_2samp(real_df.iloc[i], real_df.iloc[j])
    #         ks_matrix[i, j] = ks_stat

    # # Project to first principal component for row/column ordering
    # pca = PCA(n_components=1)
    # pc_order = np.argsort(pca.fit_transform(real_df).ravel())
    # ordered_labels = [networks[i] for i in pc_order]

    # # Reorder KS matrix
    # ks_matrix_ordered = ks_matrix[pc_order, :][:, pc_order]

    # # Plotting
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(ks_matrix_ordered, xticklabels=ordered_labels, yticklabels=ordered_labels,
    #             cmap='magma', square=True, cbar_kws={'label': 'KS Statistic'})
    # plt.title('Pairwise KS Test: Cortical Type Distributions Across Networks')
    # plt.tight_layout()
    # plt.show()




def main():
    #### Define paths
    micapipe='/local_raid/data/pbautin/software/micapipe'
    pni_deriv = '/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0'

    ### load surfaces
    surf32k_lh_infl, surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii'), read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    #### load atlases
    df_yeo_surf = load_yeo_atlas(micapipe=micapipe, surf_32k=surf_32k)
    # df_yeo_surf = load_econo_atlas(micapipe, df_yeo_surf)
    # load_baillarger_atlas(df_yeo_surf)
    # load_intrusion_atlas(df_yeo_surf)



    ######### Part 1 -- T1 map
    path_figure1_part1 = '/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure1_part1_df.tsv'
    if os.path.exists(path_figure1_part1):
        df_yeo_surf = pd.read_csv(path_figure1_part1)
        #t1_salience_profiles = load_t1_salience_profiles(pni_deriv, df_yeo_surf)
    else:
        path = pni_deriv + '/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'
        t1_salience_profiles = load_t1_salience_profiles(path, df_yeo_surf)
        df_yeo_surf, t1_salience_profiles = compute_t1_gradient(pni_deriv, df_yeo_surf, t1_salience_profiles)
        df_yeo_surf.to_csv(path_figure1_part1, index=False)
    # plot_gradient_profiles(df_yeo_surf, t1_salience_profiles)



    ######### Part 2 -- Contextualisation
    # df_yeo_surf = load_t1map(df_yeo_surf, t1_salience_profiles)
    # df_yeo_surf = load_bigbrain(df_yeo_surf)
    # df_yeo_surf = load_ahead_biel(df_yeo_surf)
    # df_yeo_surf = load_ahead_parva(df_yeo_surf)
    # context_analysis(df_yeo_surf, surf_32k, modalities=["BigBrain", "T1map", "Bielschowsky", "Parvalbumin"], n_rep=10)



    ######### Part 3 -- Cortical type comparisons
    df_yeo_surf = load_econo_atlas(micapipe, df_yeo_surf)
    cortical_type_analysis(df_yeo_surf)



    ###### Cortical type comparisons
    # Define type labels
    type_labels = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
    label_map = dict(zip(range(1, 8), type_labels))

    # Prepare spin permutations
    n_rand = 100
    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sp.fit(sphere32k_lh, points_rh=sphere32k_rh)

    # Compute and store results
    all_data = {}
    real_data = {}

    surf_type[np.isnan(surf_type)] = 7  # Replace NaNs with dummy label
    state[np.isnan(state)] = np.where(state_name == 'medial_wall')[0][0]  # Replace NaNs with dummy label

    for net_idx, net_name in enumerate(state_name):
        mask = (state == net_idx)
        mask_lh, mask_rh = mask[:32492], mask[32492:]

        # Empirical
        expected_types = np.arange(1, 8)  # Cortical types 1 to 7
        comp = surf_type[mask] * mask[mask]
        observed_types, counts = np.unique(comp, return_counts=True)
        counts_dict = dict(zip(observed_types, counts))
        full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
        percentages = (full_counts / len(comp)) * 100
        real_data[net_name] = dict(zip(expected_types, percentages))

        # comp = surf_type[mask] * mask[mask]
        # u, c = np.unique(comp, return_counts=True)
        # perc = (c / len(comp)) * 100
        # real_data[net_name] = dict(zip(u, perc))

        # Null distribution
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        comp_dict = {val: [] for val in np.unique(surf_type)}
        for n in range(n_rand):
            comp = surf_type[net_rot[n]] * net_rot[n][net_rot[n]]
            u, c = np.unique(comp, return_counts=True)
            counts_dict = dict(zip(u, c))
            full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
            perc = (full_counts / len(comp)) * 100
            for val in comp_dict:
                comp_dict[val].append(dict(zip(expected_types, perc)).get(val, 0))
        df = pd.DataFrame(comp_dict)
        df.rename(columns={k: label_map.get(k, k) for k in df.columns}, inplace=True)
        all_data[net_name] = df

    # --- Plotting ---

    # Setup: Salience in full column
    n_total = len(all_data)
    n_cols = 4
    sal_idx = np.where(state_name == "SalVentAttn")[0][0]
    other_names = [n for i, n in enumerate(state_name)
                if i != sal_idx and n != "medial_wall"]
    n_rows = int(np.ceil(len(other_names) / (n_cols - 1)))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.6)

    # Plot Salience in full column
    ax_sal = fig.add_subplot(gs[:, 0])  # full height first column
    df = all_data["SalVentAttn"]
    sns.barplot(data=df, ax=ax_sal, color='lightgrey')
    rdict = {label_map.get(k, k): v for k, v in real_data["SalVentAttn"].items()}
    sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax_sal)
    ax_sal.set_title("SalVentAttn")
    ax_sal.set_ylim(0, 60)
    ax_sal.tick_params(axis='x', labelrotation=90)

    # Plot other networks
    for i, net_name in enumerate(other_names):
        row, col = divmod(i, n_cols - 1)
        ax = fig.add_subplot(gs[row, col + 1])
        df = all_data[net_name]
        sns.barplot(data=df, ax=ax, color='lightgrey')
        rdict = {label_map.get(k, k): v for k, v in real_data[net_name].items()}
        sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax)
        ax.set_title(net_name)
        ax.set_ylim(0, 60)
        ax.tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
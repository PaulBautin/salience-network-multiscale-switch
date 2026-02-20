from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Local cortical type heterogeneity of the salience network
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1c_cortical_types.py \
#   -pni_deriv /data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0
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

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles, load_econo_atlas, convert_states_str2int
from src.plot_colors import cmap_types, cmap_types_mw


plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'

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

    #print(df_yeo_surf[df_yeo_surf.network == 'medial_wall'])
    df_yeo_surf.loc[df_yeo_surf.surf_type == 0, 'surf_type'] = 7  # Replace NaNs with dummy label
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    #state[np.isnan(state)] = np.where(state_name == 'medial_wall')[0][0]  # Replace NaNs with dummy label

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


    fig, ax_sal = plt.subplots(figsize=(6, 6))
    # Plot Salience in full column
    df = all_data["SalVentAttn"][type_labels]
    sns.barplot(data=df, ax=ax_sal, color='lightgrey')
    rdict = {label_map.get(k, k): v for k, v in real_data["SalVentAttn"].items()}
    print(rdict)
    sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, edgecolors='none', ax=ax_sal)
    ax_sal.set_title("SalVentAttn")
    ax_sal.set_ylim(0, 60)
    ax_sal.tick_params(axis='x', labelrotation=90)
    ax_sal.set_ylabel("Percentage (%)")
    plt.tight_layout()
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_1c_type_salience.svg")



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

    ######### Part 1 -- T1 map
    path_figure1_part1 = '/local_raid/data/pbautin/software/salience-network-multiscale-switch/data/dataframes/figure1_part1_df.tsv'
    if os.path.exists(path_figure1_part1):
        path = pni_deriv + '/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'
        t1_salience_profiles = load_t1_salience_profiles(path, df_yeo_surf)
        df_yeo_surf = pd.read_csv(path_figure1_part1)
    else:
        path = pni_deriv + '/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'
        t1_salience_profiles = load_t1_salience_profiles(path, df_yeo_surf)
        df_yeo_surf = compute_t1_gradient(df_yeo_surf, t1_salience_profiles, network='SalVentAttn')
        df_yeo_surf.to_csv(path_figure1_part1, index=False)


    ######### Part 3 -- Cortical type comparisons
    df_yeo_surf = load_econo_atlas(project_root, df_yeo_surf)
    screenshot_path = "/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_1c_brain_economo.svg"
    plt_values = df_yeo_surf['surf_type'].values * df_yeo_surf['salience_border'].values
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(0, 0, 0, 1), cmap=cmap_types, transparent_bg=True, screenshot=True, filename=screenshot_path)
    
    cortical_type_analysis(df_yeo_surf)



if __name__ == "__main__":
    main()
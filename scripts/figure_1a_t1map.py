from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Local microstructural heterogeneity of the salience network
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1a_t1map.py \
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

from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.gradient.gradient import GradientMaps
from brainspace.plotting import plot_hemispheres

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles
from src.gradient_computation import partial_corr_with_covariate, compute_t1_gradient

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


def plot_gradient_profiles(df_yeo_surf, t1_salience_profiles, network='SalVentAttn'):
    print(t1_salience_profiles.shape)
    # find top and bottom quantiles
    net_mask = (df_yeo_surf["network"] == network)
    low_q, high_q = np.nanquantile(df_yeo_surf["t1_gradient1_SalVentAttn"], [0.25, 0.75])

    df_yeo_surf.loc[net_mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] <= low_q), "quantiles"] = -1
    df_yeo_surf.loc[net_mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] >= high_q), "quantiles"] = 1
    profiles = np.mean(t1_salience_profiles, axis=0)
    bottom_profiles = np.mean(t1_salience_profiles[:,:,df_yeo_surf.loc[net_mask, "quantiles"] == -1], axis=0)
    top_profiles = np.mean(t1_salience_profiles[:,:,df_yeo_surf.loc[net_mask, "quantiles"] == 1], axis=0)

    fig, ax = plt.subplots(figsize=(6, 6))
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mpl.colors.Normalize(vmin=np.nanmin(df_yeo_surf.loc[net_mask, "t1_gradient1_SalVentAttn"].values), vmax=np.nanmax(df_yeo_surf.loc[net_mask, "t1_gradient1_SalVentAttn"].values))
    colors = custom_cmap(norm(df_yeo_surf.loc[net_mask, "t1_gradient1_SalVentAttn"].values))
    for i, col in enumerate(colors):
        ax.plot(profiles[:,i] / 1000, np.linspace(0,1,profiles.shape[0]), color=col, alpha=0.1, rasterized=True)
    ax.plot(np.mean(bottom_profiles, axis=1) / 1000, np.linspace(0,1,profiles.shape[0]), color='b', alpha=0.8, label='bottom 25%')
    ax.plot(np.mean(top_profiles, axis=1) / 1000, np.linspace(0,1,profiles.shape[0]), color='r', alpha=0.8, label='top 25%')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 0.1))
    plt.xlim(1.4, 2.5)
    plt.ylabel("Intracortical depth")
    plt.xlabel("long. relaxation time (s)")
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='k', linestyle='--', linewidth=1)
    plt.yticks([0, 1], ['pial', 'WM'])
    plt.gca().invert_yaxis()  # pial at top
    plt.grid(False)
    plt.tight_layout()
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_1a_profiles.svg")


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
    
    # plot figures
    # plot_gradient_profiles(df_yeo_surf, t1_salience_profiles)
    screenshot_path = "/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_1a_brain.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path)


if __name__ == "__main__":
    main()
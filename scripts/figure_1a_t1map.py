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

import argparse
import logging
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles
from src.gradient_computation import compute_t1_gradient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Matplotlib globals
plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'


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


def plot_gradient_profiles(df_yeo_surf, t1_salience_profiles, screenshot_path: Path, network: str = 'SalVentAttn'):
    net_mask = (df_yeo_surf["network"] == network)

    # find top and bottom quantiles
    low_q, high_q = np.nanquantile(df_yeo_surf["t1_gradient1_SalVentAttn"], [0.25, 0.75])
    df_yeo_surf.loc[net_mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] <= low_q), "quantiles"] = -1
    df_yeo_surf.loc[net_mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] >= high_q), "quantiles"] = 1
    profiles = np.mean(t1_salience_profiles, axis=0)
    bottom_mask = df_yeo_surf.loc[net_mask, "quantiles"] == -1
    top_mask = df_yeo_surf.loc[net_mask, "quantiles"] == 1
    bottom_profiles = np.mean(t1_salience_profiles[:, :, bottom_mask], axis=0)
    top_profiles = np.mean(t1_salience_profiles[:, :, top_mask], axis=0)

    # Plotting setup
    fig, ax = plt.subplots(figsize=(6, 6))
    custom_cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    colors = custom_cmap(norm(df_yeo_surf.loc[net_mask, f"t1_gradient1_{network}"]))
    
    y_axis = np.linspace(0, 1, profiles.shape[0])
    
    # Plot individual profiles (Consider LineCollection here in the future if this loop is slow)
    for i, col in enumerate(colors):
        ax.plot(profiles[:, i] / 1000, y_axis, color=col, alpha=0.1, rasterized=True)
        
    ax.plot(np.mean(bottom_profiles, axis=1) / 1000, y_axis, color='b', alpha=0.8, label='bottom 25%')
    ax.plot(np.mean(top_profiles, axis=1) / 1000, y_axis, color='r', alpha=0.8, label='top 25%')
    
    # Aesthetics
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 0.1))
    plt.xlim(1.4, 2.5)
    plt.ylabel("Intracortical depth")
    plt.xlabel("Long. relaxation time (s)")
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='k', linestyle='--', linewidth=1)
    plt.yticks([0, 1], ['pial', 'WM'])
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(screenshot_path)


def extract_pnc_id(path: Path) -> str:
    return str(path.parent.parent.parent.parent.name).


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
    df_pni = pd.read_csv(project_root / 'data/dataframes/MICA_PNI.csv')[['ID_PNI', 'session', 'ID_MICs']]
    print(df_pni)

    ######### Part 1 -- T1 map
    path_df_1a = project_root / 'data/dataframes/df_1a.tsv'
    if os.path.exists(path_df_1a):
        logging.info(f"Found existing dataframe at {path_df_1a}. Loading...")
        path = str(pni_deriv) + '/sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'
        t1_salience_profiles = load_t1_salience_profiles(path, df_yeo_surf, network='SalVentAttn')
        df_yeo_surf = pd.read_csv(path_df_1a)
    else:
        path = pni_deriv / 'sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'
        t1_files = list(pni_deriv.glob('sub-PNC*/ses-a1/mpc/acq-T1map/sub-PNC*_ses-a1_surf-fsLR-32k_desc-intensity_profiles.shape.gii'))
        #print(t1_files[i].parent.parent.parent.parent.name)
        file_df = pd.DataFrame({'path': t1_files}).assign(ID_PNI=lambda df: df['path'].map(extract_pnc_id))
        print(file_df)
        df_pni = df_pni.merge(file_df, on='ID_PNI', validate="many_to_one", how='left').dropna(subset='path')
        print(df_pni)
        t1_salience_profiles = load_t1_salience_profiles(path, df_yeo_surf, network='SalVentAttn')
        df_yeo_surf = compute_t1_gradient(df_yeo_surf, t1_salience_profiles, network='SalVentAttn')
        df_yeo_surf.to_csv(path_df_1a, index=False)
    
    # plot figures
    screenshot_path = project_root / "results/figures/figure_1a_profiles.svg"
    logging.info(f"Generating brain qt1 profiles figure at {screenshot_path}")
    plot_gradient_profiles(df_yeo_surf, t1_salience_profiles, screenshot_path, network='SalVentAttn')

    screenshot_path = project_root / "results/figures/figure_1a_brain.svg"
    logging.info(f"Generating brain hemispheres screenshot at {screenshot_path}")
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range=(-3,3), transparent_bg=True, screenshot=True, filename=screenshot_path)


if __name__ == "__main__":
    main()
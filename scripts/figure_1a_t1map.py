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
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.atlas_load import load_yeo_atlas


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


def main(pni_deriv):
    # Setup Relative Paths
    script_path = Path(__file__).resolve()
    print(f"Script path: {script_path}")
    project_root = script_path.parent.parent
    print(f"Project root: {project_root}")

    ### load surfaces
    surf32k_lh_infl = read_surface(project_root + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root + '/data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
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


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Process PNI derivatives and surfaces.")
    
    # Add pni_deriv as a required argument
    parser.add_argument(
        "pni_deriv", 
        type=str, 
        help="Absolute path to the PNI derivatives folder (e.g., /data/mica/...)"
    )

    args = parser.parse_args()

    # Call main with the parsed argument
    main(pni_deriv=args.pni_deriv)
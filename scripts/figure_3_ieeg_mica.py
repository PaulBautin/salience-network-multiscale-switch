from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Concatenate ieeg information from all sessions and subjects from electromica,
# derivatives, and map it to the surface using the provided contact sensitivity maps.
#
# The sensitivity maps are only for 32k surfaces and fsaverage5. The 5k surfaces 
# are not able to properly depict the sensitivity, which varies quickly in space.
# fsnative surfaces have triangles that vary widely in size (1000 times),
# which leads to some numerical issues.
#
# database
# BIDS_ieeg: 31 subjects
# The iEEG Data is in host/verges/tank/data/BIDS_iEEG
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3_ieeg_mica.py \
#   -pni_deriv /data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0 \
#   -ieeg_deriv /host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA
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
import pickle



from brainspace.plotting import plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import  mesh_elements
from brainspace.datasets import load_conte69
from brainspace.null_models import moran
from scipy.stats import spearmanr


from vtkmodules.vtkFiltersSources import vtkSphereSource
import re
import matplotlib as mp
from scipy.stats import zscore

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles
from src.gradient_computation import compute_t1_gradient
from src.ieeg_processing import load_sensitivity_info, load_original_data_files, preprocess_and_compute_psd_ieeg, extract_band_power


plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False


def get_parser():
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Process ieeg derivatives and surfaces.",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-pni_deriv",
        type=str,
        help="Absolute path to the PNI derivatives folder (e.g., /data/mica/...)"
    )
    mandatory.add_argument(
        "-ieeg_deriv",
        type=str,
        help="Absolute path to the ieeg derivatives folder (e.g., /data/mica/...)"
    )
    return parser


def frequency_band_analysis_sensitivity(df_channel, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf):
    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']
    N_LH = 32492
    
    # Setup Geometry
    surf_combined = load_conte69(join=True)
    surf_lh, surf_rh = load_conte69(join=False)
    n_vertices = surf_combined.GetPoints().shape[0]
    fs = df_channel['SamplingRate'].iloc[0]
    
    # Define Analysis Mask: SalVent network specifically within the RH
    mask = ((df_yeo_surf['hemisphere'] == 'RH') & (df_yeo_surf['network'] == 'SalVentAttn')).values

    # Find top and bottom 25% of vertices in the SalVentAttn network based on the T1 gradient
    low_q, high_q = np.nanquantile(df_yeo_surf["t1_gradient1_SalVentAttn"], [0.25, 0.75])
    df_yeo_surf.loc[mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] <= low_q), "quantiles"] = -1
    df_yeo_surf.loc[mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] >= high_q), "quantiles"] = 1

    # Pre-calculate Moran Weights
    w = mesh_elements.get_ring_distance(surf_rh, n_ring=1, mask=mask[N_LH:])
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=100, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)

    # 1. Find the length of each signal
    lengths = [len(sig) for sig in df_channel['Data']]
    min_len, max_len = min(lengths), max(lengths)
    if min_len != max_len:
        print(f"Warning: Variable lengths detected ({min_len} to {max_len} samples).")
        print(f"Truncating all channels to {min_len} samples for vectorization.")
    data_matrix = np.vstack([np.asarray(sig)[:min_len] for sig in df_channel['Data']])
    
    # Compute PSD
    f, pxx_raw = preprocess_and_compute_psd_ieeg(data_matrix, fs)
    sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)
    surf_map = (pxx_raw.T @ sens) / (np.sum(sens, axis=0) + 1e-12)

    # Plot all PSDs colored by gradient value
    fig, ax = plt.subplots(figsize=(6, 4))
    grad = df_yeo_surf['t1_gradient1_SalVentAttn'].values[32492:][mask[32492:]]
    surf_map_sal = surf_map[:, mask[32492:]].T
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-1, vmax=1)
    for i in range(surf_map_sal.shape[0]): 
        ax.loglog(f, surf_map_sal[i, :], color=custom_cmap(norm(grad[i])), alpha=0.1, rasterized=True)
    surf_map_top = np.nanmean(surf_map[:, (df_yeo_surf['quantiles'] == 1).values[32492:]],axis=1)
    ax.loglog(f, surf_map_top, color='red', alpha=0.8)
    surf_map_bottom = np.nanmean(surf_map[:, (df_yeo_surf['quantiles'] == -1).values[32492:]],axis=1)
    ax.loglog(f, surf_map_bottom, color='blue', alpha=0.8)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized PSD')
    xticks = [0.5, 4, 8, 13, 30, 80]
    xtick_labels = ["0.5", "4", "8", "13", "30", "80"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    for x in xticks:
        ax.axvline(x=x, color="grey", linestyle="--", alpha=0.4)
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_3b_ieeg_mica_psd.svg")

    # Process Bands
    fig, axes = plt.subplots(1, len(band_order), figsize=(20, 4), sharex=True, sharey=True)
    band_maps = {}
    for i, band in enumerate(band_order):
        # Extract Power in Band for each channel
        z = extract_band_power(pxx_raw, f, freq_bands[band], relative=False)
        sens = np.nan_to_num(np.vstack(df_channel['SensitivityMap_bip'].values), nan=0.0)
        surf_map = (z @ sens) / (np.sum(sens, axis=0) + 1e-12)
        surf_map[np.sum(sens, axis=0) == 0] = np.nan

        # Plot Surface Map
        surf_map[df_yeo_surf.hemisphere.isna()[32492:]] = np.nan
        surf32k_rh_infl.append_array(surf_map, name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        layout = [['rh1', 'rh2']]
        view = [['lateral', 'medial']]
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
            nan_color=(0, 0, 0, 1), cmap="Purples", transparent_bg=True, return_plotter=True)
        p.show()

        # 4. Correlation Analysis
        # Extract data only within the mask
        x_raw = surf_map[mask[32492:]]
        y = df_yeo_surf['t1_gradient1_SalVentAttn'].values[32492:][mask[32492:]]
        print(x_raw.shape, y.shape)


        # Filter: Only correlate vertices that had signal (non-zero smoothed)
        # AND are finite (no NaNs from bad electrodes)
        valid_data_mask = (x_raw != 0) & np.isfinite(x_raw) & np.isfinite(y)
        
        # if np.sum(valid_data_mask) < 10:
        #     print(f"Skipping {band}: Not enough valid vertices in mask.")
        #     continue

        # Z-score for statistics
        x_stats = zscore(x_raw[valid_data_mask])
        y_stats = zscore(y[valid_data_mask])

        surf_map = np.zeros(n_vertices)
        idx = np.flatnonzero(mask)[valid_data_mask]
        surf_map[idx] = x_stats
        surf_map[df_yeo_surf['salience_border'].isna()] = np.nan
        surf32k_rh_infl.append_array(surf_map[32492:], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        layout = [['rh1', 'rh2']]
        view = [['lateral', 'medial']]
        screenshot_path = f"/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_3b_ieeg_mica_{band}_map.svg"
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(0, 0, 0, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, screenshot=True, filename=screenshot_path)

        # Pearson
        r, _ = spearmanr(x_stats, y_stats)            
        r_null = []
        # Generate surrogates for the specific mask
        for y_surr_full in msr.randomize(y_stats):
            # Apply same valid mask filter
            r_null.append(spearmanr(x_stats, y_surr_full[valid_data_mask])[0])
        
        r_null = np.asarray(r_null)
        p_perm = np.mean(np.abs(r_null) >= np.abs(r))

        # 5. Visualization
        # Plot Scatter
        slope, intercept = np.polyfit(x_stats, y_stats, 1)
        axes[i].scatter(x_stats, y_stats, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_xlim([-3, 3])
        axes[i].set_ylim([-3, 3])
        axes[i].plot(x_stats, slope*x_stats + intercept, c=band_colors[i], lw=2.5)
        
        stats_text = f"$r={r:.2f}$\n$p_{{perm}}={p_perm:.3f}$"
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                     va='top', fontsize=11, fontweight='bold')
        axes[i].set_xlabel(band.capitalize(), color=band_colors[i])
        axes[0].set_ylabel('MPC gradient')
    plt.tight_layout()
    plt.savefig("/local_raid/data/pbautin/software/salience-network-multiscale-switch/results/figures/figure_3b_ieeg_mica_band_power_corr.svg")
    return band_maps
        

def main():
    # Setup Relative Paths
    parser = get_parser()
    args = parser.parse_args()
    ieeg_deriv = args.ieeg_deriv
    pni_deriv = args.pni_deriv
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

    # Load sensitivity for each contact information.
    df_sensitivity = load_sensitivity_info(root_dir=ieeg_deriv)

    # Load channel information
    cache_path = '/local_raid/data/pbautin/software/salience-network-multiscale-switch/data/dataframes/figure_3_channel_data_df.pkl'
    if os.path.exists(cache_path):
        print(f"Loading cached channel info from {cache_path}...")
        with open(cache_path, 'rb') as f:
            df_channel_data = pickle.load(f)
    else:
        print("Cache not found. Loading and processing channel info...")
        df_channel_data = load_original_data_files()
        with open(cache_path, 'wb') as f:
                pickle.dump(df_channel_data, f)
                print(f"Channel info saved to {cache_path}.")

    # Align sensitivity maps by contact name
    df1 = df_channel_data.merge(df_sensitivity, left_on=['Subject', 'Session', 'ContactName1'], right_on=['Subject', 'Session', 'ContactName'], how='left').rename(columns={'ContactSensitivityMap': 'Sens1'})
    df2 = df1.merge(df_sensitivity, left_on=['Subject', 'Session', 'ContactName2'], right_on=['Subject', 'Session', 'ContactName'], how='left').rename(columns={'ContactSensitivityMap': 'Sens2'})
    df2['SensitivityMap_bip'] = df2['Sens1'] - df2['Sens2']
    df2['SensitivityMap_bip'] = df2['SensitivityMap_bip'].map(lambda x: np.abs(x) if isinstance(x, np.ndarray) else np.zeros(32492))


    frequency_band_analysis_sensitivity(df2, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf)





    # print(df_channel)
    # frequency_band_analysis(df_channel, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf)
    # # #df_channel_info = df_channel_info[df_channel_info.Subject == 'PX001']
    # # # Explode the lists and drop NaNs
    # indices_rh = df_channel["ChannelIndices"].explode()
    # indices_rh = indices_rh[~indices_rh.isna()].astype(int).to_numpy()
    # plt_values = np.zeros(32492, dtype=int)
    # plt_values[indices_rh] = 1

    # # create df_electrodes with positions
    # surf_lh = read_surface('/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA/sub-PX001/ses-01/surf/sub-PX001_ses-01_hemi-L_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii')
    # surf_rh = read_surface('/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA/sub-PX001/ses-01/surf/sub-PX001_ses-01_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness.surf.gii')

    # # Append to surface
    # #salience_border = nib.load('/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA/sub-PX001/ses-01/maps/sub-PX001_ses-01_ChannelMap_hemi-R_space-nativepro_surf-fsLR-32k_label-midthickness.gii').darrays[0].data
    # surf_rh.append_array(plt_values, name="overlay2")
    # surfs = {'rh1': surf_rh, 'rh2': surf_rh}
    # layout = [['rh1', 'rh2']]
    # view = [['lateral', 'medial']]
    # p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
    #             nan_color=(220, 220, 220, 1), cmap="Purples", transparent_bg=True, return_plotter=True)
    # p.show()
    # # Add colored spheres
    # for i, pos in enumerate(df_electrode_pos[['x', 'y', 'z']].values):
    #     sphere = vtkSphereSource()
    #     sphere.SetCenter(*pos)
    #     sphere.SetRadius(1.5)
    #     sphere.Update()
    #     actor = p.renderers[0][0].AddActor()
    #     actor.SetMapper(inputData=sphere.GetOutput())
    #     #actor.GetProperty().SetColor(*rgb)
    #     actor.GetProperty().SetOpacity(1.0)
    #     actor.RotateX(-90)
    #     actor.RotateZ(90)

    # # Add colored spheres
    # for i, pos in enumerate(df_electrode_pos[['x', 'y', 'z']].values):
    #     sphere = vtkSphereSource()
    #     sphere.SetCenter(*pos)
    #     sphere.SetRadius(1.5)
    #     sphere.Update()
    #     actor = p.renderers[1][0].AddActor()
    #     actor.SetMapper(inputData=sphere.GetOutput())
    #     #actor.GetProperty().SetColor(*rgb)
    #     actor.GetProperty().SetOpacity(1.0)
    #     actor.RotateX(-90)
    #     actor.RotateZ(90)
    #     actor.RotateZ(180)
    # p.show()


if __name__ == "__main__":
    main()




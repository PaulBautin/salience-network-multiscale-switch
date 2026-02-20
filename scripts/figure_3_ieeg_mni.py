from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Concatenate ieeg information from MNI Open iEEG atlas,
#
# database
# 1772 channels with normal brain activity from 106 subjects, 
# registered to a common stereotaxic space. https://mni-open-ieegatlas.research.mcgill.ca/
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3_ieeg_mni.py \
#   -pni_deriv /data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0 \
#   -ieeg_deriv /local_raid/data/pbautin/downloads/MNI_ieeg/MatlabFile.mat
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


#### imports
import argparse
from pathlib import Path
import numpy as np
import os
import pandas as pd
import seaborn as sns

from brainspace.plotting import plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import mesh_elements
from brainspace.datasets import load_conte69
from brainspace import mesh
from brainspace.null_models import moran
from brainspace.mesh.array_operations import smooth_array

from scipy.stats import spearmanr, zscore
from scipy.spatial import cKDTree
from scipy.io import loadmat
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
import matplotlib as mp
import matplotlib.patches as patches

from vtkmodules.vtkFiltersSources import vtkSphereSource

from src.atlas_load import load_yeo_atlas, load_t1_salience_profiles, load_bigbrain_gradients, convert_states_str2int
from src.gradient_computation import compute_t1_gradient
from src.ieeg_processing import preprocess_and_compute_psd_ieeg, extract_band_power
from src.plot_colors import yeo7_rgba, yeo7_rgb

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


def plot_surface_nodes(surf, df_data, df_yeo_surf, project_root):
    # Create custom colormap
    colors = ['darkgray', 'purple', 'orange', 'red', 'blue']  # Index 0 is white
    custom_cmap = mp.colors.ListedColormap(colors)
    norm = mp.colors.Normalize(vmin=0, vmax=4)  # Normalize integers from 0–4

    channel_type_raw = df_data['ChannelType'].values
    channel_type_flat = [item[0][0] for item in channel_type_raw]
    channel_type_mapping_int = {
        'D': 1,  # Dixi intracerebral electrodes
        'M': 2,  # Homemade MNI intracerebral electrodes
        'A': 3,  # AdTech intracerebral electrodes
        'G': 4   # AdTech subdural strips and grids
    }
    channel_integers = np.array([channel_type_mapping_int.get(ct, 0) for ct in channel_type_flat])

    ##### Plotting #####
    salience_border = df_yeo_surf['salience_border'].values.astype(float)
    surf.append_array(salience_border[32492:], name="overlay2")
    surfs = {'rh1': surf, 'rh2': surf}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    #screenshot_path = f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_{band}_map_mni_atlas.svg"
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
        nan_color=(220, 220, 220, 1), cmap="Greys", transparent_bg=True, return_plotter=True)
    for i, pos in enumerate(df_data['ChannelPosition_conte69_infl']):
        val = channel_integers[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[0][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)

    # Add colored spheres
    for i, pos in enumerate(df_data['ChannelPosition_conte69_infl']):
        val = channel_integers[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[1][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)
        actor.RotateZ(180)
    screenshot_path = project_root / "results/figures/figure_3a_ieeg_mni_channel_types.svg"
    p.screenshot(screenshot_path, transparent_bg=True)

    # coverage map
    mask = np.zeros(df_yeo_surf["t1_gradient1_SalVentAttn"].values.shape)
    mask[df_data['ChannelIndices_conte69']] = 1
    # Smooth the mesh or a vertex function using Laplace smoothing.
    # Applies iterative smoothing: v_new = (1-relax)*v + relax * M*v where M is the vertex-area weighted adjacency matrix.
    sigma = 5.0
    relax = 0.1
    t = (sigma ** 2) / 2.0
    n_iter = int(np.ceil(t / relax))
    smoothed_values_gradient = smooth_array(load_conte69(join=True), point_data=mask, n_iter=n_iter, sigma=sigma, relax=relax)
    smoothed_values_gradient[df_yeo_surf.network == 'medial_wall'] = np.nan

    # Append to surface
    surf.append_array(df_yeo_surf['t1_gradient1_SalVentAttn'].values[32492:], name="overlay1")
    salience_border = df_yeo_surf['salience_border'].values.astype(float)
    surf.append_array(salience_border[32492:], name="overlay2")
    surf.append_array(smoothed_values_gradient[32492:], name="overlay3")
    surfs = {'rh1': surf, 'rh2': surf}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay3", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="Purples", return_plotter=True)
    screenshot_path = project_root / "results/figures/figure_3a_ieeg_mni_channel_coverage.svg"
    p.screenshot(screenshot_path, transparent_bg=True)


def plot_surface_nodes_gradients(surf, df_data, df_yeo_surf, project_root):
    df = df_data[df_data.network == 'SalVentAttn'].copy()
    df['t1_gradient1'] = zscore(df['t1_gradient1'].values)

    # Plot
    salience_border = np.nan_to_num(df_yeo_surf['salience_border'].values.astype(float) - 1, nan=1)
    surf.append_array(salience_border[32492:], name="overlay2")
    surfs = {'rh1': surf, 'rh2': surf}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
        nan_color=(220, 220, 220, 1), cmap="Greys", transparent_bg=True, return_plotter=True)
    
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-3, vmax=3)
    for i, pos in enumerate(df['ChannelPosition_conte69_infl']):
        val = df['t1_gradient1'].values[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[0][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)

    # Add colored spheres
    for i, pos in enumerate(df['ChannelPosition_conte69_infl']):
        val = df['t1_gradient1'].values[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[1][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)
        actor.RotateZ(180)
    screenshot_path = project_root / "results/figures/figure_3a_ieeg_mni_channel_gradient.svg"
    p.screenshot(screenshot_path, transparent_bg=True)


def load_mni_ieeg_data(ieeg_deriv, project_root, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl):
    """ 
    Load MNI open ieeg data
    """
    data_dict = loadmat(ieeg_deriv, squeeze_me=True)
    filter_keys = ['ChannelName', 'ChannelType']#, 'ChannelPosition',, 'Data_W']
    data_dict_filtered = {key: data_dict[key] for key in filter_keys if key in data_dict}
    df_data = pd.DataFrame(data_dict_filtered)
    df_data['ChannelPosition'] = data_dict['ChannelPosition'].tolist()
    df_data['Data_W'] = data_dict['Data_W'].T.tolist()
    
    # channel_name = [str(c) for c in data['ChannelName']]
    # # Data_W: matrix with one column per channel, and 13600 samples containing all the signals for wakefulness
    # channel_type_flat = [str(c) for c in data['ChannelType']]
    # channel_type_mapping_int = {
    #     'D': 1,  # Dixi intracerebral electrodes
    #     'M': 2,  # Homemade MNI intracerebral electrodes
    #     'A': 3,  # AdTech intracerebral electrodes
    #     'G': 4}  # AdTech subdural strips and grids
    # channel_integers = np.array([channel_type_mapping_int.get(ct, 0) for ct in channel_type_flat])
    # # Create custom colormap
    # colors = ['darkgray', 'purple', 'orange', 'red', 'blue']  # Index 0 is white
    # custom_cmap = mp.colors.ListedColormap(colors)
    # norm = mp.colors.Normalize(vmin=0, vmax=4)  # Normalize integers from 0–4
    # Create surface polydata objects
    surf_lh = mesh.mesh_creation.build_polydata(points=data_dict['NodesLeft'], cells=data_dict['FacesLeft'] - 1)
    surf_rh = mesh.mesh_creation.build_polydata(points=data_dict['NodesRight'], cells=data_dict['FacesRight'] - 1)
    mesh.mesh_io.write_surface(surf_lh, str(project_root / 'data/surfaces/ieeg_surfaces/surf_lh_ieeg_atlas.surf.gii'))
    mesh.mesh_io.write_surface(surf_rh, str(project_root / 'data/surfaces/ieeg_surfaces/surf_lh_ieeg_atlas.surf.gii'))

    # Electrode projection on cortical surface
    vertices = np.vstack((data_dict['NodesLeft'], data_dict['NodesRight']))
    tree = cKDTree(vertices)
    indices_surf = tree.query(np.stack(df_data['ChannelPosition'].to_numpy()))[1]
    df_data['ChannelPosition_surf_atlas'] = vertices[indices_surf].tolist()

    # electrode projection on registered (to template) cortical surface
    print(project_root)
    surf_reg_lh = read_surface(project_root / 'data/surfaces/ieeg_surfaces/L.anat.reg.surf.gii', itype='gii')
    surf_reg_rh = read_surface(project_root / 'data/surfaces/ieeg_surfaces/R.anat.reg.surf.gii', itype='gii')
    vertices_surf_reg = np.vstack((surf_reg_lh.GetPoints(), surf_reg_rh.GetPoints()))
    df_data['ChannelPosition_surf_reg'] = vertices_surf_reg[indices_surf].tolist()

    # Projection on template 32k surface
    vertices_32k = np.vstack(load_conte69(join=True).GetPoints())
    vertices_32k_infl = np.vstack((surf32k_lh_infl.GetPoints(), surf32k_rh_infl.GetPoints()))
    tree = cKDTree(vertices_32k)
    channel_indices_32k = tree.query(np.stack(df_data['ChannelPosition_surf_reg'].to_numpy()))[1]
    channel_indices_32k[channel_indices_32k < 32492] += 32492
    df_data['ChannelIndices_conte69'] = channel_indices_32k
    df_data['ChannelPosition_conte69_infl'] = vertices_32k_infl[channel_indices_32k].tolist()
    df_data['network'] = df_yeo_surf['network'][channel_indices_32k].values
    df_data['t1_gradient1'] = df_yeo_surf['t1_gradient1_SalVentAttn'][channel_indices_32k].values

    low_q, high_q = np.nanquantile(df_yeo_surf["t1_gradient1_SalVentAttn"], [0.25, 0.75])
    mask = np.zeros(df_yeo_surf["t1_gradient1_SalVentAttn"].values.shape)
    mask[channel_indices_32k] = 1
    df_yeo_surf.loc[mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] <= low_q), "quantiles"] = -1
    df_yeo_surf.loc[mask & (df_yeo_surf["t1_gradient1_SalVentAttn"] >= high_q), "quantiles"] = 1
    df_data['quantiles'] = df_yeo_surf['quantiles'][channel_indices_32k].values
    
    df_yeo_surf['bigbrain_g2'] = load_bigbrain_gradients()
    df_data['bigbrain_g2'] = df_yeo_surf['bigbrain_g2'][channel_indices_32k].values
    return df_data, data_dict['SamplingFrequency']


def frequency_band_analysis(df_data, surf32k_rh_infl, df_yeo_surf, sampling_frequency, project_root):
    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']

    # A. Setup Geometry
    surf_combined = load_conte69(join=True)
    n_vertices = surf_combined.GetPoints().shape[0]

    # B. Define Analysis Mask: SalVent network specifically within the RH
    #mask = (df_yeo_surf['hemisphere'] == 'RH') & (df_yeo_surf['network'] == 'SalVentAttn')
    mask = np.zeros(df_yeo_surf["t1_gradient1_SalVentAttn"].values.shape)
    mask[df_data.loc[df_data.network == 'SalVentAttn', 'ChannelIndices_conte69'].values] = 1
    mask = mask == 1

    # C. Pre-calculate Moran Weights
    w = mesh_elements.get_ring_distance(surf_combined, n_ring=1, mask=mask)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=100, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)

    # Compute PSD
    data_w = np.stack(df_data['Data_W'].to_numpy())
    freq, pxx = preprocess_and_compute_psd_ieeg(data_w, sampling_frequency, fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)

    # E. Process Bands
    fig, axes = plt.subplots(1, len(band_order), figsize=(20, 5), sharex=True, sharey=True)
    band_maps = {}

    # Pre-process mapping indices (Run once, use many times)
    indices_list = [np.atleast_1d(idx) for idx in df_data['ChannelIndices_conte69'].values]
    indices_list = [i[np.isfinite(i)] for i in indices_list]
    counts = np.array([idx.size for idx in indices_list])
    flat_idxs = np.concatenate(indices_list).astype(np.int64)

    for i, band in enumerate(band_order):
        # Extract Power and zscore
        z = extract_band_power(pxx, freq, freq_bands[band], relative=False)
        
        # 2. Map to Surface (Vectorized)
        flat_vals = np.repeat(z, counts)
        
        sum_per_vertex = np.bincount(flat_idxs, weights=flat_vals, minlength=n_vertices)
        hits_per_vertex = np.bincount(flat_idxs, minlength=n_vertices)

        surf_map = np.zeros(n_vertices)
        valid_verts = hits_per_vertex > 0
        surf_map[valid_verts] = sum_per_vertex[valid_verts] / hits_per_vertex[valid_verts]


        # Correlation Analysis
        x_raw = surf_map[mask]
        y = df_yeo_surf.loc[mask, 't1_gradient1_SalVentAttn'].values
        valid_data_mask = (x_raw != 0) & np.isfinite(x_raw) & np.isfinite(y)
        # Z-score for statistics
        x_stats = zscore(x_raw[valid_data_mask])
        y_stats = zscore(y[valid_data_mask])


        salience_border = np.nan_to_num(df_yeo_surf['salience_border'].values.astype(float) - 1, nan=1)
        surf32k_rh_infl.append_array(salience_border[32492:], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        layout = [['rh1', 'rh2']]
        view = [['lateral', 'medial']]
        screenshot_path = project_root / f"results/figures/figure_3a_ieeg_mni_{band}_map.svg"
        p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="Greys", transparent_bg=True, return_plotter=True)
        # Add colored spheres
        custom_cmap = plt.get_cmap(name="coolwarm")
        norm = mp.colors.Normalize(vmin=-2, vmax=2)
        for n, pos in enumerate(surf32k_rh_infl.GetPoints()[mask[32492:]][valid_data_mask]):
            val = x_stats[n]
            rgba = custom_cmap(norm(val))
            rgb = rgba[:3]
            sphere = vtkSphereSource()
            sphere.SetCenter(*pos)
            sphere.SetRadius(1.5)
            sphere.Update()
            actor = p.renderers[0][0].AddActor()
            actor.SetMapper(inputData=sphere.GetOutput())
            actor.GetProperty().SetColor(*rgb)
            actor.GetProperty().SetOpacity(1.0)
            actor.RotateX(-90)
            actor.RotateZ(90)

        # Add colored spheres
        for n, pos in enumerate(surf32k_rh_infl.GetPoints()[mask[32492:]][valid_data_mask]):
            val = x_stats[n]
            rgba = custom_cmap(norm(val))
            rgb = rgba[:3]
            sphere = vtkSphereSource()
            sphere.SetCenter(*pos)
            sphere.SetRadius(1.5)
            sphere.Update()
            actor = p.renderers[1][0].AddActor()
            actor.SetMapper(inputData=sphere.GetOutput())
            actor.GetProperty().SetColor(*rgb)
            actor.GetProperty().SetOpacity(1.0)
            actor.RotateX(-90)
            actor.RotateZ(90)
            actor.RotateZ(180)
        p.screenshot(screenshot_path, transparent_bg=True)

        # Pearson
        r, _ = spearmanr(x_stats, y_stats)            
        r_null = []
        # Generate surrogates for the specific mask
        for y_surr_full in msr.randomize(y_stats):
            # Apply same valid mask filter
            r_null.append(spearmanr(x_stats, y_surr_full[valid_data_mask])[0])
        r_null = np.asarray(r_null)
        p_perm = np.mean(np.abs(r_null) >= np.abs(r))

        # Plot Scatter
        slope, intercept = np.polyfit(x_stats, y_stats, 1)
        axes[i].scatter(x_stats, y_stats, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_xlim([-3, 3])
        axes[i].set_ylim([-3, 3])
        axes[i].plot(x_stats, slope*x_stats + intercept, c=band_colors[i], lw=2.5)
        axes[i].text(0.05, 0.95, f"r = {r:.2f}\np = {p_perm:.2e}", transform=axes[i].transAxes, va="top")
        axes[i].set_xlabel(band.capitalize(), color=band_colors[i])
        axes[i].set_aspect("equal")
        axes[0].set_ylabel('MPC gradient')
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_3a_ieeg_mni_band_power_corr.svg")
    return band_maps


def frequency_analysis(df_data, sampling_frequency, project_root):
    df = df_data[df_data.network == 'SalVentAttn'].copy()
    data_w = np.stack(df['Data_W'].to_numpy())
    freq, pxx = preprocess_and_compute_psd_ieeg(data_w, sampling_frequency, fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)
    g1_values = zscore(df['t1_gradient1'].values)
    #pxx_log = np.log10(pxx + 1e-12)
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-3, vmax=3)

    # PSD computation
    fig, ax = plt.subplots(figsize=(6, 4))
    for i in range(pxx.shape[0]): 
        plt.loglog(freq, pxx[i, :], color=custom_cmap(norm(g1_values[i])), alpha=0.1, rasterized=True)
    surf_map_top = np.nanmean(pxx[(df['quantiles'] == 1), :], axis=0)
    ax.loglog(freq, surf_map_top, color='red', alpha=0.8, label='top 25%')
    surf_map_bottom = np.nanmean(pxx[(df['quantiles'] == -1), :], axis=0)
    ax.loglog(freq, surf_map_bottom, color='blue', alpha=0.8, label='bottom 25%')
    plt.legend(frameon=False, loc='lower left', bbox_to_anchor=(0.03, 0.03))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Normalized PSD')
    xticks = [0.5, 4, 8, 13, 30, 80]
    xtick_labels = ["0.5", "4", "8", "13", "30", "80"]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)
    for x in xticks:
        ax.axvline(x=x, color="grey", linestyle="--", alpha=0.4)
    plt.savefig(project_root / "results/figures/figure_3a_ieeg_mni_psd.svg")


def correlation_analysis(df_data, sampling_frequency, project_root):
    data_w = np.stack(df_data['Data_W'].to_numpy())
    freq, pxx = preprocess_and_compute_psd_ieeg(data_w, sampling_frequency, fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)
    pxx = zscore(pxx, axis=0)
    A_psd = np.corrcoef(pxx)

    # Sort nodes by network
    node_networks = df_data['network'].values
    sort_idx = np.argsort(node_networks)
    A_sorted = A_psd[sort_idx][:, sort_idx]
    sorted_networks = node_networks[sort_idx]

    # Find network boundaries
    boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0] + 1
    boundaries = np.insert(boundaries,0,0)

    mpc_fig = A_sorted.copy()
    mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    mpc_fig = rotate(mpc_fig, angle=-45, order=0, cval=np.nan)

    # Plot correlation matrix
    fig, ax = plt.subplots()
    b_ext = np.append(boundaries, node_networks.shape[0])
    for i, b in enumerate(boundaries):
        rect = patches.Rectangle((node_networks.shape[0] / 2  * np.sqrt(2), b * np.sqrt(2)), b_ext[i+1] - b_ext[i], b_ext[i+1] - b_ext[i], linewidth=2, edgecolor=yeo7_rgb[i], facecolor='none', angle=45)
        ax.add_patch(rect)

    mpc_fig[mpc_fig > 1] = 1
    plt.imshow(mpc_fig, cmap='coolwarm', origin='upper')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_3a_ieeg_mni_corr.svg")


def correlation_analysis_scatter(surf, df_data, sampling_frequency, df_yeo_surf, project_root):
    data_w = np.stack(df_data['Data_W'].to_numpy())
    freq, pxx = preprocess_and_compute_psd_ieeg(data_w, sampling_frequency, fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)
    pxx = zscore(pxx, axis=0)
    A_psd = np.corrcoef(pxx)

    other_net = (df_data["network"] != 'medial_wall') & (df_data["network"] != 'SalVentAttn')
    A_bottom = np.mean(A_psd[df_data.quantiles.values == -1][:, other_net], axis=0)
    A_top = np.mean(A_psd[df_data.quantiles.values == 1][:, other_net], axis=0)
    df_data.loc[other_net, 'corr_diff'] = zscore(np.abs(A_top) - np.abs(A_bottom))
    df_data["bigbrain_g2"] = zscore(df_data["bigbrain_g2"].values)
    df_data['network_int'] = convert_states_str2int(df_data['network'].values)[0]
    df_data['colors'] = [yeo7_rgb[int(k)] for k in df_data["network_int"]]

    corr, pval = spearmanr(df_data['corr_diff'], df_data['bigbrain_g2'], nan_policy="omit")

    # Correlation plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(df_data['corr_diff'], df_data['bigbrain_g2'], color=np.stack(df_data['colors'].to_numpy()), s=10, alpha=0.9, rasterized=True)
    sns.regplot(x=df_data['corr_diff'], y=df_data['bigbrain_g2'], scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[0])
    axes[0].text(0.05, 0.95, f"r = {corr:.2f}\np = {pval:.2e}", transform=axes[0].transAxes, va="top")
    axes[0].set_ylabel('BigBrain Gradient 2')
    axes[0].set_xlabel("ES$_{top}$ - ES$_{bottom}$")
    axes[0].set_xlim([-3,3])
    axes[0].set_ylim([-3,3])
    axes[0].set_aspect('equal')

    df_data_net = df_data[['network', 'corr_diff', 'colors', 'bigbrain_g2']].dropna().groupby('network').mean().reset_index().sort_values(by='bigbrain_g2')
    axes[1].barh(df_data_net['network'], df_data_net['corr_diff'], color=df_data_net['colors'], edgecolor='black', alpha=0.8, capsize=3, label='Spin null mean ± 95% CI')
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set_xlabel("Mean ES$_{top}$ - ES$_{bottom}$")
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_3a_ieeg_mni_corr_scatter.svg")

        # Plot
    salience_border = np.nan_to_num(df_yeo_surf['salience_border'].values.astype(float) - 1, nan=1)
    surf.append_array(salience_border[32492:], name="overlay2")
    surfs = {'rh1': surf, 'rh2': surf}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
        nan_color=(220, 220, 220, 1), cmap="Greys", transparent_bg=True, return_plotter=True)
    df = df_data.dropna(subset=['corr_diff'])
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-3, vmax=3)
    for i, pos in enumerate(df['ChannelPosition_conte69_infl']):
        val = df['corr_diff'].values[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[0][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)

    # Add colored spheres
    for i, pos in enumerate(df['ChannelPosition_conte69_infl']):
        val = df['corr_diff'].values[i]
        rgba = custom_cmap(norm(val))
        rgb = rgba[:3]
        sphere = vtkSphereSource()
        sphere.SetCenter(*pos)
        sphere.SetRadius(1.5)
        sphere.Update()
        actor = p.renderers[1][0].AddActor()
        actor.SetMapper(inputData=sphere.GetOutput())
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetOpacity(1.0)
        actor.RotateX(-90)
        actor.RotateZ(90)
        actor.RotateZ(180)
    screenshot_path = project_root / "results/figures/figure_3a_ieeg_mni_channel_corr_diff.svg"
    p.screenshot(screenshot_path, transparent_bg=True)

    # coverage map
    mask = np.zeros(df_yeo_surf["t1_gradient1_SalVentAttn"].values.shape)
    mask[df['ChannelIndices_conte69']] = df['corr_diff']
    # Smooth the mesh or a vertex function using Laplace smoothing.
    # Applies iterative smoothing: v_new = (1-relax)*v + relax * M*v where M is the vertex-area weighted adjacency matrix.
    sigma = 5.0
    relax = 0.1
    t = (sigma ** 2) / 2.0
    n_iter = int(np.ceil(t / relax))
    smoothed_values_gradient = smooth_array(load_conte69(join=True), point_data=mask, n_iter=n_iter, sigma=sigma, relax=relax)
    smoothed_values_gradient[(df_yeo_surf.network == 'medial_wall') & (df_yeo_surf.network == 'SalVentAttn')] = np.nan

    # Append to surface
    surf.append_array(df_yeo_surf['t1_gradient1_SalVentAttn'].values[32492:], name="overlay1")
    salience_border = np.nan_to_num(df_yeo_surf['salience_border'].values.astype(float) - 1, nan=1)
    surf.append_array(salience_border[32492:], name="overlay2")
    surf.append_array(smoothed_values_gradient[32492:], name="overlay3")
    surfs = {'rh1': surf, 'rh2': surf}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay3", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
            nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym', return_plotter=True)
    screenshot_path = project_root / "results/figures/figure_3a_ieeg_mni_channel_corr_diff_smooth.svg"
    p.screenshot(screenshot_path, transparent_bg=True)


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

    ######### Part 2 -- Extract iEEG data
    df_data, sampling_frequency = load_mni_ieeg_data(ieeg_deriv, project_root, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl)
    print(df_data)

    # plot_surface_nodes(surf32k_rh_infl, df_data, df_yeo_surf, project_root)
    # plot_surface_nodes_gradients(surf32k_rh_infl, df_data, df_yeo_surf, project_root)
    # correlation_analysis(df_data, sampling_frequency, project_root)
    correlation_analysis_scatter(surf32k_rh_infl, df_data, sampling_frequency, df_yeo_surf, project_root)
    # frequency_analysis(df_data, sampling_frequency, project_root)
    # frequency_band_analysis(df_data, surf32k_rh_infl, df_yeo_surf, sampling_frequency, project_root)

    # plot_values_gradient = np.zeros(vertices_32k_infl.shape[0])
    # plot_values_gradient[channel_indices_32k] = 1
    # smoothed_values_gradient = smooth_lapy(plot_values_gradient, load_conte69(join=True), sigma=5.0, lambda_=0.1, fix_zeros=False)
    # smoothed_values_gradient[df_yeo_surf.network == 'medial_wall'] = np.nan


    # # Append to surface
    # surf32k_rh_infl.append_array(df_yeo_surf['t1_gradient1_SalVentAttn'].values[32492:], name="overlay1")
    # salience_border = df_yeo_surf['salience_border'].values.astype(float)
    # surf32k_rh_infl.append_array(salience_border[32492:], name="overlay2")
    # surf32k_rh_infl.append_array(smoothed_values_gradient[32492:], name="overlay3")
    # surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
    # layout = [['rh1', 'rh2']]
    # view = [['lateral', 'medial']]

    # # coverage map
    # screenshot_path = f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_coverage_mni_atlas.svg"
    # p = plot_surf(surfs, layout=layout, view=view, array_name="overlay3", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
    #         nan_color=(220, 220, 220, 1), cmap="Purples", transparent_bg=True, screenshot=True, filename=screenshot_path)
    
    # #data['ChannelPosition'] = data['ChannelPosition'][indices_in_salience_mask,:]
    # gradient_values = df_yeo_surf['t1_gradient1_SalVentAttn'][indices_32k_salience].values
    # p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
    #             nan_color=(220, 220, 220, 1), cmap="Greys", color_range=(0,1), transparent_bg=True, return_plotter=True)
    # # Add colored spheres
    # # salience channel index: 89
    # custom_cmap = plt.get_cmap(name="coolwarm")
    # norm = mp.colors.Normalize(vmin=-1, vmax=1)
    # for i, pos in enumerate(data['ChannelPosition'][:,:]):
    #     val = gradient_values[i]
    #     rgba = custom_cmap(norm(val))
    #     rgb = rgba[:3]
    #     sphere = vtkSphereSource()
    #     sphere.SetCenter(*pos)
    #     sphere.SetRadius(1.5)
    #     sphere.Update()
    #     actor = p.renderers[0][0].AddActor()
    #     actor.SetMapper(inputData=sphere.GetOutput())
    #     actor.GetProperty().SetColor(*rgb)
    #     actor.GetProperty().SetOpacity(1.0)
    #     actor.RotateX(-90)
    #     actor.RotateZ(90)

    # # Add colored spheres
    # for i, pos in enumerate(data['ChannelPosition'][:,:]):
    #     val = gradient_values[i]
    #     rgba = custom_cmap(norm(val))
    #     rgb = rgba[:3]
    #     sphere = vtkSphereSource()
    #     sphere.SetCenter(*pos)
    #     sphere.SetRadius(1.5)
    #     sphere.Update()
    #     actor = p.renderers[1][0].AddActor()
    #     actor.SetMapper(inputData=sphere.GetOutput())
    #     actor.GetProperty().SetColor(*rgb)
    #     actor.GetProperty().SetOpacity(1.0)
    #     actor.RotateX(-90)
    #     actor.RotateZ(90)
    #     actor.RotateZ(180)
    # screenshot_path = f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_salience_gradient_mni_atlas.svg"
    # p.screenshot(screenshot_path, transparent_bg=True)
    

    # data_w = data['Data_W'].T
    # freq, pxx = preprocess_and_compute_psd_ieeg(data_w, data['SamplingFrequency'], fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)
    # pxx = (pxx - pxx.mean(axis=0, keepdims=True)) / pxx.std(axis=0, keepdims=True)
    # A_psd = np.corrcoef(pxx)
    # # print(connectivity_psd)
    # # plt.imshow(connectivity_psd, cmap='coolwarm', vmin=-1, vmax=1)
    # # plt.show()

    # from scipy.ndimage import rotate
    # # --- Sort nodes by network ---
    # node_networks = df_yeo_surf['network'].values[channel_indices_32k]
    # print(node_networks)
    # sort_idx = np.argsort(node_networks)
    # print(node_networks[sort_idx])
    # A_sorted = A_psd[sort_idx][:, sort_idx]
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
    # plt.imshow(mpc_fig, cmap='coolwarm', origin='upper')
    # plt.axis('off')
    # plt.title('Upper Triangle of MPC Rotated by 45°')
    # plt.tight_layout()
    # plt.show()

    # A_bottom = np.mean(A_psd[df_yeo_surf.quantile_idx.values[channel_indices_32k] == -1, :], axis=0)
    # A_top = np.mean(A_psd[df_yeo_surf.quantile_idx.values[channel_indices_32k] == 1, :], axis=0)
    # A_salience = np.abs(A_top) - np.abs(A_bottom)
    # #A_salience[df_yeo_surf.quantile_idx.values[channel_indices_32k] == -1] = -1
    # #A_salience[df_yeo_surf.quantile_idx.values[channel_indices_32k] == 1] = 1
    # plot_values_gradient = np.zeros(vertices_32k_infl.shape[0])
    # plot_values_gradient[channel_indices_32k] = A_salience
    # smoothed_values_gradient = smooth_lapy(plot_values_gradient, load_conte69(join=True), sigma=2, fix_zeros=False)
    # smoothed_values_gradient[df_yeo_surf.network == 'medial_wall'] = np.nan
    # smoothed_values_gradient[smoothed_values_gradient == 0] = np.nan
    # #plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, smoothed_values_gradient)

    # # Z-score for normalization
    # n_lh = surf32k_lh_infl.n_points
    # df_corr = pd.DataFrame({
    #     'network': df_yeo_surf['network'].values[n_lh:],
    #     'corr_diff': zscore(smoothed_values_gradient[32492:], nan_policy='omit')
    # })
    # print(df_corr)

    # print(df_yeo_surf)
    # df_corr["bigbrain_g2"] = zscore(load_bigbrain_gradients()[32492:])
    # df_yeo_surf['network_int'] = convert_states_str2int(df_yeo_surf['network'].values)[0]
    # colors = [yeo7_rgb[int(k)] for k in df_yeo_surf["network_int"]][:32492]
    # corr, pval = spearmanr(df_corr['corr_diff'], df_corr['bigbrain_g2'], nan_policy="omit")


    # fig, axes = plt.subplots(1, 2,figsize=(12, 6))
    # axes[0].scatter(df_corr['corr_diff'], df_corr['bigbrain_g2'], color=colors, s=10, alpha=0.9)
    # sns.regplot(x=df_corr['corr_diff'], y=df_corr['bigbrain_g2'], scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[0])
    # axes[0].text(0.05, 0.95, f"r = {corr:.2f}\np = {pval:.2e}", transform=axes[0].transAxes, va="top")
    # axes[0].set_ylabel('BigBrain Gradient 2')
    # axes[0].set_xlabel("ES$_{top}$ - ES$_{bottom}$")
    # #plt.show()

    # # Compute mean correlation difference per network
    # df_corr_mean = (
    #     df_corr
    #     .dropna(subset=['corr_diff'])
    #     .groupby('network')['corr_diff']
    #     .mean()
    # )
   
    # # Create SpinPermutations model (1000 rotations)
    # spin_model = SpinPermutations(n_rep=100, random_state=42)
    # spin_model.fit(load_conte69(as_sphere=True)[1])

    # # Split data into hemispheres
    # n_lh = surf32k_lh_infl.n_points
    # # Generate rotated surrogate maps
    # corr_spins = spin_model.randomize(smoothed_values_gradient[n_lh:])  # shape: (1000, n_vertices)
    # print(corr_spins.shape)

    # # Compute mean per network for each permutation
    # df_corr_spin = pd.DataFrame({
    #     'network': df_yeo_surf['network'].values[n_lh:]
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
    # #fig, ax = plt.subplots(figsize=(8, 4))
    # x = np.arange(len(df_corr_mean))
    # print(df_corr_mean)
    # print(x.shape)
    # # Bars for null distribution mean ± 95% CI
    # axes[1].barh(x, spin_mean, yerr=spin_ci, color='lightgrey', edgecolor='black', alpha=0.8, capsize=3, label='Spin null mean ± 95% CI')
    # # Scatter points for empirical correlation difference
    # axes[1].scatter(df_corr_mean.values, x, color=yeo7_rgba[:-1], alpha=0.8, s=100, zorder=5, label='Empirical mean')
    # axes[1].axvline(0, color='black', linewidth=1)
    # axes[1].set_xlabel("Mean ES$_{top}$ - ES$_{bottom}$")
    # yticks = np.arange(len(df_corr_mean.index))
    # axes[1].set_yticks(yticks)
    # axes[1].set_yticklabels(df_corr_mean.index.values)
    # axes[1].yaxis.set_label_position("right")
    # axes[1].yaxis.tick_right()
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
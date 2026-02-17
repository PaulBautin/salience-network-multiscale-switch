from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Dynamics of the salience network
#
# example:
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


#### imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
import os
from pprint import pprint
import glob
from os.path import dirname as up
import nibabel as nib
from nilearn import plotting, datasets
import pandas as pd
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import array_operations, mesh_elements
from brainspace.datasets import load_conte69
from brainspace.utils.parcellation import map_to_labels, reduce_by_labels, relabel
from brainspace.datasets import load_gradient, load_marker, load_conte69, load_parcellation
from brainspace import mesh

from brainspace.null_models import SpinPermutations, spin_permutations, moran
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

from brainspace.gradient import GradientMaps, kernels
import scipy
from joblib import Parallel, delayed

from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, linregress, skew, zscore


from brainspace.vtk_interface import wrap_vtk
from brainspace.plotting.base import Plotter
from vtkmodules.vtkFiltersSources import vtkSphereSource

from scipy.signal import welch
# import pycatch22 as catch22

from lapy import TriaMesh
from scipy.integrate import simpson

from figure_2_distance import load_bigbrain_gradients


def plot_surface_nodes(surf, data, df_yeo_surf):
    # Create custom colormap
    colors = ['darkgray', 'purple', 'orange', 'red', 'blue']  # Index 0 is white
    custom_cmap = mp.colors.ListedColormap(colors)
    norm = mp.colors.Normalize(vmin=0, vmax=4)  # Normalize integers from 0–4

    channel_type_raw = data['ChannelType']
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
    for i, pos in enumerate(data['ChannelPosition'][:,:]):
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
    for i, pos in enumerate(data['ChannelPosition'][:,:]):
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
    screenshot_path = f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_channel_types_mni_atlas.svg"
    p.screenshot(screenshot_path, transparent_bg=True)



def frequency_band_analysis(data, surf32k_lh_infl, surf32k_rh_infl, state, state_name, indices_32k, df_yeo_surf):
    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']

    # A. Setup Geometry
    surf_combined = load_conte69(join=True)
    n_vertices = surf_combined.GetPoints().shape[0]
    fs = data['SamplingFrequency']

    # B. Define Analysis Mask: SalVent network specifically within the RH
    mask = (df_yeo_surf['hemisphere'] == 'RH') & (df_yeo_surf['network'] == 'SalVentAttn')
    mask_indices = np.where(mask)[0]

    # C. Pre-calculate Moran Weights
    w = mesh_elements.get_ring_distance(surf_combined, n_ring=1, mask=mask.values)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=100, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)

    # 1. Compute PSD
    lengths = [len(sig) for sig in data['Data_W'].T]
    min_len, max_len = min(lengths), max(lengths)
    if min_len != max_len:
        print(f"Warning: Variable lengths detected ({min_len} to {max_len} samples).")
        print(f"Truncating all channels to {min_len} samples for vectorization.")
    data_matrix = np.vstack([np.asarray(sig)[:min_len] for sig in data['Data_W'].T])
    f, pxx_raw = preprocess_and_compute_psd_ieeg(data_matrix, fs)

    # E. Process Bands
    fig, axes = plt.subplots(1, len(band_order), figsize=(20, 4), sharex=True, sharey=True)
    band_maps = {}

    # Pre-process mapping indices (Run once, use many times)
    indices_list = [np.atleast_1d(idx) for idx in indices_32k]
    indices_list = [i[np.isfinite(i)] for i in indices_list]
    counts = np.array([idx.size for idx in indices_list])
    flat_idxs = np.concatenate(indices_list).astype(np.int64)

    for i, band in enumerate(band_order):
        # 1. Extract Power and zscore
        z = extract_band_power(pxx_raw, f, freq_bands[band], relative=False)
        
        # 2. Map to Surface (Vectorized)
        flat_vals = np.repeat(z, counts)
        
        sum_per_vertex = np.bincount(flat_idxs, weights=flat_vals, minlength=n_vertices)
        hits_per_vertex = np.bincount(flat_idxs, minlength=n_vertices)

        surf_map = np.zeros(n_vertices)
        valid_verts = hits_per_vertex > 0
        surf_map[valid_verts] = sum_per_vertex[valid_verts] / hits_per_vertex[valid_verts]

        # 3. Smooth
        #smoothed = smooth_lapy(surf_map, surf_combined, sigma=2)
        #surf_map = smoothed
        #band_maps[band] = smoothed

        # 4. Correlation Analysis
        # Extract data only within the mask
        x_raw = surf_map[mask]
        y = df_yeo_surf.loc[mask, 't1_gradient1_SalVentAttn'].values

        # Filter: Only correlate vertices that had signal (non-zero smoothed)
        # AND are finite (no NaNs from bad electrodes)
        valid_data_mask = (x_raw != 0) & np.isfinite(x_raw) & np.isfinite(y)
        
        if np.sum(valid_data_mask) < 10:
            print(f"Skipping {band}: Not enough valid vertices in mask.")
            continue

        x_clean = x_raw[valid_data_mask]
        y_clean = y[valid_data_mask]
        
        # Z-score for statistics
        x_stats = scipy.stats.zscore(x_clean)
        y_stats = y_clean # Gradient is usually already normalized, or zscore it too

        # surf_map_masked = np.zeros(n_vertices)
        # idx = np.flatnonzero(mask)[valid_data_mask]
        # surf_map_masked[idx] = x_stats
        # surf_map_masked[df_yeo_surf['salience_border'].values] = np.nan
        # surf32k_rh_infl.append_array(surf_map_masked[32492:], name="overlay2")
        # surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        # layout = [['rh1', 'rh2']]
        # view = [['lateral', 'medial']]
        # p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 600), zoom=1.3, color_bar='bottom', share='both',
        #     nan_color=(0, 0, 0, 1), cmap="coolwarm", color_range='sym', transparent_bg=True, return_plotter=True)
        # p.show()

        salience_border = df_yeo_surf['salience_border'].values.astype(float)
        surf32k_rh_infl.append_array(salience_border[32492:], name="overlay2")
        surfs = {'rh1': surf32k_rh_infl, 'rh2': surf32k_rh_infl}
        layout = [['rh1', 'rh2']]
        view = [['lateral', 'medial']]
        screenshot_path = f"/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_{band}_map_mni_atlas.svg"
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
        
        # Null Model
        if msr is not None:
            # Note: We must randomize the FULL 'y' vector that matches 'w_masked',
            # then subset to 'valid_data_mask' for the correlation check
            y_subset_full = y # The full vector matching the weight matrix
            
            r_null = []
            # Generate surrogates for the specific mask
            for y_surr_full in msr.randomize(y_subset_full):
                # Apply same valid mask filter
                r_null.append(spearmanr(x_stats, y_surr_full[valid_data_mask])[0])
            
            r_null = np.asarray(r_null)
            p_perm = np.mean(np.abs(r_null) >= np.abs(r))
        else:
            p_perm = np.nan

        # 5. Visualization
        # Plot Scatter
        slope, intercept = np.polyfit(x_stats, y_stats, 1)
        axes[i].scatter(x_stats, y_stats, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_ylim([-1, 1])
        axes[i].plot(x_stats, slope*x_stats + intercept, c=band_colors[i], lw=2.5)
        
        stats_text = f"$r={r:.2f}$\n$p_{{perm}}={p_perm:.3f}$"
        axes[i].text(0.05, 0.95, stats_text, transform=axes[i].transAxes, 
                     va='top', fontsize=11, fontweight='bold')
        axes[i].set_xlabel(band.capitalize())
        axes[i].set_aspect('equal')
        axes[0].set_ylabel('MPC gradient')

    plt.tight_layout()
    plt.savefig("/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figures/figure3_ieeg_mni_atlas_band.svg")
    return band_maps



def main():
    #### load the conte69 hemisphere surfaces and spheres
    micapipe='/local_raid/data/pbautin/software/micapipe'
    # Load fsLR-32k inflated surface
    surf32k_lh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(micapipe + '/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf32k_lh = read_surface(micapipe + '/surfaces/fsLR-32k.L.midthickness.surf.gii', itype='gii')
    surf32k_rh = read_surface(micapipe + '/surfaces/fsLR-32k.R.midthickness.surf.gii', itype='gii')

    #### load yeo atlas 7 network
    df_yeo_surf = pd.read_csv('/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/figure2_df.tsv')

    ##### Extract the data #####
    data = scipy.io.loadmat("/local_raid/data/pbautin/downloads/MNI_ieeg/MatlabFile.mat", squeeze_me=True) # dict: data.keys()
    channel_name = [str(c) for c in data['ChannelName']]
    # Data_W: matrix with one column per channel, and 13600 samples containing all the signals for wakefulness
    channel_type_flat = [str(c) for c in data['ChannelType']]
    channel_type_mapping_int = {
        'D': 1,  # Dixi intracerebral electrodes
        'M': 2,  # Homemade MNI intracerebral electrodes
        'A': 3,  # AdTech intracerebral electrodes
        'G': 4}  # AdTech subdural strips and grids
    channel_integers = np.array([channel_type_mapping_int.get(ct, 0) for ct in channel_type_flat])
    # Create custom colormap
    colors = ['darkgray', 'purple', 'orange', 'red', 'blue']  # Index 0 is white
    custom_cmap = mp.colors.ListedColormap(colors)
    norm = mp.colors.Normalize(vmin=0, vmax=4)  # Normalize integers from 0–4
    # Create surface polydata objects
    surf_lh = mesh.mesh_creation.build_polydata(points=data['NodesLeft'], cells=data['FacesLeft'] - 1)
    surf_rh = mesh.mesh_creation.build_polydata(points=data['NodesRight'], cells=data['FacesRight'] - 1)
    mesh.mesh_io.write_surface(surf_lh, '/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/surf_lh_ieeg_atlas.surf.gii')
    mesh.mesh_io.write_surface(surf_rh, '/local_raid/data/pbautin/software/neuroimaging_scripts/salience_network/manuscript/surf_rh_ieeg_atlas.surf.gii')

    ###### Electrode projection on cortical surface ######
    vertices = np.vstack((data['NodesLeft'], data['NodesRight']))
    tree = cKDTree(vertices)
    indices_surf = tree.query(data['ChannelPosition'])[1]
    data['ChannelPosition'] = vertices[indices_surf]

    ## electrode projection on registered (to template) cortical surface
    surf_reg_lh = read_surface('/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/L.anat.reg.surf.gii', itype='gii')
    surf_reg_rh = read_surface('/local_raid/data/pbautin/software/neuroimaging_scripts/ieeg/R.anat.reg.surf.gii', itype='gii')
    vertices_surf_reg = np.vstack((surf_reg_lh.GetPoints(), surf_reg_rh.GetPoints()))
    data['ChannelPosition'] = vertices_surf_reg[indices_surf]

    # Projection on template 32k surface
    vertices_32k = np.vstack((surf32k_lh.GetPoints(), surf32k_rh.GetPoints()))
    vertices_32k_infl = np.vstack((surf32k_lh_infl.GetPoints(), surf32k_rh_infl.GetPoints()))
    tree = cKDTree(vertices_32k)
    channel_indices_32k = tree.query(data['ChannelPosition'])[1]
    channel_indices_32k[channel_indices_32k < 32492] =+ 32492
    data['ChannelPosition'] = vertices_32k_infl[channel_indices_32k]
    indices_in_salience_mask = np.asarray(df_yeo_surf['network'] == 'SalVentAttn')[channel_indices_32k] != 0
    indices_32k_salience = channel_indices_32k[indices_in_salience_mask]
    data['ChannelPosition'] = data['ChannelPosition'][indices_in_salience_mask,:]

    gradient_values = df_yeo_surf['t1_gradient1_SalVentAttn'][channel_indices_32k].values

    #frequency_analysis(data, indices_in_salience_mask, gradient_values)
    #frequency_band_analysis(data, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf.network.values, df_yeo_surf.network.unique(), channel_indices_32k, df_yeo_surf)

    # plot_values_gradient = np.zeros(vertices_32k_infl.shape[0])
    # plot_values_gradient[channel_indices_32k] = 1
    # smoothed_values_gradient = smooth_lapy(plot_values_gradient, load_conte69(join=True), sigma=5.0, lambda_=0.1, fix_zeros=False)
    # smoothed_values_gradient[df_yeo_surf.network == 'medial_wall'] = np.nan

    # plot_surface_nodes(surf32k_rh_infl, data, df_yeo_surf)

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
    

    data_w = data['Data_W'].T
    freq, pxx = preprocess_and_compute_psd_ieeg(data_w, data['SamplingFrequency'], fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)
    pxx = (pxx - pxx.mean(axis=0, keepdims=True)) / pxx.std(axis=0, keepdims=True)
    A_psd = np.corrcoef(pxx)
    # print(connectivity_psd)
    # plt.imshow(connectivity_psd, cmap='coolwarm', vmin=-1, vmax=1)
    # plt.show()

    from scipy.ndimage import rotate
    # --- Sort nodes by network ---
    node_networks = df_yeo_surf['network'].values[channel_indices_32k]
    print(node_networks)
    sort_idx = np.argsort(node_networks)
    print(node_networks[sort_idx])
    A_sorted = A_psd[sort_idx][:, sort_idx]
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
    b_ext = np.append(boundaries,node_networks.shape[0])
    for i, b in enumerate(boundaries):
        rect = patches.Rectangle((node_networks.shape[0] / 2  * np.sqrt(2), b * np.sqrt(2)), b_ext[i+1] - b_ext[i], b_ext[i+1] - b_ext[i], linewidth=2, edgecolor=yeo7_colors.colors[i], facecolor='none', angle=45)
        # Add the patch to the Axes
        ax.add_patch(rect)

    mpc_fig[mpc_fig > 1] = 1
    plt.imshow(mpc_fig, cmap='coolwarm', origin='upper')
    plt.axis('off')
    plt.title('Upper Triangle of MPC Rotated by 45°')
    plt.tight_layout()
    plt.show()

    A_bottom = np.mean(A_psd[df_yeo_surf.quantile_idx.values[channel_indices_32k] == -1, :], axis=0)
    A_top = np.mean(A_psd[df_yeo_surf.quantile_idx.values[channel_indices_32k] == 1, :], axis=0)
    A_salience = np.abs(A_top) - np.abs(A_bottom)
    #A_salience[df_yeo_surf.quantile_idx.values[channel_indices_32k] == -1] = -1
    #A_salience[df_yeo_surf.quantile_idx.values[channel_indices_32k] == 1] = 1
    plot_values_gradient = np.zeros(vertices_32k_infl.shape[0])
    plot_values_gradient[channel_indices_32k] = A_salience
    smoothed_values_gradient = smooth_lapy(plot_values_gradient, load_conte69(join=True), sigma=2, fix_zeros=False)
    smoothed_values_gradient[df_yeo_surf.network == 'medial_wall'] = np.nan
    smoothed_values_gradient[smoothed_values_gradient == 0] = np.nan
    #plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, smoothed_values_gradient)

    # Z-score for normalization
    n_lh = surf32k_lh_infl.n_points
    df_corr = pd.DataFrame({
        'network': df_yeo_surf['network'].values[n_lh:],
        'corr_diff': zscore(smoothed_values_gradient[32492:], nan_policy='omit')
    })
    print(df_corr)

    print(df_yeo_surf)
    df_corr["bigbrain_g2"] = zscore(load_bigbrain_gradients()[32492:])
    df_yeo_surf['network_int'] = convert_states_str2int(df_yeo_surf['network'].values)[0]
    colors = [yeo7_rgb[int(k)] for k in df_yeo_surf["network_int"]][:32492]
    corr, pval = spearmanr(df_corr['corr_diff'], df_corr['bigbrain_g2'], nan_policy="omit")


    fig, axes = plt.subplots(1, 2,figsize=(12, 6))
    axes[0].scatter(df_corr['corr_diff'], df_corr['bigbrain_g2'], color=colors, s=10, alpha=0.9)
    sns.regplot(x=df_corr['corr_diff'], y=df_corr['bigbrain_g2'], scatter=False, color="black", line_kws={"linewidth": 1}, ax=axes[0])
    axes[0].text(0.05, 0.95, f"r = {corr:.2f}\np = {pval:.2e}", transform=axes[0].transAxes, va="top")
    axes[0].set_ylabel('BigBrain Gradient 2')
    axes[0].set_xlabel("ES$_{top}$ - ES$_{bottom}$")
    #plt.show()

    # Compute mean correlation difference per network
    df_corr_mean = (
        df_corr
        .dropna(subset=['corr_diff'])
        .groupby('network')['corr_diff']
        .mean()
    )
   
    # Create SpinPermutations model (1000 rotations)
    spin_model = SpinPermutations(n_rep=100, random_state=42)
    spin_model.fit(load_conte69(as_sphere=True)[1])

    # Split data into hemispheres
    n_lh = surf32k_lh_infl.n_points
    # Generate rotated surrogate maps
    corr_spins = spin_model.randomize(smoothed_values_gradient[n_lh:])  # shape: (1000, n_vertices)
    print(corr_spins.shape)

    # Compute mean per network for each permutation
    df_corr_spin = pd.DataFrame({
        'network': df_yeo_surf['network'].values[n_lh:]
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
    #fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df_corr_mean))
    print(df_corr_mean)
    print(x.shape)
    # Bars for null distribution mean ± 95% CI
    axes[1].barh(x, spin_mean, yerr=spin_ci, color='lightgrey', edgecolor='black', alpha=0.8, capsize=3, label='Spin null mean ± 95% CI')
    # Scatter points for empirical correlation difference
    axes[1].scatter(df_corr_mean.values, x, color=yeo7_rgba[:-1], alpha=0.8, s=100, zorder=5, label='Empirical mean')
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set_xlabel("Mean ES$_{top}$ - ES$_{bottom}$")
    yticks = np.arange(len(df_corr_mean.index))
    axes[1].set_yticks(yticks)
    axes[1].set_yticklabels(df_corr_mean.index.values)
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    plt.tight_layout()
    plt.show()










if __name__ == "__main__":
    main()
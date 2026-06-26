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
#   -ieeg_deriv /local_raid/data/pbautin/downloads/MNI_ieeg/MatlabFile.mat
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24"
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
from brainspace.datasets import load_conte69
from brainspace import mesh
from brainspace.mesh.array_operations import smooth_array

from scipy.stats import spearmanr, zscore
from scipy.spatial import cKDTree
from scipy.io import loadmat

import matplotlib.pyplot as plt
import matplotlib as mp

import logging

from src.atlas_load import load_yeo_atlas, load_bigbrain_gradients, convert_states_str2int, compute_network_mask
from src.ieeg_processing import preprocess_and_compute_psd_ieeg, plot_surface_sphere, compute_gradient_quantiles
from src.plot_colors import yeo7_rgba, yeo7_rgb
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)

plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False


def get_parser():
    """Build argument parser for the MNI iEEG figure script."""
    parser = argparse.ArgumentParser(
        description="Process ieeg derivatives and surfaces.",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-ieeg_deriv",
        type=str,
        help="Absolute path to the MNI iEEG MatlabFile.mat (e.g., /path/to/MatlabFile.mat)"
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-hemi",
        type=str,
        default="RH",
        choices=["both", "LH", "RH"],
        help="Hemisphere for analysis: 'both', 'LH', or 'RH' (default: RH)"
    )
    optional.add_argument(
        "-network",
        type=str,
        default="SalVentAttn",
        choices=["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"],
        help="Yeo 7-network to use as the analysis target (default: SalVentAttn)"
    )
    return parser


def load_mni_ieeg_data(ieeg_deriv, project_root, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, network):
    """Load MNI open iEEG data and project channels onto the fsLR-32k template surface.

    Reads the MNI Open iEEG Atlas MATLAB file, maps each electrode contact onto the
    native MNI surface and then onto the fsLR-32k template surface via nearest-neighbour
    search, and enriches the channel DataFrame with network labels, gradient values, and
    gradient quantile assignments for the requested Yeo network.

    Args:
        ieeg_deriv (str): Path to the MNI MatlabFile.mat.
        project_root (Path): Root of the project (used to locate registration surfaces).
        df_yeo_surf (pd.DataFrame): Surface DataFrame with network labels and gradient columns.
        surf32k_lh_infl (brainspace surface): Inflated left-hemisphere 32k surface.
        surf32k_rh_infl (brainspace surface): Inflated right-hemisphere 32k surface.
        network (str): Yeo 7-network label used to select the gradient column
            (e.g. 'SalVentAttn').

    Returns:
        df_data (pd.DataFrame): Channel-level DataFrame with columns:
            - ChannelName, ChannelType, ChannelPosition
            - Data_W: raw wakefulness signal, shape (n_samples,)
            - ChannelPosition_surf_atlas: nearest vertex on native MNI surface
            - ChannelPosition_surf_reg: nearest vertex on registered surface
            - ChannelIndices_conte69: integer vertex index on the 32k surface
            - ChannelPosition_conte69_infl: 3-D position on inflated 32k surface
            - network: Yeo network label at the electrode vertex
            - t1_gradient1: T1 gradient value at the electrode vertex
            - quantiles: gradient quantile label (-1 bottom 25%, +1 top 25%, NaN otherwise)
            - bigbrain_g2: BigBrain G2 gradient value at the electrode vertex
        sampling_frequency (float): Recording sampling rate in Hz.
    """
    data_dict = loadmat(ieeg_deriv, squeeze_me=True)
    filter_keys = ['ChannelName', 'ChannelType']
    data_dict_filtered = {key: data_dict[key] for key in filter_keys if key in data_dict}
    df_data = pd.DataFrame(data_dict_filtered)
    df_data['ChannelPosition'] = data_dict['ChannelPosition'].tolist()
    df_data['Data_W'] = data_dict['Data_W'].T.tolist()

    # Build native MNI surface polydata and write for reference
    surf_lh = mesh.mesh_creation.build_polydata(points=data_dict['NodesLeft'], cells=data_dict['FacesLeft'] - 1)
    surf_rh = mesh.mesh_creation.build_polydata(points=data_dict['NodesRight'], cells=data_dict['FacesRight'] - 1)
    mesh.mesh_io.write_surface(surf_lh, str(project_root / 'data/surfaces/ieeg_surfaces/surf_lh_ieeg_atlas.surf.gii'))
    mesh.mesh_io.write_surface(surf_rh, str(project_root / 'data/surfaces/ieeg_surfaces/surf_lh_ieeg_atlas.surf.gii'))

    # Electrode projection on cortical surface
    vertices = np.vstack((data_dict['NodesLeft'], data_dict['NodesRight']))
    tree = cKDTree(vertices)
    indices_surf = tree.query(np.stack(df_data['ChannelPosition'].to_numpy()))[1]
    df_data['ChannelPosition_surf_atlas'] = vertices[indices_surf].tolist()

    # Electrode projection on registered (to template) cortical surface
    logger.debug(f"Project root: {project_root}")
    surf_reg_lh = read_surface(project_root / 'data/surfaces/ieeg_surfaces/L.anat.reg.surf.gii', itype='gii')
    surf_reg_rh = read_surface(project_root / 'data/surfaces/ieeg_surfaces/R.anat.reg.surf.gii', itype='gii')
    vertices_surf_reg = np.vstack((surf_reg_lh.GetPoints(), surf_reg_rh.GetPoints()))
    df_data['ChannelPosition_surf_reg'] = vertices_surf_reg[indices_surf].tolist()

    # Projection on template 32k surface
    vertices_32k = np.vstack(load_conte69(join=True).GetPoints())
    vertices_32k_infl = np.vstack((surf32k_lh_infl.GetPoints(), surf32k_rh_infl.GetPoints()))
    tree = cKDTree(vertices_32k)
    channel_indices_32k = tree.query(np.stack(df_data['ChannelPosition_surf_reg'].to_numpy()))[1]
    # Force right-hemisphere channels into the RH index range (32492+)
    channel_indices_32k[channel_indices_32k < 32492] += 32492
    df_data['ChannelIndices_conte69'] = channel_indices_32k
    df_data['ChannelPosition_conte69_infl'] = vertices_32k_infl[channel_indices_32k].tolist()
    df_data['network'] = df_yeo_surf['network'][channel_indices_32k].values

    gradient_col = f't1_gradient1_{network}'
    df_data['t1_gradient1'] = df_yeo_surf[gradient_col][channel_indices_32k].values
    df_data['quantiles'] = compute_gradient_quantiles(df_yeo_surf, channel_indices_32k, gradient_col)

    df_yeo_surf['bigbrain_g2'] = load_bigbrain_gradients()
    df_data['bigbrain_g2'] = df_yeo_surf['bigbrain_g2'][channel_indices_32k].values
    return df_data, data_dict['SamplingFrequency']


def correlation_analysis_scatter(surf, df_data, sampling_frequency, df_yeo_surf, project_root, network):
    """Compute and plot electrophysiological similarity differences across the gradient.

    For each non-medial-wall, non-target-network channel, computes the difference
    in mean PSD correlation between gradient-top and gradient-bottom target-network
    channels (ES_top - ES_bottom). Correlates this difference with BigBrain G2
    gradient across channels and saves scatter + bar plots. Also renders a smoothed
    surface map of the correlation-difference values.

    Args:
        surf: brainspace surface object for rendering (inflated RH).
        df_data (pd.DataFrame): Channel-level DataFrame from load_mni_ieeg_data().
        sampling_frequency (float): Recording sampling rate in Hz.
        df_yeo_surf (pd.DataFrame): Surface DataFrame with gradient and network columns.
        project_root (Path): Root of the project (output written to results/figures/).
        network (str): Yeo 7-network used as the gradient-stratification target.
    """
    data_w = np.stack(df_data['Data_W'].to_numpy())
    freq, pxx = preprocess_and_compute_psd_ieeg(data_w, sampling_frequency, fmin=0.5, fmax=80.0, fs_target=200.0, filter_order=4, window_sec=2.0, overlap_sec=1.0)
    pxx = zscore(pxx, axis=0)
    A_psd = np.corrcoef(pxx)

    other_net = (df_data["network"] != 'medial_wall') & (df_data["network"] != network)
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

    # Surface plot of ES difference per channel
    salience_border = np.nan_to_num(df_yeo_surf['salience_border'].values.astype(float) - 1, nan=1)
    surf.append_array(salience_border[32492:], name="overlay2")
    surfs = {'rh1': surf, 'rh2': surf}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
        nan_color=(220, 220, 220, 1), cmap="Greys", transparent_bg=True, return_plotter=True)
    screenshot_path = project_root / "results/figures/figure_3a_ieeg_mni_channel_corr_diff.svg"
    df = df_data.dropna(subset=['corr_diff'])
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-3, vmax=3)
    plot_surface_sphere(p, df['ChannelPosition_conte69_infl'], custom_cmap(norm(df['corr_diff'].values)), screenshot_path)

    # Smoothed surface map of ES difference
    gradient_col = f't1_gradient1_{network}'
    mask = np.zeros(df_yeo_surf[gradient_col].values.shape)
    mask[df['ChannelIndices_conte69']] = df['corr_diff']
    # Laplace smoothing: v_new = (1-relax)*v + relax * M*v (M = vertex-area weighted adjacency)
    sigma = 5.0
    relax = 0.1
    t = (sigma ** 2) / 2.0
    n_iter = int(np.ceil(t / relax))
    smoothed_values_gradient = smooth_array(load_conte69(join=True), point_data=mask, n_iter=n_iter, sigma=sigma, relax=relax)
    smoothed_values_gradient[(df_yeo_surf.network == 'medial_wall') | (df_yeo_surf.network == network)] = np.nan

    surf.append_array(df_yeo_surf[gradient_col].values[32492:], name="overlay1")
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
    network = args.network
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_3_ieeg_mni", project_root, args)
    logger.info(f"Dataset        : MNI Open iEEG Atlas (N=106 subjects, 1772 channels, wakefulness)")
    logger.info(f"Electrode types: D=Dixi, M=MNI homemade, A=AdTech depth, G=AdTech subdural")
    logger.info(f"Preprocessing  : Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned")
    logger.info(f"PSD            : Welch method, Hamming window 2s, overlap 1s, normalized to unit sum")
    logger.info(f"Frequency bands: delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, gamma 30-80 Hz")
    logger.info(f"Null model     : Moran randomization (n_rep=100, procedure=singleton, random_state=0)")
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Analysis network: {network}")

    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    # Load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    # Load atlas
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)

    ######### Part 1 -- T1 map
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(f"Gradient dataframe not found at {path_df_1a}. Run figure_1a_t1map.py with -hemi {args.hemi} first.")
    logger.info(f"Loading gradient dataframe from {path_df_1a}")
    df_yeo_surf = pd.read_csv(path_df_1a, sep="\t")

    ######### Part 2 -- Extract iEEG data
    df_data, sampling_frequency = load_mni_ieeg_data(ieeg_deriv, project_root, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, network)
    logger.info(f"MNI iEEG loaded: {len(df_data)} channels, sampling frequency={sampling_frequency} Hz")

    correlation_analysis_scatter(surf32k_rh_infl, df_data, sampling_frequency, df_yeo_surf, project_root, network)


if __name__ == "__main__":
    main()

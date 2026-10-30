#!/usr/bin/env python
"""Step 2a: electrophysiological similarity along the gradient, MNI Open iEEG Atlas (Figure 2a).

The atlas holds wakefulness recordings of 1,772 channels with normal activity from 106
patients, in a common stereotaxic space (https://mni-open-ieegatlas.research.mcgill.ca/).
Each contact is mapped onto the fsLR-32k template through the atlas surfaces registered to
it, and all channels are placed on the right hemisphere. For every channel outside the
network, the electrophysiological similarity (ES: correlation of z-scored power spectra)
with the network channels in the top quarter of the MPC gradient is compared with that
with the bottom quarter, and this difference is correlated across channels with the second
BigBrain histological gradient against a Moran null on the channel layout.

Inputs   MNI Open iEEG Atlas MatlabFile.mat and its template-registered surfaces
         (config/paths.yaml); step 1a fsLR-32k gradient (source_data).
Public   source_data/step2a_ieeg_mni/: per channel, its template vertex, network, ES
         difference and BigBrain gradient; the statistic; manifest.
Figures  results/figures/figure_2a_ieeg_mni_*.svg

Run      python pipeline/step2a_ieeg_mni.py [--stage compute|plot|all]
"""
import logging

import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainspace.datasets import load_conte69
from brainspace.mesh.array_operations import smooth_array
from brainspace.mesh.mesh_io import read_surface
from brainspace.plotting import plot_surf
from scipy.io import loadmat
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import zscore

import common
from salience_switch import config
from salience_switch.colors import yeo7_rgb
from salience_switch.ieeg import compute_gradient_quantiles, preprocess_and_compute_psd_ieeg
from salience_switch.plotting import format_pvalue, plot_surface_sphere
from salience_switch.provenance import write_manifest
from salience_switch.stats import HemisphereMoranRandomization, moran_spearman
from salience_switch.surface import convert_states_str2int, load_bigbrain_gradients, load_inflated_surfaces

STEP = "step2a_ieeg_mni"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False

N_LH = 32492
# Moran surrogates of the channel-level correlation
N_PERMUTATIONS = 1000


def load_mni_ieeg_data(mat_path, surfaces_dir, df_yeo_surf, network):
    """Load the MNI Open iEEG Atlas and map each channel onto the fsLR-32k template.

    Each contact is assigned its nearest vertex on the atlas surface, carried to the
    template through the atlas surfaces registered to it (``L.anat.reg.surf.gii``,
    ``R.anat.reg.surf.gii``), then to the nearest fsLR-32k (conte69) vertex; every channel
    is placed on the right hemisphere.

    Args:
        mat_path (Path): The atlas MatlabFile.mat.
        surfaces_dir (Path): Folder of the template-registered atlas surfaces.
        df_yeo_surf (pd.DataFrame): fsLR-32k surface table with network labels and the
            gradient column.
        network (str): Yeo 7-network whose gradient defines the quantiles.

    Returns:
        df_data (pd.DataFrame): One row per channel with ChannelName, ChannelType,
            ChannelPosition, Data_W (wakefulness signal), ChannelIndices_conte69 (template
            vertex), network, t1_gradient1, quantiles (-1 bottom 25 %, +1 top 25 %, NaN
            otherwise) and bigbrain_g2.
        sampling_frequency (float): Recording sampling rate in Hz.
    """
    data_dict = loadmat(mat_path, squeeze_me=True)
    filter_keys = ['ChannelName', 'ChannelType']
    data_dict_filtered = {key: data_dict[key] for key in filter_keys if key in data_dict}
    df_data = pd.DataFrame(data_dict_filtered)
    df_data['ChannelPosition'] = data_dict['ChannelPosition'].tolist()
    df_data['Data_W'] = data_dict['Data_W'].T.tolist()

    # Electrode projection on cortical surface
    vertices = np.vstack((data_dict['NodesLeft'], data_dict['NodesRight']))
    tree = cKDTree(vertices)
    indices_surf = tree.query(np.stack(df_data['ChannelPosition'].to_numpy()))[1]

    # Electrode projection on registered (to template) cortical surface
    surf_reg_lh = read_surface(str(surfaces_dir / 'L.anat.reg.surf.gii'), itype='gii')
    surf_reg_rh = read_surface(str(surfaces_dir / 'R.anat.reg.surf.gii'), itype='gii')
    vertices_surf_reg = np.vstack((surf_reg_lh.GetPoints(), surf_reg_rh.GetPoints()))
    df_data['ChannelPosition_surf_reg'] = vertices_surf_reg[indices_surf].tolist()

    # Projection on template 32k surface
    vertices_32k = np.vstack(load_conte69(join=True).GetPoints())
    tree = cKDTree(vertices_32k)
    channel_indices_32k = tree.query(np.stack(df_data['ChannelPosition_surf_reg'].to_numpy()))[1]
    # Force right-hemisphere channels into the RH index range (32492+)
    channel_indices_32k[channel_indices_32k < N_LH] += N_LH
    df_data['ChannelIndices_conte69'] = channel_indices_32k
    df_data['network'] = df_yeo_surf['network'][channel_indices_32k].values

    gradient_col = f't1_gradient1_{network}'
    df_data['t1_gradient1'] = df_yeo_surf[gradient_col][channel_indices_32k].values
    df_data['quantiles'] = compute_gradient_quantiles(df_yeo_surf, channel_indices_32k, gradient_col)
    df_data['bigbrain_g2'] = load_bigbrain_gradients()[channel_indices_32k]
    return df_data, data_dict['SamplingFrequency']


def es_difference_statistics(df_data: pd.DataFrame, sampling_frequency: float, network: str,
                             n_rep: int = N_PERMUTATIONS) -> tuple[pd.DataFrame, dict]:
    """Electrophysiological similarity difference per channel and its correlation with BigBrain G2.

    For each channel outside the network (and the medial wall), the mean PSD correlation
    with the network channels of the top gradient quarter minus that with the bottom
    quarter (absolute correlations), z-scored across those channels. Its Spearman
    correlation with the z-scored BigBrain G2 across the channels with both values is
    tested against a Moran spectral randomisation null on the channel layout: the
    channels are scattered points, so the spatial weights are inverse Euclidean distances
    between their template (fsLR-32k midthickness) positions, floored at 1 mm for channels
    sharing a vertex. BigBrain G2 is randomised and the ES difference held fixed; add-one
    two-sided p.

    Returns
    -------
    channels : pd.DataFrame
        One row per channel: ``vertex`` (template vertex), ``network``, ``quantile``,
        ``corr_diff`` (NaN inside the network) and ``bigbrain_g2`` (z-scored).
    stats : dict
        ``r``, ``p_moran``, ``null_lo``, ``null_hi``, ``n_channels``, ``n_permutations``.
    """
    data_w = np.stack(df_data['Data_W'].to_numpy())
    freq, pxx = preprocess_and_compute_psd_ieeg(data_w, sampling_frequency, fmin=0.5, fmax=80.0, fs_target=200.0,
                                                filter_order=4, window_sec=2.0, overlap_sec=1.0)
    pxx = zscore(pxx, axis=0)
    A_psd = np.corrcoef(pxx)

    other_net = (df_data["network"] != 'medial_wall') & (df_data["network"] != network)
    A_bottom = np.mean(A_psd[df_data.quantiles.values == -1][:, other_net], axis=0)
    A_top = np.mean(A_psd[df_data.quantiles.values == 1][:, other_net], axis=0)
    df_data.loc[other_net, 'corr_diff'] = zscore(np.abs(A_top) - np.abs(A_bottom))
    df_data["bigbrain_g2"] = zscore(df_data["bigbrain_g2"].values)

    use = (df_data['corr_diff'].notna() & df_data['bigbrain_g2'].notna()).to_numpy()
    coords = np.asarray(load_conte69(join=True).GetPoints())[df_data.loc[use, 'ChannelIndices_conte69'].to_numpy()]
    weights = 1.0 / np.maximum(cdist(coords, coords), 1.0)
    np.fill_diagonal(weights, 0.0)
    # Every channel sits on the right hemisphere, so the per-hemisphere null is one block.
    msr = HemisphereMoranRandomization(np.full(int(use.sum()), 'RH'), n_rep, random_state=0).fit(weights)
    corr, pval, r_null = moran_spearman(df_data.loc[use, 'corr_diff'].to_numpy(),
                                        df_data.loc[use, 'bigbrain_g2'].to_numpy(), msr)
    lo, hi = np.percentile(r_null, [2.5, 97.5])
    logger.info(f"[Figure 2A] ES difference vs BigBrain G2 | Spearman r={corr:+.3f}, Moran permutation p={pval:.3e} "
                f"(n_perm={n_rep}, n_channels={int(use.sum())}; null r 95% [{lo:+.3f}, {hi:+.3f}])")
    channels = pd.DataFrame({"vertex": df_data['ChannelIndices_conte69'].to_numpy(),
                             "network": df_data['network'].to_numpy(),
                             "quantile": df_data['quantiles'].to_numpy(),
                             "corr_diff": df_data['corr_diff'].to_numpy(dtype=float),
                             "bigbrain_g2": df_data['bigbrain_g2'].to_numpy(dtype=float)})
    stats = {"network": network, "r": corr, "p_moran": pval, "null_lo": lo, "null_hi": hi,
             "n_channels": int(use.sum()), "n_permutations": n_rep}
    return channels, stats


def plot_es_difference(channels: pd.DataFrame, stats: pd.Series, df_yeo_surf: pd.DataFrame, network: str) -> None:
    """Figure 2a panels: scatter and per-network bars, channel spheres and the smoothed map."""
    out = config.FIGURES_DIR
    df_data = channels.rename(columns={"vertex": "ChannelIndices_conte69"})
    df_data['network_int'] = convert_states_str2int(df_data['network'].values)[0]
    df_data['colors'] = [yeo7_rgb[int(k)] for k in df_data["network_int"]]
    corr, pval = stats["r"], stats["p_moran"]

    # Correlation plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(df_data['corr_diff'], df_data['bigbrain_g2'], color=np.stack(df_data['colors'].to_numpy()), s=10,
                    alpha=0.9, rasterized=True)
    sns.regplot(x=df_data['corr_diff'], y=df_data['bigbrain_g2'], scatter=False, color="black",
                line_kws={"linewidth": 1}, ax=axes[0], seed=0)
    axes[0].text(0.05, 0.95, f"r = {corr:.2f}\np = {format_pvalue(pval)}", transform=axes[0].transAxes, va="top")
    axes[0].set_ylabel('BigBrain Gradient 2')
    axes[0].set_xlabel("ES$_{top}$ - ES$_{bottom}$")
    axes[0].set_xlim([-3, 3])
    axes[0].set_ylim([-3, 3])
    axes[0].set_aspect('equal')

    # Per-network mean ES difference: descriptive (raw per-network means, no error bars
    # and no significance test; the channel-level correlation above carries the Moran p).
    df_data_net = (df_data[['network', 'corr_diff', 'colors', 'bigbrain_g2']].dropna().groupby('network').mean()
                   .reset_index().sort_values(by='bigbrain_g2'))
    axes[1].barh(df_data_net['network'], df_data_net['corr_diff'], color=df_data_net['colors'], edgecolor='black',
                 alpha=0.8, label='Mean ES difference per network')
    axes[1].axvline(0, color='black', linewidth=1)
    axes[1].set_xlabel("Mean ES$_{top}$ - ES$_{bottom}$")
    axes[1].yaxis.set_label_position("right")
    axes[1].yaxis.tick_right()
    plt.tight_layout()
    plt.savefig(out / "figure_2a_ieeg_mni_corr_scatter.svg")
    plt.close(fig)

    # Surface plot of ES difference per channel
    surf_lh_infl, surf = load_inflated_surfaces("fsLR-32k")
    vertices_32k_infl = np.vstack((surf_lh_infl.GetPoints(), surf.GetPoints()))
    salience_border = np.nan_to_num(df_yeo_surf['salience_border'].values.astype(float) - 1, nan=1)
    surf.append_array(salience_border[N_LH:], name="overlay2")
    surfs = {'rh1': surf, 'rh2': surf}
    layout = [['rh1', 'rh2']]
    view = [['lateral', 'medial']]
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay2", size=(1200, 500), zoom=1.4,
                  color_bar='bottom', share='both', nan_color=(220, 220, 220, 1), cmap="Greys", transparent_bg=True,
                  return_plotter=True)
    df = df_data.dropna(subset=['corr_diff'])
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-3, vmax=3)
    plot_surface_sphere(p, vertices_32k_infl[df['ChannelIndices_conte69'].to_numpy()],
                        custom_cmap(norm(df['corr_diff'].values)), out / "figure_2a_ieeg_mni_channel_corr_diff.svg")

    # Smoothed surface map of ES difference
    gradient_col = f't1_gradient1_{network}'
    mask = np.zeros(df_yeo_surf[gradient_col].values.shape)
    mask[df['ChannelIndices_conte69']] = df['corr_diff']
    # Laplace smoothing: v_new = (1-relax)*v + relax * M*v (M = vertex-area weighted adjacency)
    sigma = 5.0
    relax = 0.1
    t = (sigma ** 2) / 2.0
    n_iter = int(np.ceil(t / relax))
    smoothed_values_gradient = smooth_array(load_conte69(join=True), point_data=mask, n_iter=n_iter, sigma=sigma,
                                            relax=relax)
    smoothed_values_gradient[(df_yeo_surf.network == 'medial_wall') | (df_yeo_surf.network == network)] = np.nan

    surf.append_array(df_yeo_surf[gradient_col].values[N_LH:], name="overlay1")
    surf.append_array(salience_border[N_LH:], name="overlay2")
    surf.append_array(smoothed_values_gradient[N_LH:], name="overlay3")
    p = plot_surf(surfs, layout=layout, view=view, array_name="overlay3", size=(1200, 500), zoom=1.4,
                  color_bar='bottom', share='both', nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym',
                  return_plotter=True)
    p.screenshot(out / "figure_2a_ieeg_mni_channel_corr_diff_smooth.svg", transparent_bg=True)


def compute(settings: dict, paths: dict) -> None:
    """Channel projection, ES differences and their Moran test against BigBrain G2."""
    network = settings["network"]
    source = config.source_dir(STEP)
    logger.info("Dataset        : MNI Open iEEG Atlas (N=106 subjects, 1772 channels, wakefulness)")
    logger.info("Electrode types: D=Dixi, M=MNI homemade, A=AdTech depth, G=AdTech subdural")
    logger.info("Preprocessing  : Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned")
    logger.info("PSD            : Welch method, Hamming window 2s, overlap 1s, normalized to unit sum")
    logger.info(f"Null model     : Moran randomization on the channel layout (inverse Euclidean distance, n_rep={N_PERMUTATIONS}, procedure=singleton, random_state=0), add-one empirical p")
    logger.info("Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Analysis network: {network}")
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    df_data, sampling_frequency = load_mni_ieeg_data(paths["mni_ieeg"], paths["mni_ieeg_surfaces"], df_yeo_surf,
                                                     network)
    logger.info(f"MNI iEEG loaded: {len(df_data)} channels, sampling frequency={sampling_frequency} Hz")
    channels, stats = es_difference_statistics(df_data, sampling_frequency, network)
    config.write_table(channels, source / "channels.tsv")
    config.write_table(pd.DataFrame([stats]), source / "es_difference_stats.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network,
                   n_channels=len(channels), n_permutations=N_PERMUTATIONS)


def plot(settings: dict) -> None:
    """Figure 2a panels, from source_data only."""
    source = config.source_dir(STEP)
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    plot_es_difference(config.read_table(source / "channels.tsv"),
                       config.read_table(source / "es_difference_stats.tsv").iloc[0], df_yeo_surf, settings["network"])


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 2a: ES difference along the gradient, MNI Open iEEG Atlas (Figure 2a).")

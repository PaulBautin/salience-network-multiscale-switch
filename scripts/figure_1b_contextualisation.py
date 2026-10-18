# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 1b - Contextualisation of local microstructural heterogeneity of the salience network
# using BigBrain and Ahead datasets
# 
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1b_contextualisation.py \
#   -hemi LH
# (requires figure_1a_t1map.py to have been run first)
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
import pandas as pd
import nibabel as nib
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

from brainspace.null_models import moran
from scipy.stats import spearmanr, zscore

import logging

from src.atlas_load import load_yeo_atlas, load_bigbrain, load_ahead_biel, load_ahead_parva, compute_network_mask
from src.connectome_processing import empirical_p_twosided
from src.logging_utils import setup_manuscript_logger
from src.plot_colors import histology_cmap
from src.surface_plotting import (plot_hemispheres_zoom, salience_zoom_rois, mask_outline, plot_forest,
                                  format_pvalue, SALIENCE_Z_RANGE, SALIENCE_Z_LABELS)

logger = logging.getLogger(__name__)


plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'

def get_parser() -> argparse.ArgumentParser:
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Process PNI derivatives and surfaces.",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-hemi",
        type=str,
        default="both",
        choices=["both", "LH", "RH"],
        help="Hemisphere for gradient computation: 'both', 'LH', or 'RH' (default: both)"
    )
    return parser


# Histological maps contextualising the gradient (stats, forest and zoom rows, in row
# order), their panel titles and output-file stems.
HISTOLOGY_MAPS = ["BigBrain", "Bielschowsky", "Parvalbumin"]
CONTEXT_TITLES = {
    'BigBrain': 'BigBrain – Merker staining (neuronal cell-body density)',
    'T1map': 'MICA-PNI – qT1 MRI (intracortical myelin content)',
    'Bielschowsky': 'AHEAD – Bielschowsky staining (axonal fiber density)',
    'Parvalbumin': 'AHEAD – Parvalbumin staining (PV+ interneurons)',
}
ZOOM_STEMS = {'BigBrain': 'bigbrain', 'Bielschowsky': 'biel', 'Parvalbumin': 'parva'}
# Figure 1 zoom-row window (as figure_1a_brain_zoom*.svg), inset-row titles at 36 pt:
# the 1450-pt rows are shown ZOOM_ROW_WIDTH_MM wide in the Figure 1 layout (~0.29x).
ZOOM_KW = dict(size=(1450, 300), zoom=1.3, color_bar=True, nan_color=(220, 220, 220, 1),
               label_fontsize=30, title_fontsize=36)


def context_analysis(df_yeo_surf: pd.DataFrame, surf_32k, modalities: list[str], n_rep: int = 1000, hemisphere: str = 'both', project_root: Path | None = None) -> pd.DataFrame:
    """Spearman correlation of each map with the SN MPC gradient against a Moran null.

    Each map is correlated with the gradient over the SN vertices (of ``hemisphere``)
    where both are finite, and the within-network Moran null is fitted on that same
    finite subset (add-one two-sided p). Writes the per-map statistics, with the
    2.5th / 50th / 97.5th percentiles of the null correlations, to
    ``df_1b_context_stats.csv``, and draws the supplementary vertex scatter + |r| stem
    figure and the half-polar |r| plot.

    Returns
    -------
    pd.DataFrame
        One row per map: ``modality, r, p, null_lo, null_med, null_hi, n_vertices, n_perm``.
    """
    ## Correlation analyses
    net_mask = df_yeo_surf['network'].eq('SalVentAttn')
    if hemisphere in ('LH', 'RH'):
        net_mask = net_mask & df_yeo_surf['hemisphere'].eq(hemisphere)
    x = zscore(df_yeo_surf.loc[net_mask, 't1_gradient1_SalVentAttn'].values)
    # Full-surface indices of the network vertices, so a per-modality finite subset can
    # be mapped back to a whole-surface mask for the Moran graph (see the loop below).
    net_idx = np.flatnonzero(net_mask.values)

    # Plot
    fig, axes = plt.subplots(
        len(modalities),
        2,
        figsize=(6.0, 2.8 * len(modalities)),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.4, 'width_ratios': [1.0, 1.0]},
    )
    axes = np.atleast_2d(axes)
    stats_rows = []
    for row, label in enumerate(modalities):
        ax = axes[row, 0]
        ax_stem = axes[row, 1]
        y = df_yeo_surf.loc[net_mask, label].values
        # Score the correlation only where both the gradient and the modality are
        # finite, and build the Moran spatial graph on that same finite subset so the
        # surrogate field carries only real values (no NaN/zero fill, which would inject
        # artificial values into the spatial-autocorrelation structure); this mirrors
        # the finite-subset null in figure_3_ieeg_mica.
        finite = np.isfinite(x) & np.isfinite(y)
        finite_mask = np.zeros(df_yeo_surf.shape[0], dtype=bool)
        finite_mask[net_idx[finite]] = True
        w = mesh_elements.get_ring_distance(surf_32k, n_ring=1, mask=finite_mask)
        w.data **= -1
        msr = moran.MoranRandomization(n_rep=n_rep, procedure='singleton', tol=1e-6, random_state=0)
        msr.fit(w)
        rand = msr.randomize(y[finite])
        sns.regplot(x=x[finite], y=y[finite], ax=ax, scatter_kws={"s": 20, "alpha": 0.1, "edgecolors":'none', 'rasterized':True}, line_kws={"color": "black", "lw":2.5})
        r_obs, _ = spearmanr(x[finite], y[finite])
        r_rand = np.asarray([spearmanr(x[finite], surr)[0] for surr in rand])
        pv_rand = empirical_p_twosided(r_rand, r_obs)
        null_lo, null_med, null_hi = np.percentile(r_rand, [2.5, 50, 97.5])
        stats_rows.append(dict(modality=label, r=r_obs, p=pv_rand, null_lo=null_lo, null_med=null_med,
                               null_hi=null_hi, n_vertices=int(finite.sum()), n_perm=n_rep))
        logger.info(f"[Figure 1B] {label}: MPC-gradient vs {label} | Spearman r={r_obs:.3f}, Moran permutation p={pv_rand:.3e} "
                    f"(n_perm={n_rep}, n_vertices={int(finite.sum())}; null r 95% [{null_lo:+.3f}, {null_hi:+.3f}])")
        stats_text = f"$r={r_obs:.2f}$\n$p={format_pvalue(pv_rand)}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
        t = ax.set_title(f"{CONTEXT_TITLES.get(label, label)}", loc='left', pad=15)
        t.set_in_layout(False)
        ax.set_ylim([-4,4])
        ax.set_xlim([-3,3])
        ax.set_yticks([-2, 2])
        ax.set_ylabel(label)
        ax.set_box_aspect(1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if row < len(modalities) - 1:
            ax.set_xlabel('')
            ax.tick_params(labelbottom=False)

        # Color the marker and stem based on sign of the observed r
        color = "tab:blue" if r_obs >= 0 else "tab:red"
        markerline, stemlines, baseline = ax_stem.stem([0], [abs(r_obs)], orientation='horizontal', basefmt='k-')
        plt.setp(markerline, markersize=7, markeredgewidth=0, markerfacecolor=color, markeredgecolor=color)
        plt.setp(stemlines, linewidth=2.5, color=color)
        plt.setp(baseline, linewidth=1.0, color='0.5')
        ax_stem.axhline(0, color='0.6', lw=1, zorder=0)
        ax_stem.axvline(0, color='0.9', lw=1, zorder=0)
        ax_stem.set_xlim(0, 0.5)
        ax_stem.set_ylim(-0.6, 0.6)
        ax_stem.set_yticks([])
        ax_stem.set_xticks([0, 0.25, 0.5])
        ax_stem.set_xticklabels(['0', '0.25', '0.5'])
        # Only keep an x-axis label on the bottom-most row
        if row < len(modalities) - 1:
            ax_stem.set_xlabel('')
            ax_stem.tick_params(labelbottom=False)
        else:
            ax_stem.set_xlabel('Spearman |r|')
        ax_stem.set_title('')
        ax_stem.set_box_aspect(1)
        ax_stem.spines['right'].set_visible(False)
        ax_stem.spines['top'].set_visible(False)
        ax_stem.spines['left'].set_visible(False)
        if row < len(modalities) - 1:
            ax_stem.spines['bottom'].set_visible(False)
        ax_stem.text(abs(r_obs) + 0.08, 0, f"{abs(r_obs):.2f}", ha='center', va='center', fontsize=11, fontweight='bold')

    axes[-1, 0].set_xlabel('MPC gradient')
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_1b_correlations_supp.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

    stats = pd.DataFrame(stats_rows)
    stats_path = project_root / "data/dataframes/df_1b_context_stats.csv"
    stats.to_csv(stats_path, index=False)
    logger.info(f"[Figure 1B] context statistics saved to {stats_path}")

    r_vals, labels = [], []
    for label in modalities:
        if label in df_yeo_surf.columns:
            y = df_yeo_surf.loc[net_mask, label].values
            if len(y) > 1 and not np.all(np.isnan(y)):
                r, _ = spearmanr(x, y, nan_policy='omit')
                r_vals.append(r)
                labels.append(label)

    # Convert to numpy
    r_vals = np.array(r_vals)

    if r_vals.size == 0:
        raise ValueError("No valid correlations could be computed. Check your modality columns.")

    # Half-circle polar coordinates
    N = len(r_vals)
    theta = np.linspace(-np.pi /2 + np.pi/N*0.8, np.pi /2, N, endpoint=False)[::-1]
    radii = np.abs(r_vals)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 8))
    bars = ax.bar(theta, radii, width=np.pi/N*0.8, align="center", alpha=0.8)
    ax.set_thetamax(90)
    ax.set_thetamin(-90)
    ax.tick_params(axis='y', labelsize=12) 
    ax.set_rticks([0.0, 0.1, 0.2, 0.3])

    # Color by sign
    for bar, r in zip(bars, r_vals):
        bar.set_facecolor("tab:red" if r < 0 else "tab:blue")
    plt.grid(axis='x')
    ax.set_xticklabels([])
    ax.set_ylabel("Spearman's |r|")
    #ax.set_yticklabels([])
    plt.savefig(project_root / "results/figures/figure_1b_correlations_circle.svg")
    plt.close(fig)

    return stats


def plot_context_zoom(df_yeo_surf: pd.DataFrame, surf_lh, surf_rh, stats: pd.DataFrame,
                      out_dir: Path, hemisphere: str = 'both') -> None:
    """Figure 1b context rows: one zoom row per histological map, plus the forest.

    Each map in ``stats`` (in its row order) is z-scored over the SN vertices of
    ``hemisphere`` and drawn as the ``SALIENCE_ZOOM_ROIS`` inset row alone (no brains,
    no region names), titled with the map name, on the Figure 1 window (``ZOOM_KW``):
    ``CustomCmap_histology`` on the shared +/-2 SD scale, the SN border traced
    (``mask_outline``). ``figure_1b_forest.svg`` (``plot_forest``) then puts one row
    level with each inset row when the rows are stacked in that order.
    """
    net_mask = compute_network_mask(df_yeo_surf, 'SalVentAttn', hemisphere)
    rois = [dict(r, label=None) for r in salience_zoom_rois(df_yeo_surf, hemisphere)]
    outline = mask_outline(surf_lh, surf_rh, df_yeo_surf['network'].eq('SalVentAttn').to_numpy())
    geometry = set()
    for label in stats['modality']:
        values = np.full(len(df_yeo_surf), np.nan)
        vals = df_yeo_surf.loc[net_mask, label].to_numpy(dtype=float)
        values[net_mask] = (vals - np.nanmean(vals)) / np.nanstd(vals)  # z within the SN
        fig = plot_hemispheres_zoom(surf_lh, surf_rh, values, rois, roi_only=True, draw_color_bar=True,
                                    title=CONTEXT_TITLES.get(label, label), cmap=histology_cmap.name,
                                    color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS, cbar_title='z',
                                    outline=outline, filename=out_dir / f"figure_1b_zoom_{ZOOM_STEMS[label]}.svg",
                                    **ZOOM_KW)
        geometry.add((fig.inset_cy, fig.get_size_inches()[1] * 72))
        plt.close(fig)
    if len(geometry) != 1:
        raise RuntimeError(f"Zoom rows differ in geometry {geometry}; the forest rows cannot be aligned.")
    plot_forest(stats, *geometry.pop(), out_dir / "figure_1b_forest.svg")
    plt.close('all')


def main():
    # Setup Relative Paths
    parser = get_parser()
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    # 1000 surrogates matches figures 2 and 3 and keeps the add-one p floor at
    # 1/(1+1000) ~ 1e-3 for stable spatial-null p-values.
    n_rep = 1000

    logger = setup_manuscript_logger("figure_1b_contextualisation", project_root, args)
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Network        : SalVentAttn (Salience/Ventral Attention)")
    logger.info(f"Modalities     : BigBrain (cell staining), Bielschowsky (myelin), Parvalbumin (AHEAD); T1map (in-vivo MRI) brain map only")
    logger.info(f"Null model     : Moran randomization (n_rep={n_rep}, procedure=singleton, random_state=0), on the finite SN vertices of each map")

    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    # load atlases
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)

    ######### Part 1 -- Load gradient and T1map from figure_1a cache
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(f"Gradient dataframe not found at {path_df_1a}. Run figure_1a_t1map.py with -hemi {args.hemi} first.")
    logger.info(f"Loading gradient dataframe from {path_df_1a}")
    df_yeo_surf = pd.read_csv(path_df_1a, sep="\t")

    ######### Part 2 -- Contextualisation
    # Use 'both' to match original behavior: old loaders had no hemisphere filter
    net_mask = compute_network_mask(df_yeo_surf, 'SalVentAttn', 'both')
    df_yeo_surf.loc[net_mask, 'BigBrain'] = load_bigbrain(project_root, net_mask)
    screenshot_path = project_root / "results/figures/figure_1b_brain_bigbrain.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['BigBrain'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)
    screenshot_path = project_root / "results/figures/figure_1b_brain_t1map.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['T1map'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)
    df_yeo_surf.loc[net_mask, 'Bielschowsky'] = load_ahead_biel(project_root, net_mask)
    screenshot_path = project_root / "results/figures/figure_1b_brain_biel.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['Bielschowsky'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)
    df_yeo_surf.loc[net_mask, 'Parvalbumin'] = load_ahead_parva(project_root, net_mask)
    screenshot_path = project_root / "results/figures/figure_1b_brain_parva.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['Parvalbumin'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)

    stats = context_analysis(df_yeo_surf, surf_32k, modalities=HISTOLOGY_MAPS, n_rep=n_rep, hemisphere=args.hemi, project_root=project_root)

    ######### Part 3 -- Figure 1b context rows (zoom insets per map) + forest
    logger.info("Generating context zoom rows (figure_1b_zoom_*.svg) and forest (figure_1b_forest.svg)")
    plot_context_zoom(df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, stats, project_root / "results/figures", hemisphere=args.hemi)


if __name__ == "__main__":
    main()
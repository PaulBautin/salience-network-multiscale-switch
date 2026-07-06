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


def context_analysis(df_yeo_surf: pd.DataFrame, surf_32k, modalities: list[str], n_rep: int = 10, hemisphere: str = 'both', project_root: Path | None = None) -> None:
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
    title_txt = {
        'BigBrain': 'BigBrain – Merker staining (neuronal cell-body density)',
        'T1map': 'MICA-PNI – qT1 MRI (intracortical myelin content)',
        'Bielschowsky': 'AHEAD – Bielschowsky staining (axonal fiber density)',
        'Parvalbumin': 'AHEAD – Parvalbumin staining (PV+ interneurons)'
    }
    for row, label in enumerate(modalities):
        ax = axes[row, 0]
        ax_stem = axes[row, 1]
        y = df_yeo_surf.loc[net_mask, label].values
        # Score the correlation only where both the gradient and the modality are
        # finite, and build the Moran spatial graph on that same finite subset so the
        # surrogate field carries only real values. Filling NaNs with 0 and randomising
        # the full masked vector (the previous approach) injected artificial zeros into
        # the spatial-autocorrelation structure and biased the surrogates even at the
        # retained vertices; this mirrors the finite-subset null in figure_3_ieeg_mica.
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
        logger.info(f"[Figure 1B] {label}: MPC-gradient vs {label} | Spearman r={r_obs:.3f}, Moran permutation p={pv_rand:.3e} (n_perm={n_rep})")
        stats_text = f"$r={r_obs:.2f}$\n$p={pv_rand:.3f}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
        t = ax.set_title(f"{title_txt.get(label, label)}", loc='left', pad=15)
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
    plt.savefig(project_root / "results/figures/figure_1b_correlations.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)


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
    logger.info(f"Modalities     : BigBrain (cell staining), T1map (in-vivo MRI), Bielschowsky (myelin), Parvalbumin (AHEAD)")
    logger.info(f"Null model     : Moran randomization (n_rep={n_rep}, procedure=singleton, random_state=0)")

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

    context_analysis(df_yeo_surf, surf_32k, modalities=["BigBrain", "T1map", "Bielschowsky", "Parvalbumin"], n_rep=n_rep, hemisphere=args.hemi, project_root=project_root)


if __name__ == "__main__":
    main()
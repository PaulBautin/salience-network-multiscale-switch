from __future__ import division

# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Contextualisation of local microstructural heterogeneity of the salience network
# using BigBrain and Ahead datasets
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1b_contextualisation.py \
#   -hemi LH
# (requires figure_1a_t1map.py to have been run first)
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

from brainspace.null_models import SpinPermutations, moran
from scipy.stats import spearmanr, zscore

import logging

from src.atlas_load import load_yeo_atlas, load_bigbrain, load_ahead_biel, load_ahead_parva
from src.logging_utils import setup_manuscript_logger


plt.rcParams['font.size'] = 12
plt.rcParams['svg.fonttype'] = 'none'

def get_parser():
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


def context_analysis(df_yeo_surf, surf_32k, modalities, n_rep=10, hemisphere='both', project_root=None):
    ## Correlation analyses
    net_mask = df_yeo_surf['network'].eq('SalVentAttn')
    if hemisphere in ('LH', 'RH'):
        net_mask = net_mask & df_yeo_surf['hemisphere'].eq(hemisphere)
    x = zscore(df_yeo_surf.loc[net_mask, 't1_gradient1_SalVentAttn'].values)
    # Moran spatial autocorrelation model
    w = mesh_elements.get_ring_distance(surf_32k, n_ring=1, mask=net_mask.values)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_rep, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)

    # Plot 
    fig, axes = plt.subplots(len(modalities), 1, figsize=(3, 2.5 * len(modalities)), sharex=True, sharey=True)
    for ax, label in zip(axes, modalities):
        y = df_yeo_surf.loc[net_mask, label].values
        y = np.nan_to_num(y)
        rand = msr.randomize(y)
        sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 10, "alpha": 0.3, "edgecolors":'none', 'rasterized':True}, line_kws={"color": "black", "lw":2.5})
        r_obs, p = spearmanr(x, y, nan_policy='omit')
        r_rand = np.asarray([spearmanr(x, d, nan_policy='omit')[0] for d in rand])
        pv_rand = np.mean(np.abs(r_rand) >= np.abs(r_obs))
        logging.info(f"[Figure 1B] {label}: MPC-gradient vs {label} | Spearman r={r_obs:.3f}, Moran permutation p={pv_rand:.3e} (n_perm={n_rep})")
        stats_text = f"$r={r_obs:.2f}$\n$p_{{perm}}={pv_rand:.2e}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                     va='top', fontweight='bold')
        ax.set_ylim([-4,4])
        ax.set_xlim([-3,3])
        ax.set_yticks([-2, 2])
    ax.set_xlabel('MPC gradient')
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_1b_correlations.svg")

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
    theta = np.linspace(-np.pi /2 + np.pi/N*0.8, np.pi /2, N, endpoint=False)
    radii = np.abs(r_vals)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 8))
    bars = ax.bar(theta, radii, width=np.pi/N*0.8, align="center", alpha=0.8)
    ax.set_thetamax(90)
    ax.set_thetamin(-90)
    ax.set_rticks([0.0, 0.1, 0.2, 0.3])

    # Color by sign
    for bar, r in zip(bars, r_vals):
        bar.set_facecolor("tab:red" if r < 0 else "tab:blue")
    plt.grid(axis='x')
    ax.set_xticklabels([])
    ax.set_ylabel("Spearman's |r|")
    #ax.set_yticklabels([])
    plt.savefig(project_root / "results/figures/figure_1b_correlations_circle.svg")


def main():
    # Setup Relative Paths
    parser = get_parser()
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_1b_contextualisation", project_root, args)
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Network        : SalVentAttn (Salience/Ventral Attention)")
    logger.info(f"Modalities     : T1map (in-vivo MRI), BigBrain (cell staining), Bielschowsky (myelin), Parvalbumin (AHEAD)")
    logger.info(f"Null model     : Moran randomization (n_rep=100, procedure=singleton, random_state=0)")

    logging.info(f"Script path: {script_path}")
    logging.info(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    # load atlases
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)

    ######### Part 1 -- Load gradient and T1map from figure_1a cache
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(
            f"Gradient cache not found: {path_df_1a}\n"
            "Run figure_1a_t1map.py first to generate it."
        )
    logging.info(f"Loading gradient from figure_1a cache: {path_df_1a}")
    df_yeo_surf = pd.read_csv(path_df_1a)

    ######### Part 2 -- Contextualisation
    screenshot_path = project_root / "results/figures/figure_1b_brain_t1map.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['T1map'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path)
    df_yeo_surf = load_bigbrain(project_root, df_yeo_surf)
    screenshot_path = project_root / "results/figures/figure_1b_brain_bigbrain.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['BigBrain'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path)
    df_yeo_surf = load_ahead_biel(project_root, df_yeo_surf)
    screenshot_path = project_root / "results/figures/figure_1b_brain_biel.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['Bielschowsky'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path)
    df_yeo_surf = load_ahead_parva(project_root, df_yeo_surf)
    screenshot_path = project_root / "results/figures/figure_1b_brain_parva.svg"
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['Parvalbumin'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', transparent_bg=True, screenshot=True, filename=screenshot_path)

    context_analysis(df_yeo_surf, surf_32k, modalities=["BigBrain", "T1map", "Bielschowsky", "Parvalbumin"], n_rep=100, hemisphere=args.hemi, project_root=project_root)


if __name__ == "__main__":
    main()
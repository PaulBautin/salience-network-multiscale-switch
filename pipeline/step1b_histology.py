#!/usr/bin/env python
"""Step 1b: histological context of the salience-network gradient (Figure 1b).

Correlates the step 1a MPC gradient with three post-mortem histological maps sampled on
fsLR-32k (BigBrain Merker staining; AHEAD Bielschowsky and Parvalbumin staining) over the
network vertices, each against a Moran spectral randomization null fitted per hemisphere.

Inputs   source_data/step1a_mpc_gradient/ and the histological maps bundled in data/
         (all public, so this step needs no private data).
Public   source_data/step1b_histology/: the histological maps of the network vertices,
         the statistics, the Moran-null correlations, manifest.
Figures  results/figures/figure_1b_*.svg

Run      python pipeline/step1b_histology.py [--stage compute|plot|all]
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainspace.datasets import load_conte69
from brainspace.mesh import mesh_elements
from brainspace.plotting import plot_hemispheres
from scipy.stats import spearmanr, zscore

import common
from salience_switch import config
from salience_switch.colors import histology_cmap
from salience_switch.plotting import (SALIENCE_Z_LABELS, SALIENCE_Z_RANGE, format_pvalue, mask_outline,
                                      plot_hemispheres_zoom, plot_ridge, salience_zoom_rois)
from salience_switch.provenance import write_manifest
from salience_switch.stats import HemisphereMoranRandomization, moran_spearman
from salience_switch.surface import (compute_network_mask, load_ahead_biel, load_ahead_parva, load_bigbrain,
                                     load_inflated_surfaces)

STEP = "step1b_histology"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'

# Moran surrogates per test: 1000, as in figures 2 and 3, keeps the add-one p floor at
# 1/(1+1000) ~ 1e-3.
N_PERMUTATIONS = 1000

# Histological maps contextualising the gradient (stats, ridgeline and zoom rows, in row
# order), their panel titles and output-file stems.
HISTOLOGY_MAPS = ["BigBrain", "Bielschowsky", "Parvalbumin"]
HISTOLOGY_LOADERS = {"BigBrain": load_bigbrain, "Bielschowsky": load_ahead_biel, "Parvalbumin": load_ahead_parva}
CONTEXT_TITLES = {
    'BigBrain': 'BigBrain – Merker staining (neuronal cell-body density)',
    'T1map': 'MICA-PNI – qT1 MRI (intracortical myelin content)',
    'Bielschowsky': 'AHEAD – Bielschowsky staining (axonal fiber density)',
    'Parvalbumin': 'AHEAD – Parvalbumin staining (PV+ interneurons)',
}
ZOOM_STEMS = {'BigBrain': 'bigbrain', 'Bielschowsky': 'biel', 'Parvalbumin': 'parva'}
BRAIN_STEMS = {'BigBrain': 'bigbrain', 'T1map': 't1map', 'Bielschowsky': 'biel', 'Parvalbumin': 'parva'}
# Figure 1 zoom-row window (as figure_1a_brain_zoom*.svg), inset-row titles at 36 pt:
# the 1450-pt rows are shown ZOOM_ROW_WIDTH_MM wide in the Figure 1 layout (~0.29x).
ZOOM_KW = dict(size=(1450, 300), zoom=1.3, color_bar=True, nan_color=(220, 220, 220, 1),
               label_fontsize=30, title_fontsize=36)


def context_statistics(df_yeo_surf: pd.DataFrame, surf_32k, modalities: list[str], n_rep: int = N_PERMUTATIONS,
                       hemisphere: str = 'both') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation of each map with the SN MPC gradient against a Moran null.

    Each map is correlated with the gradient over the SN vertices (of ``hemisphere``)
    where both are finite, and the within-network Moran null is fitted on that same
    finite subset, separately within each hemisphere (``HemisphereMoranRandomization``;
    seeds 0 and 1; add-one two-sided p). The histological map is randomised and the
    gradient held fixed.

    Returns
    -------
    stats : pd.DataFrame
        One row per map: ``modality, r, p, null_lo, null_med, null_hi, n_vertices, n_perm``
        (``null_*``: 2.5th, 50th and 97.5th percentiles of the null correlations).
    null : pd.DataFrame
        One row per map and surrogate: ``modality, surrogate, r``.
    """
    net_mask = df_yeo_surf['network'].eq('SalVentAttn')
    if hemisphere in ('LH', 'RH'):
        net_mask = net_mask & df_yeo_surf['hemisphere'].eq(hemisphere)
    x = zscore(df_yeo_surf.loc[net_mask, 't1_gradient1_SalVentAttn'].values)
    # Full-surface indices of the network vertices, so a per-modality finite subset can
    # be mapped back to a whole-surface mask for the Moran graph (see the loop below).
    net_idx = np.flatnonzero(net_mask.values)

    stats_rows, null_rows = [], []
    for label in modalities:
        y = df_yeo_surf.loc[net_mask, label].values
        # Score the correlation only where both the gradient and the modality are
        # finite, and build the Moran spatial graph on that same finite subset so the
        # surrogate field carries only real values (no NaN/zero fill, which would inject
        # artificial values into the spatial-autocorrelation structure); this mirrors
        # the finite-subset null of step 2b.
        # The null is fitted and drawn separately within each hemisphere (seeds 0, 1),
        # since no spatial relation is defined between them, and the surrogates are
        # reassembled into the bilateral vector (one block for a single-hemisphere run).
        finite = np.isfinite(x) & np.isfinite(y)
        finite_mask = np.zeros(df_yeo_surf.shape[0], dtype=bool)
        finite_mask[net_idx[finite]] = True
        w = mesh_elements.get_ring_distance(surf_32k, n_ring=1, mask=finite_mask)
        w.data **= -1
        hemi_finite = df_yeo_surf['hemisphere'].to_numpy()[net_idx[finite]]
        msr = HemisphereMoranRandomization(hemi_finite, n_rep, random_state=0).fit(w)
        r_obs, pv_rand, r_rand = moran_spearman(x[finite], y[finite], msr)
        null_lo, null_med, null_hi = np.percentile(r_rand, [2.5, 50, 97.5])
        stats_rows.append(dict(modality=label, r=r_obs, p=pv_rand, null_lo=null_lo, null_med=null_med,
                               null_hi=null_hi, n_vertices=int(finite.sum()), n_perm=n_rep))
        null_rows.append(pd.DataFrame({'modality': label, 'surrogate': np.arange(r_rand.size), 'r': r_rand}))
        logger.info(f"[Figure 1B] {label}: MPC-gradient vs {label} | Spearman r={r_obs:.3f}, Moran permutation p={pv_rand:.3e} "
                    f"(n_perm={n_rep}, n_vertices={int(finite.sum())}; null r 95% [{null_lo:+.3f}, {null_hi:+.3f}])")
    return pd.DataFrame(stats_rows), pd.concat(null_rows, ignore_index=True)


def plot_context_correlations(df_yeo_surf: pd.DataFrame, stats: pd.DataFrame, modalities: list[str],
                              out_dir: Path, hemisphere: str = 'both') -> None:
    """Supplementary vertex scatter + |r| stem figure, and the half-polar |r| plot.

    The scatter shows each map against the z-scored gradient over the finite network
    vertices with a regression line (95 % bootstrap band, seed 0), annotated with the
    Moran-test r and p of ``stats``.
    """
    net_mask = df_yeo_surf['network'].eq('SalVentAttn')
    if hemisphere in ('LH', 'RH'):
        net_mask = net_mask & df_yeo_surf['hemisphere'].eq(hemisphere)
    x = zscore(df_yeo_surf.loc[net_mask, 't1_gradient1_SalVentAttn'].values)

    fig, axes = plt.subplots(
        len(modalities),
        2,
        figsize=(6.0, 2.8 * len(modalities)),
        gridspec_kw={'wspace': 0.1, 'hspace': 0.4, 'width_ratios': [1.0, 1.0]},
    )
    axes = np.atleast_2d(axes)
    for row, label in enumerate(modalities):
        ax = axes[row, 0]
        ax_stem = axes[row, 1]
        y = df_yeo_surf.loc[net_mask, label].values
        finite = np.isfinite(x) & np.isfinite(y)
        test = stats.set_index('modality').loc[label]
        r_obs, pv_rand = test['r'], test['p']
        sns.regplot(x=x[finite], y=y[finite], ax=ax, seed=0,
                    scatter_kws={"s": 20, "alpha": 0.1, "edgecolors": 'none', 'rasterized': True},
                    line_kws={"color": "black", "lw": 2.5})
        stats_text = f"$r={r_obs:.2f}$\n$p={format_pvalue(pv_rand)}$"
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
        t = ax.set_title(f"{CONTEXT_TITLES.get(label, label)}", loc='left', pad=15)
        t.set_in_layout(False)
        ax.set_ylim([-4, 4])
        ax.set_xlim([-3, 3])
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
    plt.savefig(out_dir / "figure_1b_correlations_supp.svg", transparent=True, bbox_inches="tight")
    plt.close(fig)

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
    theta = np.linspace(-np.pi / 2 + np.pi / N * 0.8, np.pi / 2, N, endpoint=False)[::-1]
    radii = np.abs(r_vals)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 8))
    bars = ax.bar(theta, radii, width=np.pi / N * 0.8, align="center", alpha=0.8)
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
    plt.savefig(out_dir / "figure_1b_correlations_circle.svg")
    plt.close(fig)


def plot_context_zoom(df_yeo_surf: pd.DataFrame, surf_lh, surf_rh, stats: pd.DataFrame, null: pd.DataFrame,
                      out_dir: Path, hemisphere: str = 'both') -> None:
    """Figure 1b context rows: one zoom row per histological map, plus the ridgeline.

    Each map in ``stats`` (in its row order) is z-scored over the SN vertices of
    ``hemisphere`` and drawn as the ``SALIENCE_ZOOM_ROIS`` inset row alone (no brains,
    no region names), titled with the map name, on the Figure 1 window (``ZOOM_KW``):
    ``CustomCmap_histology`` on the shared +/-2 SD scale, the SN border traced
    (``mask_outline``). ``figure_1b_forest.svg`` (``plot_ridge``) then puts one row
    level with each inset row when the rows are stacked in that order: the map's Moran
    null distribution (``null``) with its observed r.
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
        raise RuntimeError(f"Zoom rows differ in geometry {geometry}; the ridgeline rows cannot be aligned.")
    nulls = [null.loc[null['modality'] == m, 'r'].to_numpy() for m in stats['modality']]
    plot_ridge(stats, nulls, *geometry.pop(), out_dir / "figure_1b_forest.svg")
    plt.close('all')


def _surface_with_maps(hemi: str) -> pd.DataFrame:
    """The step 1a surface table with the histological maps of step 1b joined on."""
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / f"surface_fslr32k_{hemi}.tsv")
    maps = config.read_table(config.source_dir(STEP) / f"histology_maps_{hemi}.tsv").set_index("vertex")
    for label in HISTOLOGY_MAPS:
        df_yeo_surf[label] = maps[label].reindex(df_yeo_surf.index).to_numpy()
    return df_yeo_surf


def compute(settings: dict, paths: dict | None) -> None:
    """Histological maps of the network vertices and their Moran tests against the gradient."""
    hemi = settings["hemisphere"]
    source = config.source_dir(STEP)
    logger.info("Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info("Network        : SalVentAttn (Salience/Ventral Attention)")
    logger.info("Modalities     : BigBrain (cell staining), Bielschowsky (myelin), Parvalbumin (AHEAD); T1map (in-vivo MRI) brain map only")
    logger.info(f"Null model     : Moran randomization (n_rep={N_PERMUTATIONS}, procedure=singleton, inverse 1-ring distance), on the finite SN vertices of each map, fitted per hemisphere (random_state=0 for LH, or the only hemisphere, and 1 for RH)")

    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / f"surface_fslr32k_{hemi}.tsv")
    # The maps are loaded on the network vertices of both hemispheres, whatever `hemi`
    net_mask = compute_network_mask(df_yeo_surf, 'SalVentAttn', 'both')
    maps = pd.DataFrame({"vertex": np.flatnonzero(net_mask)})
    for label in HISTOLOGY_MAPS:
        df_yeo_surf.loc[net_mask, label] = HISTOLOGY_LOADERS[label](net_mask)
        # float64 so the text table holds the loaders' float32 values exactly
        maps[label] = df_yeo_surf.loc[net_mask, label].to_numpy(dtype=float)
    config.write_table(maps, source / f"histology_maps_{hemi}.tsv")

    stats, null = context_statistics(df_yeo_surf, load_conte69(join=True), HISTOLOGY_MAPS, hemisphere=hemi)
    config.write_table(stats, source / f"context_stats_{hemi}.tsv")
    config.write_table(null, source / f"context_null_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, hemisphere=hemi,
                   n_permutations=N_PERMUTATIONS, maps=HISTOLOGY_MAPS)


def plot(settings: dict) -> None:
    """Figure 1b panels and supplementary plots, from source_data only."""
    hemi = settings["hemisphere"]
    source, out = config.source_dir(STEP), config.FIGURES_DIR
    df_yeo_surf = _surface_with_maps(hemi)
    stats = config.read_table(source / f"context_stats_{hemi}.tsv")
    null = config.read_table(source / f"context_null_{hemi}.tsv")
    surf32k_lh_infl, surf32k_rh_infl = load_inflated_surfaces("fsLR-32k")

    for label, stem in BRAIN_STEMS.items():
        plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf[label].values, size=(1450, 300),
                         zoom=1.3, color_bar='right', share='both', nan_color=(220, 220, 220, 1), cmap='coolwarm',
                         transparent_bg=True, screenshot=True, filename=out / f"figure_1b_brain_{stem}.svg",
                         cb__numberOfLabels=0)
    plot_context_correlations(df_yeo_surf, stats, HISTOLOGY_MAPS, out, hemisphere=hemi)
    logger.info("Generating context zoom rows (figure_1b_zoom_*.svg) and Moran-null ridgeline (figure_1b_forest.svg)")
    plot_context_zoom(df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, stats, null, out, hemisphere=hemi)


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 1b: histological context of the salience-network gradient (Figure 1b).",
                private=False)

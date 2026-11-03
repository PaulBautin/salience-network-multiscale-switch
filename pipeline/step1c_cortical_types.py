#!/usr/bin/env python
"""Step 1c: cortical types of the salience network (Figure 1c).

Overlays the von Economo-Koskinas cortical types (García-Cabezas et al., 2020) on the Yeo
networks and tests, for the salience network, whether each type is over- or
under-represented relative to a spin-permutation null of the network, following the
procedure of Paquola et al.'s analysis of the default-mode network (network labels rotated
on the sphere, type composition recomputed within the rotated network, 10,000 rotations);
rotated vertices on the medial wall, which has no cortical type, are left out.

Inputs   atlases bundled in data/ (public, so this step needs no private data).
Public   source_data/step1c_cortical_types/: the type composition of every network, the
         spin-null compositions and the enrichment statistics of the salience network,
         manifest.
Figures  results/figures/figure_1c_*.svg

Run      python pipeline/step1c_cortical_types.py [--stage compute|plot|all]
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainspace.datasets import load_conte69
from brainspace.null_models import SpinPermutations
from brainspace.plotting import plot_hemispheres

import common
from salience_switch import config
from salience_switch.colors import cmap_types, cmap_types_mw
from salience_switch.plotting import ZOOM_ROW_KW, mask_outline, plot_hemispheres_zoom, salience_zoom_rois
from salience_switch.provenance import write_manifest
from salience_switch.stats import empirical_p_tails
from salience_switch.surface import compute_network_mask, load_econo_atlas, load_inflated_surfaces, load_yeo_atlas

STEP = "step1c_cortical_types"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16

# Rotations of the spin test: 10,000, as in Paquola et al.'s cortical-type analysis of the
# default-mode network (caseypaquola/DMN, scripts/cytoarchitecture.m).
N_SPINS = 10000
# Cortical types, codes 1-6 of load_econo_atlas, then 'Other' (code 0: no type) as code 7.
TYPE_LABELS = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']


def _type_percent(types: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentage of the vertices of `mask` in each type (codes 1-7, `TYPE_LABELS` order)."""
    counts = np.bincount(types[mask], minlength=len(TYPE_LABELS) + 1)[1:]
    return counts / counts.sum() * 100


def type_composition(df_yeo_surf: pd.DataFrame, network: str, hemisphere: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cortical type composition of every network, and the spin-permutation null of `network`.

    The composition of a network is the percentage of its vertices (of ``hemisphere``)
    in each type, vertices without a type counted as 'Other'. The null follows the spin
    test of Paquola et al.'s default-mode network analysis: the network mask is rotated
    on the conte69 sphere (``SpinPermutations``, ``N_SPINS`` rotations, seed 0; the
    rotation of the right sphere mirrors that of the left), and the composition is
    recomputed over the rotated network. Rotated vertices that land on the medial wall,
    which has no cortical type, are left out of that rotation's composition, so the null,
    like the network, covers cortex only.

    Returns
    -------
    composition : pd.DataFrame
        One row per network and type: ``network, type, percent``.
    null : pd.DataFrame
        One row per rotation and type of `network`: ``network, spin, type, percent``.
    """
    types = df_yeo_surf['surf_type'].to_numpy().astype(int)
    types[types == 0] = len(TYPE_LABELS)  # no type -> 'Other'
    cortex = df_yeo_surf['network'].notna().to_numpy() & df_yeo_surf['network'].ne('medial_wall').to_numpy()

    composition = []
    for net in sorted(set(df_yeo_surf.loc[cortex, 'network'])):
        percent = _type_percent(types, compute_network_mask(df_yeo_surf, net, hemisphere))
        composition.append(pd.DataFrame({'network': net, 'type': TYPE_LABELS, 'percent': percent}))
        logger.info(f"[Figure 1C] {net}: cortical type composition | "
                    + ", ".join(f"{t}={v:.1f}%" for t, v in zip(TYPE_LABELS, percent)))

    sphere_lh, sphere_rh = load_conte69(as_sphere=True)
    sp = SpinPermutations(n_rep=N_SPINS, random_state=0)
    sp.fit(sphere_lh, points_rh=sphere_rh)
    mask = compute_network_mask(df_yeo_surf, network, hemisphere)
    n_lh = sphere_lh.n_points
    rotated = np.hstack(sp.randomize(mask[:n_lh], mask[n_lh:]))  # (N_SPINS, n_vertices) bool
    on_wall = (rotated & ~cortex).sum(axis=1) / rotated.sum(axis=1)
    logger.info(f"[Figure 1C] spin test of {network}: {N_SPINS} rotations, seed 0; rotated vertices on the "
                f"medial wall left out (mean {on_wall.mean():.1%}, max {on_wall.max():.1%} of the rotated network)")
    percent = np.stack([_type_percent(types, r & cortex) for r in rotated])
    null = pd.DataFrame({'network': network, 'spin': np.repeat(np.arange(N_SPINS), len(TYPE_LABELS)),
                         'type': np.tile(TYPE_LABELS, N_SPINS), 'percent': percent.ravel()})
    return pd.concat(composition, ignore_index=True), null


def enrichment_statistics(composition: pd.DataFrame, null: pd.DataFrame, network: str) -> pd.DataFrame:
    """Spin-permutation test of the depletion and enrichment of each cortical type in `network`.

    Per type, the observed percentage is compared with the spin-null percentages by the
    two one-sided add-one p-values (`empirical_p_tails`): depletion,
    ``(1 + #{null <= observed}) / (1 + n)``, and enrichment,
    ``(1 + #{null >= observed}) / (1 + n)`` (Paquola et al. tested each direction one-sided,
    without the add-one: their script computes the lower tail and the over-representation
    they report is its complement, the upper tail); the test p is the equal-tailed
    two-sided ``min(1, 2 min(depletion, enrichment))``, which does not assume a symmetric
    null (the percentages of rare types are bounded at 0). The z-score is the deviation
    from the null mean in null standard deviations (descriptive).

    Returns
    -------
    stats : pd.DataFrame
        One row per type: ``network, type, observed, null_mean, null_sd, z, p_depletion,
        p_enrichment, p, n_spins``.
    """
    rows = []
    for lbl in TYPE_LABELS:
        obs = composition.query("network == @network and type == @lbl")['percent'].item()
        null_vals = null.query("network == @network and type == @lbl").sort_values('spin')['percent'].to_numpy()
        null_mean, null_sd = null_vals.mean(), null_vals.std(ddof=1)
        p_depletion, p_enrichment, p = empirical_p_tails(null_vals, obs)
        z = (obs - null_mean) / null_sd if null_sd > 0 else np.nan
        logger.info(f"[Figure 1C] {network} {lbl}: observed={obs:.1f}%, spin null mean={null_mean:.1f}% "
                    f"± {null_sd:.1f}%, z={z:.2f}, p_depletion={p_depletion:.3e}, p_enrichment={p_enrichment:.3e}, "
                    f"two-sided p={p:.3e}")
        rows.append(dict(network=network, type=lbl, observed=obs, null_mean=null_mean, null_sd=null_sd, z=z,
                         p_depletion=p_depletion, p_enrichment=p_enrichment, p=p, n_spins=null_vals.size))
    return pd.DataFrame(rows)


def plot_type_salience(composition: pd.DataFrame, null: pd.DataFrame, network: str, path) -> None:
    """The network's type composition (dots) over the spin-null compositions (bars).

    The bars show the mean spin-null percentage with its 95 % bootstrap interval (seed 0).
    The six cortical types only: 'Other' (unassigned vertices) is left out of this panel
    (it stays in the composition and enrichment statistics).
    """
    shown_types = [t for t in TYPE_LABELS if t != 'Other']
    df = (null.query("network == @network").pivot(index='spin', columns='type', values='percent')[shown_types]
          .rename_axis(index=None, columns=None))  # unnamed axes: no axis label from seaborn
    observed = composition.query("network == @network").set_index('type')['percent']
    # Font: the 6-in panel is shown at ~0.41x in the Figure 1 layout, so 21 pt reads like
    # the ~8.8-pt text of the zoom rows (30 pt shown at ~0.29x), as in figure_1a_profiles.
    with plt.rc_context({'font.size': 21}):
        fig, ax_sal = plt.subplots(figsize=(6, 6))
        sns.barplot(data=df, ax=ax_sal, color='lightgrey', seed=0)
        sns.scatterplot(x=shown_types, y=observed[shown_types].to_list(),
                        color=cmap_types_mw.colors[:len(shown_types)], s=250, edgecolors='none', ax=ax_sal)
        ax_sal.set_ylim(0, 60)
        ax_sal.tick_params(axis='x', labelrotation=90)
        ax_sal.set_ylabel("Salience percentage (%)")
        ax_sal.spines['top'].set_visible(False)
        ax_sal.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)


def _surface_with_types(network: str) -> pd.DataFrame:
    """fsLR-32k surface table with the cortical type of each vertex and the border of `network`."""
    return load_econo_atlas(load_yeo_atlas(load_conte69(join=True), border_network=network))


def compute(settings: dict, paths: dict | None) -> None:
    """Type composition of every network, and the spin test of the network of the settings."""
    network, hemi = settings["network"], settings["hemisphere"]
    source = config.source_dir(STEP)
    logger.info("Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Network        : {network}; hemisphere: {hemi}")
    logger.info("Cortical types : von Economo-Koskinas atlas (7 types: Kon, Eu-III, Eu-II, Eu-I, Dys, Ag, Other)")
    composition, null = type_composition(_surface_with_types(network), network, hemi)
    stats = enrichment_statistics(composition, null, network)
    config.write_table(composition, source / f"type_composition_{hemi}.tsv")
    config.write_table(null, source / f"type_null_{network}_{hemi}.tsv")
    config.write_table(stats, source / f"type_stats_{network}_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network, hemisphere=hemi,
                   n_spins=N_SPINS)


def plot(settings: dict) -> None:
    """Figure 1c panels, from the bundled atlases and source_data."""
    network, hemi = settings["network"], settings["hemisphere"]
    source, out = config.source_dir(STEP), config.FIGURES_DIR
    df_yeo_surf = _surface_with_types(network)
    surf32k_lh_infl, surf32k_rh_infl = load_inflated_surfaces("fsLR-32k")

    plt_values = df_yeo_surf['surf_type'].values * df_yeo_surf['salience_border'].values
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3,
                     color_bar='right', share='both', nan_color=(0, 0, 0, 1), cmap=cmap_types, transparent_bg=True,
                     screenshot=True, filename=out / "figure_1c_brain_economo.svg", cb__numberOfLabels=0)

    # Same brain row with circular zoom insets of the salience regions, as Figure 1a:
    # figure_1c_brain_economo_zoom.svg overlays figure_1c_brain_economo.svg, and the
    # inset row alone (_roi) stacks 300 pt below it. The salience-network border is traced
    # as in Figures 1a and 2b (mask_outline: the vertices just outside the network, a
    # line over the cortical types) instead of the black NaN border vertices above. The
    # colour bar shows the categorical type colours as blocks without end labels
    # (min/max would be meaningless for types), as in figure_1c_brain_economo.svg; the
    # type bar plot names the colours. Labels at 30 pt: the 1450-pt rows are shown at
    # ~0.29x in the Figure 1 layout (~8.8 pt).
    rois = salience_zoom_rois(df_yeo_surf, hemi)
    sn_outline = mask_outline(surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf['network'].eq(network).to_numpy())
    zoom_values = df_yeo_surf['surf_type'].values
    zoom_kwargs = dict(ZOOM_ROW_KW, cmap=cmap_types.name, nan_color=(0, 0, 0, 1), outline=sn_outline)
    logger.info("Generating brain zoom-inset figures (figure_1c_brain_economo_zoom*.svg)")
    plot_hemispheres_zoom(surf32k_lh_infl, surf32k_rh_infl, zoom_values, rois, cbar_labels=('', ''),
                          filename=out / "figure_1c_brain_economo_zoom.svg", **zoom_kwargs)
    plot_hemispheres_zoom(surf32k_lh_infl, surf32k_rh_infl, zoom_values, rois, roi_only=True,
                          filename=out / "figure_1c_brain_economo_zoom_roi.svg", **zoom_kwargs)
    plt.close('all')

    plot_type_salience(config.read_table(source / f"type_composition_{hemi}.tsv"),
                       config.read_table(source / f"type_null_{network}_{hemi}.tsv"), network,
                       out / "figure_1c_type_salience.svg")


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 1c: cortical types of the salience network (Figure 1c).", private=False)

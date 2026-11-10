#!/usr/bin/env python
"""Step 1c: cortical types of the salience network (Figure 1c).

Overlays the von Economo-Koskinas cortical types (García-Cabezas et al., 2020) on the Yeo
networks and tests, for the salience network, whether each type is over- or
under-represented relative to a spin-permutation null of the network, following the
procedure of Paquola et al.'s analysis of the default-mode network (network labels rotated
on the sphere, type composition recomputed within the rotated network, 10,000 rotations);
rotated vertices on the medial wall, which has no cortical type, are left out.

Relates the two Figure 1 descriptions of the network, the step 1a MPC gradient and the
cortical types, by the Spearman correlation of the gradient with the laminar
differentiation rank of each vertex's type (agranular 1 to koniocortex 6) over the
network vertices, against a Moran spectral randomization null of the gradient fitted
per hemisphere, as in step 1b.

Inputs   source_data/step1a_mpc_gradient/ and the atlases bundled in data/ (all public,
         so this step needs no private data).
Public   source_data/step1c_cortical_types/: the type composition of every network, the
         spin-null compositions and the enrichment statistics of the salience network,
         the gradient-by-type statistics and their Moran-null correlations, manifest.
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
from salience_switch.plotting import (ZOOM_ROW_KW, format_pvalue, mask_outline, plot_hemispheres_zoom,
                                      salience_zoom_rois)
from salience_switch.provenance import write_manifest
from salience_switch.stats import empirical_p_tails, fit_surface_moran, moran_spearman, null_interval
from salience_switch.surface import compute_network_mask, load_econo_atlas, load_inflated_surfaces, load_yeo_atlas

STEP = "step1c_cortical_types"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16

# Rotations of the spin test: 10,000, as in Paquola et al.'s cortical-type analysis of the
# default-mode network (caseypaquola/DMN, scripts/cytoarchitecture.m).
N_SPINS = 10000
# Moran surrogates of the gradient-by-type test: 1000, as in step 1b, keeps the add-one
# p floor at 1/(1+1000) ~ 1e-3.
N_PERMUTATIONS = 1000
# Cortical types, codes 1-6 of load_econo_atlas, then 'Other' (code 0: no type) as code 7.
TYPE_LABELS = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
# Rank of each type on the ordinal scale of laminar differentiation (García-Cabezas et al.,
# 2020), from agranular (least differentiated) to koniocortex; 'Other' has no rank.
TYPE_RANK = {'Ag': 1, 'Dys': 2, 'Eu-I': 3, 'Eu-II': 4, 'Eu-III': 5, 'Kon': 6}


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


def _typed_gradient(df_surf: pd.DataFrame, network: str, hemisphere: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Network vertices (of ``hemisphere``) with a cortical type and a finite gradient.

    Returns
    -------
    vertices : (n,) int
        Their surface indices, in surface order.
    gradient : (n,) float
        Their step 1a MPC gradient.
    types : (n,) str
        Their cortical type (`TYPE_LABELS`, 'Other' excluded).
    """
    codes = df_surf['surf_type'].to_numpy()
    gradient = df_surf[f't1_gradient1_{network}'].to_numpy(dtype=float)
    keep = (compute_network_mask(df_surf, network, hemisphere) & np.isfinite(gradient)
            & (codes >= 1) & (codes <= len(TYPE_RANK)))
    vertices = np.flatnonzero(keep)
    return vertices, gradient[vertices], np.array(TYPE_LABELS)[codes[vertices].astype(int) - 1]


def gradient_by_type(df_surf: pd.DataFrame, surf_32k, network: str, hemisphere: str,
                     n_rep: int = N_PERMUTATIONS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation of the network MPC gradient with the laminar differentiation of the types.

    Each network vertex (of ``hemisphere``) is given the rank of its cortical type on the
    ordinal scale of laminar differentiation (`TYPE_RANK`: agranular 1 to koniocortex 6),
    vertices without a type being left out, and the rank is correlated with the step 1a
    gradient (Spearman, the sign as the two maps produce it). The gradient is randomised
    by the within-network Moran null fitted on those vertices, separately within each
    hemisphere (`fit_surface_moran`: inverse 1-ring edge lengths, seeds 0 and 1; add-one
    two-sided p), and the type rank is held fixed: Moran spectral randomisation draws
    continuous surrogates, which could not keep the six levels of the ordinal type map.

    Returns
    -------
    stats : pd.DataFrame
        One row: ``network, r, p, null_lo, null_med, null_hi, n_vertices, n_perm``
        (``null_*``: 2.5th, 50th and 97.5th percentiles of the null correlations).
    null : pd.DataFrame
        One row per surrogate: ``network, surrogate, r``.
    """
    vertices, gradient, types = _typed_gradient(df_surf, network, hemisphere)
    rank = np.array([TYPE_RANK[t] for t in types], dtype=float)
    for lbl in TYPE_RANK:
        g = gradient[types == lbl]
        q1, med, q3 = np.percentile(g, [25, 50, 75]) if g.size else (np.nan,) * 3
        logger.info(f"[Figure 1C] {network} {lbl} (rank {TYPE_RANK[lbl]}): {g.size} vertices, "
                    f"MPC gradient median={med:+.3f} [IQR {q1:+.3f}, {q3:+.3f}]")
    mask = np.zeros(len(df_surf), dtype=bool)
    mask[vertices] = True
    msr = fit_surface_moran(surf_32k, mask, df_surf['hemisphere'].to_numpy()[vertices], n_rep)
    r_obs, p, r_null = moran_spearman(rank, gradient, msr)
    null_lo, null_med, null_hi = null_interval(r_null)
    logger.info(f"[Figure 1C] {network}: MPC gradient vs cortical-type rank | Spearman r={r_obs:.3f}, Moran "
                f"permutation p={p:.3e} (n_perm={n_rep}, n_vertices={vertices.size}; "
                f"null r 95% [{null_lo:+.3f}, {null_hi:+.3f}])")
    stats = pd.DataFrame([dict(network=network, r=r_obs, p=p, null_lo=null_lo, null_med=null_med, null_hi=null_hi,
                               n_vertices=int(vertices.size), n_perm=n_rep)])
    null = pd.DataFrame({'network': network, 'surrogate': np.arange(r_null.size), 'r': r_null})
    return stats, null


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


def plot_gradient_by_type(df_surf: pd.DataFrame, stats: pd.DataFrame, network: str, hemisphere: str, path) -> None:
    """The MPC gradient of the network vertices of each cortical type, with the Moran test.

    One horizontal box per type (quartiles and median; whiskers at 1.5 IQR; outlying
    vertices not drawn) in the type colours, the types from koniocortex (top) to
    agranular (bottom), the order of `plot_type_salience` read from top to bottom, so the
    laminar differentiation rank increases upwards and the gradient to the right.
    Annotated with the Spearman r and Moran p of ``stats``.
    """
    shown_types = [t for t in TYPE_LABELS if t != 'Other']
    _, gradient, types = _typed_gradient(df_surf, network, hemisphere)
    test = stats.query("network == @network").iloc[0]
    # Same size and font as figure_1c_type_salience, its neighbour in the Figure 1 layout.
    with plt.rc_context({'font.size': 21}):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axvline(0, color='0.8', lw=1, zorder=0)
        sns.boxplot(x=gradient, y=types, order=shown_types, hue=types, hue_order=shown_types, legend=False,
                    palette=[tuple(c) for c in cmap_types_mw.colors[:len(shown_types)]], showfliers=False,
                    width=0.65, linecolor='#1a1a1a', linewidth=1.5, ax=ax)
        ax.text(0.97, 0.03, f"r = {test['r']:+.2f}".replace('-', '−') + f"\n$p_{{m}}$ = {format_pvalue(test['p'])}",
                transform=ax.transAxes, ha='right', va='bottom')
        ax.set_xlabel("MPC gradient")
        ax.set_ylabel("")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)


def _surface_with_types(network: str) -> pd.DataFrame:
    """fsLR-32k surface table with the cortical type of each vertex and the border of `network`."""
    return load_econo_atlas(load_yeo_atlas(load_conte69(join=True), border_network=network))


def _surface_with_gradient(hemi: str) -> pd.DataFrame:
    """The step 1a fsLR-32k surface table (with the network gradient) and the cortical type of each vertex."""
    return load_econo_atlas(config.read_table(config.source_dir(GRADIENT_STEP) / f"surface_fslr32k_{hemi}.tsv"))


def compute(settings: dict, paths: dict | None) -> None:
    """Type composition of every network, the spin test of the network of the settings, and
    the Moran test of its gradient against the cortical-type rank."""
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

    logger.info(f"Gradient by type: step 1a MPC gradient vs laminar differentiation rank ({TYPE_RANK}); Moran "
                f"randomization of the gradient (n_rep={N_PERMUTATIONS}, procedure=singleton, inverse 1-ring "
                f"distance), on the typed SN vertices, fitted per hemisphere (random_state=0 for LH, or the only "
                f"hemisphere, and 1 for RH)")
    g_stats, g_null = gradient_by_type(_surface_with_gradient(hemi), load_conte69(join=True), network, hemi)
    config.write_table(g_stats, source / f"type_gradient_stats_{network}_{hemi}.tsv")
    config.write_table(g_null, source / f"type_gradient_null_{network}_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network, hemisphere=hemi,
                   n_spins=N_SPINS, n_permutations=N_PERMUTATIONS)


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
    plot_gradient_by_type(_surface_with_gradient(hemi),
                          config.read_table(source / f"type_gradient_stats_{network}_{hemi}.tsv"), network, hemi,
                          out / "figure_1c_gradient_type.svg")


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 1c: cortical types of the salience network (Figure 1c).", private=False)

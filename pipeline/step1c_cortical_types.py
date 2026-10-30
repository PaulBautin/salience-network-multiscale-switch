#!/usr/bin/env python
"""Step 1c: cortical types of the salience network (Figure 1c).

Overlays the von Economo-Koskinas cortical types (García-Cabezas et al., 2020) on the Yeo
networks and tests, for the salience network, whether each type is over- or
under-represented relative to a spin-permutation null.

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
from salience_switch.plotting import mask_outline, plot_hemispheres_zoom, salience_zoom_rois
from salience_switch.provenance import write_manifest
from salience_switch.stats import empirical_p_twosided
from salience_switch.surface import convert_states_str2int, load_econo_atlas, load_inflated_surfaces, load_yeo_atlas

STEP = "step1c_cortical_types"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'

# Spins: 1000, as in figures 2 and 3, keeps the add-one p floor at ~1e-3
N_SPINS = 1000
TYPE_LABELS = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
LABEL_MAP = dict(zip(range(1, 8), TYPE_LABELS))
FOCUS_NETWORK = "SalVentAttn"


def type_composition(df_yeo_surf: pd.DataFrame, hemisphere: str = 'both') -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cortical type composition of each network and of its spin-permutation surrogates.

    The network masks are rotated on the conte69 sphere (``SpinPermutations``,
    ``N_SPINS`` rotations, seed 0), always fitted on the full surface so the spin indices
    match the sphere size (32,492 per hemisphere); the hemisphere selection is applied to
    the network mask. Vertices of type 0 (unassigned) are counted as 'Other'.

    Returns
    -------
    composition : pd.DataFrame
        One row per network and type: ``network, type, percent``.
    null : pd.DataFrame
        One row per network, spin and type: ``network, spin, type, percent``.
    """
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)
    sp = SpinPermutations(n_rep=N_SPINS, random_state=0)
    sp.fit(sphere32k_lh, points_rh=sphere32k_rh)

    df_yeo_surf = df_yeo_surf.copy()
    # surf_type == 0 means unassigned; map to the "Other" bin (index 7) so it
    # doesn't silently inflate zero-counts when building the distribution.
    df_yeo_surf.loc[df_yeo_surf.surf_type == 0, 'surf_type'] = 7
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)
    expected_types = np.arange(1, 8)

    composition, null = [], []
    for net_idx, net_name in enumerate(state_name):
        mask = (state == net_idx)
        if hemisphere in ('LH', 'RH'):
            mask = mask & (df_yeo_surf['hemisphere'] == hemisphere).values
        mask_lh, mask_rh = mask[:32492], mask[32492:]

        comp = df_yeo_surf.surf_type.values[mask] * mask[mask]
        observed_types, counts = np.unique(comp, return_counts=True)
        counts_dict = dict(zip(observed_types, counts))
        full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
        percentages = (full_counts / len(comp)) * 100
        composition.append(pd.DataFrame({'network': net_name, 'type': TYPE_LABELS, 'percent': percentages}))
        type_summary = ", ".join(f"{LABEL_MAP[t]}={percentages[t-1]:.1f}%" for t in expected_types)
        logger.info(f"[Figure 1C] {net_name}: cortical type composition | {type_summary}")

        # Spin-permutation null distribution
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        perc = np.empty((N_SPINS, len(expected_types)))
        for n in range(N_SPINS):
            comp = df_yeo_surf.surf_type.values[net_rot[n]] * net_rot[n][net_rot[n]]
            u, c = np.unique(comp, return_counts=True)
            counts_dict = dict(zip(u, c))
            full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
            perc[n] = (full_counts / len(comp)) * 100
        null.append(pd.DataFrame({'network': net_name, 'spin': np.repeat(np.arange(N_SPINS), len(TYPE_LABELS)),
                                  'type': np.tile(TYPE_LABELS, N_SPINS), 'percent': perc.ravel()}))
    logger.info(f"[Figure 1C] Spin permutations: n_rep={N_SPINS}, random_state=0")
    return pd.concat(composition, ignore_index=True), pd.concat(null, ignore_index=True)


def enrichment_statistics(composition: pd.DataFrame, null: pd.DataFrame, network: str = FOCUS_NETWORK) -> pd.DataFrame:
    """Spin-permutation enrichment test of each cortical type in `network`.

    Per type, the two-sided add-one empirical p compares the observed percentage with
    the spin-null percentages, both centred on the null mean so the test asks for a
    deviation in either direction; the z-score is the deviation in null standard
    deviations (descriptive).

    Returns
    -------
    stats : pd.DataFrame
        One row per type: ``network, type, observed, null_mean, null_sd, z, p, n_spins``.
    """
    rows = []
    for lbl in TYPE_LABELS:
        obs = composition.query("network == @network and type == @lbl")['percent'].item()
        null_vals = null.query("network == @network and type == @lbl").sort_values('spin')['percent'].to_numpy()
        null_mean, null_sd = null_vals.mean(), null_vals.std(ddof=1)
        p = empirical_p_twosided(null_vals - null_mean, obs - null_mean)
        z = (obs - null_mean) / (null_sd + 1e-12) if null_sd > 0 else np.nan
        logger.info(f"[Figure 1C] {network} {lbl}: observed={obs:.1f}%, spin null mean={null_mean:.1f}% "
                    f"± {null_sd:.1f}%, z={z:.2f}, spin enrichment p={p:.3e}")
        rows.append(dict(network=network, type=lbl, observed=obs, null_mean=null_mean, null_sd=null_sd,
                         z=z, p=p, n_spins=null_vals.size))
    return pd.DataFrame(rows)


def plot_type_salience(composition: pd.DataFrame, null: pd.DataFrame, path) -> None:
    """Salience-network type composition (dots) over the spin-null compositions (bars).

    The bars show the mean spin-null percentage with its 95 % bootstrap interval (seed 0).
    The six cortical types only: 'Other' (unassigned vertices) is left out of this panel
    (it stays in the composition and enrichment statistics).
    """
    shown_types = [t for t in TYPE_LABELS if t != 'Other']
    df = (null.query("network == @FOCUS_NETWORK").pivot(index='spin', columns='type', values='percent')[shown_types]
          .rename_axis(index=None, columns=None))  # unnamed axes: no axis label from seaborn
    observed = composition.query("network == @FOCUS_NETWORK").set_index('type')['percent']
    # Font: the 6-in panel is shown at ~0.41x in the Figure 1 layout, so 21 pt reads like
    # the ~8.8-pt text of the zoom rows (30 pt shown at ~0.29x), as in figure_1a_profiles.
    with plt.rc_context({'font.size': 21}):
        fig, ax_sal = plt.subplots(figsize=(6, 6))
        sns.barplot(data=df, ax=ax_sal, color='lightgrey', seed=0)
        sns.scatterplot(x=shown_types, y=observed[shown_types].to_list(),
                        color=cmap_types_mw.colors[:len(shown_types)], s=100, edgecolors='none', ax=ax_sal)
        ax_sal.set_ylim(0, 60)
        ax_sal.tick_params(axis='x', labelrotation=90)
        ax_sal.set_ylabel("Salience percentage (%)")
        ax_sal.spines['top'].set_visible(False)
        ax_sal.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)


def _surface_with_types() -> pd.DataFrame:
    """fsLR-32k surface table with the cortical type of each vertex."""
    return load_econo_atlas(load_yeo_atlas(load_conte69(join=True)))


def compute(settings: dict, paths: dict | None) -> None:
    """Type composition of every network, spin nulls and the salience-network statistics."""
    hemi = settings["hemisphere"]
    source = config.source_dir(STEP)
    logger.info("Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Hemisphere     : {hemi}")
    logger.info("Cortical types : von Economo-Koskinas atlas (7 types: Kon, Eu-III, Eu-II, Eu-I, Dys, Ag, Other)")
    logger.info(f"Null model     : SpinPermutations (n_rep={N_SPINS}, random_state=0)")
    composition, null = type_composition(_surface_with_types(), hemisphere=hemi)
    stats = enrichment_statistics(composition, null)
    config.write_table(composition, source / f"type_composition_{hemi}.tsv")
    config.write_table(null.query("network == @FOCUS_NETWORK"), source / f"type_null_{FOCUS_NETWORK}_{hemi}.tsv")
    config.write_table(stats, source / f"type_stats_{FOCUS_NETWORK}_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, hemisphere=hemi, n_spins=N_SPINS,
                   network=FOCUS_NETWORK)


def plot(settings: dict) -> None:
    """Figure 1c panels, from the bundled atlases and source_data."""
    hemi = settings["hemisphere"]
    source, out = config.source_dir(STEP), config.FIGURES_DIR
    df_yeo_surf = _surface_with_types()
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
    sn_outline = mask_outline(surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf['network'].eq('SalVentAttn').to_numpy())
    zoom_values = df_yeo_surf['surf_type'].values
    zoom_kwargs = dict(size=(1450, 300), zoom=1.3, color_bar=True, cmap=cmap_types.name,
                       nan_color=(0, 0, 0, 1), label_fontsize=30, outline=sn_outline)
    logger.info("Generating brain zoom-inset figures (figure_1c_brain_economo_zoom*.svg)")
    plot_hemispheres_zoom(surf32k_lh_infl, surf32k_rh_infl, zoom_values, rois, cbar_labels=('', ''),
                          filename=out / "figure_1c_brain_economo_zoom.svg", **zoom_kwargs)
    plot_hemispheres_zoom(surf32k_lh_infl, surf32k_rh_infl, zoom_values, rois, roi_only=True,
                          filename=out / "figure_1c_brain_economo_zoom_roi.svg", **zoom_kwargs)
    plt.close('all')

    plot_type_salience(config.read_table(source / f"type_composition_{hemi}.tsv"),
                       config.read_table(source / f"type_null_{FOCUS_NETWORK}_{hemi}.tsv"),
                       out / "figure_1c_type_salience.svg")


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 1c: cortical types of the salience network (Figure 1c).", private=False)

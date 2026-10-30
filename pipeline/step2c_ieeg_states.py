#!/usr/bin/env python
"""Step 2c: state dependence of the iEEG aperiodic exponent along the gradient (Figure 2c).

The same bipolar channels of the MICA iEEG dataset were recorded in wakefulness (stage W),
N2 and N3 sleep. The first 180 s of every recording are parameterised with specparam (as
step 2b) and the aperiodic exponent is projected onto the folded fsLR-32k template through
the bipolar sensitivity maps. The primary tests correlate the exponent of each state with
the MPC gradient against the within-network Moran null, and the correlations of every
pair of states are compared against paired Moran surrogates; the within-channel change
from wakefulness to N3 is tested as well (supplementary). p values are not corrected for
multiple comparisons here.

Inputs   electroMICA derivatives and the original recordings (config/paths.yaml); step 1a
         fsLR-32k gradient (source_data).
Private  <work_dir>/step2c_ieeg_states/: the channel spectra caches (patient IDs), log.
Public   source_data/step2c_ieeg_states/: the template maps of the exponent of each state
         and of its change, the band peak powers at the tested vertices, the statistics and
         Moran nulls, manifest.
Figures  results/figures/figure_2c_ieeg_states_*.svg. The specparam fit-quality figures
         (individual channels) are drawn by the compute stage.

Run      python pipeline/step2c_ieeg_states.py [--stage compute|plot|all]
"""
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from brainspace.datasets import load_conte69
from brainspace.mesh import mesh_elements
from scipy.stats import zscore

import common
from salience_switch import config
from salience_switch.colors import ieeg_cmap
from salience_switch.ieeg import (MICA_PSD_PARAMS, SPEC_BANDS, SPEC_FIT_RANGE, build_bipolar_sensitivity,
                                  compute_spectral_parameters, fit_quality_summary, load_or_compute_channel_psd,
                                  load_sensitivity_info, mirror_to_cortex, project_channel_measure)
from salience_switch.plotting import (SALIENCE_Z_LABELS, SALIENCE_Z_RANGE, format_pvalue, mask_outline,
                                      plot_fit_quality, plot_hemispheres_zoom, plot_ridge, salience_zoom_rois)
from salience_switch.provenance import write_manifest
from salience_switch.stats import HemisphereMoranRandomization, moran_spearman, moran_spearman_difference
from salience_switch.surface import load_inflated_surfaces

STEP = "step2c_ieeg_states"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'

# fsLR-32k vertices per hemisphere (LH and RH each carry this many).
N_LH = 32492
# Moran surrogates per test
N_PERMUTATIONS = 1000
PREFIX = "figure_2c_ieeg_states"
# Vigilance states, in display order, and the analysed length of every recording: the
# shortest wakefulness recording lasts ~180 s, so every state's spectrum is estimated
# from the same amount of data.
STATES = ("W", "N2", "N3")
STATE_PAIRS = (("W", "N2"), ("W", "N3"), ("N2", "N3"))
STATE_LABELS = {"W": "Wake", "N2": "N2", "N3": "N3"}
STATE_TITLES = {s: f"Aperiodic exponent, {'wake' if s == 'W' else s}" for s in STATES}
STATE_DURATION_S = 180.0
# Figure 1 zoom-row window (figure_1a/1b/1c): 1450 x 300, zoom 1.3, colour-bar column,
# region names at 30 pt, inset-row titles at 36 pt (rows shown 150 mm wide, ~0.29x).
ZOOM_KW = dict(size=(1450, 300), zoom=1.3, color_bar=True, nan_color=(220, 220, 220, 1),
               label_fontsize=30, title_fontsize=36)


def load_state_spectra(ieeg_raw: str, cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Per-channel Welch PSD of every state, restricted to the channels recorded in all.

    Each state's spectra come from the first ``STATE_DURATION_S`` seconds of its
    recordings (cached per state). Channels are matched on subject, session and
    bipolar channel name and kept in wakefulness order, so the PSD rows of the three
    states are aligned with one channel table.

    Returns
    -------
    channels : pd.DataFrame
        Channel table of the retained channels.
    freq : np.ndarray
        Welch frequency axis (shared by the states).
    pxx : dict
        ``{state: PSD array (n_channels, n_freqs)}``, row-aligned with ``channels``.
    """
    key = ['Subject', 'Session', 'ChannelName']
    tables = {}
    for state in STATES:
        tables[state] = load_or_compute_channel_psd(
            ieeg_raw, cache_dir / f"channel_psd_{state}.pkl", stage=state,
            duration_s=STATE_DURATION_S, **MICA_PSD_PARAMS)
        if tables[state][0].duplicated(key).any():
            raise ValueError(f"Duplicate channels in the stage-{state} recordings.")
    freq = tables['W'][1]
    if any(not np.array_equal(tables[s][1], freq) for s in STATES):
        raise ValueError("The states' Welch frequency axes differ.")

    matched = tables['W'][0][key].reset_index().rename(columns={'index': 'row_W'})
    for state in STATES[1:]:
        other = tables[state][0][key].reset_index().rename(columns={'index': f'row_{state}'})
        matched = matched.merge(other, on=key, how='inner')
    n_all = {s: len(tables[s][0]) for s in STATES}
    logger.info(f"[Figure 2C] channels per state {n_all}; recorded in all three states: {len(matched)}")
    channels = tables['W'][0].iloc[matched['row_W'].to_numpy()].reset_index(drop=True)
    pxx = {s: tables[s][2][matched[f'row_{s}'].to_numpy()] for s in STATES}
    return channels, freq, pxx


def _null_percentiles(null: np.ndarray) -> dict:
    """2.5th, 50th and 97.5th percentiles of a null (NaN if it has no finite value)."""
    if not np.isfinite(null).any():
        return dict(null_lo=np.nan, null_med=np.nan, null_hi=np.nan)
    lo, med, hi = np.nanpercentile(null, [2.5, 50, 97.5])
    return dict(null_lo=lo, null_med=med, null_hi=hi)


def band_state_statistics(spec: dict, good: np.ndarray, sens_good: np.ndarray, idx_valid: np.ndarray,
                          y_valid: np.ndarray, msr, n_perm: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Supplementary: oscillatory peak power of every band vs the gradient, per state.

    For each band of ``SPEC_BANDS`` and state, the per-channel peak power above the
    aperiodic fit from the state's specparam fit (no detected peak = 0, as step 2b) is
    projected onto the template and correlated (Spearman) with the gradient on the
    inferential vertices against the within-network Moran null; for each band, the
    correlations of every pair of states are compared against paired Moran surrogates.

    Returns
    -------
    stats : pd.DataFrame
        Rows ``{band}_{state}`` and ``{band}_diff_{a}_{b}``: ``band``, ``test``, ``r``
        (correlation or difference), ``p``, null percentiles, ``n_vertices``, ``n_perm``.
    values : pd.DataFrame
        Per inferential vertex: ``vertex`` and the projected peak power ``{band}_{state}``.
    """
    rows, zvals = [], {}
    values = pd.DataFrame({"vertex": idx_valid})
    for band in SPEC_BANDS:
        for s in STATES:
            power = np.nan_to_num(spec[s]['band_power'][band], nan=0.0)[good]
            values[f"{band}_{s}"] = project_channel_measure(power, sens_good)[idx_valid]
            sd = np.nanstd(values[f"{band}_{s}"])
            zvals[(band, s)] = zscore(values[f"{band}_{s}"].to_numpy()) if sd > 0 else None
            if zvals[(band, s)] is None:
                logger.warning(f"[Figure 2C] {band} {s}: no peak-power variation across vertices; correlation undefined.")
                r, p, null = np.nan, np.nan, np.array([np.nan])
            else:
                r, p, null = moran_spearman(zvals[(band, s)], y_valid, msr)
            rows.append(dict(measure=f"{band}_{s}", band=band, test=s, r=r, p=p, **_null_percentiles(null),
                             n_vertices=int(idx_valid.size), n_perm=n_perm))
        for a, b in STATE_PAIRS:
            if zvals[(band, a)] is None or zvals[(band, b)] is None:
                d, p, d_null = np.nan, np.nan, np.array([np.nan])
            else:
                d, p, d_null = moran_spearman_difference(zvals[(band, a)], zvals[(band, b)], y_valid, msr)
            rows.append(dict(measure=f"{band}_diff_{a}_{b}", band=band, test=f"{a}-{b}", r=d, p=p,
                             **_null_percentiles(d_null), n_vertices=int(idx_valid.size), n_perm=n_perm))
    stats = pd.DataFrame(rows)
    for row in stats.itertuples(index=False):
        kind = "difference of correlations" if '-' in row.test else "Spearman r"
        logger.info(f"[Figure 2C] {row.band} {row.test}: {kind}={row.r:+.3f}, Moran permutation p={row.p:.3e}")
    return stats, values


def state_statistics(f: np.ndarray, pxx: dict, sens: np.ndarray, df_yeo_surf: pd.DataFrame,
                     hemi: str = 'LH', network: str = 'SalVentAttn', n_perm: int = N_PERMUTATIONS) -> dict:
    """Aperiodic exponent per state and its change from wakefulness to N3 vs the MPC gradient.

    Every channel spectrum of every state is parameterised once with ``specparam``
    (knee mode, ``SPEC_FIT_RANGE``); channels whose fit failed in any state are dropped so
    all measures share one channel set. The per-channel exponents and their
    within-channel change (N3 minus wakefulness) are projected onto the folded template
    through the sensitivity maps. On the analysis hemisphere's covered network vertices
    with a finite gradient, each map is correlated (Spearman) with the MPC gradient
    against a within-network Moran null (graph fitted once on that vertex set, add-one
    p), and the correlations of every pair of states are compared against paired Moran
    surrogates. The specparam fit-quality figures are drawn here, since they show
    individual channels.

    Returns
    -------
    tables : dict
        ``maps`` (per template vertex: ``inferential`` and the maps ``delta_N3_W``,
        ``exponent_W``, ``exponent_N2``, ``exponent_N3``), ``stats`` (the four map tests
        and the three state pairs: measure, r, p, null percentiles, n_vertices, n_perm),
        ``null`` (measure, surrogate, r), ``band_stats`` and ``band_values``
        (`band_state_statistics`).
    """
    hemi_offset = N_LH if hemi == 'RH' else 0
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    mask = ((df_yeo_surf['hemisphere'] == hemi) & (df_yeo_surf['network'] == network)).values
    mask_hemi = mask[hemi_offset:hemi_offset + N_LH]
    grad_hemi = df_yeo_surf[f't1_gradient1_{network}'].values[hemi_offset:hemi_offset + N_LH]

    # One specparam fit per channel and state; keep channels fitted in every state.
    spec = {s: compute_spectral_parameters(pxx[s], f, bands=SPEC_BANDS, fmin=SPEC_FIT_RANGE[0], fmax=SPEC_FIT_RANGE[1],
                                           return_model=True) for s in STATES}
    logger.info(f"[Figure 2C] specparam methods report (all states):\n{spec['W']['report']}")
    for s in STATES:
        logger.info(f"[Figure 2C] stage-{s} fit quality: {fit_quality_summary(spec[s], SPEC_FIT_RANGE[0])}")
        # Fit check (specparam guidance): worst, median and best fits, no exclusion.
        plot_fit_quality(spec[s].pop("model"), config.FIGURES_DIR / f"{PREFIX}_specparam_qc_{s}_supp_{hemi}.svg",
                         title=STATE_LABELS[s])
    exponent = {s: spec[s]['exponent'] for s in STATES}
    good = np.all([np.isfinite(exponent[s]) for s in STATES], axis=0)
    if not good.all():
        logger.warning(f"[Figure 2C] specparam fit failed in at least one state for {int((~good).sum())}/{good.size} channels; excluded.")
    for s in STATES:
        logger.info(f"[Figure 2C] stage-{s}: channel exponent median {np.median(exponent[s][good]):.3f} "
                    f"(IQR {np.percentile(exponent[s][good], 25):.3f}-{np.percentile(exponent[s][good], 75):.3f})")
    sens_good = sens[good]

    # Inferential vertex set (as step 2b): covered network vertices with a finite
    # gradient; the Moran graph is fitted once on it.
    covered = np.sum(sens_good, axis=0) > 0
    valid = covered[mask_hemi] & np.isfinite(grad_hemi[mask_hemi])
    idx_valid = np.flatnonzero(mask_hemi)[valid]
    y_valid = grad_hemi[idx_valid]
    valid_full = np.zeros(N_LH, dtype=bool)
    valid_full[idx_valid] = True
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=valid_full)
    w.data **= -1
    msr = HemisphereMoranRandomization(np.full(idx_valid.size, hemi), n_perm, random_state=0).fit(w)
    logger.info(f"[Figure 2C] inferential vertices (covered {hemi} {network} with gradient): {idx_valid.size}")

    maps = {"delta_N3_W": project_channel_measure((exponent['N3'] - exponent['W'])[good], sens_good)}
    maps.update({f"exponent_{s}": project_channel_measure(exponent[s][good], sens_good) for s in STATES})
    rows, null_rows = [], []
    for name, surf_map in maps.items():
        r, p, r_null = moran_spearman(zscore(surf_map[idx_valid]), y_valid, msr)
        rows.append(dict(measure=name, r=r, p=p, **_null_percentiles(r_null), n_vertices=int(idx_valid.size),
                         n_perm=n_perm))
        null_rows.append(pd.DataFrame({'measure': name, 'surrogate': np.arange(r_null.size), 'r': r_null}))
        logger.info(f"[Figure 2C] {name} vs MPC-gradient | Spearman r={r:+.3f}, Moran permutation p={p:.3e} "
                    f"(n_perm={n_perm}, n_vertices={idx_valid.size}; null r 95% "
                    f"[{rows[-1]['null_lo']:+.3f}, {rows[-1]['null_hi']:+.3f}])")

    # Between-state comparison of the correlations with the gradient: each pair of
    # state maps is correlated with the same Moran surrogates of the gradient, so the
    # null of the difference keeps the dependence between the two maps.
    for a, b in STATE_PAIRS:
        d, p, d_null = moran_spearman_difference(zscore(maps[f"exponent_{a}"][idx_valid]),
                                                 zscore(maps[f"exponent_{b}"][idx_valid]), y_valid, msr)
        rows.append(dict(measure=f"diff_{a}_{b}", r=d, p=p, **_null_percentiles(d_null),
                         n_vertices=int(idx_valid.size), n_perm=n_perm))
        null_rows.append(pd.DataFrame({'measure': f"diff_{a}_{b}", 'surrogate': np.arange(d_null.size), 'r': d_null}))
        logger.info(f"[Figure 2C] r({a}) - r({b}) with MPC-gradient | difference={d:+.3f}, Moran permutation p={p:.3e} "
                    f"(paired surrogates; null 95% [{rows[-1]['null_lo']:+.3f}, {rows[-1]['null_hi']:+.3f}])")

    band_stats, band_values = band_state_statistics(spec, good, sens_good, idx_valid, y_valid, msr, n_perm)
    return {"maps": pd.DataFrame({"vertex": np.arange(N_LH), "inferential": valid_full, **maps}),
            "stats": pd.DataFrame(rows), "null": pd.concat(null_rows, ignore_index=True),
            "band_stats": band_stats, "band_values": band_values}


def _draw_state_quintiles(ax, vals: dict, grad_v: np.ndarray, state_labels: list[str], ylabel: str | None,
                          legend_loc: str | None, tag: str, n_bins: int = 5) -> None:
    """Draw each MPC-gradient quintile's mean value across the states on ``ax``.

    The inferential vertices are split into ``n_bins`` equal-count gradient bins and
    each bin's mean of ``vals[state]`` is drawn across the states, in the coolwarm
    colour of its median gradient value on the shared +/-2 SD scale (as the Figure 1a
    strata caps); ``state_labels`` are the x tick labels. The means are logged with
    ``tag``.
    """
    bins = pd.qcut(grad_v, n_bins, labels=False)
    medians = [float(np.median(grad_v[bins == k])) for k in range(n_bins)]
    cmap, norm = plt.get_cmap('coolwarm'), mpl.colors.Normalize(*SALIENCE_Z_RANGE)
    xs = np.arange(len(STATES))
    for k in range(n_bins):
        means = [np.nanmean(vals[s][bins == k]) for s in STATES]
        logger.info(f"[Figure 2C] {tag} quintile {k + 1}/{n_bins} (gradient median {medians[k]:+.2f}, "
                    f"{int((bins == k).sum())} vertices): mean " + ", ".join(f"{s}={m:.3f}" for s, m in zip(STATES, means)))
        ax.plot(xs, means, '-o', color=cmap(norm(medians[k])), lw=2.5, ms=9, label=f"Q{k + 1}")
    ax.set_xticks(xs, state_labels, fontsize=17)
    ax.tick_params(axis='x', length=0)
    ax.set_xlim(-0.3, len(STATES) - 0.7)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend_loc:
        ax.legend(frameon=False, loc=legend_loc, title="MPC gradient", fontsize=17, title_fontsize=17)
    sns.despine(ax=ax)


def _fmt_r(value: float) -> str:
    """Signed correlation with a typographic minus."""
    return f"{value:+.2f}".replace('-', '−')


def plot_state_quintiles(exp_v: dict, grad_v: np.ndarray, state_stats: pd.DataFrame, path: Path) -> None:
    """Mean aperiodic exponent of each MPC-gradient quintile across the states.

    Each quintile's mean projected exponent (absolute values) is drawn across
    wakefulness, N2 and N3 (``_draw_state_quintiles``). Parallel lines mean a
    state-invariant gradient; converging lines mean the gradient's expression depends
    on the state. Under each state its Spearman r with the gradient and Moran p
    (``state_stats``). Means only: vertices share channels, so a vertex-level spread
    would be pseudo-replicated. 6-in panel at 21 pt, the Figure 1 companion-panel
    convention.
    """
    by_state = state_stats.set_index('measure')
    labels = [f"{STATE_LABELS[s]}\nr = {_fmt_r(by_state.loc[f'exponent_{s}', 'r'])}\n"
              f"$p_{{m}}$ = {format_pvalue(by_state.loc[f'exponent_{s}', 'p'])}" for s in STATES]
    with plt.rc_context({'font.size': 21}):
        fig, ax = plt.subplots(figsize=(6, 6.6))
        # Lines start high at wakefulness and fall towards N3: the upper right stays free.
        _draw_state_quintiles(ax, exp_v, grad_v, labels, "Aperiodic exponent", 'upper right', tag="exponent")
        plt.tight_layout()
        plt.savefig(path, transparent=True)
        plt.close(fig)


def plot_band_state_quintiles(band_values: pd.DataFrame, grad_v: np.ndarray, band_stats: pd.DataFrame,
                              path: Path) -> None:
    """Supplementary: the quintile panel of every frequency band's oscillatory peak power.

    One panel per band of ``SPEC_BANDS`` (6-in panels at 21 pt, as the main exponent
    panel): each quintile's mean projected peak power across the states, with each
    state's Spearman r with the gradient and its Moran p (labelled p_m) under it.
    """
    by = band_stats.set_index('measure')
    with plt.rc_context({'font.size': 21}):
        fig, axes = plt.subplots(1, len(SPEC_BANDS), figsize=(6 * len(SPEC_BANDS), 6.6))
        for i, (ax, (band, (lo, hi))) in enumerate(zip(axes, SPEC_BANDS.items())):
            labels = [f"{STATE_LABELS[s]}\nr = {_fmt_r(by.loc[f'{band}_{s}', 'r'])}\n"
                      f"$p_{{m}}$ = {format_pvalue(by.loc[f'{band}_{s}', 'p'])}" for s in STATES]
            _draw_state_quintiles(ax, {s: band_values[f"{band}_{s}"].to_numpy() for s in STATES}, grad_v, labels,
                                  "Peak power above aperiodic (log$_{10}$)" if i == 0 else None,
                                  'best' if i == 0 else None, tag=band)
            ax.set_title(f"{band.capitalize()} ({lo:g}–{hi:g} Hz)", fontsize=21)
        plt.tight_layout()
        plt.savefig(path, transparent=True)
        plt.close(fig)


def plot_states(tables: dict, df_yeo_surf: pd.DataFrame, hemi: str, network: str) -> None:
    """Figure 2c panels and supplements, from the `state_statistics` tables.

    The main panels: one Figure 1 brain row per state (both hemispheres,
    ``CustomCmap_ieeg``, +/-2 SD, network outline), the ridgeline of the Moran nulls level
    with them (states with uncorrected Moran p < 0.05 only) and the quintile trajectories
    with the per-state correlations; and, as supplementary, the change map, the inset rows
    of the change and of each state with the complete ridgeline, and every frequency
    band's peak power per state.
    """
    out_dir = config.FIGURES_DIR
    maps, stats, null = tables["maps"], tables["stats"], tables["null"]
    surf_lh_infl, surf_rh_infl = load_inflated_surfaces("fsLR-32k")
    hemi_offset = N_LH if hemi == 'RH' else 0
    grad_hemi = df_yeo_surf[f't1_gradient1_{network}'].values[hemi_offset:hemi_offset + N_LH]
    idx_valid = np.flatnonzero(maps["inferential"].to_numpy())
    y_valid = grad_hemi[idx_valid]
    null_of = {m: g['r'].to_numpy() for m, g in null.groupby('measure')}
    map_stats = stats[~stats['measure'].str.startswith('diff_')].reset_index(drop=True)
    per_state = map_stats['measure'].str.startswith('exponent_')

    # Figure 1 brain rows: network vertices of both hemispheres, z-scored there, with the
    # network border traced.
    rois = salience_zoom_rois(df_yeo_surf, 'both')
    outline = mask_outline(surf_lh_infl, surf_rh_infl, (df_yeo_surf['network'] == network).to_numpy())
    kwargs = dict(ZOOM_KW, cmap=ieeg_cmap.name, color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS,
                  cbar_title='z', outline=outline)

    def _z_display(template_map: np.ndarray) -> np.ndarray:
        values = mirror_to_cortex(template_map, df_yeo_surf, network)
        return (values - np.nanmean(values)) / np.nanstd(values)

    # Main: one brain row per state (untitled), and the ridgeline level with them showing
    # the states with an uncorrected Moran p < 0.05 (it shows p_m, as the quintile panel;
    # other slots left empty).
    geometry = set()
    for s in STATES:
        fig = plot_hemispheres_zoom(surf_lh_infl, surf_rh_infl, _z_display(maps[f"exponent_{s}"].to_numpy()), rois,
                                    filename=out_dir / f"{PREFIX}_exponent_{s}_zoom_{hemi}.svg", **kwargs)
        geometry.add((fig.inset_cy, fig.get_size_inches()[1] * 72))
        plt.close(fig)
    if len(geometry) != 1:
        raise RuntimeError(f"State rows differ in geometry {geometry}; the ridgeline rows cannot be aligned.")
    state_stats = map_stats[per_state].reset_index(drop=True)
    shown = state_stats.copy()
    shown.loc[~(shown['p'] < 0.05), ['r', 'p', 'null_lo', 'null_med', 'null_hi']] = np.nan
    logger.info(f"[Figure 2C] main ridgeline rows (significant): {', '.join(state_stats.loc[state_stats['p'] < 0.05, 'measure'])}")
    plot_ridge(shown, [null_of[m] for m in shown['measure']], *geometry.pop(), out_dir / f"{PREFIX}_forest_{hemi}.svg")
    plt.close('all')

    # Main: quintile trajectories with the per-state correlations.
    plot_state_quintiles({s: maps[f"exponent_{s}"].to_numpy()[idx_valid] for s in STATES}, y_valid, state_stats,
                         out_dir / f"{PREFIX}_quintiles_{hemi}.svg")

    # Supplementary: change map (brain row), and the inset rows of the change and of each
    # state with the complete ridgeline level with them.
    plot_hemispheres_zoom(surf_lh_infl, surf_rh_infl, _z_display(maps['delta_N3_W'].to_numpy()), rois,
                          filename=out_dir / f"{PREFIX}_delta_zoom_supp_{hemi}.svg", **kwargs)
    plt.close('all')
    titles = {"delta_N3_W": "Aperiodic exponent, N3 − wake", **{f"exponent_{s}": STATE_TITLES[s] for s in STATES}}
    roi_rows = [dict(r, label=None) for r in rois]
    geometry = set()
    for name in map_stats['measure']:
        fig = plot_hemispheres_zoom(surf_lh_infl, surf_rh_infl, _z_display(maps[name].to_numpy()), roi_rows,
                                    roi_only=True, draw_color_bar=True, title=titles[name],
                                    filename=out_dir / f"{PREFIX}_{name}_zoom_roi_supp_{hemi}.svg", **kwargs)
        geometry.add((fig.inset_cy, fig.get_size_inches()[1] * 72))
        plt.close(fig)
    if len(geometry) != 1:
        raise RuntimeError(f"Inset rows differ in geometry {geometry}; the ridgeline rows cannot be aligned.")
    plot_ridge(map_stats, [null_of[m] for m in map_stats['measure']], *geometry.pop(),
               out_dir / f"{PREFIX}_forest_supp_{hemi}.svg")
    plt.close('all')

    # Supplementary: every frequency band's oscillatory peak power per state (as step 2b).
    band_values = tables["band_values"]
    if not np.array_equal(band_values["vertex"].to_numpy(), idx_valid):
        raise ValueError("The band values are not on the inferential vertices of the maps table.")
    plot_band_state_quintiles(band_values, y_valid, tables["band_stats"],
                              out_dir / f"{PREFIX}_bands_quintiles_supp_{hemi}.svg")


def compute(settings: dict, paths: dict) -> None:
    """Per-state spectral parameters, their template maps and Moran tests."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    logger.info("Dataset        : MICA iEEG (BIDS_iEEG, sub-PX*, ses-01), stage-W wakefulness, stage-N2 and stage-N3 sleep")
    logger.info(f"Segments       : first {STATE_DURATION_S:g} s of every recording; channels recorded in all three states")
    logger.info("Preprocessing  : Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned; Welch PSD (Hamming 2 s, 1 s overlap), unit sum")
    logger.info(f"Spectral model : specparam, knee mode (knee >= 0), Gaussian peaks, {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz (as step 2b)")
    logger.info("Peak bands     : " + ", ".join(f"{b} {lo:g}-{hi:g} Hz" for b, (lo, hi) in SPEC_BANDS.items()))
    logger.info("Statistic      : per-state aperiodic exponent vs MPC gradient; between-state difference of the correlations (paired Moran surrogates); change N3 - wake (supplementary); uncorrected")
    logger.info(f"Null model     : within-network Moran randomization (n_rep={N_PERMUTATIONS}, procedure=singleton, random_state=0), add-one empirical p")
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels; statistics on {hemi}, maps on both hemispheres")
    logger.info(f"Analysis network: {network}")
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")

    logger.info(f"Raw recordings : {paths['ieeg_raw']}")
    channels, f, pxx = load_state_spectra(str(paths["ieeg_raw"]), config.work_dir(STEP, paths))
    channels[['ContactName1', 'ContactName2']] = channels[['ContactName1', 'ContactName2']].apply(lambda c: c.str.upper())
    df_sensitivity, sens_areas = load_sensitivity_info(root_dir=str(paths["electromica"]))
    logger.info(f"Sensitivity maps loaded: {df_sensitivity['Subject'].nunique()} subjects, {len(df_sensitivity)} contacts")
    sens = np.nan_to_num(build_bipolar_sensitivity(channels, df_sensitivity, sens_areas), nan=0.0)
    n_no_sens = int((sens.sum(axis=1) == 0).sum())
    if n_no_sens:
        logger.warning(f"{n_no_sens}/{len(sens)} channels have no suprathreshold sensitivity; "
                       f"they carry no weight in the surface projections.")

    tables = state_statistics(f, pxx, sens, df_yeo_surf, hemi=hemi, network=network)
    for name, table in tables.items():
        config.write_table(table, source / f"state_{name}_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network, hemisphere=hemi,
                   n_patients=int(channels['Subject'].nunique()), n_channels=len(channels),
                   segment_s=STATE_DURATION_S, n_permutations=N_PERMUTATIONS)


def plot(settings: dict) -> None:
    """Figure 2c panels and supplements, from source_data only."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    tables = {name: config.read_table(source / f"state_{name}_{hemi}.tsv")
              for name in ("maps", "stats", "null", "band_stats", "band_values")}
    plot_states(tables, df_yeo_surf, hemi, network)


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 2c: state dependence of the iEEG aperiodic exponent (Figure 2c).")

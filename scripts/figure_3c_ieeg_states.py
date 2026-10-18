# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 3c - State dependence of the iEEG aperiodic exponent along the salience-network
# microstructural gradient (wakefulness, N2 and N3 sleep).
#
# The same bipolar channels of the MICA iEEG dataset were recorded in wakefulness
# (stage-W), N2 and N3 sleep. The first 180 s of every recording are parameterised with
# specparam (as Figure 3b) and the aperiodic exponent is projected onto the folded
# fsLR-32k template through the bipolar sensitivity maps. The primary tests correlate the
# exponent of each state with the MPC gradient (within-network Moran null, FDR across
# the three states), and the correlations of every pair of states are compared against
# paired Moran surrogates (FDR across the three pairs); the within-channel change from
# wakefulness to N3 is tested as well (supplementary). Main panels: one brain row with
# zoom insets per state, the forest of the significant states level with them, and the
# exponent of each gradient quintile across the states with the per-state correlations.
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3c_ieeg_states.py \
#   -ieeg_deriv /host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA \
#   -hemi LH
# Requires df_1a_{hemi}.tsv (run figure_1a_t1map.py with matching -hemi first); falls
# back to df_1a_both.tsv restricted to the hemisphere, as figure_2_distance.py does.
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24"
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import mesh_elements
from brainspace.datasets import load_conte69
from brainspace.null_models import moran
from scipy.stats import zscore

from src.ieeg_processing import (load_sensitivity_info, build_bipolar_sensitivity, load_or_compute_channel_psd,
                                 compute_spectral_parameters, project_channel_measure, mirror_to_cortex,
                                 fit_quality_summary, plot_fit_quality, MICA_PSD_PARAMS, SPEC_FIT_RANGE,
                                 SPEC_BANDS)
from src.connectome_processing import benjamini_hochberg, moran_spearman, moran_spearman_difference
from src.plot_colors import ieeg_cmap
from src.logging_utils import setup_manuscript_logger
from src.surface_plotting import (plot_hemispheres_zoom, salience_zoom_rois, mask_outline, plot_forest,
                                  format_pvalue, SALIENCE_Z_RANGE, SALIENCE_Z_LABELS)

logger = logging.getLogger(__name__)

plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'

# fsLR-32k vertices per hemisphere (LH and RH each carry this many).
N_LH = 32492
# Vigilance states, in display order, and the analysed length of every recording: the
# shortest wakefulness recording lasts ~180 s, so every state's spectrum is estimated
# from the same amount of data.
STATES = ("W", "N2", "N3")
STATE_LABELS = {"W": "Wake", "N2": "N2", "N3": "N3"}
STATE_TITLES = {s: f"Aperiodic exponent, {'wake' if s == 'W' else s}" for s in STATES}
STATE_DURATION_S = 180.0
# Figure 1 zoom-row window (figure_1a/1b/1c): 1450 x 300, zoom 1.3, colour-bar column,
# region names at 30 pt, inset-row titles at 36 pt (rows shown 150 mm wide, ~0.29x).
ZOOM_KW = dict(size=(1450, 300), zoom=1.3, color_bar=True, nan_color=(220, 220, 220, 1),
               label_fontsize=30, title_fontsize=36)


def get_parser() -> argparse.ArgumentParser:
    """parser function"""
    parser = argparse.ArgumentParser(
        description="State dependence (wake, N2, N3) of the iEEG aperiodic exponent along the MPC gradient (Figure 3c).",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )
    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-ieeg_deriv",
        type=str,
        required=True,
        help="Absolute path to the electroMICA derivatives folder (leadfield sensitivity maps)"
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-ieeg_raw",
        type=str,
        default=None,
        help="Root of the original iEEG recordings (<root>/sub-PX*/ses-01/*stage-{W,N2,N3}.mat). "
             "Default: <ieeg_deriv>/../../original."
    )
    optional.add_argument(
        "-hemi",
        type=str,
        default="LH",
        choices=["LH", "RH"],
        help="Hemisphere of the statistics (default: LH, as the rest of the paper); the maps "
             "are shown on both hemispheres (folded sensitivity template)."
    )
    optional.add_argument(
        "-network",
        type=str,
        default="SalVentAttn",
        choices=["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"],
        help="Yeo 7-network to use as the analysis target (default: SalVentAttn)"
    )
    return parser


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
            ieeg_raw, cache_dir / f"figure_3c_channel_psd_{state}.pkl", stage=state,
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
    logger.info(f"[Figure 3C] channels per state {n_all}; recorded in all three states: {len(matched)}")
    channels = tables['W'][0].iloc[matched['row_W'].to_numpy()].reset_index(drop=True)
    pxx = {s: tables[s][2][matched[f'row_{s}'].to_numpy()] for s in STATES}
    return channels, freq, pxx


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
        logger.info(f"[Figure 3C] {tag} quintile {k + 1}/{n_bins} (gradient median {medians[k]:+.2f}, "
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


def plot_band_state_quintiles(band_vals: dict, grad_v: np.ndarray, band_stats: pd.DataFrame, path: Path) -> None:
    """Supplementary: the quintile panel of every frequency band's oscillatory peak power.

    One panel per band of ``SPEC_BANDS`` (6-in panels at 21 pt, as the main exponent
    panel): each quintile's mean projected peak power across the states, with each
    state's Spearman r with the gradient and its Moran p (labelled p_m; the FDR q
    across the five bands within the state is in the statistics table) under it.
    """
    by = band_stats.set_index('measure')
    with plt.rc_context({'font.size': 21}):
        fig, axes = plt.subplots(1, len(SPEC_BANDS), figsize=(6 * len(SPEC_BANDS), 6.6))
        for i, (ax, (band, (lo, hi))) in enumerate(zip(axes, SPEC_BANDS.items())):
            labels = [f"{STATE_LABELS[s]}\nr = {_fmt_r(by.loc[f'{band}_{s}', 'r'])}\n"
                      f"$p_{{m}}$ = {format_pvalue(by.loc[f'{band}_{s}', 'p'])}" for s in STATES]
            _draw_state_quintiles(ax, band_vals[band], grad_v, labels,
                                  "Peak power above aperiodic (log$_{10}$)" if i == 0 else None,
                                  'best' if i == 0 else None, tag=band)
            ax.set_title(f"{band.capitalize()} ({lo:g}–{hi:g} Hz)", fontsize=21)
        plt.tight_layout()
        plt.savefig(path, transparent=True)
        plt.close(fig)


def band_state_analysis(spec: dict, good: np.ndarray, sens_good: np.ndarray, idx_valid: np.ndarray,
                        y_valid: np.ndarray, msr, n_perm: int) -> tuple[pd.DataFrame, dict]:
    """Supplementary: oscillatory peak power of every band vs the gradient, per state.

    For each band of ``SPEC_BANDS`` and state, the per-channel peak power above the
    aperiodic fit from the state's specparam fit (no detected peak = 0, as Figure 3b) is
    projected onto the template and correlated (Spearman) with the gradient on the
    inferential vertices against the within-network Moran null (FDR across the five
    bands within each state, as Figure 3b); for each band, the correlations of every
    pair of states are compared against paired Moran surrogates (FDR across the three
    pairs within the band).

    Returns
    -------
    stats : pd.DataFrame
        Rows ``{band}_{state}`` and ``{band}_diff_{a}_{b}``: ``band``, ``test``, ``r``
        (correlation or difference), ``p``, ``q``, null percentiles, ``n_vertices``,
        ``n_perm``.
    band_vals : dict
        ``{band: {state: projected peak power on the inferential vertices}}``.
    """
    rows, band_vals, zvals = [], {}, {}
    for band in SPEC_BANDS:
        band_vals[band] = {}
        for s in STATES:
            power = np.nan_to_num(spec[s]['band_power'][band], nan=0.0)[good]
            band_vals[band][s] = project_channel_measure(power, sens_good)[idx_valid]
            sd = np.nanstd(band_vals[band][s])
            zvals[(band, s)] = zscore(band_vals[band][s]) if sd > 0 else None
            if zvals[(band, s)] is None:
                logger.warning(f"[Figure 3C] {band} {s}: no peak-power variation across vertices; correlation undefined.")
                r, p, null = np.nan, np.nan, np.array([np.nan])
            else:
                r, p, null = moran_spearman(zvals[(band, s)], y_valid, msr)
            rows.append(dict(measure=f"{band}_{s}", band=band, test=s, r=r, p=p, q=np.nan,
                             null_lo=np.nanpercentile(null, 2.5) if np.isfinite(null).any() else np.nan,
                             null_med=np.nanpercentile(null, 50) if np.isfinite(null).any() else np.nan,
                             null_hi=np.nanpercentile(null, 97.5) if np.isfinite(null).any() else np.nan,
                             n_vertices=int(idx_valid.size), n_perm=n_perm))
        for a, b in (("W", "N2"), ("W", "N3"), ("N2", "N3")):
            if zvals[(band, a)] is None or zvals[(band, b)] is None:
                d, p, d_null = np.nan, np.nan, np.array([np.nan])
            else:
                d, p, d_null = moran_spearman_difference(zvals[(band, a)], zvals[(band, b)], y_valid, msr)
            rows.append(dict(measure=f"{band}_diff_{a}_{b}", band=band, test=f"{a}-{b}", r=d, p=p, q=np.nan,
                             null_lo=np.nanpercentile(d_null, 2.5) if np.isfinite(d_null).any() else np.nan,
                             null_med=np.nanpercentile(d_null, 50) if np.isfinite(d_null).any() else np.nan,
                             null_hi=np.nanpercentile(d_null, 97.5) if np.isfinite(d_null).any() else np.nan,
                             n_vertices=int(idx_valid.size), n_perm=n_perm))
    stats = pd.DataFrame(rows)
    for s in STATES:  # FDR across the five bands within each state
        sel = stats['test'].eq(s) & stats['p'].notna()
        stats.loc[sel, 'q'] = benjamini_hochberg(stats.loc[sel, 'p'].to_numpy())
    for band in SPEC_BANDS:  # FDR across the three state pairs within each band
        sel = stats['band'].eq(band) & stats['test'].str.contains('-') & stats['p'].notna()
        stats.loc[sel, 'q'] = benjamini_hochberg(stats.loc[sel, 'p'].to_numpy())
    for row in stats.itertuples(index=False):
        kind = "difference of correlations" if '-' in row.test else "Spearman r"
        logger.info(f"[Figure 3C] {row.band} {row.test}: {kind}={row.r:+.3f}, Moran permutation p={row.p:.3e}, "
                    f"FDR q={row.q:.3f}")
    return stats, band_vals


def state_analysis(f: np.ndarray, pxx: dict, sens: np.ndarray, df_yeo_surf: pd.DataFrame, surf_lh_infl, surf_rh_infl,
                   project_root: Path, hemi: str = 'LH', network: str = 'SalVentAttn', n_perm: int = 1000) -> None:
    """Aperiodic exponent per state and its change from wakefulness to N3 vs the MPC gradient.

    Every channel spectrum of every state is parameterised once with ``specparam``
    (knee mode, ``SPEC_FIT_RANGE``); channels whose fit failed in any state are dropped so
    all measures share one channel set. The per-channel exponents and their
    within-channel change (N3 minus wakefulness) are projected onto the folded template
    through the sensitivity maps. On the analysis hemisphere's covered network vertices
    with a finite gradient, each map is correlated (Spearman) with the MPC gradient
    against a within-network Moran null (graph fitted once on that vertex set, add-one
    p). The three per-state correlations are FDR-corrected, and the correlations of every
    pair of states are compared against paired Moran surrogates (FDR across the three
    pairs); the change map is tested as well (supplementary). Writes
    ``df_3c_state_stats_{hemi}.csv``; the main panels: one Figure 1 brain row per state
    (both hemispheres, ``CustomCmap_ieeg``, +/-2 SD, network outline), the forest level
    with them (states with uncorrected Moran p < 0.05 only) and the quintile trajectories with the
    per-state correlations; and, as supplementary, the change map,
    the inset rows of the change and of each state with the complete forest, and every
    frequency band's peak power per state (``band_state_analysis``).
    """
    hemi_offset = N_LH if hemi == 'RH' else 0
    out_dir = project_root / "results/figures"
    prefix = "figure_3c_ieeg_states"

    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    mask = ((df_yeo_surf['hemisphere'] == hemi) & (df_yeo_surf['network'] == network)).values
    mask_hemi = mask[hemi_offset:hemi_offset + N_LH]
    grad_hemi = df_yeo_surf[f't1_gradient1_{network}'].values[hemi_offset:hemi_offset + N_LH]

    # One specparam fit per channel and state; keep channels fitted in every state.
    spec = {s: compute_spectral_parameters(pxx[s], f, bands=SPEC_BANDS, fmin=SPEC_FIT_RANGE[0], fmax=SPEC_FIT_RANGE[1],
                                           return_model=True) for s in STATES}
    logger.info(f"[Figure 3C] specparam methods report (all states):\n{spec['W']['report']}")
    for s in STATES:
        logger.info(f"[Figure 3C] stage-{s} fit quality: {fit_quality_summary(spec[s], SPEC_FIT_RANGE[0])}")
        # Fit check (specparam guidance): worst, median and best fits, no exclusion.
        plot_fit_quality(spec[s].pop("model"), out_dir / f"{prefix}_specparam_qc_{s}_supp_{hemi}.svg",
                         title=STATE_LABELS[s])
    exponent = {s: spec[s]['exponent'] for s in STATES}
    good = np.all([np.isfinite(exponent[s]) for s in STATES], axis=0)
    if not good.all():
        logger.warning(f"[Figure 3C] specparam fit failed in at least one state for {int((~good).sum())}/{good.size} channels; excluded.")
    for s in STATES:
        logger.info(f"[Figure 3C] stage-{s}: channel exponent median {np.median(exponent[s][good]):.3f} "
                    f"(IQR {np.percentile(exponent[s][good], 25):.3f}-{np.percentile(exponent[s][good], 75):.3f})")
    sens_good = sens[good]

    # Inferential vertex set (as Figure 3b): covered network vertices with a finite
    # gradient; the Moran graph is fitted once on it.
    covered = np.sum(sens_good, axis=0) > 0
    valid = covered[mask_hemi] & np.isfinite(grad_hemi[mask_hemi])
    idx_valid = np.flatnonzero(mask_hemi)[valid]
    y_valid = grad_hemi[idx_valid]
    valid_full = np.zeros(N_LH, dtype=bool)
    valid_full[idx_valid] = True
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=valid_full)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_perm, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)
    logger.info(f"[Figure 3C] inferential vertices (covered {hemi} {network} with gradient): {idx_valid.size}")

    maps = {"delta_N3_W": project_channel_measure((exponent['N3'] - exponent['W'])[good], sens_good)}
    maps.update({f"exponent_{s}": project_channel_measure(exponent[s][good], sens_good) for s in STATES})
    rows = []
    for name, surf_map in maps.items():
        r, p, r_null = moran_spearman(zscore(surf_map[idx_valid]), y_valid, msr)
        null_lo, null_med, null_hi = np.percentile(r_null, [2.5, 50, 97.5])
        rows.append(dict(measure=name, r=r, p=p, q=np.nan, null_lo=null_lo, null_med=null_med, null_hi=null_hi,
                         n_vertices=int(idx_valid.size), n_perm=n_perm))
        logger.info(f"[Figure 3C] {name} vs MPC-gradient | Spearman r={r:+.3f}, Moran permutation p={p:.3e} "
                    f"(n_perm={n_perm}, n_vertices={idx_valid.size}; null r 95% [{null_lo:+.3f}, {null_hi:+.3f}])")
    stats = pd.DataFrame(rows)
    per_state = stats['measure'].str.startswith('exponent_')
    stats.loc[per_state, 'q'] = benjamini_hochberg(stats.loc[per_state, 'p'].to_numpy())
    for row in stats[per_state].itertuples(index=False):
        logger.info(f"[Figure 3C] {row.measure}: FDR q={row.q:.3f} (across the three states)")

    # Between-state comparison of the correlations with the gradient: each pair of
    # state maps is correlated with the same Moran surrogates of the gradient, so the
    # null of the difference keeps the dependence between the two maps.
    pair_rows = []
    for a, b in (("W", "N2"), ("W", "N3"), ("N2", "N3")):
        d, p, d_null = moran_spearman_difference(zscore(maps[f"exponent_{a}"][idx_valid]),
                                                 zscore(maps[f"exponent_{b}"][idx_valid]), y_valid, msr)
        null_lo, null_med, null_hi = np.percentile(d_null, [2.5, 50, 97.5])
        pair_rows.append(dict(measure=f"diff_{a}_{b}", state_a=a, state_b=b, r=d, p=p, q=np.nan, null_lo=null_lo,
                              null_med=null_med, null_hi=null_hi, n_vertices=int(idx_valid.size), n_perm=n_perm))
        logger.info(f"[Figure 3C] r({a}) - r({b}) with MPC-gradient | difference={d:+.3f}, Moran permutation p={p:.3e} "
                    f"(paired surrogates; null 95% [{null_lo:+.3f}, {null_hi:+.3f}])")
    pairs = pd.DataFrame(pair_rows)
    pairs['q'] = benjamini_hochberg(pairs['p'].to_numpy())
    for row in pairs.itertuples(index=False):
        logger.info(f"[Figure 3C] {row.measure}: FDR q={row.q:.3f} (across the three pairs)")
    stats_path = project_root / f"data/dataframes/df_3c_state_stats_{hemi}.csv"
    pd.concat([stats, pairs.drop(columns=['state_a', 'state_b'])], ignore_index=True).to_csv(stats_path, index=False)
    logger.info(f"[Figure 3C] state statistics saved to {stats_path}")

    # Figure 1 brain rows: network vertices of both hemispheres, z-scored there, with the
    # network border traced.
    rois = salience_zoom_rois(df_yeo_surf, 'both')
    outline = mask_outline(surf_lh_infl, surf_rh_infl, (df_yeo_surf['network'] == network).to_numpy())
    kwargs = dict(ZOOM_KW, cmap=ieeg_cmap.name, color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS,
                  cbar_title='z', outline=outline)

    def _z_display(template_map: np.ndarray) -> np.ndarray:
        values = mirror_to_cortex(template_map, df_yeo_surf, network)
        return (values - np.nanmean(values)) / np.nanstd(values)

    # Main: one brain row per state (untitled), and the forest level with them showing the
    # states with an uncorrected Moran p < 0.05 (the forest shows p_m, as the quintile panel;
    # the FDR q across the three states stays in the CSV; other slots left empty).
    geometry = set()
    for s in STATES:
        fig = plot_hemispheres_zoom(surf_lh_infl, surf_rh_infl, _z_display(maps[f"exponent_{s}"]), rois,
                                    filename=out_dir / f"{prefix}_exponent_{s}_zoom_{hemi}.svg", **kwargs)
        geometry.add((fig.inset_cy, fig.get_size_inches()[1] * 72))
        plt.close(fig)
    if len(geometry) != 1:
        raise RuntimeError(f"State rows differ in geometry {geometry}; the forest rows cannot be aligned.")
    state_stats = stats[per_state].reset_index(drop=True)
    shown = state_stats.copy()
    shown.loc[~(shown['p'] < 0.05), ['r', 'p', 'q', 'null_lo', 'null_med', 'null_hi']] = np.nan
    logger.info(f"[Figure 3C] main forest rows (significant): {', '.join(state_stats.loc[state_stats['p'] < 0.05, 'measure'])}")
    plot_forest(shown, *geometry.pop(), out_dir / f"{prefix}_forest_{hemi}.svg")
    plt.close('all')

    # Main: quintile trajectories with the per-state correlations.
    plot_state_quintiles({s: maps[f"exponent_{s}"][idx_valid] for s in STATES}, y_valid, state_stats,
                         out_dir / f"{prefix}_quintiles_{hemi}.svg")

    # Supplementary: change map (brain row), and the inset rows of the change and of each
    # state with the complete forest level with them.
    plot_hemispheres_zoom(surf_lh_infl, surf_rh_infl, _z_display(maps['delta_N3_W']), rois,
                          filename=out_dir / f"{prefix}_delta_zoom_supp_{hemi}.svg", **kwargs)
    plt.close('all')
    titles = {"delta_N3_W": "Aperiodic exponent, N3 − wake", **{f"exponent_{s}": STATE_TITLES[s] for s in STATES}}
    roi_rows = [dict(r, label=None) for r in rois]
    geometry = set()
    for name in stats['measure']:
        fig = plot_hemispheres_zoom(surf_lh_infl, surf_rh_infl, _z_display(maps[name]), roi_rows, roi_only=True,
                                    draw_color_bar=True, title=titles[name],
                                    filename=out_dir / f"{prefix}_{name}_zoom_roi_supp_{hemi}.svg", **kwargs)
        geometry.add((fig.inset_cy, fig.get_size_inches()[1] * 72))
        plt.close(fig)
    if len(geometry) != 1:
        raise RuntimeError(f"Inset rows differ in geometry {geometry}; the forest rows cannot be aligned.")
    plot_forest(stats, *geometry.pop(), out_dir / f"{prefix}_forest_supp_{hemi}.svg")
    plt.close('all')

    # Supplementary: every frequency band's oscillatory peak power per state (as Figure 3b).
    band_stats, band_vals = band_state_analysis(spec, good, sens_good, idx_valid, y_valid, msr, n_perm)
    band_path = project_root / f"data/dataframes/df_3c_band_state_stats_{hemi}.csv"
    band_stats.to_csv(band_path, index=False)
    logger.info(f"[Figure 3C] band statistics saved to {band_path}")
    plot_band_state_quintiles(band_vals, y_valid, band_stats, out_dir / f"{prefix}_bands_quintiles_supp_{hemi}.svg")


def main():
    parser = get_parser()
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_3c_ieeg_states", project_root, args)
    logger.info("Dataset        : MICA iEEG (BIDS_iEEG, sub-PX*, ses-01), stage-W wakefulness, stage-N2 and stage-N3 sleep")
    logger.info(f"Segments       : first {STATE_DURATION_S:g} s of every recording; channels recorded in all three states")
    logger.info("Preprocessing  : Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned; Welch PSD (Hamming 2 s, 1 s overlap), unit sum")
    logger.info(f"Spectral model : specparam, knee mode (knee >= 0), Gaussian peaks, {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz (as Figure 3b)")
    logger.info("Peak bands     : " + ", ".join(f"{b} {lo:g}-{hi:g} Hz" for b, (lo, hi) in SPEC_BANDS.items()))
    logger.info("Statistic      : per-state aperiodic exponent vs MPC gradient (FDR across states); between-state difference of the correlations (paired Moran surrogates, FDR across pairs); change N3 - wake (supplementary)")
    logger.info("Null model     : within-network Moran randomization (n_rep=1000, procedure=singleton, random_state=0), add-one empirical p")
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels; statistics on {args.hemi}, maps on both hemispheres")
    logger.info(f"Analysis network: {args.network}")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')

    # MPC gradient (as figure_3_ieeg_mica / figure_2_distance): the hemisphere's own
    # table, else the 'both' table (a safe superset; the network mask subsets it).
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        path_df_1a = project_root / 'data/dataframes/df_1a_both.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(f"No gradient dataframe (df_1a_{args.hemi}.tsv or df_1a_both.tsv) in "
                                f"{path_df_1a.parent}. Run figure_1a_t1map.py with -hemi {args.hemi} first.")
    logger.info(f"Gradient       : SN MPC gradient from {path_df_1a.name}")
    df_yeo_surf = pd.read_csv(path_df_1a, sep="\t")

    ieeg_raw = args.ieeg_raw or str(Path(args.ieeg_deriv).parent.parent / 'original')
    logger.info(f"Raw recordings : {ieeg_raw}")
    channels, f, pxx = load_state_spectra(ieeg_raw, project_root / 'data/dataframes')
    channels[['ContactName1', 'ContactName2']] = channels[['ContactName1', 'ContactName2']].apply(lambda c: c.str.upper())

    df_sensitivity, sens_areas = load_sensitivity_info(root_dir=args.ieeg_deriv)
    logger.info(f"Sensitivity maps loaded: {df_sensitivity['Subject'].nunique()} subjects, {len(df_sensitivity)} contacts")
    sens = np.nan_to_num(build_bipolar_sensitivity(channels, df_sensitivity, sens_areas), nan=0.0)
    n_no_sens = int((sens.sum(axis=1) == 0).sum())
    if n_no_sens:
        logger.warning(f"{n_no_sens}/{len(sens)} channels have no suprathreshold sensitivity; "
                       f"they carry no weight in the surface projections.")

    state_analysis(f, pxx, sens, df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, project_root,
                   hemi=args.hemi, network=args.network)


if __name__ == "__main__":
    main()

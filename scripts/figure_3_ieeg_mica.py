# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Concatenate ieeg information from all sessions and subjects from electromica,
# derivatives, and map it to the surface using the provided contact sensitivity maps.
#
# The sensitivity maps are only for 32k surfaces and fsaverage5. The 5k surfaces 
# are not able to properly depict the sensitivity, which varies quickly in space.
# fsnative surfaces have triangles that vary widely in size (1000 times),
# which leads to some numerical issues.
#
# The bipolar sensitivities of both hemispheres are folded onto one fsLR-32k template,
# so the spectral maps are shown on both hemispheres (as the Figure 1 maps), while the
# statistics are run on one hemisphere (-hemi, default LH, as the rest of the paper)
# against that hemisphere's salience-network MPC gradient.
#
# database
# BIDS_ieeg: 31 subjects
# The iEEG Data is in host/verges/tank/data/BIDS_iEEG
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3_ieeg_mica.py \
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

from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import  mesh_elements
from brainspace.datasets import load_conte69
from brainspace.null_models import moran
from scipy.stats import spearmanr


import re
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from scipy.stats import zscore
from scipy.ndimage import rotate
from scipy.spatial.distance import cdist

import logging

from src.atlas_load import convert_states_str2int, compute_network_mask
from src.ieeg_processing import (load_sensitivity_info, build_bipolar_sensitivity, load_or_compute_channel_psd,
                                 compute_spectral_parameters, aperiodic_component, project_channel_measure,
                                 mirror_to_cortex, fit_quality_summary, plot_fit_quality, MICA_PSD_PARAMS,
                                 SPEC_FIT_RANGE, SPEC_BANDS)
from src.connectome_processing import empirical_p_twosided, benjamini_hochberg, moran_spearman, _weighted_mean_projection
from src.plot_colors import yeo7_rgb, yeo7_abbrev, ieeg_cmap
from src.logging_utils import setup_manuscript_logger
from src.surface_plotting import (plot_hemispheres_zoom, salience_zoom_rois, mask_outline, plot_forest,
                                  format_pvalue, SALIENCE_Z_RANGE, SALIENCE_Z_LABELS)

logger = logging.getLogger(__name__)

# fsLR-32k vertices per hemisphere (LH and RH each carry this many).
N_LH = 32492

BAND_SYMBOLS = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}

# Figure 1 zoom-row window (figure_1a/1b/1c): 1450 x 300, zoom 1.3, colour-bar column,
# region names at 30 pt and inset-row titles at 36 pt; the rows are shown
# ZOOM_ROW_WIDTH_MM wide in the figure layout (~0.29x, so 30 pt reads as ~8.8 pt).
ZOOM_KW = dict(size=(1450, 300), zoom=1.3, color_bar=True, nan_color=(220, 220, 220, 1),
               label_fontsize=30, title_fontsize=36)


def _orient_fc_gradient(fc_vals: np.ndarray, networks: np.ndarray, label: str = "FC gradient") -> np.ndarray:
    """Orient an FC gradient so the default-mode network sits at the low end.

    The diffusion-map eigenvector polarity is arbitrary, so the gradient is
    oriented by anatomy rather than a hardcoded sign (mirrors
    ``figure_2_distance._load_fc_gradient``): it is flipped so the default-mode
    network occupies the low (default-mode) pole and the task-positive systems
    the high pole, matching the projection reading (high P = coupling to the
    task-positive end of the FC gradient). The chosen sign is logged so a change
    in the source gradient's polarity surfaces in the logs rather than silently
    inverting results.

    Parameters
    ----------
    fc_vals : np.ndarray
        Per-vertex FC gradient values (NaN allowed off-cortex).
    networks : np.ndarray
        Per-vertex Yeo network labels, aligned with ``fc_vals``.
    label : str
        Name used in the log message.

    Returns
    -------
    np.ndarray
        FC gradient with default-mode at the low end.
    """
    finite = np.isfinite(fc_vals)
    default_mean = np.nanmean(fc_vals[finite & (networks == "Default")])
    cortical_mean = np.nanmean(fc_vals[finite])
    if np.isfinite(default_mean) and default_mean > cortical_mean:
        logger.info(f"[{label}] flipped sign so the default-mode network sits at the low end.")
        return -fc_vals
    logger.info(f"[{label}] sign kept as loaded (default-mode already at the low end).")
    return fc_vals


def _pct_color_range(values: np.ndarray, lo: float = 0.0, hi: float = 100.0):
    """Percentile colour range for a surface map (``None`` if no finite values).

    With the default [0, 100] bounds this spans the full finite data range (no
    clipping); narrower ``lo``/``hi`` percentiles would instead spread skewed data
    across a sequential colormap by trimming the extremes.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    vmin, vmax = np.percentile(finite, [lo, hi])
    if vmin == vmax:
        return None
    return (float(vmin), float(vmax))


def _zscore_finite(values: np.ndarray) -> np.ndarray:
    """z-score over the finite entries (NaN kept); 0 where the values do not vary."""
    sd = np.nanstd(values)
    if not sd > 0:
        return np.where(np.isfinite(values), 0.0, np.nan)
    return (values - np.nanmean(values)) / sd


def _plot_measure_zoom(values: np.ndarray, prefix: str, title: str, *, rois: list[dict], surf_lh, surf_rh,
                       out_dir: Path, suffix: str, cmap: str, color_range: tuple[float, float],
                       cbar_labels: tuple[str, str], cbar_title: str | None,
                       outline: np.ndarray | None = None) -> tuple[float, float]:
    """Figure 1-style brain row with zoom insets, plus the inset row alone.

    Writes ``{prefix}_zoom_{suffix}.svg`` (brain row, marker circles, insets and region
    names, as ``figure_1a_brain_zoom.svg``) and ``{prefix}_zoom_roi_{suffix}.svg`` (the
    inset row alone, titled, colour bar drawn, no region names, as the Figure 1b
    zoom rows), both on the Figure 1 window (``ZOOM_KW``), with ``outline`` (e.g. the
    network border from ``mask_outline``) traced on the brains and insets.

    Returns
    -------
    tuple[float, float]
        Inset-row centre and height (pt) of the ``_zoom_roi`` figure, which place the
        forest-plot rows level with the stacked inset rows.
    """
    kwargs = dict(ZOOM_KW, cmap=cmap, color_range=color_range, cbar_labels=cbar_labels, cbar_title=cbar_title,
                  outline=outline)
    plot_hemispheres_zoom(surf_lh, surf_rh, values, rois, filename=out_dir / f"{prefix}_zoom_{suffix}.svg", **kwargs)
    fig = plot_hemispheres_zoom(surf_lh, surf_rh, values, [dict(r, label=None) for r in rois], roi_only=True,
                                draw_color_bar=True, title=title,
                                filename=out_dir / f"{prefix}_zoom_roi_{suffix}.svg", **kwargs)
    geometry = (fig.inset_cy, fig.get_size_inches()[1] * 72)
    plt.close('all')
    return geometry


def _plot_gradient_spectra(freq: np.ndarray, psd_v: np.ndarray, flat_v: np.ndarray, grad_v: np.ndarray,
                           path: Path, n_bins: int = 5) -> None:
    """Spectra and flattened spectra averaged within MPC-gradient quantile bins.

    The inferential vertices are split into ``n_bins`` equal-count bins of the
    gradient (quintiles by default), and each bin's mean sensitivity-weighted
    spectrum (left, log-log) and mean flattened spectrum (right: log10 power above
    each channel's fitted aperiodic component) are drawn in the coolwarm colour of the
    bin's median gradient value on the shared +/-2 SD scale, as the caps of the
    Figure 1a strata. Canonical bands are shaded alternately and labelled. Means
    only: vertices share channels, so a vertex-level spread would be pseudo-replicated.

    Parameters
    ----------
    freq : np.ndarray, shape (n_freqs,)
    psd_v, flat_v : np.ndarray, shape (n_freqs, n_vertices)
        Per-vertex spectrum (linear power) and flattened spectrum (log10).
    grad_v : np.ndarray, shape (n_vertices,)
        MPC gradient at the same vertices.
    path : Path
        Output SVG path.
    n_bins : int
        Number of equal-count gradient bins.
    """
    bins = pd.qcut(grad_v, n_bins, labels=False)
    medians = [float(np.median(grad_v[bins == k])) for k in range(n_bins)]
    for k in range(n_bins):
        logger.info(f"[Figure 3B] spectra bin {k + 1}/{n_bins}: {int((bins == k).sum())} vertices, "
                    f"gradient median {medians[k]:+.2f}")
    cmap, norm = plt.get_cmap('coolwarm'), mpl.colors.Normalize(*SALIENCE_Z_RANGE)
    xticks = [1, 4, 8, 13, 30, 45]

    # Two 6-in panels at 21 pt, the Figure 1a panel convention (shown at ~0.41x).
    with plt.rc_context({'font.size': 21}):
        fig, (ax_psd, ax_flat) = plt.subplots(1, 2, figsize=(12, 6))
        for i, (band, (lo, hi)) in enumerate(SPEC_BANDS.items()):
            lo = max(lo, freq[0])
            for ax in (ax_psd, ax_flat):
                if i % 2 == 0:
                    ax.axvspan(lo, hi, color='0.93', lw=0, zorder=0)
            ax_flat.text(np.sqrt(lo * hi), 1.0, BAND_SYMBOLS[band], transform=ax_flat.get_xaxis_transform(),
                         ha='center', va='bottom')
        for k in range(n_bins):
            color = cmap(norm(medians[k]))
            ax_psd.loglog(freq, np.nanmean(psd_v[:, bins == k], axis=1), color=color, lw=2.5, label=f"Q{k + 1}")
            ax_flat.semilogx(freq, np.nanmean(flat_v[:, bins == k], axis=1), color=color, lw=2.5)
        ax_flat.axhline(0, color='k', ls='--', lw=1, zorder=1)
        ax_psd.set_ylabel("Normalized PSD")
        ax_flat.set_ylabel("Power above aperiodic fit (log$_{10}$)")
        for ax in (ax_psd, ax_flat):
            ax.set_xlim(freq[0], freq[-1])
            ax.set_xticks(xticks, [str(x) for x in xticks])
            ax.xaxis.set_minor_locator(ticker.NullLocator())
            ax.set_xlabel("Frequency (Hz)")
            sns.despine(ax=ax)
        ax_psd.legend(frameon=False, loc='lower left', title="MPC gradient")
        plt.tight_layout()
        plt.savefig(path, transparent=True)
        plt.close(fig)


plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False


def get_parser() -> argparse.ArgumentParser:
    """parser function"""
    parser = argparse.ArgumentParser(
        description="Process ieeg derivatives and surfaces.",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-ieeg_deriv",
        type=str,
        help="Absolute path to the ieeg derivatives folder (e.g., /data/mica/...)"
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-ieeg_raw",
        type=str,
        default=None,
        help="Root of the original iEEG recordings (<root>/sub-PX*/ses-01/*stage-W.mat). "
             "Default: the 'original' folder of the BIDS dataset that holds -ieeg_deriv "
             "(<ieeg_deriv>/../../original)."
    )
    optional.add_argument(
        "-hemi",
        type=str,
        default="LH",
        choices=["LH", "RH"],
        help="Hemisphere of the statistics: 'LH' or 'RH' (default: LH). Bipolar "
             "sensitivities from both hemispheres are folded onto a single fsLR-32k "
             "template (see build_bipolar_sensitivity), so the maps are shown on both "
             "hemispheres, while the gradient correlations and their Moran null are "
             "evaluated on the network vertices of this hemisphere, against its MPC "
             "gradient (df_1a_<hemi>.tsv, else df_1a_both.tsv restricted to it)."
    )
    optional.add_argument(
        "-network",
        type=str,
        default="SalVentAttn",
        choices=["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"],
        help="Yeo 7-network to use as the analysis target (default: SalVentAttn)"
    )
    return parser


def frequency_band_analysis_sensitivity(f: np.ndarray, pxx_raw: np.ndarray, sens: np.ndarray, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf: pd.DataFrame, project_root: Path, hemi: str = 'RH', network: str = 'SalVentAttn', n_perm: int = 1000) -> None:
    """Sensitivity-weighted spectral measures versus the within-network MPC gradient.

    The per-channel spectra are parameterised once with ``specparam`` (one fit per
    channel) and projected onto the fsLR-32k template through the folded bipolar
    sensitivity maps. **Statistics** run on the analysis hemisphere ``hemi``: its
    source-network vertices that have coverage and a finite MPC gradient are correlated
    (Spearman) against that gradient. The **primary**
    measure is the aperiodic (1/f) exponent — the theoretically grounded
    electrophysiological correlate of cortical hierarchy; the **secondary** measure is
    the oscillatory (periodic) peak power in five canonical bands, taken from the same
    fit so it is orthogonal to the exponent rather than re-encoding the 1/f change, and
    FDR-corrected across bands. Significance is the within-network Moran spatial null
    with the add-one empirical p, the graph fitted once on the (finite) inferential
    vertex set; r, p, q and the central 95 % of each null are written to
    ``df_3b_spectral_stats_{hemi}.csv``.

    **Display** follows Figure 1: since the template stands for both hemispheres, each
    measure is shown on the network vertices of both hemispheres, z-scored there, on the
    shared +/-2 SD scale in ``CustomCmap_ieeg``, as a brain row with the
    ``SALIENCE_ZOOM_ROIS`` insets and as the inset row alone. A forest plot level with
    the stacked inset rows summarises the tests, and the spectra and flattened spectra
    averaged within gradient quintiles show the data behind them. Coverage maps and the
    per-measure scatter plots (supplementary) are also written.

    Parameters
    ----------
    f : np.ndarray
        Welch frequency bins, shared with the similarity analysis (computed once in ``main``).
    pxx_raw : np.ndarray, shape (n_channels, n_freqs)
        Per-channel Welch PSD, shared with the similarity analysis.
    sens : np.ndarray, shape (n_channels, N_LH)
        Stacked absolute bipolar sensitivity maps (NaN already zeroed).
    surf32k_lh_infl, surf32k_rh_infl
        Inflated fsLR-32k surfaces for the brain figures.
    df_yeo_surf : pd.DataFrame
        Per-vertex surface table (both hemispheres) with the MPC gradient column.
    project_root : Path
    hemi : {'LH', 'RH'}
        Analysis hemisphere of the statistics.
    network : str
        Yeo 7-network used as the analysis target.
    n_perm : int
        Number of Moran surrogates.
    """
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']
    hemi_offset = N_LH if hemi == 'RH' else 0
    out_dir = project_root / "results/figures"
    prefix = "figure_3b_ieeg_mica"

    # Setup Geometry
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh

    # Define analysis mask: target network for the specified hemisphere.
    mask = ((df_yeo_surf['hemisphere'] == hemi) & (df_yeo_surf['network'] == network)).values
    mask_hemi = mask[hemi_offset:hemi_offset + N_LH]
    grad_hemi = df_yeo_surf[f't1_gradient1_{network}'].values[hemi_offset:hemi_offset + N_LH]

    # Parameterise every channel spectrum ONCE (aperiodic exponent + oscillatory band
    # power from the same fit), then drop channels whose specparam fit failed to
    # converge so both measures and the coverage map share one successfully-fit set.
    spec = compute_spectral_parameters(pxx_raw, f, bands=SPEC_BANDS, fmin=SPEC_FIT_RANGE[0], fmax=SPEC_FIT_RANGE[1],
                                       return_model=True)
    exp_ch = spec['exponent']
    good = np.isfinite(exp_ch)
    if not good.all():
        logger.warning(f"[Figure 3B] specparam fit failed for {int((~good).sum())}/{good.size} channels; excluded.")
    logger.info(f"[Figure 3B] specparam methods report:\n{spec['report']}")
    logger.info(f"[Figure 3B] fit quality: {fit_quality_summary(spec, SPEC_FIT_RANGE[0])}")
    # Fit check (specparam guidance): worst, median and best fits, no exclusion.
    plot_fit_quality(spec.pop('model'), out_dir / f"{prefix}_specparam_qc_supp_{hemi}.svg", title="wakefulness")
    sens_good = sens[good]

    # Inferential vertex set: source-network vertices with iEEG coverage (from the
    # fit channels) AND a finite gradient value. The Moran null is fitted once on this
    # finite subset's geometry (no NaN/zero fill), keeping the surrogate field free of
    # artificial values.
    sens_good_sum = np.sum(sens_good, axis=0)
    covered_hemi = sens_good_sum > 0
    y_mask = grad_hemi[mask_hemi]
    valid = covered_hemi[mask_hemi] & np.isfinite(y_mask)
    y_valid = y_mask[valid]
    y_valid_z = zscore(y_valid)
    idx_valid = np.flatnonzero(mask_hemi)[valid]  # template indices of the inferential vertices
    valid_full = np.zeros(N_LH, dtype=bool)
    valid_full[idx_valid] = True
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=valid_full)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_perm, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)
    logger.info(f"[Figure 3B] inferential vertices (covered source-network with gradient): {int(valid.sum())}")

    # Figure 1 brain rows: both hemispheres, the four salience zoom regions, and the
    # target-network border traced on every map.
    net_outline = mask_outline(surf32k_lh_infl, surf32k_rh_infl, (df_yeo_surf['network'] == network).to_numpy())
    zoom_kw = dict(rois=salience_zoom_rois(df_yeo_surf, 'both'), surf_lh=surf32k_lh_infl,
                   surf_rh=surf32k_rh_infl, out_dir=out_dir, suffix=hemi, outline=net_outline)

    def _show_measure(template_map: np.ndarray, stem: str, title: str) -> tuple[float, float]:
        """Network-vertex map on both hemispheres, z-scored there, on the shared +/-2 SD scale."""
        values = _zscore_finite(mirror_to_cortex(template_map, df_yeo_surf, network))
        return _plot_measure_zoom(values, f"{prefix}_{stem}", title, cmap=ieeg_cmap.name,
                                  color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS,
                                  cbar_title='z', **zoom_kw)

    stats_rows, row_geometry = [], []

    def _add_stats(measure: str, r: float, p: float, r_null: np.ndarray) -> None:
        null_lo, null_med, null_hi = (np.percentile(r_null, [2.5, 50, 97.5]) if r_null.size
                                      else (np.nan, np.nan, np.nan))
        stats_rows.append(dict(measure=measure, r=r, p=p, q=np.nan, null_lo=null_lo, null_med=null_med,
                               null_hi=null_hi, n_vertices=int(valid.sum()), n_perm=n_perm))

    # ----- PRIMARY measure: aperiodic (1/f) exponent ---------------------------------
    # The aperiodic exponent is the theoretically grounded electrophysiological
    # correlate of cortical hierarchy / microstructure (Gao et al., 2020; Donoghue et
    # al., 2020): flatter spectra (low exponent) at differentiated, task-positive
    # apex regions, steeper spectra at the integrative pole.
    surf_exp = project_channel_measure(exp_ch[good], sens_good)
    x_exp = zscore(surf_exp[idx_valid])
    r_exp, p_exp, null_exp = moran_spearman(x_exp, y_valid, msr)
    _add_stats("exponent", r_exp, p_exp, null_exp)
    logger.info(f"[Figure 3B] aperiodic exponent vs MPC-gradient | Spearman r={r_exp:+.3f}, "
                f"Moran permutation p={p_exp:.3e} (n_perm={n_perm}, n_vertices={int(valid.sum())})")
    row_geometry.append(_show_measure(surf_exp, "exponent", "Aperiodic exponent"))

    # ----- SECONDARY measure: oscillatory band power (FDR-corrected across bands) -----
    # Per-band peak power from the same specparam fit (periodic component, orthogonal to
    # the aperiodic exponent). A vertex with no detected peak carries 0 oscillatory power.
    band_x, band_r, band_p = [], [], []
    for band, (lo, hi) in SPEC_BANDS.items():
        z = np.nan_to_num(spec['band_power'][band], nan=0.0)
        surf_b = project_channel_measure(z[good], sens_good)
        vals = surf_b[idx_valid]
        if np.nanstd(vals) > 0:
            x_b = zscore(vals)
            r_b, p_b, null_b = moran_spearman(x_b, y_valid, msr)
        else:
            x_b = np.zeros_like(vals)
            r_b, p_b, null_b = np.nan, np.nan, np.array([])
            logger.warning(f"[Figure 3B] Band {band}: no oscillatory-power variation across vertices; correlation undefined.")
        band_x.append(x_b); band_r.append(r_b); band_p.append(p_b)
        _add_stats(band, r_b, p_b, null_b)
        logger.info(f"[Figure 3B] Band {band}: oscillatory power vs MPC-gradient | Spearman r={r_b:+.3f}, "
                    f"Moran permutation p={p_b:.3e} (n_perm={n_perm}, n_vertices={int(valid.sum())})")
        row_geometry.append(_show_measure(surf_b, band, f"{band.capitalize()} peak power ({lo:g}–{hi:g} Hz)"))

    # FDR (Benjamini-Hochberg) across the five band tests.
    band_q = benjamini_hochberg(np.array(band_p))
    for band, q_b in zip(SPEC_BANDS, band_q):
        logger.info(f"[Figure 3B] Band {band}: FDR q={q_b:.3f}")

    # Test summary: CSV (one row per measure) + forest plot level with the inset rows.
    stats = pd.DataFrame(stats_rows)
    stats.loc[stats['measure'].isin(list(SPEC_BANDS)), 'q'] = band_q
    stats_path = project_root / f"data/dataframes/df_3b_spectral_stats_{hemi}.csv"
    stats.to_csv(stats_path, index=False)
    for row in stats.itertuples(index=False):
        logger.info(f"[Figure 3B] {row.measure}: Moran-null r 95% interval [{row.null_lo:+.3f}, {row.null_hi:+.3f}], "
                    f"median {row.null_med:+.3f}")
    logger.info(f"[Figure 3B] test summary saved to {stats_path}")
    if len(set(row_geometry)) != 1:
        raise RuntimeError(f"Inset rows differ in geometry {set(row_geometry)}; the forest rows cannot be aligned.")
    # Main forest: the tests with an uncorrected Moran p < 0.05 (the forest shows p_m; the
    # band FDR q stays in the CSV), level with their inset rows stacked in table order; the
    # complete forest (all six tests) is supplementary.
    significant = (stats['p'] < 0.05).to_numpy()
    logger.info(f"[Figure 3B] main forest rows (significant): {', '.join(stats.loc[significant, 'measure'])}")
    plot_forest(stats[significant], *row_geometry[0], out_dir / f"{prefix}_forest_{hemi}.svg")
    plot_forest(stats, *row_geometry[0], out_dir / f"{prefix}_forest_supp_{hemi}.svg")
    plt.close('all')

    # Spectra and flattened spectra (log10 power above each channel's aperiodic fit),
    # projected onto the inferential vertices and averaged within gradient quintiles.
    # A channel whose log spectrum or aperiodic fit is not finite over the range (e.g. a
    # knee below -1) would spread NaN to every vertex through the projection, so it is
    # left out of this display only (both panels share the remaining channels).
    fit = (f >= SPEC_FIT_RANGE[0]) & (f <= SPEC_FIT_RANGE[1])
    pxx_fit = pxx_raw[good][:, fit]
    with np.errstate(divide='ignore', invalid='ignore'):
        flat_ch = np.log10(pxx_fit) - aperiodic_component(f[fit], spec['offset'][good], spec['knee'][good], exp_ch[good])
    finite_ch = np.isfinite(flat_ch).all(axis=1)
    if not finite_ch.all():
        logger.warning(f"[Figure 3B] spectra panel: {int((~finite_ch).sum())}/{finite_ch.size} fitted channels with a "
                       f"non-finite spectrum or aperiodic fit over {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz left out.")
    sens_valid = sens_good[finite_ch][:, idx_valid]
    _plot_gradient_spectra(f[fit], project_channel_measure(pxx_fit[finite_ch].T, sens_valid),
                           project_channel_measure(flat_ch[finite_ch].T, sens_valid), y_valid,
                           out_dir / f"{prefix}_spectra_{hemi}.svg")

    # Sensitivity coverage (whole cortex + target network): total per-vertex sensitivity
    # from the fit channels, on both hemispheres, Purples over its 10-99th percentiles.
    surf_cov = np.where(sens_good_sum > 0, sens_good_sum, np.nan)
    for stem, net, title in (("coverage", None, "iEEG coverage"), ("coverage_sn", network, f"iEEG coverage ({network})")):
        values = mirror_to_cortex(surf_cov, df_yeo_surf, net)
        color_range = _pct_color_range(values, 10, 99)
        if color_range is None:
            logger.warning(f"[Figure 3B] {stem}: no coverage variation; map skipped.")
            continue
        _plot_measure_zoom(values, f"{prefix}_{stem}", title, cmap='Purples', color_range=color_range,
                           cbar_labels=tuple(f"{v:.2g}" for v in color_range), cbar_title=None, **zoom_kw)

    # ----- Supplementary: per-measure vertex scatter plots ----------------------------
    fig, axp = plt.subplots(figsize=(3.4, 3.2))
    slope_e, intercept_e = np.polyfit(x_exp, y_valid_z, 1)
    axp.scatter(x_exp, y_valid_z, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
    axp.set_xlim([-3, 3])
    axp.set_ylim([-3, 3])
    axp.plot(x_exp, slope_e * x_exp + intercept_e, c='black', lw=2.5)
    axp.text(0.05, 0.95, f"$r={r_exp:+.2f}$\n$p={format_pvalue(p_exp)}$", transform=axp.transAxes,
             va="top", fontweight="bold", fontsize=12)
    axp.set_xlabel('Aperiodic exponent', fontsize=16)
    axp.set_ylabel('MPC gradient', fontsize=16)
    axp.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axp.set_box_aspect(1)
    sns.despine(ax=axp)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_slope_corr_supp_{hemi}.svg", bbox_inches='tight')
    plt.close(fig)

    # Band scatter grid (per-panel size matched to figure 2a / figure 3c).
    fig, axes = plt.subplots(1, len(SPEC_BANDS), figsize=(3.0 * len(SPEC_BANDS), 3.2), sharex=True, sharey=True)
    for i, band in enumerate(SPEC_BANDS):
        slope_b, intercept_b = np.polyfit(band_x[i], y_valid_z, 1)
        axes[i].scatter(band_x[i], y_valid_z, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_xlim([-3, 3])
        axes[i].set_ylim([-3, 3])
        axes[i].plot(band_x[i], slope_b * band_x[i] + intercept_b, c=band_colors[i], lw=2.5)
        axes[i].text(0.05, 0.95, f"$r={band_r[i]:+.2f}$\n$q={format_pvalue(band_q[i])}$",
                     transform=axes[i].transAxes, va="top", fontweight="bold", fontsize=12)
        axes[i].set_xlabel(band.capitalize(), color=band_colors[i], fontsize=16)
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[i].set_box_aspect(1)
    axes[0].set_ylabel('MPC gradient', fontsize=16)
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_band_power_corr_supp_{hemi}.svg", bbox_inches='tight')
    plt.close(fig)


# Canonical Yeo network -> integer index matching the rows of `yeo7_rgb`
# (the same convention `figure_3_ieeg_mni` uses via `convert_states_str2int`).
_NET_NAMES = np.array(["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"])
_NET_TO_INT = {name: int(i) for name, i in zip(*[_NET_NAMES, convert_states_str2int(_NET_NAMES)[0].astype(int)])}


def salience_network_electrophysiological_similarity(
    pxx_raw: np.ndarray,
    sens: np.ndarray,
    surf32k_lh_infl,
    surf32k_rh_infl,
    df_yeo_surf: pd.DataFrame,
    project_root: Path,
    hemi: str = 'RH',
    network: str = 'SalVentAttn',
    n_perm: int = 1000,
    min_valid: int = 10,
    leak_cos_thresh: float = 0.1,
) -> None:
    """Spectral-similarity projection of the FC gradient (supplementary; group level).

    This is a **spectral-similarity** measure, not a connectivity measure: it quantifies
    how alike two vertices' power spectra are, not whether their signals are temporally
    coupled (cross-subject-averaged spectra cannot define connectivity). Each surface
    vertex carries a sensitivity-weighted PSD fingerprint; the spectral similarity (SS)
    between two vertices is the positive part of their PSD correlation. For each
    source-network (`network`) vertex i the projection score is the SS-weighted mean of
    the FC gradient across its non-network targets,

        P[i] = sum_j SS+_ij * g_FC[j] / sum_j SS+_ij .

    Because vertices sampled by overlapping leadfields share a sensitivity-averaged
    spectrum and are therefore trivially similar, source-target pairs whose sensitivity
    profiles overlap (cosine > `leak_cos_thresh`) are excluded as instrumental leakage.
    To show the spectral similarity carries information beyond geometry, the same
    projection is also computed with uniform and inverse-distance weights over the same
    target set, and the three correlations are reported together. The group statistic is
    a single Spearman(g_MPC[i], P[i]) across source vertices, with significance from the
    within-network Moran spatial null and the add-one empirical p. Coverage is the
    non-uniform iEEG electrode sampling, so both sources and targets are a convenience
    subset of cortex — a limitation noted in the Methods. All outputs are supplementary:
    the Figure-2a-style scatter (with the uniform and distance baseline correlations), the
    SS projection brain map (analysis hemisphere only, since P is defined on its source
    vertices) and the channel-level PSD correlation matrix (the spectral-similarity
    measure itself).
    """
    hemi_offset = N_LH if hemi == 'RH' else 0
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    df_hemi = df_yeo_surf.iloc[hemi_offset:hemi_offset + N_LH].reset_index(drop=True)
    net_labels = df_hemi['network'].values

    # Project channel PSD to surface vertices via sensitivity-weighted average
    # (pxx_raw and sens are computed once in main and shared with the band analysis).
    sens_sum = np.sum(sens, axis=0)  # (32492,)
    surf_psd = (pxx_raw.T @ sens) / (sens_sum + 1e-12)  # (n_freqs, 32492)
    covered = sens_sum > 0
    surf_psd[:, ~covered] = np.nan
    surf_psd_v = surf_psd.T  # (32492, n_freqs) vertex-level PSD

    # Z-score each vertex's PSD across frequencies; uncovered vertices -> 0 so their
    # similarity rows/columns vanish from the projection. The spectral similarity between
    # vertices i and j is then SS_ij = (z_i . z_j) / n_freqs (Pearson over frequencies).
    psd_mean = np.nanmean(surf_psd_v, axis=1, keepdims=True)
    psd_std = np.nanstd(surf_psd_v, axis=1, keepdims=True)
    surf_psd_z = np.where(covered[:, None], (surf_psd_v - psd_mean) / (psd_std + 1e-12), 0.0)
    n_freqs = surf_psd_z.shape[1]

    # Source = source-network vertices; targets = other cortical networks (FC-gradient
    # axis defined there). The FC gradient is oriented by anatomy (default-mode low).
    sal_mask = compute_network_mask(df_hemi, network, 'both')
    other_mask = (covered & (net_labels != network) & (net_labels != 'medial_wall')
                  & pd.notna(net_labels))
    grad = df_hemi[f't1_gradient1_{network}'].values.astype(float)

    # Use the project's own FC gradient (same source as figure 2) at fsLR-32k to match
    # this analysis's surface space, rather than brainspace's generic built-in gradient.
    fc_lh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-32k_lh.shape.gii").darrays[0].data
    fc_rh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-32k_rh.shape.gii").darrays[0].data
    fc_raw = np.concatenate([fc_lh, fc_rh])[hemi_offset:hemi_offset + N_LH]
    fc_g1_hemi = _orient_fc_gradient(fc_raw, net_labels, label="FC gradient")
    other_mask = other_mask & np.isfinite(fc_g1_hemi)
    logger.info(f"[Figure 3B] source {network} vertices: {sal_mask.sum()} "
                f"(covered+gradient: {(sal_mask & covered & np.isfinite(grad)).sum()}); "
                f"target vertices: {other_mask.sum()}")

    # Spectral-similarity-weighted projection of the FC gradient (source x target block).
    z_src = surf_psd_z[sal_mask]                         # (n_sal, n_freqs)
    z_tgt = surf_psd_z[other_mask]                       # (n_tgt, n_freqs)
    W_block = (z_src @ z_tgt.T) / n_freqs                # (n_sal, n_tgt) PSD correlations

    # Leakage mask: zero pairs whose sensitivity profiles overlap (shared electrodes),
    # because such vertices share a sensitivity-averaged spectrum and are trivially
    # similar for instrumental, not neural, reasons. Overlap = cosine of the two
    # vertices' sensitivity profiles.
    sens_src_n = sens[:, sal_mask] / (np.linalg.norm(sens[:, sal_mask], axis=0) + 1e-12)
    sens_tgt_n = sens[:, other_mask] / (np.linalg.norm(sens[:, other_mask], axis=0) + 1e-12)
    leak = (sens_src_n.T @ sens_tgt_n) > leak_cos_thresh
    W_block[leak] = 0.0
    logger.info(f"[Figure 3B] leakage mask removed {int(leak.sum())}/{leak.size} "
                f"source-target pairs (cosine > {leak_cos_thresh}).")

    W_pos = np.where(W_block > 0, W_block, 0.0)          # positive similarities only
    g_tgt = zscore(fc_g1_hemi[other_mask])               # FC gradient at targets, SD units

    # P[i] = sum_j w_ij g_tgt[j] / sum_j w_ij via the shared Figure-2 projection core
    # (weights here are already non-negative and finite, so its NaN/positive filtering
    # is a no-op).
    P = _weighted_mean_projection(W_pos, g_tgt, min_valid=min_valid)

    # Baselines on the same usable target set, isolating the spectral-similarity signal
    # beyond geometry: uniform weights and inverse-Euclidean-distance weights.
    usable = (W_pos > 0).astype(float)
    coords = np.asarray(surf_hemi.GetPoints())
    dist = cdist(coords[sal_mask], coords[other_mask])
    P_unif = _weighted_mean_projection(usable, g_tgt, min_valid=min_valid)
    P_dist = _weighted_mean_projection(np.where(usable > 0, 1.0 / (dist + 1e-6), 0.0), g_tgt, min_valid=min_valid)

    # Group statistic: Spearman(MPC gradient, SS projection) over source vertices.
    g_sal = grad[sal_mask]
    finite = np.isfinite(g_sal) & np.isfinite(P)
    r_group, _ = spearmanr(g_sal[finite], P[finite])

    def _spearman_vs_mpc(Pv: np.ndarray) -> float:
        f_ = np.isfinite(g_sal) & np.isfinite(Pv)
        return float(spearmanr(g_sal[f_], Pv[f_])[0]) if f_.sum() > 2 else np.nan
    r_unif, r_dist = _spearman_vs_mpc(P_unif), _spearman_vs_mpc(P_dist)
    logger.info(f"[Figure 3B] baselines vs MPC-gradient | uniform r={r_unif:+.3f}, "
                f"distance r={r_dist:+.3f} (spectral-similarity r={r_group:+.3f})")

    # Within-network Moran spatial null (single hemisphere -> one connected component).
    # Build the spatial graph on the finite source vertices only, so the surrogate
    # field carries the real gradient values: filling NaNs with 0 (the previous
    # approach) would inject artificial values into the spatial autocorrelation
    # structure and bias the surrogates at the retained vertices too.
    sal_finite_mask = np.zeros(N_LH, dtype=bool)
    sal_finite_mask[np.flatnonzero(sal_mask)[finite]] = True
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=sal_finite_mask)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_perm, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)
    r_null = np.array([spearmanr(surr, P[finite])[0]
                       for surr in msr.randomize(g_sal[finite])])
    p_moran = empirical_p_twosided(r_null, r_group)
    logger.info(f"[Figure 3B] spectral-similarity projection vs {network} MPC-gradient | "
                f"Spearman r_group={r_group:+.3f}, Moran permutation p={p_moran:.3e} "
                f"(n_perm={n_perm}, n_src={int(finite.sum())})")

    # Per-source-vertex dominant target network (scatter colour) and per-target-network
    # mean projection (lollipop), both from the same positive-similarity block W_pos.
    tgt_networks = net_labels[other_mask]
    tgt_net_list = [n for n in _NET_NAMES if (tgt_networks == n).any()]
    net_weight = np.column_stack([W_pos[:, tgt_networks == n].sum(axis=1) for n in tgt_net_list])
    has_weight = net_weight.sum(axis=1) > 0
    dominant_int = np.full(sal_mask.sum(), _NET_TO_INT[network])  # fallback: focus colour
    if has_weight.any():
        dom = np.argmax(net_weight[has_weight], axis=1)
        dominant_int[has_weight] = [_NET_TO_INT[tgt_net_list[d]] for d in dom]
    point_colors = yeo7_rgb[dominant_int]

    # Mean spectral-similarity projection per target network: the projection restricted to
    # that network's targets, averaged over source vertices (reuses W_pos; target cols change).
    P_t_mean = {}
    for net in tgt_net_list:
        cols = (tgt_networks == net)
        den_t = W_pos[:, cols].sum(axis=1)
        num_t = W_pos[:, cols] @ g_tgt[cols]
        P_t = np.where(den_t > 0, num_t / np.where(den_t > 0, den_t, np.nan), np.nan)
        P_t_mean[net] = float(np.nanmean(P_t)) if np.isfinite(P_t).any() else np.nan

    # Figure 2a-style layout: scatter (MPC gradient vs SS projection) + per-target-network
    # lollipop, both target-network coloured.
    P_z = zscore(P, nan_policy='omit')
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(7.0, 3.2),
        gridspec_kw={'wspace': 0.45, 'width_ratios': [1.0, 1.0]},
    )
    ax.scatter(g_sal[finite], P_z[finite], s=15, alpha=0.75,
               c=point_colors[finite], edgecolor='none', rasterized=True)
    sns.regplot(x=g_sal[finite], y=P_z[finite], scatter=False, color='black',
                line_kws={'linewidth': 2.5}, ax=ax)
    ax.text(0.05, 0.95, f"$r={r_group:+.2f}$\n$p={format_pvalue(p_moran)}$",
            transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
    # Geometry baselines on the same target set (uniform / inverse-distance weights).
    ax.text(0.97, 0.03, f"distance $r={r_dist:+.2f}$\nuniform $r={r_unif:+.2f}$",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='0.35')
    t = ax.set_title("Spectral similarity – iEEG power-spectrum fingerprint",
                     loc='left', pad=15)
    t.set_in_layout(False)
    ax.set_xlabel("MPC gradient")
    ax.set_ylabel("Spectral-similarity proj.")
    ax.set_ylim(-5, 5)
    ax.set_yticks([-4, 0, 4])
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_box_aspect(1)
    sns.despine(ax=ax)

    # Horizontal lollipop: one stem per target network, length = mean spectral-similarity
    # projection, coloured by target network, ordered by value.
    nets_sorted = sorted((n for n in tgt_net_list if np.isfinite(P_t_mean[n])),
                         key=lambda n: P_t_mean[n])
    for y, net in enumerate(nets_sorted):
        color = tuple(yeo7_rgb[_NET_TO_INT[net]])
        axl.hlines(y, 0, P_t_mean[net], colors=[color], lw=2.5)
        axl.scatter(P_t_mean[net], y, s=55, facecolors=[color], edgecolors=[color], zorder=3)
    axl.axvline(0, color='0.6', lw=1, zorder=0)
    axl.set_yticks(range(len(nets_sorted)))
    axl.set_yticklabels([yeo7_abbrev.get(n, n) for n in nets_sorted])
    for tick, net in zip(axl.get_yticklabels(), nets_sorted):
        tick.set_color(tuple(yeo7_rgb[_NET_TO_INT[net]]))
    axl.tick_params(axis='y', length=0)
    axl.set_xlabel("Mean spectral-similarity")
    axl.set_box_aspect(1)
    axl.spines['right'].set_visible(False)
    axl.spines['top'].set_visible(False)
    axl.spines['left'].set_visible(False)

    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_es_scatter_supp_{hemi}.svg",
                bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Channel-level PSD correlation matrix (the spectral-similarity measure), network-sorted.
    has_sens = sens.sum(axis=1) > 0
    peak_idx = np.argmax(sens, axis=1)  # (n_channels,)
    channel_networks = net_labels[peak_idx]
    # Exclude channels without sensitivity (argmax would return vertex 0 and borrow its
    # label), plus 'medial_wall' and NaN labels: a NaN slips past the string compare
    # and would make `channel_networks_valid` a mixed str/float object array, which
    # `np.argsort` below cannot order.
    valid_mask = has_sens & pd.notna(channel_networks) & (channel_networks != 'medial_wall')
    channel_networks_valid = channel_networks[valid_mask]
    logger.info(f"[Figure 3B] channel PSD correlation matrix: {int(valid_mask.sum())}/{valid_mask.size} channels "
                f"({int((~has_sens).sum())} without sensitivity excluded)")

    pxx_z = zscore(pxx_raw[valid_mask], axis=0)
    A_psd = np.corrcoef(pxx_z)

    sort_idx = np.argsort(channel_networks_valid)
    A_sorted = A_psd[sort_idx][:, sort_idx]
    sorted_networks = channel_networks_valid[sort_idx]

    boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0] + 1
    boundaries = np.insert(boundaries, 0, 0)
    b_ext = np.append(boundaries, len(channel_networks_valid))

    mpc_fig = A_sorted.copy()
    mpc_fig[np.tri(mpc_fig.shape[0], mpc_fig.shape[0]) == 1] = np.nan
    mpc_fig = rotate(mpc_fig, angle=-45, order=0, cval=np.nan)

    fig, ax = plt.subplots()
    for i, b in enumerate(boundaries):
        net_name = sorted_networks[b]
        color = yeo7_rgb[_NET_TO_INT.get(net_name, 7)]
        rect = patches.Rectangle(
            (len(channel_networks_valid) / 2 * np.sqrt(2), b * np.sqrt(2)),
            b_ext[i + 1] - b_ext[i], b_ext[i + 1] - b_ext[i],
            linewidth=2, edgecolor=color, facecolor='none', angle=45
        )
        ax.add_patch(rect)

    mpc_fig[mpc_fig > 1] = 1
    plt.imshow(mpc_fig, cmap='coolwarm', origin='upper')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_corr_supp_{hemi}.svg")
    plt.close()

    # Spectral-similarity projection brain map: z-scored P on the analysis hemisphere's
    # source vertices (the other hemisphere stays empty), Figure 1 zoom rows.
    p_map = np.full(2 * N_LH, np.nan)
    p_map[hemi_offset:hemi_offset + N_LH][sal_mask] = P_z
    _plot_measure_zoom(p_map, "figure_3b_ieeg_mica_es", "Spectral-similarity projection",
                       rois=salience_zoom_rois(df_yeo_surf, hemi), surf_lh=surf32k_lh_infl,
                       surf_rh=surf32k_rh_infl, out_dir=project_root / "results/figures",
                       suffix=f"supp_{hemi}", cmap=ieeg_cmap.name, color_range=SALIENCE_Z_RANGE,
                       cbar_labels=SALIENCE_Z_LABELS, cbar_title='z',
                       outline=mask_outline(surf32k_lh_infl, surf32k_rh_infl,
                                            (df_yeo_surf['network'] == network).to_numpy()))


def main():
    # Setup Relative Paths
    parser = get_parser()
    args = parser.parse_args()
    ieeg_deriv = args.ieeg_deriv
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_3_ieeg_mica", project_root, args)
    logger.info(f"Dataset        : MICA iEEG (BIDS_iEEG, sub-PX*, ses-01, stage-W wakefulness)")
    logger.info(f"Sensitivity maps: electroMICA leadfield (signed), fsLR-32k; per-hemi bipolar difference |L1-L2| -> area-normalised density thresholds (0.001 Vm/A absolute, 5% of the channel's second-largest density) -> LH+RH fold")
    logger.info(f"Preprocessing  : truncated to the shortest recording, Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned")
    logger.info(f"PSD            : Welch method, Hamming window 2s, overlap 1s, normalized to unit sum")
    logger.info(f"Spectral model : specparam, knee mode (knee >= 0), Gaussian peaks, {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz "
                f"(below the 60 Hz notch and the 80 Hz low-pass transition band); peak width 1-12 Hz, <= 6 peaks, "
                f"min height 0.05, threshold 2 SD")
    logger.info("Peak bands     : " + ", ".join(f"{b} {lo:g}-{hi:g} Hz" for b, (lo, hi) in SPEC_BANDS.items()))
    logger.info(f"Statistic      : aperiodic exponent (primary) + FDR-corrected band power vs MPC gradient; supplementary: spectral-similarity-weighted projection of the FC gradient (group level)")
    logger.info(f"Null model     : within-network Moran randomization (n_rep=1000, procedure=singleton, random_state=0), add-one empirical p")
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels; maps shown on both hemispheres (LH+RH folded template)")
    logger.info(f"Analysis network: {args.network}")

    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')

    ######### Part 1 -- T1 gradient (output of figure_1a_t1map.py)
    # As figure_2_distance.py: the hemisphere's own gradient table is preferred; a 'both'
    # table is a safe superset (the per-hemisphere network mask subsets it).
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        path_df_1a = project_root / 'data/dataframes/df_1a_both.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(f"No gradient dataframe (df_1a_{args.hemi}.tsv or df_1a_both.tsv) in "
                                f"{path_df_1a.parent}. Run figure_1a_t1map.py with -hemi {args.hemi} first.")
    logger.info(f"Gradient       : SN MPC gradient from {path_df_1a.name}; statistics on {args.hemi} network vertices")
    logger.info(f"Loading gradient dataframe from {path_df_1a}")
    df_yeo_surf = pd.read_csv(path_df_1a, sep="\t")

    # Load signed per-hemisphere contact sensitivities and per-subject vertex areas.
    df_sensitivity, sens_areas = load_sensitivity_info(root_dir=ieeg_deriv)
    logger.info(f"Sensitivity maps loaded: {df_sensitivity['Subject'].nunique()} subjects, {len(df_sensitivity)} contacts")

    # Per-channel Welch PSD of the raw recordings, computed ONCE (cached against the
    # recordings) and shared by the spectral (slope/band) and spectral-similarity analyses.
    ieeg_raw = args.ieeg_raw or str(Path(ieeg_deriv).parent.parent / 'original')
    logger.info(f"Raw recordings : {ieeg_raw}")
    cache_path = project_root / 'data/dataframes/figure_3_channel_psd.pkl'
    df_channel_data, f, pxx_raw = load_or_compute_channel_psd(ieeg_raw, cache_path, stage='W', **MICA_PSD_PARAMS)
    logger.info(f"Channel data: {df_channel_data['Subject'].nunique()} subjects, {len(df_channel_data)} bipolar channels")

    # Contact names index into the electroMICA leadfields (upper-cased there too).
    df_channel_data[['ContactName1', 'ContactName2']] = df_channel_data[['ContactName1', 'ContactName2']].apply(lambda c: c.str.upper())

    # electroMICA-faithful bipolar sensitivity (row-aligned with df_channel_data / pxx_raw):
    # per-hemisphere signed difference -> area-scaled thresholds -> abs -> LH+RH fold.
    sens = np.nan_to_num(build_bipolar_sensitivity(df_channel_data, df_sensitivity, sens_areas), nan=0.0)
    n_no_sens = int((sens.sum(axis=1) == 0).sum())
    if n_no_sens:
        logger.warning(f"{n_no_sens}/{len(sens)} channels have no suprathreshold sensitivity; "
                       f"they carry no weight in the surface projections.")
    logger.info(f"PSD computed once: {pxx_raw.shape[0]} channels x {pxx_raw.shape[1]} freqs; sensitivity stack {sens.shape}")

    # Primary: aperiodic-exponent (1/f slope) and secondary FDR band power vs MPC gradient.
    frequency_band_analysis_sensitivity(f, pxx_raw, sens, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi, network=args.network)

    # Supplementary spectral-similarity projection: regional power-spectrum similarity (NOT
    # connectivity) used to project the FC gradient and test it against the MPC gradient.
    salience_network_electrophysiological_similarity(pxx_raw, sens, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi, network=args.network)


if __name__ == "__main__":
    main()




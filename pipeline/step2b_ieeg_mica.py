#!/usr/bin/env python
"""Step 2b: iEEG spectra along the salience-network gradient, MICA dataset (Figure 2b).

Wakefulness recordings of the MICA iEEG dataset (electroMICA derivatives) are cut into
bipolar channels, their Welch spectra parameterised with specparam (aperiodic exponent
and per-band oscillatory peak power from one fit per channel), and each channel measure is
projected onto the fsLR-32k template through the contact sensitivity maps. The bipolar
sensitivities of both hemispheres are folded onto one template, so the maps are shown on
both hemispheres; the statistics run on one hemisphere (``hemisphere``) against its
values of the bilateral step 1a gradient. The primary test is the aperiodic exponent
against the gradient; the five band powers are secondary; each is tested against the
within-network Moran null with the add-one empirical p, uncorrected. A supplementary
analysis projects the FC gradient through the spectral similarity of the vertices.

Inputs   electroMICA derivatives and the original recordings (config/paths.yaml); step 1a
         fsLR-32k gradient (source_data).
Private  <work_dir>/step2b_ieeg_mica/: the channel spectra cache (patient IDs), log.
Public   source_data/step2b_ieeg_mica/: the template maps of every measure and of the
         coverage, the statistics and Moran nulls, the spectra averaged within gradient
         quintiles, the spectral-similarity projection and its statistics, manifest.
Figures  results/figures/figure_2b_ieeg_mica_*.svg. The two figures of individual channels
         (specparam fit quality, channel spectral-similarity matrix) are drawn by the compute
         stage, since their data are not public.

Run      python pipeline/step2b_ieeg_mica.py [--stage compute|plot|all]
"""
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from brainspace.datasets import load_conte69
from brainspace.mesh import mesh_elements
from scipy.ndimage import rotate
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, zscore

import common
from salience_switch import config
from salience_switch.colors import ieeg_cmap, yeo7_abbrev, yeo7_rgb
from salience_switch.connectivity import weighted_mean_projection
from salience_switch.ieeg import (MICA_PSD_PARAMS, SPEC_BANDS, SPEC_FIT_RANGE, aperiodic_component,
                                  build_bipolar_sensitivity, compute_spectral_parameters, fit_quality_summary,
                                  load_or_compute_channel_psd, load_sensitivity_info, mirror_to_cortex,
                                  project_channel_measure)
from salience_switch.plotting import (SALIENCE_Z_LABELS, SALIENCE_Z_RANGE, format_pvalue, mask_outline,
                                      plot_fit_quality, plot_hemispheres_zoom, plot_ridge, salience_zoom_rois)
from salience_switch.provenance import write_manifest
from salience_switch.stats import HemisphereMoranRandomization, empirical_p_twosided, moran_spearman
from salience_switch.surface import DATA_DIR, compute_network_mask, convert_states_str2int, load_inflated_surfaces

STEP = "step2b_ieeg_mica"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['text.usetex'] = False

# fsLR-32k vertices per hemisphere (LH and RH each carry this many).
N_LH = 32492
# Moran surrogates per test
N_PERMUTATIONS = 1000
PREFIX = "figure_2b_ieeg_mica"
MEASURES = ("exponent", *SPEC_BANDS)
BAND_SYMBOLS = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
BAND_COLORS = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']

# Figure 1 zoom-row window (figure_1a/1b/1c): 1450 x 300, zoom 1.3, colour-bar column,
# region names at 30 pt and inset-row titles at 36 pt; the rows are shown
# ZOOM_ROW_WIDTH_MM wide in the figure layout (~0.29x, so 30 pt reads as ~8.8 pt).
ZOOM_KW = dict(size=(1450, 300), zoom=1.3, color_bar=True, nan_color=(220, 220, 220, 1),
               label_fontsize=30, title_fontsize=36)

# Canonical Yeo network -> integer index matching the rows of `yeo7_rgb`
# (the same convention as step 2a via `convert_states_str2int`).
_NET_NAMES = np.array(["Cont", "Default", "DorsAttn", "Limbic", "SalVentAttn", "SomMot", "Vis"])
_NET_TO_INT = {name: int(i) for name, i in zip(*[_NET_NAMES, convert_states_str2int(_NET_NAMES)[0].astype(int)])}


def _orient_fc_gradient(fc_vals: np.ndarray, networks: np.ndarray, label: str = "FC gradient") -> np.ndarray:
    """Orient an FC gradient so the default-mode network sits at the low end.

    The diffusion-map eigenvector polarity is arbitrary, so the gradient is
    oriented by anatomy rather than a hardcoded sign (as ``surface.load_fc_gradient_32k``):
    it is flipped so the default-mode network occupies the low pole and the task-positive
    systems the high pole, matching the projection reading (high P = coupling to the
    task-positive end of the FC gradient). The chosen sign is logged so a change in the
    source gradient's polarity surfaces in the logs rather than silently inverting results.

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
        ridgeline-plot rows level with the stacked inset rows.
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


# ---------------------------------------------------------------------------
# Spectral measures versus the gradient
# ---------------------------------------------------------------------------
def gradient_bin_spectra(freq: np.ndarray, psd_v: np.ndarray, flat_v: np.ndarray, grad_v: np.ndarray,
                         n_bins: int = 5) -> pd.DataFrame:
    """Spectra and flattened spectra averaged within MPC-gradient quantile bins.

    The inferential vertices are split into ``n_bins`` equal-count bins of the gradient
    (quintiles by default), and each bin's mean sensitivity-weighted spectrum and mean
    flattened spectrum (log10 power above each channel's fitted aperiodic component) are
    taken. Means only: vertices share channels, so a vertex-level spread would be
    pseudo-replicated.

    Parameters
    ----------
    freq : np.ndarray, shape (n_freqs,)
    psd_v, flat_v : np.ndarray, shape (n_freqs, n_vertices)
        Per-vertex spectrum (linear power) and flattened spectrum (log10).
    grad_v : np.ndarray, shape (n_vertices,)
        MPC gradient at the same vertices.
    n_bins : int
        Number of equal-count gradient bins.

    Returns
    -------
    spectra : pd.DataFrame
        One row per bin and frequency: ``bin, gradient_median, n_vertices, frequency,
        psd, flat``.
    """
    bins = pd.qcut(grad_v, n_bins, labels=False)
    rows = []
    for k in range(n_bins):
        median = float(np.median(grad_v[bins == k]))
        logger.info(f"[Figure 2B] spectra bin {k + 1}/{n_bins}: {int((bins == k).sum())} vertices, "
                    f"gradient median {median:+.2f}")
        rows.append(pd.DataFrame({"bin": k + 1, "gradient_median": median, "n_vertices": int((bins == k).sum()),
                                  "frequency": freq, "psd": np.nanmean(psd_v[:, bins == k], axis=1),
                                  "flat": np.nanmean(flat_v[:, bins == k], axis=1)}))
    return pd.concat(rows, ignore_index=True)


def _plot_gradient_spectra(spectra: pd.DataFrame, path: Path) -> None:
    """Spectra and flattened spectra of each gradient quantile bin (`gradient_bin_spectra`).

    Each bin's mean spectrum (left, log-log) and mean flattened spectrum (right) are drawn
    in the coolwarm colour of the bin's median gradient value on the shared +/-2 SD scale,
    as the caps of the Figure 1a strata. Canonical bands are shaded alternately and labelled.
    """
    freq = spectra.loc[spectra["bin"] == 1, "frequency"].to_numpy()
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
        for k, rows in spectra.groupby("bin", sort=True):
            color = cmap(norm(rows["gradient_median"].iloc[0]))
            ax_psd.loglog(freq, rows["psd"].to_numpy(), color=color, lw=2.5, label=f"Q{k}")
            ax_flat.semilogx(freq, rows["flat"].to_numpy(), color=color, lw=2.5)
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


def spectral_statistics(f: np.ndarray, pxx_raw: np.ndarray, sens: np.ndarray, df_yeo_surf: pd.DataFrame,
                        hemi: str = 'RH', network: str = 'SalVentAttn',
                        n_perm: int = N_PERMUTATIONS) -> dict[str, pd.DataFrame]:
    """Sensitivity-weighted spectral measures versus the within-network MPC gradient.

    The per-channel spectra are parameterised once with ``specparam`` (one fit per
    channel) and projected onto the fsLR-32k template through the folded bipolar
    sensitivity maps. **Statistics** run on the analysis hemisphere ``hemi``: its
    source-network vertices that have coverage and a finite MPC gradient are correlated
    (Spearman) against that gradient. The **primary** measure is the aperiodic (1/f)
    exponent, the electrophysiological correlate of cortical hierarchy; the **secondary**
    measure is the oscillatory (periodic) peak power in five canonical bands, taken from
    the same fit so it is orthogonal to the exponent rather than re-encoding the 1/f
    change. Significance is the within-network Moran spatial null with the add-one
    empirical p (uncorrected), the graph fitted once on the (finite) inferential vertex
    set. The specparam fit-quality figure (worst, median and best fits) is drawn here,
    since it shows individual channels.

    Parameters
    ----------
    f : np.ndarray
        Welch frequency bins.
    pxx_raw : np.ndarray, shape (n_channels, n_freqs)
        Per-channel Welch PSD.
    sens : np.ndarray, shape (n_channels, N_LH)
        Stacked absolute bipolar sensitivity maps (NaN already zeroed).
    df_yeo_surf : pd.DataFrame
        Per-vertex surface table (both hemispheres) with the MPC gradient column.
    hemi : {'LH', 'RH'}
        Analysis hemisphere of the statistics.
    network : str
        Yeo 7-network used as the analysis target.
    n_perm : int
        Number of Moran surrogates.

    Returns
    -------
    tables : dict
        ``maps`` (per template vertex: coverage, inferential, one column per measure),
        ``stats`` (one row per measure: r, p, null_lo, null_med, null_hi, n_vertices,
        n_perm), ``null`` (measure, surrogate, r) and ``spectra`` (`gradient_bin_spectra`).
    """
    hemi_offset = N_LH if hemi == 'RH' else 0

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
        logger.warning(f"[Figure 2B] specparam fit failed for {int((~good).sum())}/{good.size} channels; excluded.")
    logger.info(f"[Figure 2B] specparam methods report:\n{spec['report']}")
    logger.info(f"[Figure 2B] fit quality: {fit_quality_summary(spec, SPEC_FIT_RANGE[0])}")
    # Fit check (specparam guidance): worst, median and best fits, no exclusion.
    plot_fit_quality(spec.pop('model'), config.FIGURES_DIR / f"{PREFIX}_specparam_qc_supp_{hemi}.svg",
                     title="wakefulness")
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
    idx_valid = np.flatnonzero(mask_hemi)[valid]  # template indices of the inferential vertices
    valid_full = np.zeros(N_LH, dtype=bool)
    valid_full[idx_valid] = True
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=valid_full)
    w.data **= -1
    msr = HemisphereMoranRandomization(np.full(idx_valid.size, hemi), n_perm, random_state=0).fit(w)
    logger.info(f"[Figure 2B] inferential vertices (covered source-network with gradient): {int(valid.sum())}")

    maps = pd.DataFrame({"vertex": np.arange(N_LH), "coverage": sens_good_sum, "inferential": valid_full})
    stats_rows, null_rows = [], []

    def _add_stats(measure: str, r: float, p: float, r_null: np.ndarray) -> None:
        null_lo, null_med, null_hi = (np.percentile(r_null, [2.5, 50, 97.5]) if r_null.size
                                      else (np.nan, np.nan, np.nan))
        stats_rows.append(dict(measure=measure, r=r, p=p, null_lo=null_lo, null_med=null_med,
                               null_hi=null_hi, n_vertices=int(valid.sum()), n_perm=n_perm))
        null_rows.append(pd.DataFrame({'measure': measure, 'surrogate': np.arange(r_null.size), 'r': r_null}))

    # ----- PRIMARY measure: aperiodic (1/f) exponent ---------------------------------
    # The aperiodic exponent is the theoretically grounded electrophysiological
    # correlate of cortical hierarchy / microstructure (Gao et al., 2020; Donoghue et
    # al., 2020): flatter spectra (low exponent) at differentiated, task-positive
    # apex regions, steeper spectra at the integrative pole.
    surf_exp = project_channel_measure(exp_ch[good], sens_good)
    maps["exponent"] = surf_exp
    x_exp = zscore(surf_exp[idx_valid])
    r_exp, p_exp, null_exp = moran_spearman(x_exp, y_valid, msr)
    _add_stats("exponent", r_exp, p_exp, null_exp)
    logger.info(f"[Figure 2B] aperiodic exponent vs MPC-gradient | Spearman r={r_exp:+.3f}, "
                f"Moran permutation p={p_exp:.3e} (n_perm={n_perm}, n_vertices={int(valid.sum())})")

    # ----- SECONDARY measure: oscillatory band power ---------------------------------
    # Per-band peak power from the same specparam fit (periodic component, orthogonal to
    # the aperiodic exponent). A vertex with no detected peak carries 0 oscillatory power.
    for band in SPEC_BANDS:
        z = np.nan_to_num(spec['band_power'][band], nan=0.0)
        surf_b = project_channel_measure(z[good], sens_good)
        maps[band] = surf_b
        vals = surf_b[idx_valid]
        if np.nanstd(vals) > 0:
            r_b, p_b, null_b = moran_spearman(zscore(vals), y_valid, msr)
        else:
            r_b, p_b, null_b = np.nan, np.nan, np.array([])
            logger.warning(f"[Figure 2B] Band {band}: no oscillatory-power variation across vertices; correlation undefined.")
        _add_stats(band, r_b, p_b, null_b)
        logger.info(f"[Figure 2B] Band {band}: oscillatory power vs MPC-gradient | Spearman r={r_b:+.3f}, "
                    f"Moran permutation p={p_b:.3e} (n_perm={n_perm}, n_vertices={int(valid.sum())})")

    stats = pd.DataFrame(stats_rows)
    for row in stats.itertuples(index=False):
        logger.info(f"[Figure 2B] {row.measure}: Moran-null r 95% interval [{row.null_lo:+.3f}, {row.null_hi:+.3f}], "
                    f"median {row.null_med:+.3f}")

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
        logger.warning(f"[Figure 2B] spectra panel: {int((~finite_ch).sum())}/{finite_ch.size} fitted channels with a "
                       f"non-finite spectrum or aperiodic fit over {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz left out.")
    sens_valid = sens_good[finite_ch][:, idx_valid]
    spectra = gradient_bin_spectra(f[fit], project_channel_measure(pxx_fit[finite_ch].T, sens_valid),
                                   project_channel_measure(flat_ch[finite_ch].T, sens_valid), y_valid)
    return {"maps": maps, "stats": stats, "null": pd.concat(null_rows, ignore_index=True), "spectra": spectra}


def plot_spectral_measures(tables: dict, df_yeo_surf: pd.DataFrame, hemi: str, network: str) -> None:
    """Figure 2b panels of the spectral measures, from the `spectral_statistics` tables.

    **Display** follows Figure 1: since the template stands for both hemispheres, each
    measure is shown on the network vertices of both hemispheres, z-scored there, on the
    shared +/-2 SD scale in ``CustomCmap_ieeg``, as a brain row with the
    ``SALIENCE_ZOOM_ROIS`` insets and as the inset row alone. A ridgeline plot of the
    Moran nulls with the observed r, level with the stacked inset rows, summarises the
    tests (main: the tests with an uncorrected p < 0.05; supplementary: all six), and the
    spectra and flattened spectra averaged within gradient quintiles show the data behind
    them. Coverage maps and the per-measure scatter plots (supplementary) are also drawn.
    """
    out_dir = config.FIGURES_DIR
    maps, stats, null = tables["maps"], tables["stats"], tables["null"]
    surf32k_lh_infl, surf32k_rh_infl = load_inflated_surfaces("fsLR-32k")
    hemi_offset = N_LH if hemi == 'RH' else 0
    grad_hemi = df_yeo_surf[f't1_gradient1_{network}'].values[hemi_offset:hemi_offset + N_LH]
    idx_valid = np.flatnonzero(maps["inferential"].to_numpy())
    y_valid_z = zscore(grad_hemi[idx_valid])

    # Figure 1 brain rows: both hemispheres, the four salience zoom regions, and the
    # target-network border traced on every map.
    net_outline = mask_outline(surf32k_lh_infl, surf32k_rh_infl, (df_yeo_surf['network'] == network).to_numpy())
    zoom_kw = dict(rois=salience_zoom_rois(df_yeo_surf, 'both'), surf_lh=surf32k_lh_infl,
                   surf_rh=surf32k_rh_infl, out_dir=out_dir, suffix=hemi, outline=net_outline)

    def _show_measure(template_map: np.ndarray, stem: str, title: str) -> tuple[float, float]:
        """Network-vertex map on both hemispheres, z-scored there, on the shared +/-2 SD scale."""
        values = _zscore_finite(mirror_to_cortex(template_map, df_yeo_surf, network))
        return _plot_measure_zoom(values, f"{PREFIX}_{stem}", title, cmap=ieeg_cmap.name,
                                  color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS,
                                  cbar_title='z', **zoom_kw)

    row_geometry = [_show_measure(maps["exponent"].to_numpy(), "exponent", "Aperiodic exponent")]
    for band, (lo, hi) in SPEC_BANDS.items():
        row_geometry.append(_show_measure(maps[band].to_numpy(), band,
                                          f"{band.capitalize()} peak power ({lo:g}–{hi:g} Hz)"))
    if len(set(row_geometry)) != 1:
        raise RuntimeError(f"Inset rows differ in geometry {set(row_geometry)}; the ridgeline rows cannot be aligned.")
    # Main ridgeline: the tests with an uncorrected Moran p < 0.05, level with their inset
    # rows stacked in table order; the complete ridgeline (all six tests) is supplementary.
    significant = (stats['p'] < 0.05).to_numpy()
    nulls = [null.loc[null['measure'] == m, 'r'].to_numpy() for m in stats['measure']]
    logger.info(f"[Figure 2B] main ridgeline rows (significant): {', '.join(stats.loc[significant, 'measure'])}")
    plot_ridge(stats[significant], [n for n, keep in zip(nulls, significant) if keep], *row_geometry[0],
               out_dir / f"{PREFIX}_forest_{hemi}.svg")
    plot_ridge(stats, nulls, *row_geometry[0], out_dir / f"{PREFIX}_forest_supp_{hemi}.svg")
    plt.close('all')

    _plot_gradient_spectra(tables["spectra"], out_dir / f"{PREFIX}_spectra_{hemi}.svg")

    # Sensitivity coverage (whole cortex + target network): total per-vertex sensitivity
    # from the fit channels, on both hemispheres, Purples over its 10-99th percentiles.
    coverage = maps["coverage"].to_numpy()
    surf_cov = np.where(coverage > 0, coverage, np.nan)
    for stem, net, title in (("coverage", None, "iEEG coverage"), ("coverage_sn", network, f"iEEG coverage ({network})")):
        values = mirror_to_cortex(surf_cov, df_yeo_surf, net)
        color_range = _pct_color_range(values, 10, 99)
        if color_range is None:
            logger.warning(f"[Figure 2B] {stem}: no coverage variation; map skipped.")
            continue
        _plot_measure_zoom(values, f"{PREFIX}_{stem}", title, cmap='Purples', color_range=color_range,
                           cbar_labels=tuple(f"{v:.2g}" for v in color_range), cbar_title=None, **zoom_kw)

    # ----- Supplementary: per-measure vertex scatter plots ----------------------------
    test = stats.set_index('measure')
    x_exp = zscore(maps["exponent"].to_numpy()[idx_valid])
    fig, axp = plt.subplots(figsize=(3.4, 3.2))
    slope_e, intercept_e = np.polyfit(x_exp, y_valid_z, 1)
    axp.scatter(x_exp, y_valid_z, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
    axp.set_xlim([-3, 3])
    axp.set_ylim([-3, 3])
    axp.plot(x_exp, slope_e * x_exp + intercept_e, c='black', lw=2.5)
    axp.text(0.05, 0.95, f"$r={test.loc['exponent', 'r']:+.2f}$\n$p={format_pvalue(test.loc['exponent', 'p'])}$",
             transform=axp.transAxes, va="top", fontweight="bold", fontsize=12)
    axp.set_xlabel('Aperiodic exponent', fontsize=16)
    axp.set_ylabel('MPC gradient', fontsize=16)
    axp.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axp.set_box_aspect(1)
    sns.despine(ax=axp)
    plt.tight_layout()
    plt.savefig(out_dir / f"{PREFIX}_slope_corr_supp_{hemi}.svg", bbox_inches='tight')
    plt.close(fig)

    # Band scatter grid (per-panel size matched to figure 3a / figure 2c).
    fig, axes = plt.subplots(1, len(SPEC_BANDS), figsize=(3.0 * len(SPEC_BANDS), 3.2), sharex=True, sharey=True)
    for i, band in enumerate(SPEC_BANDS):
        vals = maps[band].to_numpy()[idx_valid]
        x_b = zscore(vals) if np.nanstd(vals) > 0 else np.zeros_like(vals)
        slope_b, intercept_b = np.polyfit(x_b, y_valid_z, 1)
        axes[i].scatter(x_b, y_valid_z, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_xlim([-3, 3])
        axes[i].set_ylim([-3, 3])
        axes[i].plot(x_b, slope_b * x_b + intercept_b, c=BAND_COLORS[i], lw=2.5)
        axes[i].text(0.05, 0.95, f"$r={test.loc[band, 'r']:+.2f}$\n$p={format_pvalue(test.loc[band, 'p'])}$",
                     transform=axes[i].transAxes, va="top", fontweight="bold", fontsize=12)
        axes[i].set_xlabel(band.capitalize(), color=BAND_COLORS[i], fontsize=16)
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[i].set_box_aspect(1)
    axes[0].set_ylabel('MPC gradient', fontsize=16)
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(out_dir / f"{PREFIX}_band_power_corr_supp_{hemi}.svg", bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary: spectral-similarity projection of the FC gradient
# ---------------------------------------------------------------------------
def spectral_similarity_projection(
    pxx_raw: np.ndarray,
    sens: np.ndarray,
    df_yeo_surf: pd.DataFrame,
    hemi: str = 'RH',
    network: str = 'SalVentAttn',
    n_perm: int = N_PERMUTATIONS,
    min_valid: int = 10,
    leak_cos_thresh: float = 0.1,
) -> dict[str, pd.DataFrame]:
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
    subset of cortex. The channel-level PSD correlation matrix (the spectral-similarity
    measure itself) is drawn here, since it shows individual channels.

    Returns
    -------
    tables : dict
        ``vertices`` (per source vertex: vertex, g_mpc, P, dominant_network),
        ``targets`` (per target network: network, mean_projection) and ``stats``
        (r_group, p_moran, r_uniform, r_distance, n_sources, n_perm, ...).
    """
    hemi_offset = N_LH if hemi == 'RH' else 0
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    df_hemi = df_yeo_surf.iloc[hemi_offset:hemi_offset + N_LH].reset_index(drop=True)
    net_labels = df_hemi['network'].values

    # Project channel PSD to surface vertices via sensitivity-weighted average
    # (pxx_raw and sens are computed once and shared with the band analysis).
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

    # The FC gradient of figure 3 at its native fsLR-32k resolution, to match this
    # analysis's surface space: the bundled file is BrainSpace's conte69 template FC
    # gradient (brainspace.datasets.load_gradient('fc', idx=0)), identical to float32.
    fc_lh = nib.load(DATA_DIR / "parcellations/fc_gradient_fslr-32k_lh.shape.gii").darrays[0].data
    fc_rh = nib.load(DATA_DIR / "parcellations/fc_gradient_fslr-32k_rh.shape.gii").darrays[0].data
    fc_raw = np.concatenate([fc_lh, fc_rh])[hemi_offset:hemi_offset + N_LH]
    fc_g1_hemi = _orient_fc_gradient(fc_raw, net_labels, label="FC gradient")
    other_mask = other_mask & np.isfinite(fc_g1_hemi)
    logger.info(f"[Figure 2B] source {network} vertices: {sal_mask.sum()} "
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
    logger.info(f"[Figure 2B] leakage mask removed {int(leak.sum())}/{leak.size} "
                f"source-target pairs (cosine > {leak_cos_thresh}).")

    W_pos = np.where(W_block > 0, W_block, 0.0)          # positive similarities only
    g_tgt = zscore(fc_g1_hemi[other_mask])               # FC gradient at targets, SD units

    # P[i] = sum_j w_ij g_tgt[j] / sum_j w_ij via the shared Figure-3 projection core
    # (weights here are already non-negative and finite, so its NaN/positive filtering
    # is a no-op).
    P = weighted_mean_projection(W_pos, g_tgt, min_valid=min_valid)

    # Baselines on the same usable target set, isolating the spectral-similarity signal
    # beyond geometry: uniform weights and inverse-Euclidean-distance weights.
    usable = (W_pos > 0).astype(float)
    coords = np.asarray(surf_hemi.GetPoints())
    dist = cdist(coords[sal_mask], coords[other_mask])
    P_unif = weighted_mean_projection(usable, g_tgt, min_valid=min_valid)
    P_dist = weighted_mean_projection(np.where(usable > 0, 1.0 / (dist + 1e-6), 0.0), g_tgt, min_valid=min_valid)

    # Group statistic: Spearman(MPC gradient, SS projection) over source vertices.
    g_sal = grad[sal_mask]
    finite = np.isfinite(g_sal) & np.isfinite(P)
    r_group, _ = spearmanr(g_sal[finite], P[finite])

    def _spearman_vs_mpc(Pv: np.ndarray) -> float:
        f_ = np.isfinite(g_sal) & np.isfinite(Pv)
        return float(spearmanr(g_sal[f_], Pv[f_])[0]) if f_.sum() > 2 else np.nan
    r_unif, r_dist = _spearman_vs_mpc(P_unif), _spearman_vs_mpc(P_dist)
    logger.info(f"[Figure 2B] baselines vs MPC-gradient | uniform r={r_unif:+.3f}, "
                f"distance r={r_dist:+.3f} (spectral-similarity r={r_group:+.3f})")

    # Within-network Moran spatial null (single hemisphere -> one connected component).
    # Build the spatial graph on the finite source vertices only, so the surrogate
    # field carries the real gradient values.
    sal_finite_mask = np.zeros(N_LH, dtype=bool)
    sal_finite_mask[np.flatnonzero(sal_mask)[finite]] = True
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=sal_finite_mask)
    w.data **= -1
    msr = HemisphereMoranRandomization(np.full(int(finite.sum()), hemi), n_perm, random_state=0).fit(w)
    r_null = np.array([spearmanr(surr, P[finite])[0]
                       for surr in msr.randomize(g_sal[finite])])
    p_moran = empirical_p_twosided(r_null, r_group)
    logger.info(f"[Figure 2B] spectral-similarity projection vs {network} MPC-gradient | "
                f"Spearman r_group={r_group:+.3f}, Moran permutation p={p_moran:.3e} "
                f"(n_perm={n_perm}, n_src={int(finite.sum())})")

    # Per-source-vertex dominant target network (scatter colour) and per-target-network
    # mean projection (lollipop), both from the same positive-similarity block W_pos.
    tgt_networks = net_labels[other_mask]
    tgt_net_list = [n for n in _NET_NAMES if (tgt_networks == n).any()]
    net_weight = np.column_stack([W_pos[:, tgt_networks == n].sum(axis=1) for n in tgt_net_list])
    has_weight = net_weight.sum(axis=1) > 0
    dominant = np.full(sal_mask.sum(), network, dtype=object)  # fallback: focus network
    if has_weight.any():
        dom = np.argmax(net_weight[has_weight], axis=1)
        dominant[has_weight] = [tgt_net_list[d] for d in dom]

    # Mean spectral-similarity projection per target network: the projection restricted to
    # that network's targets, averaged over source vertices (reuses W_pos; target cols change).
    P_t_mean = {}
    for net in tgt_net_list:
        cols = (tgt_networks == net)
        den_t = W_pos[:, cols].sum(axis=1)
        num_t = W_pos[:, cols] @ g_tgt[cols]
        P_t = np.where(den_t > 0, num_t / np.where(den_t > 0, den_t, np.nan), np.nan)
        P_t_mean[net] = float(np.nanmean(P_t)) if np.isfinite(P_t).any() else np.nan

    _plot_channel_similarity(pxx_raw, sens, net_labels, hemi)
    vertices = pd.DataFrame({"vertex": np.flatnonzero(sal_mask), "g_mpc": g_sal, "P": P, "dominant_network": dominant})
    targets = pd.DataFrame({"network": tgt_net_list, "mean_projection": [P_t_mean[n] for n in tgt_net_list]})
    stats = pd.DataFrame([{"network": network, "hemisphere": hemi, "r_group": float(r_group), "p_moran": p_moran,
                           "r_uniform": r_unif, "r_distance": r_dist, "n_sources": int(finite.sum()),
                           "n_perm": n_perm, "leak_cos_thresh": leak_cos_thresh,
                           "n_pairs_removed": int(leak.sum()), "n_pairs": int(leak.size)}])
    return {"vertices": vertices, "targets": targets, "stats": stats}


def _plot_channel_similarity(pxx_raw: np.ndarray, sens: np.ndarray, net_labels: np.ndarray, hemi: str) -> None:
    """Channel-level PSD correlation matrix (the spectral-similarity measure), network-sorted."""
    has_sens = sens.sum(axis=1) > 0
    peak_idx = np.argmax(sens, axis=1)  # (n_channels,)
    channel_networks = net_labels[peak_idx]
    # Exclude channels without sensitivity (argmax would return vertex 0 and borrow its
    # label), plus 'medial_wall' and NaN labels: a NaN slips past the string compare
    # and would make `channel_networks_valid` a mixed str/float object array, which
    # `np.argsort` below cannot order.
    valid_mask = has_sens & pd.notna(channel_networks) & (channel_networks != 'medial_wall')
    channel_networks_valid = channel_networks[valid_mask]
    logger.info(f"[Figure 2B] channel PSD correlation matrix: {int(valid_mask.sum())}/{valid_mask.size} channels "
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
    plt.savefig(config.FIGURES_DIR / f"{PREFIX}_corr_supp_{hemi}.svg")
    plt.close()


def plot_spectral_similarity(tables: dict, df_yeo_surf: pd.DataFrame, hemi: str, network: str) -> None:
    """Supplementary spectral-similarity panels: scatter + target-network lollipop, and the map."""
    vertices, targets, stats = tables["vertices"], tables["targets"], tables["stats"].iloc[0]
    hemi_offset = N_LH if hemi == 'RH' else 0
    g_sal, P = vertices["g_mpc"].to_numpy(dtype=float), vertices["P"].to_numpy(dtype=float)
    finite = np.isfinite(g_sal) & np.isfinite(P)
    point_colors = yeo7_rgb[np.array([_NET_TO_INT[n] for n in vertices["dominant_network"]])]
    P_t_mean = dict(zip(targets["network"], targets["mean_projection"]))

    # Figure 3a-style layout: scatter (MPC gradient vs SS projection) + per-target-network
    # lollipop, both target-network coloured.
    P_z = zscore(P, nan_policy='omit')
    fig, (ax, axl) = plt.subplots(
        1, 2, figsize=(7.0, 3.2),
        gridspec_kw={'wspace': 0.45, 'width_ratios': [1.0, 1.0]},
    )
    ax.scatter(g_sal[finite], P_z[finite], s=15, alpha=0.75,
               c=point_colors[finite], edgecolor='none', rasterized=True)
    sns.regplot(x=g_sal[finite], y=P_z[finite], scatter=False, color='black',
                line_kws={'linewidth': 2.5}, ax=ax, seed=0)
    ax.text(0.05, 0.95, f"$r={stats['r_group']:+.2f}$\n$p={format_pvalue(stats['p_moran'])}$",
            transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
    # Geometry baselines on the same target set (uniform / inverse-distance weights).
    ax.text(0.97, 0.03, f"distance $r={stats['r_distance']:+.2f}$\nuniform $r={stats['r_uniform']:+.2f}$",
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
    nets_sorted = sorted((n for n in targets["network"] if np.isfinite(P_t_mean[n])),
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

    plt.savefig(config.FIGURES_DIR / f"{PREFIX}_es_scatter_supp_{hemi}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Spectral-similarity projection brain map: z-scored P on the analysis hemisphere's
    # source vertices (the other hemisphere stays empty), Figure 1 zoom rows.
    surf32k_lh_infl, surf32k_rh_infl = load_inflated_surfaces("fsLR-32k")
    p_map = np.full(2 * N_LH, np.nan)
    p_map[hemi_offset + vertices["vertex"].to_numpy()] = P_z
    _plot_measure_zoom(p_map, f"{PREFIX}_es", "Spectral-similarity projection",
                       rois=salience_zoom_rois(df_yeo_surf, hemi), surf_lh=surf32k_lh_infl,
                       surf_rh=surf32k_rh_infl, out_dir=config.FIGURES_DIR,
                       suffix=f"supp_{hemi}", cmap=ieeg_cmap.name, color_range=SALIENCE_Z_RANGE,
                       cbar_labels=SALIENCE_Z_LABELS, cbar_title='z',
                       outline=mask_outline(surf32k_lh_infl, surf32k_rh_infl,
                                            (df_yeo_surf['network'] == network).to_numpy()))


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def load_channel_spectra(paths: dict, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-channel Welch spectra (stage W) and their bipolar sensitivity maps.

    Returns the frequency bins, the (n_channels, n_freqs) PSD, the (n_channels, N_LH)
    absolute bipolar sensitivity maps (NaN zeroed), row-aligned, and the number of patients.
    """
    df_sensitivity, sens_areas = load_sensitivity_info(root_dir=str(paths["electromica"]))
    logger.info(f"Sensitivity maps loaded: {df_sensitivity['Subject'].nunique()} subjects, {len(df_sensitivity)} contacts")
    logger.info(f"Raw recordings : {paths['ieeg_raw']}")
    df_channel_data, f, pxx_raw = load_or_compute_channel_psd(str(paths["ieeg_raw"]), cache_dir / "channel_psd_W.pkl",
                                                              stage='W', **MICA_PSD_PARAMS)
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
    return f, pxx_raw, sens, df_channel_data['Subject'].nunique()


def compute(settings: dict, paths: dict) -> None:
    """Spectral measures and the spectral-similarity projection, with their Moran tests."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    logger.info("Dataset        : MICA iEEG (BIDS_iEEG, sub-PX*, ses-01, stage-W wakefulness)")
    logger.info("Sensitivity maps: electroMICA leadfield (signed), fsLR-32k; per-hemi bipolar difference |L1-L2| -> area-normalised density thresholds (0.001 Vm/A absolute, 5% of the channel's second-largest density) -> LH+RH fold")
    logger.info("Preprocessing  : truncated to the shortest recording, Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned")
    logger.info("PSD            : Welch method, Hamming window 2s, overlap 1s, normalized to unit sum")
    logger.info(f"Spectral model : specparam, knee mode (knee >= 0), Gaussian peaks, {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz "
                f"(below the 60 Hz notch and the 80 Hz low-pass transition band); peak width 1-12 Hz, <= 6 peaks, "
                f"min height 0.05, threshold 2 SD")
    logger.info("Peak bands     : " + ", ".join(f"{b} {lo:g}-{hi:g} Hz" for b, (lo, hi) in SPEC_BANDS.items()))
    logger.info("Statistic      : aperiodic exponent (primary) + band power (secondary) vs MPC gradient, uncorrected; supplementary: spectral-similarity-weighted projection of the FC gradient (group level)")
    logger.info(f"Null model     : within-network Moran randomization (n_rep={N_PERMUTATIONS}, procedure=singleton, random_state=0), add-one empirical p")
    logger.info("Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels; maps shown on both hemispheres (LH+RH folded template)")
    logger.info(f"Analysis network: {network}; statistics on the {hemi} network vertices of the bilateral step 1a gradient")
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    f, pxx_raw, sens, n_patients = load_channel_spectra(paths, config.work_dir(STEP, paths))

    spectral = spectral_statistics(f, pxx_raw, sens, df_yeo_surf, hemi=hemi, network=network)
    similarity = spectral_similarity_projection(pxx_raw, sens, df_yeo_surf, hemi=hemi, network=network)
    for name, table in spectral.items():
        config.write_table(table, source / f"spectral_{name}_{hemi}.tsv")
    for name, table in similarity.items():
        config.write_table(table, source / f"similarity_{name}_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network, hemisphere=hemi,
                   n_patients=int(n_patients), n_channels=int(pxx_raw.shape[0]), n_permutations=N_PERMUTATIONS)


def plot(settings: dict) -> None:
    """Figure 2b panels and supplements, from source_data only."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    spectral = {name: config.read_table(source / f"spectral_{name}_{hemi}.tsv")
                for name in ("maps", "stats", "null", "spectra")}
    similarity = {name: config.read_table(source / f"similarity_{name}_{hemi}.tsv")
                  for name in ("vertices", "targets", "stats")}
    plot_spectral_measures(spectral, df_yeo_surf, hemi, network)
    plot_spectral_similarity(similarity, df_yeo_surf, hemi, network)


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 2b: iEEG spectra along the salience-network gradient (Figure 2b).")

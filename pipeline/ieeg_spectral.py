"""Shared by steps 2a and 2b: the iEEG spectral measures along the gradient and their figures.

Both cohorts run the same analysis once their channels are on the folded fsLR-32k
template: the per-channel spectra (n_channels, n_freqs) and the channel sensitivity maps
(n_channels, N_LH) are all it needs. Step 2b (MICA) builds the sensitivities from the
electroMICA leadfields, step 2a (MNI Open iEEG Atlas) from Gaussian footprints around the
channel positions.

``spectral_statistics`` parameterises every channel spectrum with specparam, projects the
aperiodic exponent (primary) and the peak power of five bands (secondary) onto the
template, and tests each against the network gradient of one hemisphere (within-network
Moran null, add-one empirical p, uncorrected). ``spectral_similarity_projection`` is the
supplementary projection of the FC gradient through the spectral similarity of the
vertices. ``plot_spectral_measures`` and ``plot_spectral_similarity`` draw their panels
from the public tables. Every function takes the panel stem ``prefix`` and the log tag
``tag`` of its step.
"""
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from brainspace.datasets import load_conte69
from scipy.ndimage import rotate
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, zscore

from salience_switch import config
from salience_switch.colors import ieeg_cmap, network_rgb, yeo7_abbrev
from salience_switch.connectivity import weighted_mean_projection
from salience_switch.ieeg import (N_LH, SPEC_BANDS, SPEC_FIT_RANGE, aperiodic_component, compute_spectral_parameters,
                                  covered_vertices, fit_quality_summary, hemisphere_slice, mirror_to_cortex,
                                  project_channel_measure)
from salience_switch.plotting import (SALIENCE_Z_LABELS, SALIENCE_Z_RANGE, ZOOM_ROW_KW, format_pvalue, mask_outline,
                                      plot_fit_quality, plot_ridge, plot_zoom_rows, salience_zoom_rois,
                                      shared_row_geometry)
from salience_switch.stats import empirical_p_twosided, fit_surface_moran, moran_spearman, null_interval
from salience_switch.surface import compute_network_mask, load_fc_gradient_32k, load_inflated_surfaces

logger = logging.getLogger("ieeg_spectral")

# Moran surrogates per test
N_PERMUTATIONS = 1000
MEASURES = ("exponent", *SPEC_BANDS)
BAND_SYMBOLS = {"delta": "δ", "theta": "θ", "alpha": "α", "beta": "β", "gamma": "γ"}
BAND_COLORS = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']

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


# ---------------------------------------------------------------------------
# Spectral measures versus the gradient
# ---------------------------------------------------------------------------
def gradient_bin_spectra(freq: np.ndarray, psd_v: np.ndarray, flat_v: np.ndarray, grad_v: np.ndarray,
                         n_bins: int = 5, *, tag: str) -> pd.DataFrame:
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
    tag : str
        Log tag of the step, e.g. ``'Figure 2B'``.

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
        logger.info(f"[{tag}] spectra bin {k + 1}/{n_bins}: {int((bins == k).sum())} vertices, "
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
                        hemi: str = 'RH', network: str = 'SalVentAttn', n_perm: int = N_PERMUTATIONS, *,
                        prefix: str, tag: str) -> dict[str, pd.DataFrame]:
    """Sensitivity-weighted spectral measures versus the within-network MPC gradient.

    The per-channel spectra are parameterised once with ``specparam`` (one fit per
    channel) and projected onto the fsLR-32k template through the folded channel
    sensitivity maps. **Statistics** run on the analysis hemisphere ``hemi``: its
    source-network vertices that have coverage and a finite MPC gradient are correlated
    (Spearman) against that gradient. The **primary** measure is the aperiodic (1/f)
    exponent; the **secondary** measure is the oscillatory (periodic) peak power in five
    canonical bands, taken from the same fit and measured relative to its aperiodic
    component. Significance is the within-network Moran spatial null with the add-one
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
        Non-negative channel sensitivity maps on the folded template (NaN already zeroed).
    df_yeo_surf : pd.DataFrame
        Per-vertex surface table (both hemispheres) with the MPC gradient column.
    hemi : {'LH', 'RH'}
        Analysis hemisphere of the statistics.
    network : str
        Yeo 7-network used as the analysis target.
    n_perm : int
        Number of Moran surrogates.
    prefix : str
        Panel file stem of the step, e.g. ``'figure_2b_ieeg_mica'``.
    tag : str
        Log tag of the step, e.g. ``'Figure 2B'``.

    Returns
    -------
    tables : dict
        ``maps`` (per template vertex: coverage, inferential, one column per measure),
        ``stats`` (one row per measure: r, p, null_lo, null_med, null_hi, n_vertices,
        n_perm), ``null`` (measure, surrogate, r) and ``spectra`` (`gradient_bin_spectra`).
    """
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    hemi_rows = hemisphere_slice(hemi)
    mask_hemi = compute_network_mask(df_yeo_surf, network, hemi)[hemi_rows]
    grad_hemi = df_yeo_surf[f't1_gradient1_{network}'].to_numpy()[hemi_rows]

    # Parameterise every channel spectrum ONCE (aperiodic exponent + oscillatory band
    # power from the same fit), then drop channels whose specparam fit failed to
    # converge so both measures and the coverage map share one successfully-fit set.
    spec = compute_spectral_parameters(pxx_raw, f, bands=SPEC_BANDS, fmin=SPEC_FIT_RANGE[0], fmax=SPEC_FIT_RANGE[1],
                                       return_model=True)
    exp_ch = spec['exponent']
    good = np.isfinite(exp_ch)
    if not good.all():
        logger.warning(f"[{tag}] specparam fit failed for {int((~good).sum())}/{good.size} channels; excluded.")
    logger.info(f"[{tag}] specparam methods report:\n{spec['report']}")
    logger.info(f"[{tag}] fit quality: {fit_quality_summary(spec, SPEC_FIT_RANGE[0])}")
    # Fit check (specparam guidance): worst, median and best fits, no exclusion.
    plot_fit_quality(spec.pop('model'), config.FIGURES_DIR / f"{prefix}_specparam_qc_supp_{hemi}.svg",
                     title="wakefulness")
    sens_good = sens[good]

    # Inferential vertex set: source-network vertices with iEEG coverage (from the
    # fit channels) AND a finite gradient value. The Moran null is fitted once on this
    # finite subset's geometry (no NaN/zero fill), keeping the surrogate field free of
    # artificial values.
    idx_valid = covered_vertices(sens_good, mask_hemi, grad_hemi)  # template indices
    y_valid = grad_hemi[idx_valid]
    valid_full = np.zeros(N_LH, dtype=bool)
    valid_full[idx_valid] = True
    msr = fit_surface_moran(surf_hemi, valid_full, np.full(idx_valid.size, hemi), n_perm)
    sens_good_sum = np.sum(sens_good, axis=0)
    logger.info(f"[{tag}] inferential vertices (covered source-network with gradient): {idx_valid.size}")

    maps = pd.DataFrame({"vertex": np.arange(N_LH), "coverage": sens_good_sum, "inferential": valid_full})
    stats_rows, null_rows = [], []

    def _add_stats(measure: str, r: float, p: float, r_null: np.ndarray) -> None:
        null_lo, null_med, null_hi = null_interval(r_null)
        stats_rows.append(dict(measure=measure, r=r, p=p, null_lo=null_lo, null_med=null_med,
                               null_hi=null_hi, n_vertices=idx_valid.size, n_perm=n_perm))
        null_rows.append(pd.DataFrame({'measure': measure, 'surrogate': np.arange(r_null.size), 'r': r_null}))

    # ----- PRIMARY measure: aperiodic (1/f) exponent ---------------------------------
    # Related to the balance of synaptic excitation and inhibition (Gao et al., 2017).
    surf_exp = project_channel_measure(exp_ch[good], sens_good)
    maps["exponent"] = surf_exp
    x_exp = zscore(surf_exp[idx_valid])
    r_exp, p_exp, null_exp = moran_spearman(x_exp, y_valid, msr)
    _add_stats("exponent", r_exp, p_exp, null_exp)
    logger.info(f"[{tag}] aperiodic exponent vs MPC-gradient | Spearman r={r_exp:+.3f}, "
                f"Moran permutation p={p_exp:.3e} (n_perm={n_perm}, n_vertices={idx_valid.size})")

    # ----- SECONDARY measure: oscillatory band power ---------------------------------
    # Per-band peak power from the same specparam fit (periodic component, measured above
    # the aperiodic fit). A vertex with no detected peak carries 0 oscillatory power.
    for band in SPEC_BANDS:
        z = np.nan_to_num(spec['band_power'][band], nan=0.0)
        surf_b = project_channel_measure(z[good], sens_good)
        maps[band] = surf_b
        vals = surf_b[idx_valid]
        if np.nanstd(vals) > 0:
            r_b, p_b, null_b = moran_spearman(zscore(vals), y_valid, msr)
        else:
            r_b, p_b, null_b = np.nan, np.nan, np.array([])
            logger.warning(f"[{tag}] Band {band}: no oscillatory-power variation across vertices; correlation undefined.")
        _add_stats(band, r_b, p_b, null_b)
        logger.info(f"[{tag}] Band {band}: oscillatory power vs MPC-gradient | Spearman r={r_b:+.3f}, "
                    f"Moran permutation p={p_b:.3e} (n_perm={n_perm}, n_vertices={idx_valid.size})")

    stats = pd.DataFrame(stats_rows)
    for row in stats.itertuples(index=False):
        logger.info(f"[{tag}] {row.measure}: Moran-null r 95% interval [{row.null_lo:+.3f}, {row.null_hi:+.3f}], "
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
        logger.warning(f"[{tag}] spectra panel: {int((~finite_ch).sum())}/{finite_ch.size} fitted channels with a "
                       f"non-finite spectrum or aperiodic fit over {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz left out.")
    sens_valid = sens_good[finite_ch][:, idx_valid]
    spectra = gradient_bin_spectra(f[fit], project_channel_measure(pxx_fit[finite_ch].T, sens_valid),
                                   project_channel_measure(flat_ch[finite_ch].T, sens_valid), y_valid, tag=tag)
    return {"maps": maps, "stats": stats, "null": pd.concat(null_rows, ignore_index=True), "spectra": spectra}


def plot_spectral_measures(tables: dict, df_yeo_surf: pd.DataFrame, hemi: str, network: str, *,
                           prefix: str, tag: str, main_measures: tuple[str, ...] | None = None) -> None:
    """Panels of the spectral measures, from the `spectral_statistics` tables.

    **Display** follows Figure 1: since the template stands for both hemispheres, each
    measure is shown on the network vertices of both hemispheres, z-scored there, on the
    shared +/-2 SD scale in ``CustomCmap_ieeg``, as a brain row with the
    ``SALIENCE_ZOOM_ROIS`` insets and as the inset row alone. A ridgeline plot of the
    Moran nulls with the observed r, level with the stacked inset rows, summarises the
    tests (main: ``main_measures``, by default the tests with an uncorrected p < 0.05,
    not drawn if there is none; supplementary: all six), and the spectra and flattened
    spectra averaged within gradient quintiles show the data behind them. Coverage maps
    and the per-measure scatter plots (supplementary) are also drawn. Panels are named
    ``<prefix>_*``.

    Parameters
    ----------
    tables : dict
        ``maps``, ``stats``, ``null`` and ``spectra`` tables of `spectral_statistics`.
    df_yeo_surf : pd.DataFrame
        Per-vertex surface table (both hemispheres) with the MPC gradient column.
    hemi : {'LH', 'RH'}
        Analysis hemisphere of the statistics.
    network : str
        Yeo 7-network of the analysis.
    prefix : str
        Panel file stem of the step, e.g. ``'figure_2b_ieeg_mica'``.
    tag : str
        Log tag of the step, e.g. ``'Figure 2B'``.
    main_measures : tuple of str, optional
        Measures of the main ridgeline, drawn in table order whatever their p (e.g. the
        rows of another figure's main panel); ``None`` selects the tests with an
        uncorrected p < 0.05.

    Raises
    ------
    ValueError
        If a measure of ``main_measures`` is not in the statistics table.
    """
    out_dir = config.FIGURES_DIR
    maps, stats, null = tables["maps"], tables["stats"], tables["null"]
    surf32k_lh_infl, surf32k_rh_infl = load_inflated_surfaces("fsLR-32k")
    grad_hemi = df_yeo_surf[f't1_gradient1_{network}'].to_numpy()[hemisphere_slice(hemi)]
    idx_valid = np.flatnonzero(maps["inferential"].to_numpy())
    y_valid_z = zscore(grad_hemi[idx_valid])

    # Figure 1 brain rows: both hemispheres, the four salience zoom regions, and the
    # target-network border traced on every map.
    net_outline = mask_outline(surf32k_lh_infl, surf32k_rh_infl, (df_yeo_surf['network'] == network).to_numpy())
    zoom_kw = dict(ZOOM_ROW_KW, rois=salience_zoom_rois(df_yeo_surf, 'both'), surf_lh=surf32k_lh_infl,
                   surf_rh=surf32k_rh_infl, out_dir=out_dir, suffix=hemi, outline=net_outline)

    def _show_measure(template_map: np.ndarray, stem: str, title: str) -> tuple[float, float]:
        """Network-vertex map on both hemispheres, z-scored there, on the shared +/-2 SD scale."""
        values = _zscore_finite(mirror_to_cortex(template_map, df_yeo_surf, network))
        return plot_zoom_rows(values=values, stem=f"{prefix}_{stem}", title=title, cmap=ieeg_cmap.name,
                              color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS, cbar_title='z', **zoom_kw)

    row_geometry = [_show_measure(maps["exponent"].to_numpy(), "exponent", "Aperiodic exponent")]
    for band, (lo, hi) in SPEC_BANDS.items():
        row_geometry.append(_show_measure(maps[band].to_numpy(), band,
                                          f"{band.capitalize()} peak power ({lo:g}–{hi:g} Hz)"))
    geometry = shared_row_geometry(row_geometry)
    # Main ridgeline: the tests with an uncorrected Moran p < 0.05 (or `main_measures`),
    # level with their inset rows stacked in table order; the complete ridgeline (all six
    # tests) is supplementary.
    if main_measures is None:
        main, rule = (stats['p'] < 0.05).to_numpy(), "significant"
    else:
        unknown = sorted(set(main_measures) - set(stats['measure']))
        if unknown:
            raise ValueError(f"main_measures {unknown} are not in the statistics table.")
        main, rule = stats['measure'].isin(main_measures).to_numpy(), "fixed"
    nulls = [null.loc[null['measure'] == m, 'r'].to_numpy() for m in stats['measure']]
    main_path = out_dir / f"{prefix}_forest_{hemi}.svg"
    if main.any():
        logger.info(f"[{tag}] main ridgeline rows ({rule}): {', '.join(stats.loc[main, 'measure'])}")
        plot_ridge(stats[main], [n for n, keep in zip(nulls, main) if keep], *geometry, main_path)
    else:
        # No row to draw: remove a panel left by an earlier run, so the panels match the tables.
        main_path.unlink(missing_ok=True)
        logger.info(f"[{tag}] main ridgeline not drawn: no test with an uncorrected Moran p < 0.05.")
    plot_ridge(stats, nulls, *geometry, out_dir / f"{prefix}_forest_supp_{hemi}.svg")
    plt.close('all')

    _plot_gradient_spectra(tables["spectra"], out_dir / f"{prefix}_spectra_{hemi}.svg")

    # Sensitivity coverage (whole cortex + target network): total per-vertex sensitivity
    # from the fit channels, on both hemispheres, Purples over its 10-99th percentiles.
    coverage = maps["coverage"].to_numpy()
    surf_cov = np.where(coverage > 0, coverage, np.nan)
    for stem, net, title in (("coverage", None, "iEEG coverage"), ("coverage_sn", network, f"iEEG coverage ({network})")):
        values = mirror_to_cortex(surf_cov, df_yeo_surf, net)
        color_range = _pct_color_range(values, 10, 99)
        if color_range is None:
            logger.warning(f"[{tag}] {stem}: no coverage variation; map skipped.")
            continue
        plot_zoom_rows(values=values, stem=f"{prefix}_{stem}", title=title, cmap='Purples', color_range=color_range,
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
    plt.savefig(out_dir / f"{prefix}_slope_corr_supp_{hemi}.svg", bbox_inches='tight')
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
    plt.savefig(out_dir / f"{prefix}_band_power_corr_supp_{hemi}.svg", bbox_inches='tight')
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
    *,
    prefix: str,
    tag: str,
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

    Because vertices sampled by overlapping sensitivity maps share a sensitivity-averaged
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

    Parameters
    ----------
    pxx_raw : np.ndarray, shape (n_channels, n_freqs)
        Per-channel Welch PSD.
    sens : np.ndarray, shape (n_channels, N_LH)
        Non-negative channel sensitivity maps on the folded template.
    df_yeo_surf : pd.DataFrame
        Per-vertex surface table (both hemispheres) with the MPC gradient column.
    hemi : {'LH', 'RH'}
        Analysis hemisphere.
    network : str
        Source Yeo 7-network.
    n_perm : int
        Number of Moran surrogates.
    min_valid : int
        Least number of contributing targets of a defined projection score.
    leak_cos_thresh : float
        Sensitivity-profile cosine above which a source-target pair is excluded.
    prefix : str
        Panel file stem of the step, e.g. ``'figure_2b_ieeg_mica'``.
    tag : str
        Log tag of the step, e.g. ``'Figure 2B'``.

    Returns
    -------
    tables : dict
        ``vertices`` (per source vertex: vertex, g_mpc, P, dominant_network),
        ``targets`` (per target network: network, mean_projection) and ``stats``
        (r_group, p_moran, r_uniform, r_distance, n_sources, n_perm, ...).
    """
    hemi_rows = hemisphere_slice(hemi)
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    df_hemi = df_yeo_surf.iloc[hemi_rows].reset_index(drop=True)
    net_labels = df_hemi['network'].values

    # Sensitivity-weighted PSD of every covered vertex, (32492, n_freqs).
    covered = np.sum(sens, axis=0) > 0
    surf_psd_v = project_channel_measure(pxx_raw.T, sens).T

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

    # The FC gradient of Figure 3 at its native fsLR-32k resolution, to match this
    # analysis's surface space, in the float32 precision it is stored in.
    fc_g1_hemi = load_fc_gradient_32k(df_yeo_surf)['fc_g1'].to_numpy(dtype=np.float32)[hemi_rows]
    other_mask = other_mask & np.isfinite(fc_g1_hemi)
    logger.info(f"[{tag}] source {network} vertices: {sal_mask.sum()} "
                f"(covered+gradient: {(sal_mask & covered & np.isfinite(grad)).sum()}); "
                f"target vertices: {other_mask.sum()}")

    # Spectral-similarity-weighted projection of the FC gradient (source x target block).
    z_src = surf_psd_z[sal_mask]                         # (n_sal, n_freqs)
    z_tgt = surf_psd_z[other_mask]                       # (n_tgt, n_freqs)
    W_block = (z_src @ z_tgt.T) / n_freqs                # (n_sal, n_tgt) PSD correlations

    # Leakage mask: zero pairs whose sensitivity profiles overlap (shared channels),
    # because such vertices share a sensitivity-averaged spectrum and are trivially
    # similar for instrumental, not neural, reasons. Overlap = cosine of the two
    # vertices' sensitivity profiles.
    sens_src_n = sens[:, sal_mask] / (np.linalg.norm(sens[:, sal_mask], axis=0) + 1e-12)
    sens_tgt_n = sens[:, other_mask] / (np.linalg.norm(sens[:, other_mask], axis=0) + 1e-12)
    leak = (sens_src_n.T @ sens_tgt_n) > leak_cos_thresh
    W_block[leak] = 0.0
    logger.info(f"[{tag}] leakage mask removed {int(leak.sum())}/{leak.size} "
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
    logger.info(f"[{tag}] baselines vs MPC-gradient | uniform r={r_unif:+.3f}, "
                f"distance r={r_dist:+.3f} (spectral-similarity r={r_group:+.3f})")

    # Within-network Moran spatial null (single hemisphere -> one connected component).
    # Build the spatial graph on the finite source vertices only, so the surrogate
    # field carries the real gradient values.
    sal_finite_mask = np.zeros(N_LH, dtype=bool)
    sal_finite_mask[np.flatnonzero(sal_mask)[finite]] = True
    msr = fit_surface_moran(surf_hemi, sal_finite_mask, np.full(int(finite.sum()), hemi), n_perm)
    r_null = np.array([spearmanr(surr, P[finite])[0]
                       for surr in msr.randomize(g_sal[finite])])
    p_moran = empirical_p_twosided(r_null, r_group)
    logger.info(f"[{tag}] spectral-similarity projection vs {network} MPC-gradient | "
                f"Spearman r_group={r_group:+.3f}, Moran permutation p={p_moran:.3e} "
                f"(n_perm={n_perm}, n_src={int(finite.sum())})")

    # Per-source-vertex dominant target network (scatter colour) and per-target-network
    # mean projection (lollipop), both from the same positive-similarity block W_pos.
    tgt_networks = net_labels[other_mask]
    tgt_net_list = [n for n in sorted(yeo7_abbrev) if (tgt_networks == n).any()]
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

    _plot_channel_similarity(pxx_raw, sens, net_labels, config.FIGURES_DIR / f"{prefix}_corr_supp_{hemi}.svg", tag=tag)
    vertices = pd.DataFrame({"vertex": np.flatnonzero(sal_mask), "g_mpc": g_sal, "P": P, "dominant_network": dominant})
    targets = pd.DataFrame({"network": tgt_net_list, "mean_projection": [P_t_mean[n] for n in tgt_net_list]})
    stats = pd.DataFrame([{"network": network, "hemisphere": hemi, "r_group": float(r_group), "p_moran": p_moran,
                           "r_uniform": r_unif, "r_distance": r_dist, "n_sources": int(finite.sum()),
                           "n_perm": n_perm, "leak_cos_thresh": leak_cos_thresh,
                           "n_pairs_removed": int(leak.sum()), "n_pairs": int(leak.size)}])
    return {"vertices": vertices, "targets": targets, "stats": stats}


def _plot_channel_similarity(pxx_raw: np.ndarray, sens: np.ndarray, net_labels: np.ndarray, path: Path, *,
                             tag: str) -> None:
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
    logger.info(f"[{tag}] channel PSD correlation matrix: {int(valid_mask.sum())}/{valid_mask.size} channels "
                f"({int((~has_sens).sum())} without sensitivity excluded)")

    pxx_z = zscore(pxx_raw[valid_mask], axis=0)
    A_psd = np.corrcoef(pxx_z)

    sort_idx = np.argsort(channel_networks_valid)
    A_sorted = A_psd[sort_idx][:, sort_idx]
    sorted_networks = channel_networks_valid[sort_idx]

    boundaries = np.where(sorted_networks[:-1] != sorted_networks[1:])[0] + 1
    boundaries = np.insert(boundaries, 0, 0)
    b_ext = np.append(boundaries, len(channel_networks_valid))

    upper = A_sorted.copy()
    upper[np.tri(upper.shape[0], upper.shape[0]) == 1] = np.nan
    upper = rotate(upper, angle=-45, order=0, cval=np.nan)

    fig, ax = plt.subplots()
    for i, b in enumerate(boundaries):
        net_name = sorted_networks[b]
        color = network_rgb(net_name)
        rect = patches.Rectangle(
            (len(channel_networks_valid) / 2 * np.sqrt(2), b * np.sqrt(2)),
            b_ext[i + 1] - b_ext[i], b_ext[i + 1] - b_ext[i],
            linewidth=2, edgecolor=color, facecolor='none', angle=45
        )
        ax.add_patch(rect)

    upper[upper > 1] = 1
    plt.imshow(upper, cmap='coolwarm', origin='upper')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_spectral_similarity(tables: dict, df_yeo_surf: pd.DataFrame, hemi: str, network: str, *,
                             prefix: str) -> None:
    """Supplementary spectral-similarity panels: scatter + target-network lollipop, and the map."""
    vertices, targets, stats = tables["vertices"], tables["targets"], tables["stats"].iloc[0]
    g_sal, P = vertices["g_mpc"].to_numpy(dtype=float), vertices["P"].to_numpy(dtype=float)
    finite = np.isfinite(g_sal) & np.isfinite(P)
    point_colors = np.array([network_rgb(n) for n in vertices["dominant_network"]])
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
        color = tuple(network_rgb(net))
        axl.hlines(y, 0, P_t_mean[net], colors=[color], lw=2.5)
        axl.scatter(P_t_mean[net], y, s=55, facecolors=[color], edgecolors=[color], zorder=3)
    axl.axvline(0, color='0.6', lw=1, zorder=0)
    axl.set_yticks(range(len(nets_sorted)))
    axl.set_yticklabels([yeo7_abbrev.get(n, n) for n in nets_sorted])
    for tick, net in zip(axl.get_yticklabels(), nets_sorted):
        tick.set_color(tuple(network_rgb(net)))
    axl.tick_params(axis='y', length=0)
    axl.set_xlabel("Mean spectral-similarity")
    axl.set_box_aspect(1)
    axl.spines['right'].set_visible(False)
    axl.spines['top'].set_visible(False)
    axl.spines['left'].set_visible(False)

    plt.savefig(config.FIGURES_DIR / f"{prefix}_es_scatter_supp_{hemi}.svg", bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Spectral-similarity projection brain map: z-scored P on the analysis hemisphere's
    # source vertices (the other hemisphere stays empty), Figure 1 zoom rows.
    surf32k_lh_infl, surf32k_rh_infl = load_inflated_surfaces("fsLR-32k")
    p_map = np.full(2 * N_LH, np.nan)
    p_map[hemisphere_slice(hemi).start + vertices["vertex"].to_numpy()] = P_z
    plot_zoom_rows(surf32k_lh_infl, surf32k_rh_infl, p_map, salience_zoom_rois(df_yeo_surf, hemi), f"{prefix}_es",
                   "Spectral-similarity projection", out_dir=config.FIGURES_DIR, suffix=f"supp_{hemi}",
                   **ZOOM_ROW_KW, cmap=ieeg_cmap.name, color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS,
                   cbar_title='z', outline=mask_outline(surf32k_lh_infl, surf32k_rh_infl,
                                                        (df_yeo_surf['network'] == network).to_numpy()))

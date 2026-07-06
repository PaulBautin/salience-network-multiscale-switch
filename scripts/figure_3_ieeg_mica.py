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
# database
# BIDS_ieeg: 31 subjects
# The iEEG Data is in host/verges/tank/data/BIDS_iEEG
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3_ieeg_mica.py \
#   -ieeg_deriv /host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA \
#   -hemi RH
# Requires df_1a_{hemi}.tsv to exist (run figure_1a_t1map.py with matching -hemi first)
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
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl



import seaborn as sns

from brainspace.plotting import plot_surf
from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh import  mesh_elements
from brainspace.datasets import load_conte69
from brainspace.null_models import moran
from scipy.stats import spearmanr


import re
import matplotlib as mp
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from scipy.stats import zscore
from scipy.ndimage import rotate
from scipy.spatial.distance import cdist

import logging

from src.atlas_load import convert_states_str2int, compute_network_mask
from src.ieeg_processing import load_sensitivity_info, build_bipolar_sensitivity, load_original_data_files, preprocess_and_compute_psd_ieeg, compute_spectral_parameters, compute_gradient_quantiles
from src.connectome_processing import empirical_p_twosided, benjamini_hochberg, _weighted_mean_projection
from src.plot_colors import yeo7_rgb, yeo7_abbrev
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)

# fsLR-32k vertices per hemisphere (LH and RH each carry this many).
N_LH = 32492


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


def _plot_surf_safe(*args, **kwargs):
    """``plot_surf`` wrapper that degrades gracefully when VTK rendering fails.

    Mirrors ``figure_2_distance.save_brain_map``: some VTK builds raise
    ``AttributeError`` / ``RuntimeError`` while building the colorbar lookup table.
    In that case a warning is logged and the screenshot is skipped, rather than
    aborting the whole script (the analytic results and the matplotlib figures are
    unaffected).
    """
    try:
        return plot_surf(*args, **kwargs)
    except (AttributeError, RuntimeError) as e:
        fname = kwargs.get("filename", "<unknown>")
        logger.warning(f"plot_surf rendering failed for {fname} "
                       f"({type(e).__name__}: {e}); skipping screenshot.")
        return None


def _project_channel_measure(channel_vals: np.ndarray, sens: np.ndarray) -> np.ndarray:
    """Sensitivity-weighted projection of a per-channel scalar onto the surface.

    Each surface vertex receives the sensitivity-weighted average of the per-channel
    values; vertices with no sensitivity (uncovered) are returned as NaN.
    """
    sens_sum = np.sum(sens, axis=0)
    surf = (channel_vals @ sens) / (sens_sum + 1e-12)
    surf[sens_sum == 0] = np.nan
    return surf


def _screenshot_hemi(surf_infl, arr: np.ndarray, filename, *, name: str = "overlay2", **kwargs) -> None:
    """Attach ``arr`` to the inflated hemisphere and screenshot it lateral+medial.

    Wraps the repeated ``append_array`` -> single-hemisphere ``{lateral, medial}``
    ``_plot_surf_safe`` screenshot pattern; per-call appearance (``cmap``,
    ``color_range``, ``size``, ``color_bar``, ...) is forwarded through ``kwargs``.
    """
    surf_infl.append_array(arr, name=name)
    surfs = {'hemi1': surf_infl, 'hemi2': surf_infl}
    _plot_surf_safe(surfs, layout=[['hemi1', 'hemi2']], view=[['lateral', 'medial']],
                    array_name=name, transparent_bg=True, screenshot=True,
                    filename=str(filename), **kwargs)


def _moran_spearman(x: np.ndarray, y: np.ndarray, msr) -> tuple[float, float]:
    """Spearman(x, y) with a within-network Moran spatial null on ``y``.

    ``x`` is held fixed and ``y`` is spatially randomised by the pre-fitted
    ``MoranRandomization`` ``msr`` (whose graph must already be restricted to the
    same vertices ``x``/``y`` index). Returns the observed correlation and the
    add-one two-sided empirical p-value.
    """
    r, _ = spearmanr(x, y)
    r_null = np.array([spearmanr(x, zscore(surr))[0] for surr in msr.randomize(y)])
    return float(r), float(empirical_p_twosided(r_null, r))


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
        "-hemi",
        type=str,
        default="LH",
        choices=["LH", "RH"],
        help="Hemisphere for analysis: 'LH' or 'RH' (default: LH). 'both' is not "
             "supported here: bipolar sensitivities from both hemispheres are folded "
             "onto a single fsLR-32k template (see build_bipolar_sensitivity) and the "
             "gradient/Moran geometry are evaluated one hemisphere at a time, so the "
             "analysis runs per hemisphere."
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
    channel), projected onto the analysis hemisphere through the bipolar sensitivity
    maps, restricted to source-network vertices that have coverage and a finite
    gradient, and correlated (Spearman) against that network's MPC gradient. The
    **primary** measure is the aperiodic (1/f) exponent — the theoretically grounded
    electrophysiological correlate of cortical hierarchy; the **secondary** measure is
    the oscillatory (periodic) peak power in five canonical bands, taken from the same
    fit so it is orthogonal to the exponent rather than re-encoding the 1/f change, and
    FDR-corrected across bands. Significance is the within-network Moran spatial null
    with the add-one empirical p, the graph fitted once on the (finite) inferential
    vertex set. Renders the PSD plot, sensitivity-coverage maps, the aperiodic-exponent
    scatter + brain map, and the per-band surface maps and correlation scatter grid.

    Parameters
    ----------
    f : np.ndarray
        Welch frequency bins, shared with the similarity analysis (computed once in ``main``).
    pxx_raw : np.ndarray, shape (n_channels, n_freqs)
        Per-channel Welch PSD, shared with the similarity analysis.
    sens : np.ndarray, shape (n_channels, N_LH)
        Stacked absolute bipolar sensitivity maps (NaN already zeroed).
    surf32k_lh_infl, surf32k_rh_infl
        Inflated fsLR-32k surfaces for screenshots.
    df_yeo_surf : pd.DataFrame
        Per-vertex surface table with the MPC gradient column.
    project_root : Path
    hemi : {'LH', 'RH'}
    network : str
        Yeo 7-network used as the analysis target.
    n_perm : int
        Number of Moran surrogates.
    """
    freq_bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 80)}
    band_order = ["delta", "theta", "alpha", "beta", "gamma"]
    band_colors = ['#1f77b4', '#9467bd', '#e377c2', '#2ca02c', '#17becf']
    hemi_offset = N_LH if hemi == 'RH' else 0

    # Setup Geometry
    surf_combined = load_conte69(join=True)
    surf_lh, surf_rh = load_conte69(join=False)
    surf_hemi = surf_rh if hemi == 'RH' else surf_lh
    surf_hemi_infl = surf32k_rh_infl if hemi == 'RH' else surf32k_lh_infl
    n_vertices = surf_combined.GetPoints().shape[0]

    # Define analysis mask: target network for the specified hemisphere.
    mask = ((df_yeo_surf['hemisphere'] == hemi) & (df_yeo_surf['network'] == network)).values
    mask_hemi = mask[hemi_offset:hemi_offset + N_LH]

    gradient_col = f't1_gradient1_{network}'
    compute_gradient_quantiles(df_yeo_surf, np.where(mask)[0], gradient_col)
    grad_hemi = df_yeo_surf[gradient_col].values[hemi_offset:hemi_offset + N_LH]

    # Parameterise every channel spectrum ONCE (aperiodic exponent + oscillatory band
    # power from the same fit), then drop channels whose specparam fit failed to
    # converge so both measures and the coverage map share one successfully-fit set.
    spec = compute_spectral_parameters(pxx_raw, f, bands=freq_bands)
    exp_ch = spec['exponent']
    good = np.isfinite(exp_ch)
    if not good.all():
        logger.warning(f"[Figure 3B] specparam fit failed for {int((~good).sum())}/{good.size} channels; excluded.")
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
    valid_full = np.zeros(N_LH, dtype=bool)
    valid_full[np.flatnonzero(mask_hemi)[valid]] = True
    w = mesh_elements.get_ring_distance(surf_hemi, n_ring=1, mask=valid_full)
    w.data **= -1
    msr = moran.MoranRandomization(n_rep=n_perm, procedure='singleton', tol=1e-6, random_state=0)
    msr.fit(w)
    logger.info(f"[Figure 3B] inferential vertices (covered source-network with gradient): {int(valid.sum())}")

    def _embed_hemi(xvals: np.ndarray) -> np.ndarray:
        """Place valid-vertex values back on the full cortex, NaN off the salience border, slice hemi."""
        full = np.full(n_vertices, np.nan)
        full[np.flatnonzero(mask)[valid]] = xvals
        full[df_yeo_surf['salience_border'].isna()] = np.nan
        return full[hemi_offset:hemi_offset + N_LH]

    # Sensitivity-weighted PSD on the surface (pxx_raw and sens are computed once in
    # main and shared with the spectral-similarity analysis).
    surf_psd = (pxx_raw.T @ sens) / (np.sum(sens, axis=0) + 1e-12)

    # Plot all PSDs coloured by the MPC gradient, with the gradient-extreme means overlaid.
    fig, ax = plt.subplots(figsize=(6, 4))
    grad = grad_hemi[mask_hemi]
    surf_psd_sal = surf_psd[:, mask_hemi].T
    custom_cmap = plt.get_cmap(name="coolwarm")
    norm = mp.colors.Normalize(vmin=-1, vmax=1)
    for i in range(surf_psd_sal.shape[0]):
        ax.loglog(f, surf_psd_sal[i, :], color=custom_cmap(norm(grad[i])), alpha=0.1, rasterized=True)
    surf_psd_top = np.nanmean(surf_psd[:, (df_yeo_surf['quantiles'] == 1).values[hemi_offset:hemi_offset + N_LH]], axis=1)
    ax.loglog(f, surf_psd_top, color=custom_cmap(norm(1.0)), lw=2.5, alpha=0.9, label='top 25%')
    surf_psd_bottom = np.nanmean(surf_psd[:, (df_yeo_surf['quantiles'] == -1).values[hemi_offset:hemi_offset + N_LH]], axis=1)
    ax.loglog(f, surf_psd_bottom, color=custom_cmap(norm(-1.0)), lw=2.5, alpha=0.9, label='bottom 25%')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Normalized PSD')
    xticks = [0.5, 4, 8, 13, 30, 80]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    for x in xticks:
        ax.axvline(x=x, color="grey", linestyle="--", alpha=0.3, zorder=0)
    ax.legend(frameon=False, loc='lower left')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_psd_{hemi}.svg", bbox_inches='tight')
    plt.close(fig)

    # ----- PRIMARY measure: aperiodic (1/f) exponent ---------------------------------
    # The aperiodic exponent is the theoretically grounded electrophysiological
    # correlate of cortical hierarchy / microstructure (Gao et al., 2020; Donoghue et
    # al., 2020): flatter spectra (low exponent) at differentiated, task-positive
    # apex regions, steeper spectra at the integrative pole.
    surf_exp = _project_channel_measure(exp_ch[good], sens_good)
    x_exp = zscore(surf_exp[mask_hemi][valid])
    r_exp, p_exp = _moran_spearman(x_exp, y_valid, msr)
    logger.info(f"[Figure 3B] aperiodic exponent vs MPC-gradient | Spearman r={r_exp:+.3f}, "
                f"Moran permutation p={p_exp:.3e} (n_perm={n_perm}, n_vertices={int(valid.sum())})")

    fig, axp = plt.subplots(figsize=(3.4, 3.2))
    slope_e, intercept_e = np.polyfit(x_exp, y_valid_z, 1)
    axp.scatter(x_exp, y_valid_z, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
    axp.set_xlim([-3, 3])
    axp.set_ylim([-3, 3])
    axp.plot(x_exp, slope_e * x_exp + intercept_e, c='black', lw=2.5)
    axp.text(0.05, 0.95, f"$r={r_exp:+.2f}$\n$p={p_exp:.3f}$", transform=axp.transAxes,
             va="top", fontweight="bold", fontsize=12)
    axp.set_xlabel('Aperiodic exponent', fontsize=16)
    axp.set_ylabel('MPC gradient', fontsize=16)
    axp.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    axp.set_box_aspect(1)
    sns.despine(ax=axp)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_slope_corr_{hemi}.svg", bbox_inches='tight')
    plt.close(fig)

    # Aperiodic-exponent brain map (z-scored, salience border only).
    _screenshot_hemi(surf_hemi_infl, _embed_hemi(x_exp),
        project_root / f"results/figures/figure_3b_ieeg_mica_slope_map_{hemi}.svg",
        size=(725, 300), zoom=1.4, share='both', nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym')

    # Sensitivity coverage maps (whole-brain + salience): total per-vertex sensitivity
    # from the fit channels, spread across the Purples map over the full data range.
    surf_cov = sens_good_sum.astype(float)
    surf_cov[surf_cov == 0] = np.nan
    cov_wb = surf_cov.copy()
    cov_wb[df_yeo_surf.hemisphere.isna()[hemi_offset:hemi_offset + N_LH].values] = np.nan
    _screenshot_hemi(surf_hemi_infl, cov_wb,
        project_root / f"results/figures/figure_3b_ieeg_mica_sensitivity_map_{hemi}.svg",
        size=(725, 300), zoom=1.3, color_bar='right', share='both', nan_color=(220, 220, 220, 1),
        cmap="Purples", color_range=_pct_color_range(cov_wb, 10, 99))
    cov_sal = surf_cov.copy()
    cov_sal[~mask_hemi] = np.nan
    _screenshot_hemi(surf_hemi_infl, cov_sal,
        project_root / f"results/figures/figure_3b_ieeg_mica_sensitivity_map_{hemi}_salience.svg",
        size=(725, 300), zoom=1.3, color_bar='right', share='both', nan_color=(220, 220, 220, 1),
        cmap="Purples", color_range=_pct_color_range(cov_sal, 10, 99))

    # ----- SECONDARY measure: oscillatory band power (FDR-corrected across bands) -----
    # Per-band peak power from the same specparam fit (periodic component, orthogonal to
    # the aperiodic exponent). A vertex with no detected peak carries 0 oscillatory power.
    band_x, band_r, band_p = [], [], []
    for band in band_order:
        z = np.nan_to_num(spec['band_power'][band], nan=0.0)
        surf_b = _project_channel_measure(z[good], sens_good)
        vals = surf_b[mask_hemi][valid]
        if np.nanstd(vals) > 0:
            x_b = zscore(vals)
            r_b, p_b = _moran_spearman(x_b, y_valid, msr)
        else:
            x_b = np.zeros_like(vals)
            r_b, p_b = np.nan, np.nan
            logger.warning(f"[Figure 3B] Band {band}: no oscillatory-power variation across vertices; correlation undefined.")
        band_x.append(x_b); band_r.append(r_b); band_p.append(p_b)
        logger.info(f"[Figure 3B] Band {band}: oscillatory power vs MPC-gradient | Spearman r={r_b:+.3f}, "
                    f"Moran permutation p={p_b:.3e} (n_perm={n_perm}, n_vertices={int(valid.sum())})")

        # Per-band surface map (z-scored oscillatory power on the salience border).
        _screenshot_hemi(surf_hemi_infl, _embed_hemi(x_b),
            project_root / f"results/figures/figure_3b_ieeg_mica_{band}_map_{hemi}.svg",
            size=(725, 300), zoom=1.4, share='both', nan_color=(220, 220, 220, 1), cmap="coolwarm", color_range='sym')

    # FDR (Benjamini-Hochberg) across the five band tests.
    band_q = benjamini_hochberg(np.array(band_p))
    for band, r_b, p_b, q_b in zip(band_order, band_r, band_p, band_q):
        logger.info(f"[Figure 3B] Band {band}: FDR q={q_b:.3f}")

    # Band scatter grid (secondary; per-panel size matched to figure 2a / figure 3c).
    fig, axes = plt.subplots(1, len(band_order), figsize=(3.0 * len(band_order), 3.2), sharex=True, sharey=True)
    for i, band in enumerate(band_order):
        slope_b, intercept_b = np.polyfit(band_x[i], y_valid_z, 1)
        axes[i].scatter(band_x[i], y_valid_z, s=10, alpha=0.3, c='gray', edgecolors='none', rasterized=True)
        axes[i].set_xlim([-3, 3])
        axes[i].set_ylim([-3, 3])
        axes[i].plot(band_x[i], slope_b * band_x[i] + intercept_b, c=band_colors[i], lw=2.5)
        axes[i].text(0.05, 0.95, f"$r={band_r[i]:+.2f}$\n$q={band_q[i]:.3f}$",
                     transform=axes[i].transAxes, va="top", fontweight="bold", fontsize=12)
        axes[i].set_xlabel(band.capitalize(), color=band_colors[i], fontsize=16)
        axes[i].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        axes[i].set_box_aspect(1)
    axes[0].set_ylabel('MPC gradient', fontsize=16)
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_band_power_corr_{hemi}.svg", bbox_inches='tight')
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
    """Spectral-similarity projection of the FC gradient (Figure 2a focus row, group level).

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
    subset of cortex — a limitation noted in the Methods. Renders the Figure-2a-style
    scatter and the SS projection brain map; the channel-level PSD correlation matrix is
    kept as a supplement (it is the spectral-similarity measure itself).
    """
    hemi_offset = N_LH if hemi == 'RH' else 0
    surf_hemi_infl = surf32k_rh_infl if hemi == 'RH' else surf32k_lh_infl
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
    ax.text(0.05, 0.95, f"$r={r_group:+.2f}$\n$p={p_moran:.3f}$",
            transform=ax.transAxes, va='top', fontweight='bold', fontsize=12)
    # Uniform / inverse-distance baseline correlations are reported in the log only.
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

    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_es_scatter_{hemi}.svg",
                bbox_inches='tight', transparent=True)
    plt.close(fig)

    # Channel-level PSD correlation matrix (the spectral-similarity measure), network-sorted.
    peak_idx = np.argmax(sens, axis=1)  # (n_channels,)
    channel_networks = net_labels[peak_idx]
    # Exclude both 'medial_wall' and NaN labels: a NaN slips past the string compare
    # and would make `channel_networks_valid` a mixed str/float object array, which
    # `np.argsort` below cannot order.
    valid_mask = pd.notna(channel_networks) & (channel_networks != 'medial_wall')
    channel_networks_valid = channel_networks[valid_mask]

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
    plt.savefig(project_root / f"results/figures/figure_3b_ieeg_mica_corr_{hemi}.svg")
    plt.close()

    # Spectral-similarity projection brain map (z-scored P embedded on the analysis hemisphere).
    p_map = np.full(N_LH, np.nan)
    p_map[sal_mask] = P_z
    _screenshot_hemi(surf_hemi_infl, p_map,
        project_root / f"results/figures/figure_3b_ieeg_mica_es_map_{hemi}.svg", name='es_projection',
        size=(1200, 500), zoom=1.4, color_bar='bottom', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range='sym',
        cb__numberOfLabels=3, cb__labelTextProperty={'fontSize': 36, 'bold': False})


def main():
    # Setup Relative Paths
    parser = get_parser()
    args = parser.parse_args()
    ieeg_deriv = args.ieeg_deriv
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_3_ieeg_mica", project_root, args)
    logger.info(f"Dataset        : MICA iEEG (BIDS_iEEG, sub-PX*, ses-01, stage-W wakefulness)")
    logger.info(f"Sensitivity maps: electroMICA leadfield (signed), fsLR-32k; per-hemi bipolar difference |L1-L2| -> area-scaled thresholds (0.001, 0.05xmax) -> LH+RH fold")
    logger.info(f"Preprocessing  : Butterworth bandpass 0.5-80 Hz (order 4), downsampled to 200 Hz, demeaned")
    logger.info(f"PSD            : Welch method, Hamming window 2s, overlap 1s, normalized to unit sum")
    logger.info(f"Frequency bands: delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, beta 13-30 Hz, gamma 30-80 Hz")
    logger.info(f"Statistic      : aperiodic exponent (primary) + FDR-corrected band power vs MPC gradient; spectral-similarity-weighted projection of the FC gradient (group level)")
    logger.info(f"Null model     : within-network Moran randomization (n_rep=1000, procedure=singleton, random_state=0), add-one empirical p")
    logger.info(f"Surface space  : fsLR-32k {args.hemi}, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Analysis network: {args.network}")

    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    # load surfaces
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')

    ######### Part 1 -- T1 gradient (output of figure_1a_t1map.py)
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(f"Gradient dataframe not found at {path_df_1a}. Run figure_1a_t1map.py with -hemi {args.hemi} first.")
    logger.info(f"Loading gradient dataframe from {path_df_1a}")
    df_yeo_surf = pd.read_csv(path_df_1a, sep="\t")

    # Load signed per-hemisphere contact sensitivities and per-subject vertex areas.
    df_sensitivity, sens_areas = load_sensitivity_info(root_dir=ieeg_deriv)
    logger.info(f"Sensitivity maps loaded: {df_sensitivity['Subject'].nunique()} subjects, {len(df_sensitivity)} contacts")

    # Load channel information
    cache_path = project_root / 'data/dataframes/figure_3_channel_data_df.pkl'
    if cache_path.exists():
        logger.info(f"Loading cached channel info from {cache_path}...")
        with open(cache_path, 'rb') as f:
            df_channel_data = pickle.load(f)
    else:
        logger.info("Cache not found. Loading and processing channel info...")
        df_channel_data = load_original_data_files()
        with open(cache_path, 'wb') as f:
            pickle.dump(df_channel_data, f)
        logger.info(f"Channel info saved to {cache_path}.")
    logger.info(f"Channel data: {df_channel_data['Subject'].nunique()} subjects, {len(df_channel_data)} bipolar channels")

    # Attach each bipolar channel's two contacts' signed per-hemisphere leadfields
    # (Sens1_{L,R} = first contact, Sens2_{L,R} = second contact).
    df_channel_data[['ContactName1', 'ContactName2']] = df_channel_data[['ContactName1', 'ContactName2']].apply(lambda c: c.str.upper())
    df1 = df_channel_data.merge(
        df_sensitivity, left_on=['Subject', 'Session', 'ContactName1'],
        right_on=['Subject', 'Session', 'ContactName'], how='left'
    ).rename(columns={'Sens_L': 'Sens1_L', 'Sens_R': 'Sens1_R'}).drop(columns='ContactName')
    df2 = df1.merge(
        df_sensitivity, left_on=['Subject', 'Session', 'ContactName2'],
        right_on=['Subject', 'Session', 'ContactName'], how='left'
    ).rename(columns={'Sens_L': 'Sens2_L', 'Sens_R': 'Sens2_R'}).drop(columns='ContactName')

    # Compute the Welch PSD and stack the bipolar sensitivity maps ONCE; both the
    # spectral (slope/band) and spectral-similarity analyses reuse them (the PSD is the
    # expensive step, previously run twice).
    lengths = [len(sig) for sig in df2['Data']]
    min_len, max_len = min(lengths), max(lengths)
    if min_len != max_len:
        logger.warning(f"Variable channel lengths ({min_len} to {max_len} samples); truncating all to {min_len}.")
    data_matrix = np.vstack([np.asarray(sig)[:min_len] for sig in df2['Data']])
    fs = df2['SamplingRate'].iloc[0]
    f, pxx_raw = preprocess_and_compute_psd_ieeg(data_matrix, fs)
    # electroMICA-faithful bipolar sensitivity (row-aligned with df2 / pxx_raw):
    # per-hemisphere signed difference -> area-scaled thresholds -> abs -> LH+RH fold.
    sens = np.nan_to_num(build_bipolar_sensitivity(df2, sens_areas), nan=0.0)
    logger.info(f"PSD computed once: {pxx_raw.shape[0]} channels x {pxx_raw.shape[1]} freqs; sensitivity stack {sens.shape}")

    # Primary: aperiodic-exponent (1/f slope) and secondary FDR band power vs MPC gradient.
    frequency_band_analysis_sensitivity(f, pxx_raw, sens, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi, network=args.network)

    # Spectral-similarity projection: regional power-spectrum similarity (NOT connectivity)
    # used to project the FC gradient and test it against the within-network MPC gradient.
    salience_network_electrophysiological_similarity(pxx_raw, sens, surf32k_lh_infl, surf32k_rh_infl, df_yeo_surf, project_root, hemi=args.hemi, network=args.network)


if __name__ == "__main__":
    main()




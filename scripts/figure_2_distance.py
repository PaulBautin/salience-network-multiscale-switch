# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 2 - Gradient-weighted connectivity projection
#
# Tests whether the MPC (T1) gradient within a network predicts where those vertices
# connect, and whether that organisation mirrors the whole-brain sensory→transmodal
# (principal FC) gradient. For each network vertex i and subject s the projection score
# is the connectivity-weighted mean of the FC gradient across that vertex's targets:
#
#   P_s[i] = sum_{j in T_i,s} w_ij * g_FC[j] / sum_{j in T_i,s} w_ij ,
#   T_i,s  = { j : w_ij > 0, j not in network, j != i }
#
# A high P_s[i] means SN vertex i preferentially couples with targets at the high
# (task-positive) end of the FC gradient; low P_s[i] means coupling with the low
# (default-mode) end. Per subject, r_s = Spearman_i(g_MPC[i], P_s[i]) across network
# vertices. Group inference is a two-stage random-effects test: Fisher-z(r_s) then a
# one-sample t-test against zero across the 18 subjects, with a spin-permutation null
# (Alexander-Bloch 2018) and a Moran spectral-randomization null (within-network,
# SAC-preserving).
#
# Modality routing:
#   - SC : per-subject SIFT2 weights masked by the Betzel distance-stratified consensus
#          mask (built once across all subjects, removes non-reproducible / sparsity-
#          driven edges while preserving long-range edges); log10(SC*G/eps) on positives.
#   - GD : 1/GD (proximity), within-hemisphere only.
#   - MPC: rank variant (per-network-vertex Spearman across targets, then
#          Spearman across network vertices). Weighted-mean is ill-defined for MPC's
#          negative partial-correlation values.
#
# All matrices loaded at fsLR-5k (9684 vertices). Subject is the unit of inference.
#
# Figure 2A: SalVentAttn × {SC, GD, MPC} - projection map + group r/p per modality.
# Figure 2B: All 7 Yeo networks × {SC} - replicates the test per network. The headline
#            panel is a forest + per-subject beeswarm summary (group r ± 95% CI, FDR
#            stars) that makes the cross-network effect legible at a glance; the
#            per-network scatter grid is retained as a supplement.
#
# Outputs:
#   results/figures/figure_2a_distance_metric.svg
#   results/figures/figure_2a_brain_{SC,GD,MPC}_rho.svg
#   results/figures/figure_2b_distance_network_SC.svg          (per-network scatter grid)
#   results/figures/figure_2b_network_summary_SC.svg           (forest + beeswarm summary)
#   results/figures/figure_2b_brain_SC_rho_{network}.svg
#   data/dataframes/df_2b_label_{hemisphere}.csv               (vertex-level cache; new schema)
#   data/dataframes/df_2b_network_stats_{measure}_{hemi}.csv   (per-network group stats)
#   data/dataframes/df_2b_network_subject_r_{measure}_{hemi}.csv (per-subject r, for the beeswarm)
#
# The -panel {both,2a,2b} flag selects which panel to compute; '2b' regenerates the
# cross-network summary without rerunning the 2A modality sweep.
#
# Requires figure_1a_t1map.py to have been run first (produces
#   data/dataframes/figure_1a_pni_to_mics_5k.csv).
#
# Example:
#   python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_2_distance.py -hemi LH
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24"
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import argparse
import logging
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns

from brainspace.plotting import plot_surf
import brainspace.plotting.surface_plotting as _bsp_sp
from brainspace.plotting.utils import _gen_grid as _orig_gen_grid
from brainspace.mesh.mesh_io import read_surface
from brainspace.null_models import SpinPermutations

import matplotlib.pyplot as plt

from src.atlas_load import (
    load_yeo_surf_5k, load_t1_salience_profiles,
    convert_states_str2int, compute_network_mask,
)
from src.gradient_computation import compute_t1_gradient
from src.plot_colors import yeo7_rgb, yeo7_abbrev
from src.logging_utils import setup_manuscript_logger
from src.connectome_processing import (
    build_consensus_mask, compute_projection_subjects,
    compute_spin_null_projection,
    compute_moran_null_projection, compute_dominant_target_network,
    benjamini_hochberg, load_subject_matrix,
)

logger = logging.getLogger(__name__)

plt.rcParams["font.size"] = 16
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False

N_LH_5K = 4842
N_TOTAL_5K = 9684


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-network gradient-weighted connectivity projection at fsLR-5k. "
                    "Tests whether the within-network MPC gradient predicts each "
                    "vertex's expected FC-gradient position across its connectivity targets.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-hemi", type=str, default="both", choices=["both", "LH", "RH"],
        help="Hemisphere for analysis: 'both', 'LH', or 'RH' (default: both)"
    )
    optional.add_argument(
        "-panel", type=str, default="both", choices=["both", "2a", "2b"],
        help="Which panel to compute: 'both', '2a', or '2b' (default: both). "
             "Use '2b' to regenerate the cross-network summary without rerunning 2A."
    )
    return parser


def save_brain_map(surf_lh, surf_rh, values: np.ndarray, array_name: str,
                   filename: Path, hemisphere: str = "both",
                   color_range=(-3, 3)) -> None:
    """Append per-vertex values to fsLR-5k inflated surfaces and save a screenshot."""
    surf_lh.append_array(values[:N_LH_5K], name=array_name)
    surf_rh.append_array(values[N_LH_5K:], name=array_name)
    surfs = {"rh1": surf_rh, "lh1": surf_lh}
    if hemisphere == "LH":
        layout, view = [["lh1", "lh1"]], [["lateral", "medial"]]
    elif hemisphere == "RH":
        layout, view = [["rh1", "rh1"]], [["lateral", "medial"]]
    else:
        layout, view = [["lh1", "rh1"]], [["lateral", "medial"]]
    _bsp_sp._gen_grid = partial(_orig_gen_grid, size_bar=0.20)
    try:
        plot_surf(
            surfs, layout=layout, view=view,
            array_name=array_name, size=(1200, 500), zoom=1.4, color_bar="bottom",
            share="both", nan_color=(220, 220, 220, 1), cmap="coolwarm",
            color_range=color_range, transparent_bg=True, screenshot=True, filename=filename,
            cb__numberOfLabels=3,
            cb__labelTextProperty={'fontSize': 36, 'bold': False})
    except (AttributeError, RuntimeError) as e:
        logger.warning(f"save_brain_map: rendering failed for {filename.name} "
                       f"({type(e).__name__}: {e}). Skipping screenshot.")
    finally:
        _bsp_sp._gen_grid = _orig_gen_grid


def _load_fc_gradient(project_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Load fsLR-5k FC gradient GIFTIs and attach fc_g1, fc_g1_network, network_int."""
    fc_lh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-5k_lh.shape.gii").darrays[0].data
    fc_rh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-5k_rh.shape.gii").darrays[0].data
    df = df.copy()
    df["fc_g1"] = -np.concatenate([fc_lh, fc_rh])
    df.loc[df["hemisphere"].isna(), "fc_g1"] = np.nan
    df["fc_g1_network"] = df.groupby("network")["fc_g1"].transform("mean")
    df["network_int"] = convert_states_str2int(df["network"].values)[0]
    return df


def _prepare_network_gradient(
    df: pd.DataFrame, network: str, df_pni: pd.DataFrame, hemisphere: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (cache-aware) the T1 MPC gradient for `network`; return masks + values.

    Returns
    -------
    g_mpc_at_sn : np.ndarray, shape (n_sn,)
        MPC gradient values at source-network vertices, in cortex order.
    sn_mask_cortex : np.ndarray of bool, shape (n_cortex,)
        Source-network vertices with a valid (non-NaN) gradient value.
    other_mask_cortex : np.ndarray of bool, shape (n_cortex,)
        Target-set mask (cortex, non-network, hemisphere-filtered).
    """
    cortex_mask = df["hemisphere"].notna().values

    grad_col = f"t1_gradient1_{network}"
    if grad_col not in df.columns:
        net_mask_full_for_grad = compute_network_mask(df, network, hemisphere)
        t1_profiles = load_t1_salience_profiles(
            df_pni["path_t1_profile_5k"].tolist(), net_mask_full_for_grad
        )
        grad = compute_t1_gradient(t1_profiles)
        df.loc[net_mask_full_for_grad, grad_col] = grad

    net_mask_full = cortex_mask & (df["network"] == network).values
    if hemisphere == "LH":
        net_mask_full &= (df["hemisphere"] == "LH").values
    elif hemisphere == "RH":
        net_mask_full &= (df["hemisphere"] == "RH").values

    grad_full = df[grad_col].values.astype(float)
    grad_cortex = grad_full[cortex_mask]
    sn_mask_cortex = net_mask_full[cortex_mask] & ~np.isnan(grad_cortex)

    if hemisphere == "both":
        other_mask_full = cortex_mask & (df["network"] != network).values
    else:
        other_mask_full = (cortex_mask & (df["network"] != network).values
                           & (df["hemisphere"] == hemisphere).values)
    other_mask_cortex = other_mask_full[cortex_mask]

    g_mpc_at_sn = grad_cortex[sn_mask_cortex]
    return g_mpc_at_sn, sn_mask_cortex, other_mask_cortex


def _run_projection(
    modality: str, files: list, df: pd.DataFrame,
    g_fc_cortex: np.ndarray, g_mpc_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    cortex_mask_full: np.ndarray, gd_cortex: np.ndarray,
    target_network_labels: np.ndarray,
    spin_model: SpinPermutations, n_rand: int,
    *, mask_G: np.ndarray | None = None,
    sc_subjects: list[np.ndarray] | None = None,
) -> dict:
    """One-stop: per-subject projection + Moran + spin nulls.

    Moran is the primary spatial null (within-SN, preserves SAC of g_MPC).
    Spin is a secondary cortex-wide conservative check.
    """
    result = compute_projection_subjects(
        files=files, modality=modality,
        g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=g_mpc_at_sn,
        sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
        df_yeo_surf_5k=df,
        mask_G=mask_G, sc_subjects=sc_subjects,
        target_network_labels=target_network_labels,
    )

    gd_among_sn = gd_cortex[np.ix_(sn_mask_cortex, sn_mask_cortex)]
    moran = compute_moran_null_projection(
        g_mpc_at_sn, sn_mask_cortex, gd_among_sn, result, n_rand,
    )

    spin = compute_spin_null_projection(
        g_mpc_at_sn, sn_mask_cortex, cortex_mask_full, result, spin_model, n_rand,
    )
    return {**result, **moran, **spin}


def _embed_in_full_cortex(
    values_at_sn: np.ndarray, sn_mask_cortex: np.ndarray, cortex_mask_full: np.ndarray,
) -> np.ndarray:
    """Map per-SN values back to the full 9684-vertex surface (NaN elsewhere)."""
    out = np.full(N_TOTAL_5K, np.nan)
    cortex_indices = np.flatnonzero(cortex_mask_full)
    out[cortex_indices[sn_mask_cortex]] = values_at_sn
    return out


def _sig_stars(q: float) -> str:
    """FDR q-value to significance annotation (n.s. / * / ** / ***)."""
    if not np.isfinite(q):
        return ""
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 5e-2:
        return "*"
    return "n.s."


def plot_network_forest(
    results_per_net: list, df_yeo_surf_5k: pd.DataFrame, measure: str,
    project_root: Path, hemisphere: str,
) -> None:
    """Figure 2B summary: forest plot of the group effect across networks.

    A single horizontal row that makes the cross-network replication legible at a
    glance and stacks directly beneath the one-row per-network scatter grid. Each
    network occupies one column: the per-subject Spearman coefficients r_s (MPC
    gradient vs connectivity projection P) are drawn as a network-coloured beeswarm
    (x-jittered), the group estimate is a filled marker, the 95% CI a vertical bar,
    and FDR-corrected (Moran) significance is annotated with stars. Networks are
    taken in the order of `results_per_net` (the caller sorts by the signed group
    effect, shared with the scatter grid so columns align). Coefficients are shown
    exactly as the projection results produce them (no sign manipulation).
    """
    rng = np.random.default_rng(0)
    net_int_map = (df_yeo_surf_5k[["network", "network_int"]]
                   .drop_duplicates().dropna()
                   .set_index("network")["network_int"].to_dict())

    recs = list(results_per_net)
    n = len(recs)

    fig, ax = plt.subplots(figsize=(3.0 * n, 3.4))

    bounds = []
    for r in recs:
        rs = r["res"]["r_subjects"]
        bounds += [r["res"]["ci_low"], r["res"]["ci_high"]]
        bounds += list(rs[np.isfinite(rs)])
    ymax = float(np.nanmax(np.abs(bounds))) * 1.15 + 0.04
    ax.set_ylim(-ymax, ymax)
    ax.set_xlim(-0.6, n - 0.4)
    ax.axhline(0, color="0.35", ls="--", lw=0.9, zorder=1)

    for x, rec in enumerate(recs):
        res = rec["res"]
        color = yeo7_rgb[int(net_int_map[rec["network"]])]
        rs = res["r_subjects"][np.isfinite(res["r_subjects"])]
        jitter = rng.uniform(-0.16, 0.16, size=rs.size)
        ax.scatter(np.full(rs.size, x) + jitter, rs, s=16, color=color,
                   alpha=0.35, edgecolor="none", zorder=2, rasterized=True)
        ax.plot([x, x], [res["ci_low"], res["ci_high"]], color="black",
                lw=2.0, solid_capstyle="round", zorder=3)
        ax.scatter(x, res["r_group"], s=95, color=color,
                   edgecolor="black", linewidth=1.1, zorder=4)
        stars = _sig_stars(rec.get("q_moran", np.nan))
        ax.annotate(stars, xy=(x, ymax * 0.98), ha="center", va="top",
                    fontsize=11 if stars != "n.s." else 9,
                    color="black" if stars not in ("", "n.s.") else "0.6")

    ax.set_xticks(range(n))
    ax.set_xticklabels([yeo7_abbrev.get(r["network"], r["network"]) for r in recs])
    for tick, rec in zip(ax.get_xticklabels(), recs):
        tick.set_color(yeo7_rgb[int(net_int_map[rec["network"]])])
        tick.set_fontweight("bold")
    ax.set_ylabel(f"Group effect $r$")
    ax.tick_params(axis="x", length=0)

    sns.despine(ax=ax, bottom=True)
    plt.tight_layout()
    out = project_root / f"results/figures/figure_2b_network_summary_{measure}.svg"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Figure 2B] forest summary written to {out.name}")


def _scatter_colors_by_target_network(
    res: dict, df_yeo_surf_5k: pd.DataFrame, fallback_color,
) -> tuple[np.ndarray, dict]:
    """Return per-SN-vertex RGB colors based on dominant target network + palette."""
    dominant, names = compute_dominant_target_network(res)
    n_sn = res["P_subjects_sn"].shape[1]
    network_int_map = (df_yeo_surf_5k[["network", "network_int"]]
                       .drop_duplicates().dropna()
                       .set_index("network")["network_int"].to_dict())
    palette = {name: yeo7_rgb[int(network_int_map[name])]
               for name in names if name in network_int_map}
    if dominant is None:
        return np.tile(fallback_color, (n_sn, 1)), {}
    colors = np.tile(np.array(fallback_color), (n_sn, 1))
    for idx, name in enumerate(names):
        if name not in palette:
            continue
        colors[dominant == idx] = palette[name]
    return colors, palette


def struct_conn_metric_analysis(
    df_yeo_surf_5k: pd.DataFrame, surf5k_lh_infl, surf5k_rh_infl,
    df_pni: pd.DataFrame, project_root: Path, spin_model: SpinPermutations,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray], gd_cortex: np.ndarray,
    network: str = "SalVentAttn", n_rand: int = 100, hemisphere: str = "both",
) -> pd.DataFrame:
    """Figure 2A: SalVentAttn × {SC, GD, MPC} projection + group inference + Moran/spin nulls."""

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)
    cortex_mask_full = df_yeo_surf_5k["hemisphere"].notna().values
    g_fc_cortex = df_yeo_surf_5k["fc_g1"].values[cortex_mask_full]
    target_net_labels = df_yeo_surf_5k.loc[cortex_mask_full, "network"].values

    g_mpc_at_sn, sn_mask_cortex, other_mask_cortex = _prepare_network_gradient(
        df_yeo_surf_5k, network, df_pni, hemisphere,
    )

    modalities = {
        "SC":  {"files": df_pni["path_sc_5k"].tolist(),
                "mask_G": mask_G, "sc_subjects": sc_subjects},
        "GD":  {"files": df_pni["path_dist_5k"].tolist(),
                "mask_G": None, "sc_subjects": None},
        "MPC": {"files": df_pni["path_mpc_5k"].tolist(),
                "mask_G": None, "sc_subjects": None},
    }

    fig, axes = plt.subplots(1, 3, figsize=(4 * 4, 5), squeeze=False)
    network_color = yeo7_rgb[int(
        df_yeo_surf_5k.loc[df_yeo_surf_5k["network"] == network, "network_int"].values[0]
    )]
    legend_handles, legend_labels = None, None

    for i, (name, mcfg) in enumerate(modalities.items()):
        res = _run_projection(
            modality=name, files=mcfg["files"], df=df_yeo_surf_5k,
            g_fc_cortex=g_fc_cortex, g_mpc_at_sn=g_mpc_at_sn,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            cortex_mask_full=cortex_mask_full, gd_cortex=gd_cortex,
            target_network_labels=target_net_labels,
            spin_model=spin_model, n_rand=n_rand,
            mask_G=mcfg["mask_G"], sc_subjects=mcfg["sc_subjects"],
        )

        logger.info(
            f"[Figure 2A | {name}] r_group={res['r_group']:+.3f} "
            f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] "
            f"t={res['t']:+.2f} p={res['p']:.3e} | "
            f"p_moran={res['p_moran']:.3e} (primary) p_spin={res['p_spin']:.3e} "
            f"(n_perm={n_rand}) | n={res['n']}"
        )

        P_full = _embed_in_full_cortex(res["P_mean"], sn_mask_cortex, cortex_mask_full)
        col_P = f"{network}_{name}_P"
        df_yeo_surf_5k[col_P] = P_full

        save_brain_map(
            surf5k_lh_infl, surf5k_rh_infl, P_full,
            array_name=f"overlay_2a_{name}",
            filename=project_root / f"results/figures/figure_2a_brain_{name}_rho.svg",
            hemisphere=hemisphere,
            color_range=(np.nanpercentile(P_full, 5), np.nanpercentile(P_full, 95))
                if np.isfinite(P_full).any() else (-1, 1),
        )

        ax_top = axes[0, i]
        valid = np.isfinite(g_mpc_at_sn) & np.isfinite(res["P_mean"])
        colors_per_sn, palette = _scatter_colors_by_target_network(
            res, df_yeo_surf_5k, fallback_color=network_color,
        )
        ax_top.scatter(res["P_mean"][valid], g_mpc_at_sn[valid],
                       s=15, alpha=0.75, c=colors_per_sn[valid],
                       edgecolor="none", rasterized=True)
        sns.regplot(x=res["P_mean"][valid], y=g_mpc_at_sn[valid],
                    scatter=False, color="black", line_kws={"linewidth": 1}, ax=ax_top)
        ax_top.text(0.05, 0.95,
                    f"r = {res['r_group']:+.2f}\n"
                    f"p$_{{moran}}$ = {res['p_moran']:.3f}\n"
                    f"n = {res['n']}",
                    transform=ax_top.transAxes, va="top", fontsize=11)
        ax_top.set_xlabel(f"{name} projection P (FC-G1 units)")
        ax_top.set_ylabel(f"MPC gradient ({network})")

        if palette and legend_handles is None:
            from matplotlib.lines import Line2D
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                       markersize=8, label=yeo7_abbrev.get(n, n))
                for n, c in palette.items()
            ]
            legend_labels = list(palette.keys())

    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.06), ncol=len(legend_handles),
                   frameon=False, fontsize=11)

    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_2a_distance_metric.svg",
                bbox_inches="tight")
    plt.close(fig)
    return df_yeo_surf_5k


def struct_conn_network_analysis(
    df_yeo_surf_5k: pd.DataFrame, surf5k_lh_infl, surf5k_rh_infl,
    df_pni: pd.DataFrame, project_root: Path, spin_model: SpinPermutations,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray], gd_cortex: np.ndarray,
    networks: list[str] = ("SalVentAttn", "Limbic"),
    n_rand: int = 100, hemisphere: str = "both", measure: str = "SC",
) -> pd.DataFrame:
    """Figure 2B: replicate the projection across networks for one modality.

    Reports Moran (primary, within-region) and spin (secondary) nulls; applies
    Benjamini-Hochberg FDR correction across networks for both p_moran and
    p_spin and logs the q-values.
    """

    if measure not in ("SC", "GD", "MPC"):
        raise ValueError(f"measure must be one of 'SC', 'GD', 'MPC'; got '{measure}'")

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)
    cortex_mask_full = df_yeo_surf_5k["hemisphere"].notna().values
    g_fc_cortex = df_yeo_surf_5k["fc_g1"].values[cortex_mask_full]
    target_net_labels = df_yeo_surf_5k.loc[cortex_mask_full, "network"].values

    if measure == "SC":
        files = df_pni["path_sc_5k"].tolist()
        m_mask_G, m_sc_subjects = mask_G, sc_subjects
    elif measure == "GD":
        files = df_pni["path_dist_5k"].tolist()
        m_mask_G, m_sc_subjects = None, None
    else:
        files = df_pni["path_mpc_5k"].tolist()
        m_mask_G, m_sc_subjects = None, None

    results_per_net = []
    for i, network in enumerate(networks):
        logger.info(f"[Figure 2B | {measure}] processing network: {network}")
        g_mpc_at_sn, sn_mask_cortex, other_mask_cortex = _prepare_network_gradient(
            df_yeo_surf_5k, network, df_pni, hemisphere,
        )

        res = _run_projection(
            modality=measure, files=files, df=df_yeo_surf_5k,
            g_fc_cortex=g_fc_cortex, g_mpc_at_sn=g_mpc_at_sn,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            cortex_mask_full=cortex_mask_full, gd_cortex=gd_cortex,
            target_network_labels=target_net_labels,
            spin_model=spin_model, n_rand=n_rand,
            mask_G=m_mask_G, sc_subjects=m_sc_subjects,
        )

        logger.info(
            f"[Figure 2B | {network} | {measure}] r_group={res['r_group']:+.3f} "
            f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] "
            f"t={res['t']:+.2f} p={res['p']:.3e} | "
            f"p_moran={res['p_moran']:.3e} (primary) p_spin={res['p_spin']:.3e} "
            f"(n_perm={n_rand}) | n={res['n']}"
        )

        P_full = _embed_in_full_cortex(res["P_mean"], sn_mask_cortex, cortex_mask_full)
        df_yeo_surf_5k[f"{network}_{measure}_P"] = P_full

        save_brain_map(
            surf5k_lh_infl, surf5k_rh_infl, P_full,
            array_name=f"overlay_2b_{measure}_{network}",
            filename=project_root / f"results/figures/figure_2b_brain_{measure}_rho_{network}.svg",
            hemisphere=hemisphere,
            color_range=(np.nanpercentile(P_full, 5), np.nanpercentile(P_full, 95))
                if np.isfinite(P_full).any() else (-1, 1),
        )

        results_per_net.append({
            "network": network, "res": res, "g_mpc_at_sn": g_mpc_at_sn,
            "sn_mask_cortex": sn_mask_cortex,
        })

    q_moran = benjamini_hochberg(np.array([r["res"]["p_moran"] for r in results_per_net]))
    q_spin  = benjamini_hochberg(np.array([r["res"]["p_spin"]  for r in results_per_net]))
    for rec, qm, qs in zip(results_per_net, q_moran, q_spin):
        rec["q_moran"], rec["q_spin"] = qm, qs
        logger.info(
            f"[Figure 2B FDR | {measure} | {rec['network']}] "
            f"q_moran={qm:.3e} q_spin={qs:.3e}"
        )

    # Order networks once by the signed group effect; both the scatter row and the
    # forest summary iterate this list so their columns line up when stacked.
    results_per_net.sort(key=lambda r: r["res"]["r_group"], reverse=True)

    # Cache per-network group stats and the per-subject coefficients so the
    # summary forest plot can be regenerated without recomputing the nulls.
    stats_rows = [{
        "network": rec["network"], "measure": measure, "n": rec["res"]["n"],
        "r_group": rec["res"]["r_group"],
        "ci_low": rec["res"]["ci_low"], "ci_high": rec["res"]["ci_high"],
        "t": rec["res"]["t"], "p": rec["res"]["p"],
        "p_moran": rec["res"]["p_moran"], "q_moran": rec["q_moran"],
        "p_spin": rec["res"]["p_spin"], "q_spin": rec["q_spin"],
    } for rec in results_per_net]
    pd.DataFrame(stats_rows).to_csv(
        project_root / f"data/dataframes/df_2b_network_stats_{measure}_{hemisphere}.csv",
        index=False)
    pd.DataFrame({rec["network"]: rec["res"]["r_subjects"] for rec in results_per_net}).to_csv(
        project_root / f"data/dataframes/df_2b_network_subject_r_{measure}_{hemisphere}.csv",
        index_label="subject")

    plot_network_forest(results_per_net, df_yeo_surf_5k, measure, project_root, hemisphere)

    # Per-network scatter grid: one row, effect-ordered to stack above the forest
    # summary. Numeric stats live in the forest row below, so the panels carry only
    # a colour-coded network title and the shared axes (Tufte minimalism).
    n_net = len(results_per_net)
    fig, axes = plt.subplots(1, n_net, figsize=(3.0 * n_net, 3.4),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, rec in zip(axes, results_per_net):
        network, res = rec["network"], rec["res"]
        g_mpc_at_sn = rec["g_mpc_at_sn"]
        net_color = yeo7_rgb[int(
            df_yeo_surf_5k.loc[df_yeo_surf_5k["network"] == network, "network_int"].values[0]
        )]
        valid = np.isfinite(g_mpc_at_sn) & np.isfinite(res["P_mean"])
        colors_per_sn, _ = _scatter_colors_by_target_network(
            res, df_yeo_surf_5k, fallback_color=net_color,
        )
        ax.scatter(res["P_mean"][valid], g_mpc_at_sn[valid],
                   s=10, alpha=0.7, c=colors_per_sn[valid],
                   edgecolor="none", rasterized=True)
        sns.regplot(x=res["P_mean"][valid], y=g_mpc_at_sn[valid],
                    scatter=False, color="black", line_kws={"linewidth": 1}, ax=ax)
        ax.set_title(yeo7_abbrev.get(network, network), fontdict={"color": net_color})
    axes[0].set_ylabel("MPC gradient")
    fig.supxlabel(f"{measure} projection P")

    sns.despine(fig=fig)
    plt.savefig(project_root / f"results/figures/figure_2b_distance_network_{measure}.svg")
    plt.close(fig)
    return df_yeo_surf_5k


def main():
    parser = get_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_2_distance", project_root, args)
    logger.info("Surface space  : fsLR-5k, Yeo 7-network labels")
    logger.info("Statistic      : per-network-vertex connectivity-weighted projection P[i]")
    logger.info("Group test     : Fisher-z r_s + one-sample t-test (random effects)")
    logger.info("SC weights     : SIFT2 masked by Betzel consensus (per-subject log10)")
    logger.info("GD weights     : 1/GD (within-hemisphere)")
    logger.info("MPC variant    : per-vertex Spearman across targets (rank version)")
    logger.info("Null model     : spin permutation (Alexander-Bloch 2018, n_rep=1000)")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    surf5k_lh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.L.inflated.surf.gii", itype="gii")
    surf5k_rh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.R.inflated.surf.gii", itype="gii")

    df_pni = pd.read_csv(project_root / "data/dataframes/figure_1a_pni_to_mics_5k.csv")
    n_rand = 1000
    spin_model_5k = SpinPermutations(n_rep=n_rand, random_state=42)
    sphere_5k_lh = read_surface(project_root / "data/surfaces/fsLR-5k.L.sphere.surf.gii", itype="gii")
    sphere_5k_rh = read_surface(project_root / "data/surfaces/fsLR-5k.R.sphere.surf.gii", itype="gii")
    spin_model_5k.fit(sphere_5k_lh, sphere_5k_rh)

    df_yeo_surf_5k = load_yeo_surf_5k(micapipe=project_root)

    path_df_1a_5k = project_root / f"data/dataframes/df_1a_{args.hemi}_fslr5k.tsv"
    path_df_1a_5k_both = project_root / "data/dataframes/df_1a_both_fslr5k.tsv"
    if not path_df_1a_5k.exists() and not path_df_1a_5k_both.exists():
        raise FileNotFoundError(
            f"fsLR-5k gradient dataframe not found at {path_df_1a_5k} or {path_df_1a_5k_both}. "
            f"Run figure_1a_t1map.py first."
        )
    logger.info(f"fsLR-5k gradient dataframe found at "
                f"{path_df_1a_5k if path_df_1a_5k.exists() else path_df_1a_5k_both}")

    logger.info("Building Betzel distance-stratified consensus mask + loading SC subjects...")
    mask_G, sc_subjects = build_consensus_mask(
        df_pni["path_sc_5k"].tolist(), df_pni["path_sc_dist_5k"].tolist(),
        df_yeo_surf_5k, nbins=10,
    )

    cortex_mask_full = df_yeo_surf_5k["hemisphere"].notna().values
    logger.info("Loading group-mean geodesic distance (for Moran spatial-weight matrix)...")
    gd_stack = [load_subject_matrix(f, cortex_mask_full)
                for f in df_pni["path_dist_5k"].tolist()]
    gd_cortex = np.mean(np.stack(gd_stack, axis=0), axis=0)
    del gd_stack

    if args.panel in ("both", "2a"):
        df_yeo_surf_5k = struct_conn_metric_analysis(
            df_yeo_surf_5k, surf5k_lh_infl, surf5k_rh_infl,
            df_pni, project_root, spin_model_5k,
            mask_G=mask_G, sc_subjects=sc_subjects, gd_cortex=gd_cortex,
            network="SalVentAttn", n_rand=n_rand, hemisphere=args.hemi,
        )

    if args.panel in ("both", "2b"):
        # SC is the primary modality (axonal, independent of the MPC gradient).
        networks = ["Limbic", "Default", "Cont", "SalVentAttn", "DorsAttn", "Vis", "SomMot"]
        df_yeo_surf_5k = struct_conn_network_analysis(
            df_yeo_surf_5k, surf5k_lh_infl, surf5k_rh_infl,
            df_pni, project_root, spin_model_5k,
            mask_G=mask_G, sc_subjects=sc_subjects, gd_cortex=gd_cortex,
            networks=networks, n_rand=n_rand, hemisphere=args.hemi, measure="SC",
        )

    df_yeo_surf_5k.to_csv(project_root / f"data/dataframes/df_2b_label_{args.hemi}.csv", index=False)


if __name__ == "__main__":
    main()

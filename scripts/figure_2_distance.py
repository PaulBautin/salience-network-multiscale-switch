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
# one-sample t-test against zero across the 18 subjects (this t-test/CI reflects
# subject-level reliability of the mean alignment, NOT spatial-autocorrelation-aware
# significance). Spatial significance is the within-network Moran spectral-
# randomization null (SAC-preserving): surrogates of g_MPC are generated per
# hemisphere block (the inverse-geodesic-distance graph is disconnected across
# hemispheres) and correlated against each subject's P_s. Empirical p uses the
# add-one estimator (1+k)/(1+n_perm).
#
# Modality routing (every modality uses only positive connections and the same
# weighted-mean projection):
#   - SC : per-subject SIFT2 weights masked by the Betzel distance-stratified consensus
#          mask (built once across all subjects, removes non-reproducible / sparsity-
#          driven edges while preserving long-range edges); log10(SC*G/eps) on positives.
#          SC indexes axonal connectivity independent of the MPC gradient — the primary,
#          microstructure-independent test; it fixes the cross-network ordering.
#   - GD : 1/GD (proximity), within-hemisphere only. A geometry-only reading.
#   - MPC: positive partial correlations only (negative/zero entries dropped). Partly
#          self-referential: g_MPC is the within-SN MPC embedding while the projection
#          uses SN→target MPC, so both derive from the same qT1 profiles (reads as
#          microstructure-function coupling rather than an independent test).
#   - FC : positive correlations only (resting-state, micapipe 7T PNI;
#          anticorrelated/zero entries dropped). Because the target axis g_FC is
#          itself the principal FC gradient, this column is a convergence reading
#          (how strongly functional coupling recapitulates the gradient) rather
#          than a modality-independent test like SC.
#
# All matrices loaded at fsLR-5k (9684 vertices). Subject is the unit of inference.
#
# Figure 2A: SalVentAttn × {SC, GD, MPC, FC} - projection map + group r/p per modality.
# Figure 2B: All 7 Yeo networks × {SC, GD, MPC, FC} - replicates the test per network
#            for every connectivity measure. The headline panel is a bubble matrix
#            (rows = networks, columns = measures aligned with the 2A panels; disc
#            colour = group r, area = |r|, black ring + stars = FDR-Moran
#            significance) that makes the cross-network and cross-modality effect
#            legible at a glance; per-measure scatter grids are retained as supplements.
#
# Outputs:
#   results/figures/figure_2a_distance_metric.svg
#   results/figures/figure_2a_brain_{SC,GD,MPC,FC}_rho.svg
#   results/figures/figure_2b_distance_network_{measure}.svg   (per-measure scatter grid)
#   results/figures/figure_2b_network_summary_{hemi}.svg       (bubble matrix, all measures)
#   results/figures/figure_2b_brain_{measure}_rho_{network}.svg
#   data/dataframes/df_2b_label_{hemisphere}.csv               (vertex-level cache; new schema)
#   data/dataframes/df_2b_network_stats_{measure}_{hemi}.csv   (per-network group stats)
#   data/dataframes/df_2b_network_subject_r_{measure}_{hemi}.csv (per-subject r; row index = subject ID, one column per network)
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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.atlas_load import (
    load_yeo_surf_5k, load_t1_salience_profiles,
    convert_states_str2int, compute_network_mask,
)
from src.gradient_computation import compute_t1_gradient
from src.plot_colors import yeo7_rgb, yeo7_abbrev
from src.logging_utils import setup_manuscript_logger
from src.connectome_processing import (
    build_consensus_mask, compute_projection_subjects,
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


def _ensure_fc_paths(df_pni: pd.DataFrame) -> pd.DataFrame:
    """Add a `path_fc_5k` column to `df_pni` if the cached CSV predates FC support.

    The resting-state FC connectome lives in the same micapipe PNI derivatives
    tree as the (PNI) MPC/GD inputs. The derivatives root is recovered from an
    existing PNI path column (`path_dist_5k`: .../sub-X/ses-Y/dist/file, so
    parents[3] is the micapipe_v0.2.0 root). Subjects without a session-matched
    FC connectome get `NaN` here and are simply excluded from the FC modality
    (callers pass `df_pni["path_fc_5k"].dropna()`); the other modalities keep
    their full subject set, since each modality is an independent per-subject
    group test.
    """
    if "path_fc_5k" in df_pni.columns:
        return df_pni

    df_pni = df_pni.copy()
    fc_paths = []
    for _, row in df_pni.iterrows():
        pni_root = Path(row["path_dist_5k"]).parents[3]
        fc = (pni_root / f"sub-{row['ID_PNI']}/ses-{row['session']}/func/"
              f"desc-me_task-rest_bold/surf/"
              f"sub-{row['ID_PNI']}_ses-{row['session']}_surf-fsLR-5k_desc-FC.shape.gii")
        fc_paths.append(str(fc) if fc.exists() else np.nan)
    df_pni["path_fc_5k"] = fc_paths

    missing = df_pni.loc[df_pni["path_fc_5k"].isna(), "ID_PNI"].tolist()
    if missing:
        logger.warning(f"[FC] no session-matched fsLR-5k FC connectome for {missing}; "
                       f"excluded from the FC modality only.")
    logger.info(f"[FC] resting-state FC connectomes resolved for "
                f"N={df_pni['path_fc_5k'].notna().sum()} / {len(df_pni)} subjects.")
    return df_pni


def _load_fc_gradient(project_root: Path, df: pd.DataFrame) -> pd.DataFrame:
    """Load fsLR-5k FC gradient GIFTIs and attach fc_g1, fc_g1_network, network_int.

    The diffusion-map eigenvector polarity is arbitrary, so the gradient is
    oriented by anatomy rather than a hardcoded sign: it is flipped so the
    default-mode network sits at the low end (task-positive networks high),
    matching the projection-score reading (high P = coupling to the task-positive
    end of the FC gradient). The chosen sign is logged so a change in the source
    GIFTI's polarity surfaces in the logs rather than silently inverting results.
    """
    fc_lh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-5k_lh.shape.gii").darrays[0].data
    fc_rh = nib.load(project_root / "data/parcellations/fc_gradient_fslr-5k_rh.shape.gii").darrays[0].data
    df = df.copy()
    df["fc_g1"] = np.concatenate([fc_lh, fc_rh])
    df.loc[df["hemisphere"].isna(), "fc_g1"] = np.nan

    cortical = df["hemisphere"].notna()
    default_mean = df.loc[cortical & (df["network"] == "Default"), "fc_g1"].mean()
    cortical_mean = df.loc[cortical, "fc_g1"].mean()
    if default_mean > cortical_mean:
        df["fc_g1"] = -df["fc_g1"]
        logger.info("[FC gradient] flipped sign so the default-mode network sits "
                    "at the low (default-mode) end.")
    else:
        logger.info("[FC gradient] sign kept as loaded (default-mode already at "
                    "the low end).")

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
    gd_cortex: np.ndarray, target_network_labels: np.ndarray,
    n_rand: int,
    *, mask_G: np.ndarray | None = None,
    sc_subjects: list[np.ndarray] | None = None,
) -> dict:
    """One-stop: per-subject projection + within-network Moran null.

    The Moran spectral randomisation preserves the SAC of g_MPC and matches the test
    footprint; surrogates are generated per hemisphere block inside
    `compute_moran_null_projection`.
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
    return {**result, **moran}


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


def plot_network_bubble_matrix(
    results_by_measure: dict, measures: list, df_yeo_surf_5k: pd.DataFrame,
    project_root: Path, hemisphere: str,
) -> None:
    """Figure 2B summary: bubble matrix of the group effect for every network × measure.

    A compact network (row) × measure (column) grid whose measure columns align with
    the four Figure 2A modality panels (same width). Each cell is a disc whose colour
    encodes the group correlation $\\bar r$ (MPC gradient vs connectivity projection
    P) on a diverging scale and whose area encodes $|\\bar r|$; the FDR-corrected
    (Moran) significance is the disc's black edge ring and the stars above it, and the
    signed value is printed below. Networks share a single row order (set by the
    primary measure's effect) so rows align across measures.

    Because the within-network MPC gradient is a diffusion-map eigenvector whose
    polarity is arbitrary per network, coefficients are shown exactly as produced
    (no sign manipulation): the sign is interpretable *within* a row (same g_MPC
    across measures) but not *between* rows, where only $|\\bar r|$ and significance
    are comparable — hence area, not colour, carries the cross-network message. The
    smaller subject set for FC leaves its cells unchanged in encoding.
    """
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    net_int_map = (df_yeo_surf_5k[["network", "network_int"]]
                   .drop_duplicates().dropna()
                   .set_index("network")["network_int"].to_dict())

    measures = list(measures)
    n_meas = len(measures)
    networks = [rec["network"] for rec in results_by_measure[measures[0]]]
    n_net = len(networks)

    R = np.full((n_meas, n_net), np.nan)
    Q = np.full((n_meas, n_net), np.nan)
    for i, m in enumerate(measures):
        by_net = {rec["network"]: rec for rec in results_by_measure[m]}
        for j, net in enumerate(networks):
            R[i, j] = by_net[net]["res"]["r_group"]
            Q[i, j] = by_net[net].get("q_moran", np.nan)

    vmax = float(np.nanmax(np.abs(R))) * 1.05
    norm = Normalize(-vmax, vmax)
    cmap = plt.cm.coolwarm
    smin, smax = 90.0, 1300.0

    def _disc_size(r: float) -> float:
        return smin + (abs(r) / vmax) * (smax - smin)

    def _sig_lw(q: float) -> float:
        if not np.isfinite(q) or q >= 5e-2:
            return 0.4
        if q < 1e-3:
            return 2.0
        if q < 1e-2:
            return 1.5
        return 1.0

    # Width matches Figure 2A (4 * n modalities); measures are the columns so the
    # four columns line up with the 2A modality panels when the figures are stacked.
    fig, ax = plt.subplots(figsize=(4 * n_meas, 0.9 * n_net + 1.8))

    xs, ys, sizes, colors, edgecolors, lws = [], [], [], [], [], []
    for i in range(n_meas):          # measures -> columns (x)
        for j in range(n_net):       # networks -> rows (y)
            r, q = R[i, j], Q[i, j]
            if not np.isfinite(r):
                continue
            y = (n_net - 1) - j
            xs.append(i); ys.append(y)
            sizes.append(_disc_size(r))
            colors.append(cmap(norm(r)))
            sig = np.isfinite(q) and q < 5e-2
            edgecolors.append("black" if sig else "0.7")
            lws.append(_sig_lw(q))
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors=edgecolors,
               linewidths=lws, zorder=3)

    for i in range(n_meas):
        for j in range(n_net):
            r, q = R[i, j], Q[i, j]
            if not np.isfinite(r):
                continue
            y = (n_net - 1) - j
            stars = _sig_stars(q)
            if stars not in ("", "n.s."):
                ax.annotate(stars, xy=(i, y + 0.30), ha="center", va="bottom",
                            fontsize=12, color="black")
            ax.annotate(f"{r:+.2f}", xy=(i, y - 0.32), ha="center", va="top",
                        fontsize=8.5, color="0.25")

    ax.set_xlim(-0.6, n_meas - 0.4)
    ax.set_ylim(-0.8, n_net - 0.2)
    # Measure column headers go on top.
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticks(range(n_meas))
    ax.set_xticklabels(measures)
    ax.set_yticks([(n_net - 1) - j for j in range(n_net)])
    ax.set_yticklabels([yeo7_abbrev.get(net, net) for net in networks])
    for tick, net in zip(ax.get_yticklabels(), networks):
        tick.set_color(yeo7_rgb[int(net_int_map[net])])
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, color="0.93", lw=0.8)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Group $r$ (sign as-produced)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.subplots_adjust(bottom=0.20, top=0.88, left=0.08, right=0.90)

    out = project_root / f"results/figures/figure_2b_network_summary_{hemisphere}.svg"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Figure 2B] bubble-matrix summary written to {out.name}")


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
    df_pni: pd.DataFrame, project_root: Path,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray], gd_cortex: np.ndarray,
    network: str = "SalVentAttn", n_rand: int = 100, hemisphere: str = "both",
) -> pd.DataFrame:
    """Figure 2A: SalVentAttn × {SC, GD, MPC, FC} projection + group inference + Moran null."""

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)
    cortex_mask_full = df_yeo_surf_5k["hemisphere"].notna().values
    g_fc_cortex = df_yeo_surf_5k["fc_g1"].values[cortex_mask_full]
    target_net_labels = df_yeo_surf_5k.loc[cortex_mask_full, "network"].values

    g_mpc_at_sn, sn_mask_cortex, other_mask_cortex = _prepare_network_gradient(
        df_yeo_surf_5k, network, df_pni, hemisphere,
    )

    modalities = ("SC", "GD", "MPC", "FC")
    fig, axes = plt.subplots(1, len(modalities), figsize=(4 * len(modalities), 5),
                             squeeze=False)
    network_color = yeo7_rgb[int(
        df_yeo_surf_5k.loc[df_yeo_surf_5k["network"] == network, "network_int"].values[0]
    )]

    for i, name in enumerate(modalities):
        _, files, m_mask_G, m_sc_subjects = _measure_inputs(
            name, df_pni, mask_G, sc_subjects,
        )
        res = _run_projection(
            modality=name, files=files, df=df_yeo_surf_5k,
            g_fc_cortex=g_fc_cortex, g_mpc_at_sn=g_mpc_at_sn,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            gd_cortex=gd_cortex, target_network_labels=target_net_labels,
            n_rand=n_rand, mask_G=m_mask_G, sc_subjects=m_sc_subjects,
        )

        logger.info(
            f"[Figure 2A | {name}] r_group={res['r_group']:+.3f} "
            f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] "
            f"t={res['t']:+.2f} p={res['p']:.3e} (subject-level) | "
            f"p_moran={res['p_moran']:.3e} (spatial null, n_perm={n_rand}) | n={res['n']}"
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
        colors_per_sn, _ = _scatter_colors_by_target_network(
            res, df_yeo_surf_5k, fallback_color=network_color,
        )
        ax_top.scatter(res["P_mean"][valid], g_mpc_at_sn[valid],
                       s=15, alpha=0.75, c=colors_per_sn[valid],
                       edgecolor="none", rasterized=True)
        sns.regplot(x=res["P_mean"][valid], y=g_mpc_at_sn[valid],
                    scatter=False, color="black", line_kws={"linewidth": 1}, ax=ax_top)
        ax_top.text(0.05, 0.95,
                    f"r = {res['r_group']:+.2f}\n"
                    f"p$_{{moran}}$ = {res['p_moran']:.3f}",
                    transform=ax_top.transAxes, va="top", fontsize=11)
        ax_top.set_xlabel(f"{name} projection")
        ax_top.set_ylabel(f"MPC gradient ({network})" if i == 0 else "")

    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_2a_distance_metric.svg",
                bbox_inches="tight")
    plt.close(fig)
    return df_yeo_surf_5k


# Connectivity measure -> the df_pni column holding its per-subject file paths.
_MEASURE_PATH_COL = {
    "SC": "path_sc_5k", "GD": "path_dist_5k",
    "MPC": "path_mpc_5k", "FC": "path_fc_5k",
}


def _measure_inputs(
    measure: str, df_pni: pd.DataFrame,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray],
) -> tuple[list, list, np.ndarray | None, list[np.ndarray] | None]:
    """Resolve (subject_ids, files, mask_G, sc_subjects) for one connectivity measure.

    `subject_ids` and `files` are read from the same df_pni rows, so they stay
    aligned with the per-subject results positionally (subject ``s`` of
    ``r_subjects`` is ``subject_ids[s]``). This is the single routing point for
    both Figure 2A and 2B. SC carries the Betzel consensus mask and the
    pre-loaded per-subject weights (built from the full df_pni, so SC keeps every
    subject); GD/MPC are loaded per file. FC is the one measure with a reduced
    subject set: rows whose `path_fc_5k` is NaN (no session-matched connectome)
    are dropped from both `subject_ids` and `files`.
    """
    col = _MEASURE_PATH_COL.get(measure)
    if col is None:
        raise ValueError(f"measure must be one of 'SC', 'GD', 'MPC', 'FC'; got '{measure}'")
    rows = df_pni[df_pni[col].notna()] if measure == "FC" else df_pni
    subject_ids = rows["ID_PNI"].tolist()
    files = rows[col].tolist()
    if measure == "SC":
        return subject_ids, files, mask_G, sc_subjects
    return subject_ids, files, None, None


def _plot_network_scatter_grid(
    results_per_net: list, df_yeo_surf_5k: pd.DataFrame, measure: str,
    project_root: Path,
) -> None:
    """Per-network scatter grid (one row) for a single measure, effect-ordered.

    Numeric stats live in the bubble-matrix summary, so each panel carries only a
    colour-coded network title and the shared axes (Tufte minimalism).
    """
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
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
    axes[0].set_ylabel("MPC gradient")
    fig.supxlabel(f"{measure} projection P")

    sns.despine(fig=fig)
    plt.savefig(project_root / f"results/figures/figure_2b_distance_network_{measure}.svg")
    plt.close(fig)


def struct_conn_network_analysis(
    df_yeo_surf_5k: pd.DataFrame, surf5k_lh_infl, surf5k_rh_infl,
    df_pni: pd.DataFrame, project_root: Path,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray], gd_cortex: np.ndarray,
    networks: list[str] = ("SalVentAttn", "Limbic"),
    measures: tuple[str, ...] = ("SC", "GD", "MPC", "FC"),
    n_rand: int = 100, hemisphere: str = "both",
) -> pd.DataFrame:
    """Figure 2B: replicate the projection across networks for every connectivity measure.

    Each network's MPC gradient is computed once and reused across measures; for
    each measure the projection is run per network. Significance comes from the
    within-network Moran null (the sole spatial null; surrogates per hemisphere
    block) with Benjamini-Hochberg FDR correction across the seven networks within
    each measure. The headline panel is a bubble matrix (rows = networks, columns =
    measures); per-measure scatter grids and per-measure/network brain maps are
    retained as supplements. A single network ordering (by the primary measure's
    signed group effect) is shared across every measure column.
    """
    for measure in measures:
        if measure not in ("SC", "GD", "MPC", "FC"):
            raise ValueError(f"measure must be one of 'SC', 'GD', 'MPC', 'FC'; got '{measure}'")

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)
    cortex_mask_full = df_yeo_surf_5k["hemisphere"].notna().values
    g_fc_cortex = df_yeo_surf_5k["fc_g1"].values[cortex_mask_full]
    target_net_labels = df_yeo_surf_5k.loc[cortex_mask_full, "network"].values

    # MPC gradient is measure-independent: compute (cache-aware) once per network.
    net_gradients = {}
    for network in networks:
        net_gradients[network] = _prepare_network_gradient(
            df_yeo_surf_5k, network, df_pni, hemisphere,
        )

    results_by_measure: dict[str, list] = {}
    subject_ids_by_measure: dict[str, list] = {}
    for measure in measures:
        subject_ids, files, m_mask_G, m_sc_subjects = _measure_inputs(
            measure, df_pni, mask_G, sc_subjects,
        )
        subject_ids_by_measure[measure] = subject_ids

        results_per_net = []
        for network in networks:
            logger.info(f"[Figure 2B | {measure}] processing network: {network}")
            g_mpc_at_sn, sn_mask_cortex, other_mask_cortex = net_gradients[network]

            res = _run_projection(
                modality=measure, files=files, df=df_yeo_surf_5k,
                g_fc_cortex=g_fc_cortex, g_mpc_at_sn=g_mpc_at_sn,
                sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
                gd_cortex=gd_cortex, target_network_labels=target_net_labels,
                n_rand=n_rand, mask_G=m_mask_G, sc_subjects=m_sc_subjects,
            )

            logger.info(
                f"[Figure 2B | {network} | {measure}] r_group={res['r_group']:+.3f} "
                f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] "
                f"t={res['t']:+.2f} p={res['p']:.3e} (subject-level) | "
                f"p_moran={res['p_moran']:.3e} (spatial null, n_perm={n_rand}) | n={res['n']}"
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
        for rec, qm in zip(results_per_net, q_moran):
            rec["q_moran"] = qm
            logger.info(f"[Figure 2B FDR | {measure} | {rec['network']}] q_moran={qm:.3e}")

        results_by_measure[measure] = results_per_net

    # Fix a single network ordering (by the primary measure's signed group effect)
    # and apply it to every measure so the bubble matrix's columns line up.
    primary = measures[0]
    ordered_networks = [rec["network"] for rec in sorted(
        results_by_measure[primary], key=lambda r: r["res"]["r_group"], reverse=True)]
    order_index = {net: i for i, net in enumerate(ordered_networks)}
    for measure in measures:
        results_by_measure[measure].sort(key=lambda r: order_index[r["network"]])

    # Cache per-network group stats and per-subject coefficients per measure so the
    # summary bubble matrix can be regenerated without recomputing the nulls.
    for measure in measures:
        results_per_net = results_by_measure[measure]
        stats_rows = [{
            "network": rec["network"], "measure": measure, "n": rec["res"]["n"],
            "r_group": rec["res"]["r_group"],
            "ci_low": rec["res"]["ci_low"], "ci_high": rec["res"]["ci_high"],
            "t": rec["res"]["t"], "p": rec["res"]["p"],
            "p_moran": rec["res"]["p_moran"], "q_moran": rec["q_moran"],
        } for rec in results_per_net]
        pd.DataFrame(stats_rows).to_csv(
            project_root / f"data/dataframes/df_2b_network_stats_{measure}_{hemisphere}.csv",
            index=False)
        pd.DataFrame({rec["network"]: rec["res"]["r_subjects"]
                      for rec in results_per_net},
                     index=subject_ids_by_measure[measure]).to_csv(
            project_root / f"data/dataframes/df_2b_network_subject_r_{measure}_{hemisphere}.csv",
            index_label="subject")
        _plot_network_scatter_grid(results_per_net, df_yeo_surf_5k, measure, project_root)

    plot_network_bubble_matrix(
        results_by_measure, list(measures), df_yeo_surf_5k, project_root, hemisphere,
    )
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
    logger.info("MPC/FC weights : positive connections only (weighted-mean projection)")
    logger.info("Null model     : within-network Moran spectral randomisation (per hemisphere block)")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    surf5k_lh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.L.inflated.surf.gii", itype="gii")
    surf5k_rh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.R.inflated.surf.gii", itype="gii")

    pni_csv = project_root / "data/dataframes/figure_1a_pni_to_mics_5k.csv"
    if not pni_csv.exists():
        raise FileNotFoundError(
            f"fsLR-5k subject table not found at {pni_csv}. Run figure_1a_t1map.py first."
        )
    df_pni = pd.read_csv(pni_csv)
    df_pni = _ensure_fc_paths(df_pni)
    n_rand = 1000

    df_yeo_surf_5k = load_yeo_surf_5k(micapipe=project_root)

    # Consume the fsLR-5k MPC gradient that figure_1a (Part 2) cached, instead of
    # recomputing the diffusion-map embedding here: copy its t1_gradient1_*
    # column(s) into the surface table so _prepare_network_gradient short-circuits
    # (its `if grad_col not in df.columns` guard). The cache rows are in the same
    # order as load_yeo_surf_5k, so a positional assignment is aligned. An
    # exact-hemisphere cache is preferred; a 'both' cache is a safe superset for a
    # single-hemisphere run (the per-hemisphere mask subsets it), but an LH/RH
    # cache is NOT used for a 'both' run (it would silently drop the other
    # hemisphere's vertices). Anything else falls back to an in-figure recompute,
    # which is deterministic and uses identical inputs.
    path_df_1a_5k = project_root / f"data/dataframes/df_1a_{args.hemi}_fslr5k.tsv"
    path_df_1a_5k_both = project_root / "data/dataframes/df_1a_both_fslr5k.tsv"
    if path_df_1a_5k.exists():
        grad_cache = path_df_1a_5k
    elif args.hemi != "both" and path_df_1a_5k_both.exists():
        grad_cache = path_df_1a_5k_both
    else:
        grad_cache = None
    if grad_cache is not None:
        cached = pd.read_csv(grad_cache, sep="\t")
        grad_cols = [c for c in cached.columns if c.startswith("t1_gradient1_")]
        for c in grad_cols:
            df_yeo_surf_5k[c] = cached[c].to_numpy()
        logger.info(f"Loaded cached fsLR-5k MPC gradient ({', '.join(grad_cols) or 'none'}) "
                    f"from {grad_cache.name}")
    else:
        logger.info("No matching fsLR-5k gradient cache; computing the gradient in-figure.")

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
            df_pni, project_root,
            mask_G=mask_G, sc_subjects=sc_subjects, gd_cortex=gd_cortex,
            network="SalVentAttn", n_rand=n_rand, hemisphere=args.hemi,
        )

    if args.panel in ("both", "2b"):
        # All four modalities across all networks. SC is the primary modality
        # (axonal, independent of the MPC gradient) and fixes the network ordering.
        networks = ["Limbic", "Default", "Cont", "SalVentAttn", "DorsAttn", "Vis", "SomMot"]
        measures = ("SC", "GD", "MPC", "FC")
        df_yeo_surf_5k = struct_conn_network_analysis(
            df_yeo_surf_5k, surf5k_lh_infl, surf5k_rh_infl,
            df_pni, project_root,
            mask_G=mask_G, sc_subjects=sc_subjects, gd_cortex=gd_cortex,
            networks=networks, measures=measures, n_rand=n_rand, hemisphere=args.hemi,
            )

    df_yeo_surf_5k.to_csv(project_root / f"data/dataframes/df_2b_label_{args.hemi}.csv", index=False)


if __name__ == "__main__":
    main()

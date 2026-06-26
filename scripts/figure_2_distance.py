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
# Figure 2A: figure-1B-style grid - one row per modality {SC, GD, MPC, FC}, two
#            columns. Column 1 is the SalVentAttn scatter with the within-network MPC
#            gradient on a single shared bottom x-axis and that measure's projection P
#            on y (group r and spatial-null p_moran). Column 2 is a horizontal lollipop
#            placing all 7 Yeo networks (stem length = |group r| on a shared bottom |r|
#            axis, network-coloured, FDR-Moran-significant networks filled + starred).
#            All inputs are views of the per-network computation (nothing recomputed).
# Figure 2B: All 7 Yeo networks × {SC, GD, MPC, FC} - replicates the test per network
#            for every connectivity measure. The headline panel is a bubble matrix
#            (rows = networks, columns = measures; disc colour = group r, area = |r|,
#            black ring + stars = FDR-Moran significance); per-measure scatter grids
#            and per-network brain maps are retained as supplements.
#
# Outputs:
#   results/figures/figure_2a_distance_metric.svg               (scatter + lollipop, 1B layout)
#   results/figures/figure_2a_brain_{SC,GD,MPC,FC}_rho.svg      (SalVentAttn projection maps)
#   results/figures/figure_2b_distance_network_{measure}.svg   (per-measure scatter grid)
#   results/figures/figure_2b_network_summary_{hemi}.svg       (bubble matrix, all measures)
#   results/figures/figure_2b_brain_{measure}_rho_{network}.svg
#   data/dataframes/df_2b_label_{hemisphere}.csv               (vertex cache: _P + _dominant cols)
#   data/dataframes/df_2b_network_stats_{measure}_{hemi}.csv   (per-network group stats)
#   data/dataframes/df_2b_network_subject_r_{measure}_{hemi}.csv (per-subject r; row index = subject ID, one column per network)
#
# Two flags. -stage {both,compute,plot} separates the expensive computation from
# drawing: 'both'/'compute' run the projection + Moran nulls, write every figure-data
# cache above (the per-network projection always runs over all 7 networks), AND draw
# the figures (so a fresh compute always refreshes them); 'plot' skips all heavy loads
# and redraws figures from the caches in seconds (fast aesthetic iteration). -panel
# {both,2a,2b} then selects which figures are rendered. ('-stage plot' needs a prior
# compute run.)
#
# Requires figure_1a_t1map.py to have been run first (produces
#   data/dataframes/figure_1a_pni_to_mics_5k.csv).
#
# Example:
#   python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_2_distance.py -hemi LH
#   # then iterate on figure aesthetics without recomputing (seconds):
#   python .../scripts/figure_2_distance.py -hemi LH -stage plot
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24"
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################


import argparse
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore, spearmanr

from brainspace.plotting import plot_hemispheres
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
    compute_moran_null_projection, compute_topological_null_projection,
    compute_spin_null_projection, make_fc_spin_surrogates,
    compute_dominant_target_network,
    benjamini_hochberg, load_subject_matrix, _fisher_z_group,
)

logger = logging.getLogger(__name__)

plt.rcParams["font.size"] = 16
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False

N_LH_5K = 4842
N_TOTAL_5K = 9684

# Modalities that receive the geometry-preserving topological null. GD is excluded: its
# weights are 1/geodesic-distance, so a within-distance-bin target reassignment barely
# changes them and the null would be uninformative.
TOPO_MODALITIES = ("SC", "MPC", "FC")


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
        help="Which panel to render: 'both', '2a', or '2b' (default: both). "
             "Use '2b' to regenerate the cross-network summary without redrawing 2A."
    )
    optional.add_argument(
        "-stage", type=str, default="both", choices=["both", "compute", "plot"],
        help="Pipeline stage (default: both). 'both'/'compute' run the projection + "
             "nulls, write all figure-data caches, AND draw the figures (so a fresh "
             "compute always refreshes the figures); 'plot' skips the heavy computation "
             "and redraws figures from the caches (fast aesthetic iteration). 'plot' "
             "requires a prior 'compute'/'both' run."
    )
    optional.add_argument(
        "-n_rand", type=int, default=1000,
        help="Number of surrogates for the Moran and topological nulls (default: 1000). "
             "Lower (e.g. 300) for faster iteration; the add-one empirical p floor is "
             "1/(1+n_rand), so keep 1000 for the final run."
    )
    return parser


def save_brain_map(surf_lh, surf_rh, values: np.ndarray,
                   filename: Path, color_range=(-3, 3)) -> None:
    """Plot per-vertex fsLR-5k `values` on both hemispheres and save a screenshot.

    Renders the canonical four-view both-hemisphere layout (LH lateral/medial,
    RH lateral/medial) with the colorbar on the right, matching the figure-1 brain
    plots. A single-hemisphere analysis simply leaves the other hemisphere grey (its
    values in `values` are NaN).
    """
    # Use plot_hemispheres (the same call figure 1 uses) rather than a direct
    # plot_surf with a custom colorbar layout: the latter trips a brainspace
    # 0.1.22 colorbar bug ('VTKMethodWrapper' object has no attribute
    # 'lookupTable') that silently skipped every figure-2 brain screenshot.
    # plot_hemispheres splits the LH/RH arrays internally, so pass `values` whole.
    for n_labels in (3, 0):
        try:
            plot_hemispheres(
                surf_lh, surf_rh, array_name=values,
                size=(1450, 300), zoom=1.3, color_bar="right", share="both",
                nan_color=(220, 220, 220, 1), cmap="coolwarm",
                color_range=color_range, transparent_bg=True,
                screenshot=True, filename=filename,
                cb__numberOfLabels=n_labels)
            return
        except (AttributeError, RuntimeError) as e:
            if n_labels:  # retry once with figure-1's label-free colorbar
                logger.warning(
                    f"save_brain_map: {filename.name} colorbar with "
                    f"{n_labels} labels failed ({type(e).__name__}: {e}); "
                    f"retrying with no numeric labels.")
                continue
            logger.error(f"save_brain_map: rendering FAILED for {filename.name} "
                         f"({type(e).__name__}: {e}). No brain screenshot written.")


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
    preloaded_subjects: list[np.ndarray] | None = None,
    g_fc_spun_cortex: np.ndarray | None = None,
) -> dict:
    """One-stop: per-subject projection + Moran, topological and spin nulls.

    Three complementary nulls. The Moran spectral randomisation preserves the SAC of
    g_MPC and controls for map smoothness (per hemisphere block, all measures). The
    geometry-preserving topological null (SC/MPC/FC only) rewires the connectome within
    geodesic-distance bins and controls for connectome geometry, isolating wiring
    specificity; GD gets none (its weights are pure distance), so `p_topo` is NaN there.
    The target-side spin null rotates g_FC and controls for the FC gradient's
    autocorrelation independent of its anatomical position; being target-side it applies
    to every measure (GD included) and complements the topological null's direction-
    scrambling blind spot. `p_spin` is NaN only when no spun field is supplied.
    """
    result = compute_projection_subjects(
        files=files, modality=modality,
        g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=g_mpc_at_sn,
        sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
        df_yeo_surf_5k=df,
        mask_G=mask_G, preloaded_subjects=preloaded_subjects,
        target_network_labels=target_network_labels,
    )

    gd_among_sn = gd_cortex[np.ix_(sn_mask_cortex, sn_mask_cortex)]
    moran = compute_moran_null_projection(
        g_mpc_at_sn, sn_mask_cortex, gd_among_sn, result, n_rand,
    )

    if g_fc_spun_cortex is not None:
        spin = compute_spin_null_projection(
            modality=modality, files=files, df_yeo_surf_5k=df,
            g_fc_spun_cortex=g_fc_spun_cortex, g_mpc_cortex_at_sn=g_mpc_at_sn,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            result=result, preloaded_subjects=preloaded_subjects, mask_G=mask_G,
        )
    else:
        spin = {"null_group_spin": None, "p_spin": np.nan, "null_std_spin": np.nan}

    if modality in TOPO_MODALITIES:
        gd_sn_to_other = gd_cortex[np.ix_(sn_mask_cortex, other_mask_cortex)]
        # Dense connectomes (MPC/FC) use the analytic-moment CLT sampler (hundreds-thousands
        # of edges/vertex make the per-vertex numerator ~Gaussian); sparse SC keeps the exact
        # resampler (cheap, and it drives the positive/negative control).
        topo_method = "clt" if modality in ("MPC", "FC") else "exact"
        topo = compute_topological_null_projection(
            modality=modality, files=files, df_yeo_surf_5k=df,
            g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=g_mpc_at_sn,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            gd_sn_to_other=gd_sn_to_other, result=result, n_rand=n_rand,
            preloaded_subjects=preloaded_subjects, mask_G=mask_G, method=topo_method,
        )
    else:
        topo = {"null_group_topo": None, "p_topo": np.nan, "null_std_topo": np.nan}

    return {**result, **moran, **topo, **spin}


def _embed_in_full_cortex(
    values_at_sn: np.ndarray, sn_mask_cortex: np.ndarray, cortex_mask_full: np.ndarray,
) -> np.ndarray:
    """Map per-SN values back to the full 9684-vertex surface (NaN elsewhere)."""
    out = np.full(N_TOTAL_5K, np.nan)
    cortex_indices = np.flatnonzero(cortex_mask_full)
    out[cortex_indices[sn_mask_cortex]] = values_at_sn
    return out


def _net_int_map(df: pd.DataFrame) -> dict:
    """network -> network_int lookup from the surface table (NaN rows dropped)."""
    return (df[["network", "network_int"]]
            .drop_duplicates().dropna()
            .set_index("network")["network_int"].to_dict())


def _pct_range(values: np.ndarray) -> tuple[float, float]:
    """Symmetric-ish 5th/95th-percentile colour range; (-1, 1) if all-NaN."""
    if np.isfinite(values).any():
        return (np.nanpercentile(values, 5), np.nanpercentile(values, 95))
    return (-1, 1)


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

    net_int_map = _net_int_map(df_yeo_surf_5k)

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


def _dominant_network_int(res: dict, net_int_map: dict) -> np.ndarray:
    """Per-SN-vertex `network_int` of the dominant target network (NaN where none).

    Computed once (compute stage) from the projection result's group-mean
    target-network weights, then cached as the `{network}_{measure}_dominant`
    surface column so the scatter coloring survives into the plot-only stage.
    """
    dominant, names = compute_dominant_target_network(res)
    n_sn = res["P_subjects_sn"].shape[1]
    out = np.full(n_sn, np.nan)
    if dominant is None:
        return out
    for idx, name in enumerate(names):
        if name in net_int_map:
            out[dominant == idx] = net_int_map[name]
    return out


def _colors_from_dominant_int(dominant_int: np.ndarray, fallback_color) -> np.ndarray:
    """Map per-vertex dominant `network_int` values to RGB (fallback where NaN)."""
    dominant_int = np.asarray(dominant_int, dtype=float)
    colors = np.tile(np.asarray(fallback_color, dtype=float), (dominant_int.shape[0], 1))
    finite = np.isfinite(dominant_int)
    colors[finite] = yeo7_rgb[dominant_int[finite].astype(int)]
    return colors


def plot_figure_2a_scatter_lollipop(
    results_by_measure: dict, measures: list, df_yeo_surf_5k: pd.DataFrame,
    surf5k_lh_infl, surf5k_rh_infl, project_root: Path,
    focus_network: str = "SalVentAttn",
) -> None:
    """Figure 2A: figure-1B-style grid — one modality row, scatter + lollipop columns.

    Transposed to mirror ``figure_1b_contextualisation.context_analysis``: each
    connectivity measure (SC, GD, MPC, FC) is a row, with two columns and a single
    clear x-axis at the bottom of each.

    - Column 0 (scatter): the focus network's within-network MPC gradient on the
      shared x-axis (the same vector for every modality, so it is labelled only on
      the bottom row) against that measure's connectivity-weighted projection P on y
      (the per-row y-label names the modality). Points are coloured by dominant
      target network; black regression line; group $r$ / spatial-null
      $p_{\\mathrm{moran}}$ text box.
    - Column 1 (lollipop): a horizontal stem per Yeo network, length $|\\bar r|$ on
      the shared bottom |r| axis, network-coloured, FDR-Moran significant networks
      ($q<0.05$) filled + starred (others faded/open), the value printed at the
      marker. Networks share the single Figure 2B ordering (strongest at top).

    Magnitude (not signed $\\bar r$) is shown because each network's MPC gradient is
    a diffusion-map eigenvector with arbitrary polarity, so only $|\\bar r|$ is
    comparable across networks. All inputs are views of ``results_by_measure`` and
    the cached ``{network}_{measure}_P`` columns — nothing is recomputed. The
    focus-network projection maps are re-emitted under
    ``figure_2a_brain_{measure}_rho.svg``.
    """
    measures = list(measures)
    n_meas = len(measures)
    net_int_map = _net_int_map(df_yeo_surf_5k)
    focus_color = yeo7_rgb[int(net_int_map[focus_network])]

    # Full descriptive titles per modality (left-aligned over each scatter, fig-1B style).
    title_txt = {
        "SC":  "Structural connectivity – SIFT2-weighted tractography",
        "GD":  "Geodesic distance – cortical surface proximity",
        "MPC": "Microstructural profile covariance – qT1 profile similarity",
        "FC":  "Functional connectivity – resting-state BOLD",
    }

    # Shared network order (strongest at top) and the focus MPC gradient, which is
    # identical across measures, so it sets one shared scatter x-range.
    networks = [r["network"] for r in results_by_measure[measures[0]]]
    n_net = len(networks)
    ypos = {net: (n_net - 1) - j for j, net in enumerate(networks)}
    focus_g = next(r for r in results_by_measure[measures[0]]
                   if r["network"] == focus_network)["g_mpc_at_sn"]
    gfin = focus_g[np.isfinite(focus_g)]
    if gfin.size:
        gpad = 0.05 * (gfin.max() - gfin.min() or 1.0)
        gxlim = (gfin.min() - gpad, gfin.max() + gpad)
    else:
        gxlim = (-3, 3)

    # Shared |r| axis with headroom for the printed value + stars to the marker's right.
    all_abs_r = [abs(rec["res"]["r_group"])
                 for m in measures for rec in results_by_measure[m]
                 if np.isfinite(rec["res"]["r_group"])]
    rmax = (max(all_abs_r) * 1.45) if all_abs_r else 1.0

    fig, axes = plt.subplots(
        n_meas, 2, figsize=(6.0, 2.8 * n_meas), squeeze=False,
        gridspec_kw={"wspace": 0.35, "hspace": 0.4, "width_ratios": [1.0, 1.0]},
    )

    for row, measure in enumerate(measures):
        records = results_by_measure[measure]          # already in shared order
        focus_rec = next(r for r in records if r["network"] == focus_network)
        res = focus_rec["res"]
        g_mpc = focus_rec["g_mpc_at_sn"]

        # Re-emit the focus-network projection map (from the cached _P column).
        P_full = df_yeo_surf_5k[f"{focus_network}_{measure}_P"].values
        save_brain_map(
            surf5k_lh_infl, surf5k_rh_infl, P_full,
            filename=project_root / f"results/figures/figure_2a_brain_{measure}_rho.svg",
            color_range=_pct_range(P_full),
        )

        # Column 0 - scatter: x = MPC gradient (shared), y = projection P.
        ax = axes[row, 0]
        valid = np.isfinite(g_mpc) & np.isfinite(res["P_mean"])
        colors_per_sn = _colors_from_dominant_int(focus_rec["dominant_int"], focus_color)
        ax.scatter(g_mpc[valid], res["P_mean"][valid],
                   s=15, alpha=0.75, c=colors_per_sn[valid],
                   edgecolor="none", rasterized=True)
        sns.regplot(x=g_mpc[valid], y=res["P_mean"][valid],
                    scatter=False, color="black", line_kws={"linewidth": 2.5}, ax=ax)
        # Up to three nulls: p_moran (map smoothness, all measures), p_topo (connectome
        # geometry, SC/MPC/FC) and p_spin (FC-gradient anatomy, all measures).
        stat_txt = (f"$r={res['r_group']:+.2f}$\n"
                    f"$p_{{moran}}={res['p_moran']:.3f}$")
        if np.isfinite(res.get("p_topo", np.nan)):
            stat_txt += f"\n$p_{{topo}}={res['p_topo']:.3f}$"
        if np.isfinite(res.get("p_spin", np.nan)):
            stat_txt += f"\n$p_{{spin}}={res['p_spin']:.3f}$"
        ax.text(0.05, 0.95, stat_txt,
                transform=ax.transAxes, va="top", fontweight="bold", fontsize=12)
        t = ax.set_title(title_txt.get(measure, measure), loc="left", pad=15)
        t.set_in_layout(False)
        ax.set_ylabel(f"{measure} projection")
        ax.set_xlim(*gxlim)
        # Shared y-range across all modality rows (z-scored P, SD units); ticks only
        # at -4, 0, 4.
        ax.set_ylim(-5, 5)
        ax.set_yticks([-4, 0, 4])
        # Integer-only x ticks (x = z-scored MPC gradient, SD units).
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_box_aspect(1)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        if row < n_meas - 1:
            ax.tick_params(labelbottom=False)

        # Column 1 - horizontal lollipop: y = networks, x = |r| (shared).
        axl = axes[row, 1]
        for rec in records:
            net = rec["network"]
            r_abs = abs(rec["res"]["r_group"])
            q = rec.get("q_moran", np.nan)
            color = tuple(yeo7_rgb[int(net_int_map[net])])
            sig = np.isfinite(q) and q < 5e-2
            y = ypos[net]
            axl.hlines(y, 0, r_abs, colors=[color], lw=2.5, alpha=1.0 if sig else 0.45)
            axl.scatter(r_abs, y, s=55, facecolors=(color if sig else "white"),
                        edgecolors=[color], linewidths=1.8,
                        alpha=1.0 if sig else 0.85, zorder=3)
            stars = _sig_stars(q)
            label = f"{r_abs:.2f}{stars if sig and stars not in ('', 'n.s.') else ''}"
            axl.annotate(label, xy=(r_abs, y), xytext=(6, 0),
                         textcoords="offset points", ha="left", va="center",
                         fontsize=8.5, color="0.25")
        axl.set_xlim(0, rmax)
        axl.set_ylim(-0.6, n_net - 0.4)
        axl.set_yticks([ypos[n] for n in networks])
        axl.set_yticklabels([yeo7_abbrev.get(n, n) for n in networks])
        for tick, n in zip(axl.get_yticklabels(), networks):
            tick.set_color(tuple(yeo7_rgb[int(net_int_map[n])]))
        axl.tick_params(axis="y", length=0)
        axl.set_box_aspect(1)
        axl.spines["right"].set_visible(False)
        axl.spines["top"].set_visible(False)
        if row < n_meas - 1:
            axl.tick_params(labelbottom=False)
            axl.spines["bottom"].set_visible(False)

    axes[-1, 0].set_xlabel("MPC gradient")
    axes[-1, 1].set_xlabel(r"Spearman $|r|$")

    sns.despine(fig=fig)
    plt.savefig(project_root / "results/figures/figure_2a_distance_metric.svg",
                bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info("[Figure 2A] scatter + lollipop figure written to "
                "figure_2a_distance_metric.svg")


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
    net_int_map = _net_int_map(df_yeo_surf_5k)
    fig, axes = plt.subplots(1, n_net, figsize=(3.0 * n_net, 3.4),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, rec in zip(axes, results_per_net):
        network, res = rec["network"], rec["res"]
        g_mpc_at_sn = rec["g_mpc_at_sn"]
        net_color = yeo7_rgb[int(net_int_map[network])]
        valid = np.isfinite(g_mpc_at_sn) & np.isfinite(res["P_mean"])
        colors_per_sn = _colors_from_dominant_int(rec["dominant_int"], net_color)
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


def compute_network_results(
    df_yeo_surf_5k: pd.DataFrame, df_pni: pd.DataFrame, project_root: Path,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray], gd_cortex: np.ndarray,
    networks: list[str] = ("SalVentAttn", "Limbic"),
    measures: tuple[str, ...] = ("SC", "GD", "MPC", "FC"),
    n_rand: int = 100, hemisphere: str = "both",
    sphere_lh=None, sphere_rh=None,
) -> tuple[pd.DataFrame, dict]:
    """Compute stage: per network x measure projection + Moran null; persist all caches.

    The expensive work (per-subject projection + within-network Moran nulls) runs
    once across every network and measure. Each network's MPC gradient is computed
    (cache-aware) once and reused across measures; significance is the within-network
    Moran null (per hemisphere block) with Benjamini-Hochberg FDR across the seven
    networks within each measure, sharing one network ordering (primary measure's
    signed group effect).

    No figures are drawn here. Results are returned AND persisted so the plot stage
    can redraw without recomputing:
      - ``df_2b_network_stats_{measure}_{hemi}.csv``      (per-network group stats)
      - ``df_2b_network_subject_r_{measure}_{hemi}.csv``  (per-subject r)
      - surface columns ``{network}_{measure}_P`` and ``{network}_{measure}_dominant``
        (the latter = dominant target network's ``network_int`` per SN vertex; NaN
        elsewhere) — written into the df the caller persists as ``df_2b_label``.

    Returns
    -------
    df_yeo_surf_5k : pd.DataFrame
        The surface table with the per-network/measure ``_P`` and ``_dominant`` columns.
    results_by_measure : dict[str, list]
        Per measure, the per-network records (``network``, ``res``, ``g_mpc_at_sn``,
        ``dominant_int``, ``q_moran``) in the shared network order. The plot stage
        reconstructs the same record shape via ``load_results_from_cache``.
    """
    for measure in measures:
        if measure not in ("SC", "GD", "MPC", "FC"):
            raise ValueError(f"measure must be one of 'SC', 'GD', 'MPC', 'FC'; got '{measure}'")

    df_yeo_surf_5k = _load_fc_gradient(project_root, df_yeo_surf_5k)
    cortex_mask_full = df_yeo_surf_5k["hemisphere"].notna().values
    g_fc_cortex = df_yeo_surf_5k["fc_g1"].values[cortex_mask_full]
    target_net_labels = df_yeo_surf_5k.loc[cortex_mask_full, "network"].values
    net_int_map = _net_int_map(df_yeo_surf_5k)

    # Target-side spin null: rotate the (oriented) FC gradient once, reuse the spun field
    # across every network and measure. Requires the spheres; skipped if not supplied.
    if sphere_lh is not None and sphere_rh is not None:
        logger.info(f"Generating {n_rand} FC-gradient spin surrogates (target-side null)...")
        g_fc_spun_cortex = make_fc_spin_surrogates(
            df_yeo_surf_5k, sphere_lh, sphere_rh, n_rand,
        )
    else:
        g_fc_spun_cortex = None
        logger.info("No spheres supplied; skipping the target-side spin null (p_spin=NaN).")

    # MPC gradient is measure-independent: compute (cache-aware) once per network.
    net_gradients = {}
    for network in networks:
        net_gradients[network] = _prepare_network_gradient(
            df_yeo_surf_5k, network, df_pni, hemisphere,
        )

    results_by_measure: dict[str, list] = {}
    subject_ids_by_measure: dict[str, list] = {}
    for measure in measures:
        subject_ids, files, m_mask_G, preloaded = _measure_inputs(
            measure, df_pni, mask_G, sc_subjects,
        )
        subject_ids_by_measure[measure] = subject_ids

        # Preload this measure's per-subject connectomes ONCE and reuse them across all
        # networks and both nulls (SC arrives already preloaded). This avoids re-reading
        # the same ~350 MB matrices from disk for every network x {projection,
        # topological null}. Held one measure at a time, so peak memory stays ~one
        # measure's worth on top of the persistent SC stack.
        if preloaded is None:
            logger.info(f"[{measure}] preloading {len(files)} connectomes once for "
                        f"reuse across networks...")
            preloaded = [load_subject_matrix(f, cortex_mask_full) for f in files]

        results_per_net = []
        for network in networks:
            logger.info(f"[compute | {measure}] processing network: {network}")
            g_mpc_at_sn, sn_mask_cortex, other_mask_cortex = net_gradients[network]

            res = _run_projection(
                modality=measure, files=files, df=df_yeo_surf_5k,
                g_fc_cortex=g_fc_cortex, g_mpc_at_sn=g_mpc_at_sn,
                sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
                gd_cortex=gd_cortex, target_network_labels=target_net_labels,
                n_rand=n_rand, mask_G=m_mask_G, preloaded_subjects=preloaded,
                g_fc_spun_cortex=g_fc_spun_cortex,
            )

            logger.info(
                f"[compute | {network} | {measure}] r_group={res['r_group']:+.3f} "
                f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] "
                f"t={res['t']:+.2f} p={res['p']:.3e} (subject-level) | "
                f"p_moran={res['p_moran']:.3e} (spatial null) | "
                f"p_topo={res['p_topo']:.3e} (topological null) | "
                f"p_spin={res['p_spin']:.3e} (spin null) | "
                f"n={res['n']} (n_perm={n_rand})"
            )

            # Sparsity diagnostic: how many targets each SN vertex actually connects
            # to, and how often the min_valid=10 floor binds (an underpowered regime).
            nt = res["n_targets_per_sn"]
            nt = nt[np.isfinite(nt)]
            if nt.size:
                logger.info(
                    f"[sparsity | {network} | {measure}] targets/SN-vertex "
                    f"median={np.median(nt):.0f} "
                    f"IQR=[{np.percentile(nt, 25):.0f}, {np.percentile(nt, 75):.0f}] | "
                    f"frac SN vertices with <=10 targets: {np.mean(nt <= 10):.2f}"
                )

            # Standardize the group-mean projection per network x measure for display
            # and caching (brain-map embed + scatter y-axis in SD units). This is a
            # monotone rescale: the rank-based r_subjects / r_group and the Moran null
            # (computed from res["P_subjects_sn"]) are untouched.
            res["P_mean"] = zscore(res["P_mean"], nan_policy="omit")
            P_full = _embed_in_full_cortex(res["P_mean"], sn_mask_cortex, cortex_mask_full)
            df_yeo_surf_5k[f"{network}_{measure}_P"] = P_full

            dominant_int = _dominant_network_int(res, net_int_map)
            df_yeo_surf_5k[f"{network}_{measure}_dominant"] = _embed_in_full_cortex(
                dominant_int, sn_mask_cortex, cortex_mask_full)

            results_per_net.append({
                "network": network, "res": res, "g_mpc_at_sn": g_mpc_at_sn,
                "dominant_int": dominant_int,
            })

        q_moran = benjamini_hochberg(np.array([r["res"]["p_moran"] for r in results_per_net]))
        q_topo = benjamini_hochberg(np.array([r["res"]["p_topo"] for r in results_per_net]))
        q_spin = benjamini_hochberg(np.array([r["res"]["p_spin"] for r in results_per_net]))
        for rec, qm, qt, qs in zip(results_per_net, q_moran, q_topo, q_spin):
            rec["q_moran"] = qm
            rec["q_topo"] = qt
            rec["q_spin"] = qs
            logger.info(
                f"[compute FDR | {measure} | {rec['network']}] "
                f"q_moran={qm:.3e} q_topo={qt:.3e} q_spin={qs:.3e}"
            )

        results_by_measure[measure] = results_per_net

    # Fix a single network ordering (by the primary measure's signed group effect)
    # and apply it to every measure so the figures' network order lines up.
    primary = measures[0]
    ordered_networks = [rec["network"] for rec in sorted(
        results_by_measure[primary], key=lambda r: r["res"]["r_group"], reverse=True)]
    order_index = {net: i for i, net in enumerate(ordered_networks)}
    for measure in measures:
        results_by_measure[measure].sort(key=lambda r: order_index[r["network"]])

    # Always persist the per-network group stats and per-subject coefficients so the
    # plot stage can redraw every figure without recomputing the nulls.
    for measure in measures:
        results_per_net = results_by_measure[measure]
        stats_rows = [{
            "network": rec["network"], "measure": measure, "n": rec["res"]["n"],
            "r_group": rec["res"]["r_group"],
            "ci_low": rec["res"]["ci_low"], "ci_high": rec["res"]["ci_high"],
            "t": rec["res"]["t"], "p": rec["res"]["p"],
            "p_moran": rec["res"]["p_moran"], "q_moran": rec["q_moran"],
            "p_topo": rec["res"]["p_topo"], "q_topo": rec["q_topo"],
            "p_spin": rec["res"]["p_spin"], "q_spin": rec["q_spin"],
        } for rec in results_per_net]
        pd.DataFrame(stats_rows).to_csv(
            project_root / f"data/dataframes/df_2b_network_stats_{measure}_{hemisphere}.csv",
            index=False)
        pd.DataFrame({rec["network"]: rec["res"]["r_subjects"]
                      for rec in results_per_net},
                     index=subject_ids_by_measure[measure]).to_csv(
            project_root / f"data/dataframes/df_2b_network_subject_r_{measure}_{hemisphere}.csv",
            index_label="subject")

    return df_yeo_surf_5k, results_by_measure


def _obs_r_group(synth_g: np.ndarray, P_subjects_sn: np.ndarray, min_valid: int = 10) -> float:
    """Fisher-z group correlation of a synthetic map against each subject's projection.

    Mirrors the per-subject Spearman + Fisher-z aggregation used for the real statistic,
    so the observed value is comparable to the topological null's `null_group`.
    """
    rs = []
    valid_g = np.isfinite(synth_g)
    for s in range(P_subjects_sn.shape[0]):
        P_s = P_subjects_sn[s]
        m = valid_g & np.isfinite(P_s)
        if m.sum() >= min_valid:
            rs.append(spearmanr(synth_g[m], P_s[m])[0])
    # Reuse the shared Fisher-z aggregation so the control's observed value stays
    # defined identically to the real statistic (per-subject mask varies, so the
    # Spearman loop itself cannot use the vectorised `_rank_corr_columns`).
    return _fisher_z_group(np.asarray(rs, dtype=float))["r_group"]


def validate_topological_null(
    results_by_measure: dict, df_yeo_surf_5k: pd.DataFrame, df_pni: pd.DataFrame,
    project_root: Path, sc_subjects: list, mask_G: np.ndarray, gd_cortex: np.ndarray,
    focus_network: str = "SalVentAttn", n_rand: int = 1000, hemisphere: str = "both",
) -> None:
    """Power/specificity control: does the SC topological null discriminate wiring from
    geometry at the real connection density?

    Two synthetic source-network maps are tested against the SC topological null,
    holding the real per-subject SC projections fixed:

    - **Positive control** — a wiring-aligned map (the group-mean SC projection itself):
      its observed alignment is near-perfect by construction, so a null that destroys
      specific targeting should reject (small `p_topo`), confirming power.
    - **Negative control** — a geometry-only map (the GD/proximity projection): aligned
      to distance but not to specific SC wiring, so the geometry-centred topological
      null should *not* reject (`p_topo` n.s.), confirming specificity.

    Emits `results/figures/figure_2_supp_topo_control.svg` (the two null distributions
    with the observed values) and logs both `p_topo`.
    """
    if "SC" not in results_by_measure or "GD" not in results_by_measure:
        logger.info("[control] SC and GD measures required; skipping topological-null control.")
        return
    try:
        rec_sc = next(r for r in results_by_measure["SC"] if r["network"] == focus_network)
        rec_gd = next(r for r in results_by_measure["GD"] if r["network"] == focus_network)
    except StopIteration:
        logger.info(f"[control] focus network {focus_network} not found; skipping control.")
        return

    _, sn_mask_cortex, other_mask_cortex = _prepare_network_gradient(
        df_yeo_surf_5k, focus_network, df_pni, hemisphere,
    )
    cortex_mask_full = df_yeo_surf_5k["hemisphere"].notna().values
    g_fc_cortex = df_yeo_surf_5k["fc_g1"].values[cortex_mask_full]
    gd_sn_to_other = gd_cortex[np.ix_(sn_mask_cortex, other_mask_cortex)]

    P_subjects_sn = rec_sc["res"]["P_subjects_sn"]
    g_pos = rec_sc["res"]["P_mean"]   # wiring-aligned (SC projection)
    g_neg = rec_gd["res"]["P_mean"]   # geometry-only (GD projection)

    controls = {}
    for label, synth_g in (("positive", g_pos), ("negative", g_neg)):
        obs = _obs_r_group(synth_g, P_subjects_sn)
        topo = compute_topological_null_projection(
            modality="SC", files=None, df_yeo_surf_5k=df_yeo_surf_5k,
            g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=synth_g,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            gd_sn_to_other=gd_sn_to_other, result={"r_group": obs}, n_rand=n_rand,
            preloaded_subjects=sc_subjects, mask_G=mask_G,
        )
        controls[label] = {"obs": obs, **topo}
        logger.info(
            f"[control | {label}] obs r_group={obs:+.3f}, p_topo={topo['p_topo']:.3e} "
            f"(expected {'reject' if label == 'positive' else 'n.s.'})"
        )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    titles = {
        "positive": "Positive control\n(wiring-aligned map)",
        "negative": "Negative control\n(geometry-only map)",
    }
    for ax, label in zip(axes, ("positive", "negative")):
        null = controls[label]["null_group_topo"]
        null = null[np.isfinite(null)]
        ax.hist(null, bins=30, color="0.7", edgecolor="white")
        ax.axvline(controls[label]["obs"], color="crimson", lw=2.5,
                   label=f"observed\n$p_{{topo}}={controls[label]['p_topo']:.3f}$")
        ax.set_title(titles[label])
        ax.set_xlabel(r"group $r$ under SC topological null")
        ax.legend(fontsize=9, loc="upper center")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0].set_ylabel("surrogate count")
    out = project_root / "results/figures/figure_2_supp_topo_control.svg"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[control] topological-null control figure written to {out.name}")


def plot_figure_2b(
    results_by_measure: dict, df_yeo_surf_5k: pd.DataFrame,
    surf5k_lh_infl, surf5k_rh_infl, project_root: Path,
    measures: tuple[str, ...] = ("SC", "GD", "MPC", "FC"), hemisphere: str = "both",
) -> None:
    """Figure 2B rendering: per-network brain maps, per-measure scatter grids, bubble matrix.

    Pure plotting — reads ``results_by_measure`` plus the cached
    ``{network}_{measure}_P`` surface columns (for the brain maps), so it runs
    identically in the compute+plot and the plot-only stages.
    """
    for measure in measures:
        results_per_net = results_by_measure[measure]
        for rec in results_per_net:
            network = rec["network"]
            P_full = df_yeo_surf_5k[f"{network}_{measure}_P"].values
            save_brain_map(
                surf5k_lh_infl, surf5k_rh_infl, P_full,
                filename=project_root / f"results/figures/figure_2b_brain_{measure}_rho_{network}.svg",
                color_range=_pct_range(P_full),
            )
        _plot_network_scatter_grid(results_per_net, df_yeo_surf_5k, measure, project_root)

    plot_network_bubble_matrix(
        results_by_measure, list(measures), df_yeo_surf_5k, project_root, hemisphere,
    )


def load_results_from_cache(
    project_root: Path, hemisphere: str, measures: tuple[str, ...],
    df_yeo_surf_5k: pd.DataFrame,
) -> dict:
    """Plot stage: rebuild ``results_by_measure`` from the on-disk caches.

    Reads ``df_2b_network_stats_{measure}_{hemi}.csv`` (group stats: ``r_group``,
    ``ci_*``, ``t``, ``p``, ``p_moran``, ``q_moran``, ``n``) plus the per-vertex
    ``df_2b_label`` columns ``{network}_{measure}_P``, ``t1_gradient1_{network}`` and
    ``{network}_{measure}_dominant``. Per (measure, network) the SN vertices are
    exactly where ``{network}_{measure}_P`` is finite, taken in surface-row order —
    the same order the compute stage used — so the reconstructed ``g_mpc_at_sn`` /
    ``P_mean`` / ``dominant_int`` align with one another and with the stored stats.
    Networks keep the cached (shared) order from the stats file. Raises
    ``FileNotFoundError`` if a cache is missing (run ``-stage compute`` first).
    """
    results_by_measure: dict[str, list] = {}
    for measure in measures:
        stats_path = (project_root /
                      f"data/dataframes/df_2b_network_stats_{measure}_{hemisphere}.csv")
        if not stats_path.exists():
            raise FileNotFoundError(
                f"Figure-data cache {stats_path} not found; run with "
                f"'-stage compute' (or 'both') before '-stage plot'."
            )
        stats = pd.read_csv(stats_path).set_index("network")

        records = []
        for network in stats.index:  # cached shared order
            p_col = f"{network}_{measure}_P"
            g_col = f"t1_gradient1_{network}"
            d_col = f"{network}_{measure}_dominant"
            for c in (p_col, g_col):
                if c not in df_yeo_surf_5k.columns:
                    raise FileNotFoundError(
                        f"Column '{c}' missing from the df_2b_label cache; "
                        f"run with '-stage compute' (or 'both') first."
                    )
            P_full = df_yeo_surf_5k[p_col].to_numpy(dtype=float)
            sn = np.isfinite(P_full)
            g_mpc_at_sn = df_yeo_surf_5k[g_col].to_numpy(dtype=float)[sn]
            dominant_int = (df_yeo_surf_5k[d_col].to_numpy(dtype=float)[sn]
                            if d_col in df_yeo_surf_5k.columns
                            else np.full(int(sn.sum()), np.nan))
            srow = stats.loc[network]
            res = {
                "P_mean": P_full[sn],
                "r_group": float(srow["r_group"]),
                "ci_low": float(srow["ci_low"]), "ci_high": float(srow["ci_high"]),
                "t": float(srow["t"]), "p": float(srow["p"]),
                "p_moran": float(srow["p_moran"]), "n": int(srow["n"]),
                # p_topo/q_topo absent in caches predating the topological null;
                # p_spin/q_spin absent in caches predating the spin null.
                "p_topo": float(srow["p_topo"]) if "p_topo" in srow.index else np.nan,
                "p_spin": float(srow["p_spin"]) if "p_spin" in srow.index else np.nan,
            }
            records.append({
                "network": network, "res": res, "g_mpc_at_sn": g_mpc_at_sn,
                "dominant_int": dominant_int, "q_moran": float(srow["q_moran"]),
                "q_topo": float(srow["q_topo"]) if "q_topo" in srow.index else np.nan,
                "q_spin": float(srow["q_spin"]) if "q_spin" in srow.index else np.nan,
            })
        results_by_measure[measure] = records
    return results_by_measure


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
    logger.info("Null models    : Moran spectral randomisation (map smoothness, all measures) + "
                "geometry-preserving topological null (wiring specificity, SC/MPC/FC) + "
                "target-side FC-gradient spin null (anatomical alignment, all measures)")
    logger.info(f"Stage / panel  : stage={args.stage}, panel={args.panel}, hemi={args.hemi}")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    surf5k_lh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.L.inflated.surf.gii", itype="gii")
    surf5k_rh_infl = read_surface(project_root / "data/surfaces/fsLR-5k.R.inflated.surf.gii", itype="gii")
    # Spheres drive the target-side FC-gradient spin null (compute stage only).
    surf5k_lh_sphere = read_surface(project_root / "data/surfaces/fsLR-5k.L.sphere.surf.gii", itype="gii")
    surf5k_rh_sphere = read_surface(project_root / "data/surfaces/fsLR-5k.R.sphere.surf.gii", itype="gii")

    # SC is the primary modality (axonal, independent of the MPC gradient) and fixes
    # the shared network ordering reused by every figure.
    networks = ["Limbic", "Default", "Cont", "SalVentAttn", "DorsAttn", "Vis", "SomMot"]
    measures = ("SC", "GD", "MPC", "FC")
    n_rand = args.n_rand
    df_label_path = project_root / f"data/dataframes/df_2b_label_{args.hemi}.csv"

    if args.stage in ("compute", "both"):
        # --- Compute stage: run the projection + nulls and write the figure caches. ---
        pni_csv = project_root / "data/dataframes/figure_1a_pni_to_mics_5k.csv"
        if not pni_csv.exists():
            raise FileNotFoundError(
                f"fsLR-5k subject table not found at {pni_csv}. Run figure_1a_t1map.py first."
            )
        df_pni = pd.read_csv(pni_csv)
        df_pni = _ensure_fc_paths(df_pni)

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
        # Accumulate a running sum (one float64 buffer + one transient matrix per
        # subject) instead of materialising the whole stack and an np.stack copy,
        # which would peak at ~2x the (n_sub, n_cortex, n_cortex) array.
        gd_files = df_pni["path_dist_5k"].tolist()
        gd_acc = None
        for f in gd_files:
            M = load_subject_matrix(f, cortex_mask_full)
            gd_acc = M.astype(np.float64) if gd_acc is None else gd_acc + M
        gd_cortex = (gd_acc / len(gd_files)).astype(np.float32)
        del gd_acc

        df_yeo_surf_5k, results_by_measure = compute_network_results(
            df_yeo_surf_5k, df_pni, project_root,
            mask_G=mask_G, sc_subjects=sc_subjects, gd_cortex=gd_cortex,
            networks=networks, measures=measures, n_rand=n_rand, hemisphere=args.hemi,
            sphere_lh=surf5k_lh_sphere, sphere_rh=surf5k_rh_sphere,
        )
        df_yeo_surf_5k.to_csv(df_label_path, index=False)
        logger.info(f"[compute] figure-data caches written: {df_label_path.name}, "
                    f"df_2b_network_stats_*_{args.hemi}.csv, df_2b_network_subject_r_*_{args.hemi}.csv")

        # Synthetic power/specificity control: confirm the SC topological null
        # discriminates wiring from geometry at the real connection density.
        validate_topological_null(
            results_by_measure, df_yeo_surf_5k, df_pni, project_root,
            sc_subjects=sc_subjects, mask_G=mask_G, gd_cortex=gd_cortex,
            focus_network="SalVentAttn", n_rand=n_rand, hemisphere=args.hemi,
        )
    else:
        # --- Plot stage: skip the heavy loads; rebuild results from the caches. ---
        if not df_label_path.exists():
            raise FileNotFoundError(
                f"Vertex cache {df_label_path} not found; run with '-stage compute' "
                f"(or 'both') before '-stage plot'."
            )
        logger.info("[plot] loading figure-data caches (skipping projection + nulls)")
        df_yeo_surf_5k = pd.read_csv(df_label_path)
        results_by_measure = load_results_from_cache(
            project_root, args.hemi, measures, df_yeo_surf_5k,
        )

    # --- Render figures (every stage draws so a fresh compute always refreshes the
    # figures; only the heavy computation is gated by -stage). Figures are gated by
    # -panel. ---
    if args.panel in ("both", "2a"):
        plot_figure_2a_scatter_lollipop(
            results_by_measure, list(measures), df_yeo_surf_5k,
            surf5k_lh_infl, surf5k_rh_infl, project_root,
            focus_network="SalVentAttn",
        )
    if args.panel in ("both", "2b"):
        plot_figure_2b(
            results_by_measure, df_yeo_surf_5k,
            surf5k_lh_infl, surf5k_rh_infl, project_root,
            measures=measures, hemisphere=args.hemi,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Step 3: gradient-weighted connectivity projection (Figure 3, fsLR-5k).

Tests whether the MPC gradient within a network predicts where its vertices connect,
and whether that organisation mirrors the whole-brain sensory-to-transmodal (principal
FC) gradient. For each network vertex i and subject s the projection score is the
connectivity-weighted mean of the FC gradient across that vertex's targets:

    P_s[i] = sum_{j in T_i,s} w_ij * g_FC[j] / sum_{j in T_i,s} w_ij ,
    T_i,s  = { j : w_ij > 0, j not in network, j != i }

A high P_s[i] means vertex i preferentially couples with targets at the high
(task-positive) end of the FC gradient; low P_s[i] with the low (default-mode) end. Per
subject, r_s = Spearman_i(g_MPC[i], P_s[i]) across network vertices. Group inference is
a two-stage random-effects test: Fisher-z(r_s), then a one-sample t-test against zero
across subjects (subject-level reliability of the mean alignment, not a spatial test).
Spatial significance is the within-network Moran spectral-randomization null: surrogates
of g_MPC are generated per hemisphere (the inverse-geodesic-distance graph is
disconnected across hemispheres) and correlated against each subject's P_s, with the
add-one empirical p (1 + k) / (1 + n). A geometry-preserving topological null (SC, MPC,
FC) and a target-side FC-gradient spin null (all measures) complement it. p values are
not corrected for multiple comparisons here.

Measures (each uses only positive connections and the same weighted-mean projection):
  SC   per-subject SIFT2 weights masked by the Betzel distance-stratified consensus mask
       (built once across all subjects); log10(SC * G / eps) on positives. Independent of
       the MPC gradient: the primary test, which sets the network order of Figure 3B.
  GD   1/GD (proximity), within-hemisphere only: a geometry-only reading.
  MPC  positive partial correlations. Partly self-referential: g_MPC and the MPC
       projection derive from the same qT1 profiles.
  FC   positive correlations (resting state). The target axis is itself the principal
       FC gradient, so this is a convergence reading rather than an independent test.

Figure 3A: per measure, the network's z-scored projection as a brain row with the
salience zoom insets and the inset row alone, level with a lollipop block of the group r
of all seven networks over the central 95 % of each network's Moran null (dot filled
where the uncorrected Moran p < 0.05). Figure 3B (supplementary): every network x
measure, as a bubble matrix, test grids and per-network brain rows.

Inputs   step 1a: the fsLR-5k subject list with connectome paths (private) and the
         fsLR-5k MPC gradient (source_data); the connectomes it lists (micapipe).
Private  <work_dir>/step3_connectivity/: per-subject correlations (subject IDs), log.
Public   source_data/step3_connectivity/: vertex-wise group projections, statistics,
         null distributions, topological-null control, manifest.
Figures  results/figures/figure_3a_*.svg, figure_3b_*.svg, figure_3_supp_*.svg
         (file names as linked by results/figure3.svg)

Run      python pipeline/step3_connectivity.py [--stage compute|plot|all]
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from brainspace.mesh.mesh_io import read_surface

import common
from salience_switch import config
from salience_switch.colors import connectivity_cmap, yeo7_abbrev, yeo7_rgb
from salience_switch.connectivity import (build_consensus_mask, compute_dominant_target_network,
                                          compute_projection_nulls, compute_topological_null_projection,
                                          group_spearman_r, load_group_mean_matrix, load_subject_matrix,
                                          make_fc_spin_surrogates)
from salience_switch.microstructure import compute_t1_gradient, load_t1_salience_profiles
from salience_switch.plotting import (SALIENCE_Z_LABELS, SALIENCE_Z_RANGE, format_pvalue, mask_outline,
                                      plot_forest, plot_hemispheres_zoom, plot_network_lollipop, plot_ridge,
                                      plot_test_grid, salience_zoom_rois)
from salience_switch.provenance import write_manifest
from salience_switch.surface import (DATA_DIR, compute_network_mask, load_fc_gradient_5k, load_inflated_surfaces,
                                     load_yeo_surf_5k)

STEP = "step3_connectivity"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams["font.size"] = 16
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False

N_TOTAL_5K = 9684

# SC (axonal, independent of the MPC gradient) is the primary measure: its signed group
# effect sets the network order shared by every Figure 3B panel.
MEASURES = ("SC", "GD", "MPC", "FC")
NETWORKS = ("Limbic", "Default", "Cont", "SalVentAttn", "DorsAttn", "Vis", "SomMot")
FOCUS_NETWORK = "SalVentAttn"
NULL_MODELS = ("moran", "topo", "spin")

# Figure data written by the compute stage to source_data/ and read by the plot stage.
# Bump CACHE_SCHEMA whenever a table's columns change, so the plot stage refuses tables of
# an older run. The per-subject correlations (subject IDs) are written to the work dir.
CACHE_SCHEMA = 3
CACHE_COLUMNS = {
    "vertex": ("network", "measure", "vertex", "hemisphere", "g_mpc", "P", "n_targets",
               "dominant_network"),
    "stats": ("network", "measure", "n", "n_vertices", "r_group", "ci_low", "ci_high", "t", "p",
              "p_moran", "p_topo", "p_spin", "n_perm"),
    "null": ("network", "measure", "null_model", "surrogate", "r"),
    "topo_control": ("control", "r_obs", "p_topo", "n_perm"),
}

# Figure 1 zoom-row window (as figure_1a/1b/1c and figure_2b): 1450 x 300, zoom 1.3,
# colour-bar column, region names at 30 pt and inset-row titles at 36 pt; the rows are
# shown ZOOM_ROW_WIDTH_MM wide in the figure layout (~0.29x, so 30 pt reads as ~8.8 pt).
# outline_sharpness=3 crisps the SN border: the fsLR-5k mesh is coarse, so the default
# soft one-triangle trace reads as a thick fuzzy halo here, unlike the fine fsLR-32k
# borders of figures 1 and 2; a higher falloff power concentrates it into a thin line.
ZOOM_KW = dict(size=(1450, 300), zoom=1.3, color_bar=True, nan_color=(220, 220, 220, 1),
               label_fontsize=30, title_fontsize=36, outline_sharpness=3.0)

# Connectivity measure -> the df_pni column holding its per-subject file paths.
_MEASURE_PATH_COL = {
    "SC": "path_sc_5k", "GD": "path_dist_5k",
    "MPC": "path_mpc_5k", "FC": "path_fc_5k",
}

# Full descriptive titles per modality (left-aligned over each panel, fig-1/2 style).
_MODALITY_TITLES = {
    "SC":  "Structural connectivity – SIFT2-weighted tractography",
    "GD":  "Geodesic distance – cortical surface proximity",
    "MPC": "Microstructural profile covariance – qT1 profile similarity",
    "FC":  "Functional connectivity – resting-state BOLD",
}

# yeo7_rgb rows follow the alphabetical network order (that of convert_states_str2int);
# its last row is the background colour.
_NETWORK_INDEX = {net: i for i, net in enumerate(sorted(yeo7_abbrev))}


# =======================================================================================
# Compute stage: data processing only (no matplotlib)
# =======================================================================================

def _measure_inputs(
    measure: str, df_pni: pd.DataFrame,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray],
) -> tuple[list, list, np.ndarray | None, list[np.ndarray] | None]:
    """Resolve (subject_ids, files, mask_G, sc_subjects) for one connectivity measure.

    `subject_ids` and `files` are read from the same df_pni rows, so they stay
    aligned with the per-subject results positionally (subject ``s`` of
    ``r_subjects`` is ``subject_ids[s]``). SC carries the Betzel consensus mask and
    the pre-loaded per-subject weights (built from the full df_pni, so SC keeps every
    subject); GD/MPC/FC are loaded per file. FC is the one measure with a reduced
    subject set: rows whose `path_fc_5k` is NaN (no session-matched connectome) are
    dropped from both `subject_ids` and `files`.
    """
    col = _MEASURE_PATH_COL.get(measure)
    if col is None:
        raise ValueError(f"measure must be one of {MEASURES}; got '{measure}'")
    rows = df_pni[df_pni[col].notna()] if measure == "FC" else df_pni
    subject_ids = rows["ID_PNI"].tolist()
    files = rows[col].tolist()
    if measure == "SC":
        return subject_ids, files, mask_G, sc_subjects
    return subject_ids, files, None, None


def _network_gradient(
    df: pd.DataFrame, network: str, df_pni: pd.DataFrame, hemisphere: str,
) -> np.ndarray:
    """Full-surface MPC gradient of `network` (NaN elsewhere).

    Uses the ``t1_gradient1_{network}`` column copied from step 1a when present; otherwise the gradient is computed here from the subjects' fsLR-5k qT1
    profiles with the same (seeded) pipeline.
    """
    col = f"t1_gradient1_{network}"
    if col in df.columns:
        return df[col].to_numpy(dtype=float)
    mask = compute_network_mask(df, network, hemisphere)
    profiles = load_t1_salience_profiles(df_pni["path_t1_profile_5k"].tolist(), mask)
    grad = np.full(len(df), np.nan)
    grad[mask] = compute_t1_gradient(profiles)
    return grad


def _prepare_network_gradient(
    df: pd.DataFrame, network: str, df_pni: pd.DataFrame, hemisphere: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MPC gradient of `network` and its source/target masks, in cortex order.

    Returns
    -------
    g_mpc_at_sn : np.ndarray, shape (n_sn,)
        MPC gradient values at source-network vertices, in cortex order.
    sn_mask_cortex : np.ndarray of bool, shape (n_cortex,)
        Source-network vertices (hemisphere-filtered) with a finite gradient value.
    other_mask_cortex : np.ndarray of bool, shape (n_cortex,)
        Target-set mask (cortex, non-network, hemisphere-filtered).
    """
    cortex_mask = df["hemisphere"].notna().to_numpy()
    grad_cortex = _network_gradient(df, network, df_pni, hemisphere)[cortex_mask]
    sn_mask_cortex = (compute_network_mask(df, network, hemisphere)[cortex_mask]
                      & np.isfinite(grad_cortex))
    other = df["network"].ne(network).to_numpy()
    if hemisphere in ("LH", "RH"):
        other &= df["hemisphere"].eq(hemisphere).to_numpy()
    return grad_cortex[sn_mask_cortex], sn_mask_cortex, other[cortex_mask]


def _attach_gradient_cache(df: pd.DataFrame, hemisphere: str) -> pd.DataFrame:
    """Copy the fsLR-5k MPC gradient columns of step 1a into the surface table.

    The step 1a rows are in the same order as `load_yeo_surf_5k`, so a positional copy is
    aligned. An exact-hemisphere table is preferred; a 'both' table is a safe superset for
    a single-hemisphere run (the per-hemisphere mask subsets it), but an LH/RH table is
    NOT used for a 'both' run (it would silently drop the other hemisphere's vertices).
    Networks without a column are computed by `_network_gradient`.
    """
    source = config.source_dir(GRADIENT_STEP)
    exact = source / f"surface_fslr5k_{hemisphere}.tsv"
    both = source / "surface_fslr5k_both.tsv"
    cache = exact if exact.exists() else (both if hemisphere != "both" and both.exists() else None)
    if cache is None:
        logger.info("No matching fsLR-5k gradient table from step 1a; computing the gradients here.")
        return df
    cached = config.read_table(cache)
    if len(cached) != len(df):
        raise ValueError(f"{cache.name} has {len(cached)} rows; expected {len(df)} (fsLR-5k).")
    grad_cols = [c for c in cached.columns if c.startswith("t1_gradient1_")]
    for c in grad_cols:
        df[c] = cached[c].to_numpy()
    logger.info(f"Loaded cached fsLR-5k MPC gradient ({', '.join(grad_cols) or 'none'}) "
                f"from {cache.name}")
    return df


def _log_result(network: str, measure: str, res: dict, n_rand: int) -> None:
    """Log one network x measure result and its connection-sparsity diagnostic."""
    logger.info(
        f"[compute | {network} | {measure}] r_group={res['r_group']:+.3f} "
        f"[{res['ci_low']:+.3f}, {res['ci_high']:+.3f}] "
        f"t={res['t']:+.2f} p={res['p']:.3e} (subject-level) | "
        f"p_moran={res['p_moran']:.3e} (spatial null) | "
        f"p_topo={res['p_topo']:.3e} (topological null) | "
        f"p_spin={res['p_spin']:.3e} (spin null) | "
        f"n={res['n']} (n_perm={n_rand})"
    )
    # How many targets each source vertex actually connects to, and how often the
    # min_valid=10 floor binds (an underpowered regime).
    nt = res["n_targets_per_sn"]
    nt = nt[np.isfinite(nt)]
    if nt.size:
        logger.info(
            f"[sparsity | {network} | {measure}] targets/SN-vertex "
            f"median={np.median(nt):.0f} "
            f"IQR=[{np.percentile(nt, 25):.0f}, {np.percentile(nt, 75):.0f}] | "
            f"frac SN vertices with <=10 targets: {np.mean(nt <= 10):.2f}"
        )


def compute_network_results(
    df_surf: pd.DataFrame, df_pni: pd.DataFrame, net_gradients: dict,
    mask_G: np.ndarray, sc_subjects: list[np.ndarray], gd_cortex: np.ndarray,
    g_fc_spun_cortex: np.ndarray | None, *, n_rand: int,
) -> tuple[dict, dict]:
    """Projection + Moran, topological and spin nulls for every network x measure.

    Each measure's connectomes are loaded once and reused across all networks and
    nulls (SC arrives pre-loaded); one measure is held at a time, so peak memory is one
    measure's matrices on top of the persistent SC stack.

    Returns
    -------
    results : dict
        ``{(network, measure): res}``, ``res`` from `compute_projection_nulls` plus
        ``g_mpc_at_sn`` and ``sn_mask_cortex``.
    subject_ids : dict
        ``{measure: [subject IDs]}``, aligned with each result's ``r_subjects``.
    """
    cortex_mask = df_surf["hemisphere"].notna().to_numpy()
    g_fc_cortex = df_surf["fc_g1"].to_numpy(dtype=float)[cortex_mask]
    target_net_labels = df_surf.loc[cortex_mask, "network"].to_numpy()
    networks = list(net_gradients)

    results, subject_ids = {}, {}
    for measure in MEASURES:
        ids, files, m_mask_G, preloaded = _measure_inputs(measure, df_pni, mask_G, sc_subjects)
        subject_ids[measure] = ids
        if preloaded is None:
            logger.info(f"[{measure}] preloading {len(files)} connectomes once for "
                        f"reuse across networks...")
            preloaded = [load_subject_matrix(f, cortex_mask) for f in files]

        for network in networks:
            logger.info(f"[compute | {measure}] processing network: {network}")
            g_mpc_at_sn, sn_mask_cortex, other_mask_cortex = net_gradients[network]
            res = compute_projection_nulls(
                modality=measure, files=files, df_yeo_surf_5k=df_surf,
                g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=g_mpc_at_sn,
                sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
                gd_cortex=gd_cortex, n_rand=n_rand,
                target_network_labels=target_net_labels, mask_G=m_mask_G,
                preloaded_subjects=preloaded, g_fc_spun_cortex=g_fc_spun_cortex,
            )
            res["g_mpc_at_sn"] = g_mpc_at_sn
            res["sn_mask_cortex"] = sn_mask_cortex
            _log_result(network, measure, res, n_rand)
            results[(network, measure)] = res
    return results, subject_ids


def compute_topological_control(
    results: dict, df_surf: pd.DataFrame, net_gradients: dict,
    sc_subjects: list[np.ndarray], mask_G: np.ndarray, gd_cortex: np.ndarray,
    *, n_rand: int, focus_network: str = FOCUS_NETWORK,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Power/specificity control: does the SC topological null discriminate wiring from
    geometry at the real connection density?

    Two synthetic source-network maps are tested against the SC topological null,
    holding the real per-subject SC projections of `focus_network` fixed:

    - **Positive control** — a wiring-aligned map (the group-mean SC projection itself):
      its observed alignment is near-perfect by construction, so a null that destroys
      specific targeting should reject (small `p_topo`), confirming power.
    - **Negative control** — a geometry-only map (the GD/proximity projection): aligned
      to distance but not to specific SC wiring, so the geometry-centred topological
      null should *not* reject (`p_topo` n.s.), confirming specificity.

    Returns
    -------
    control : pd.DataFrame
        One row per control: ``control``, ``r_obs``, ``p_topo``, ``n_perm``.
    null : pd.DataFrame
        The control null distributions in the layout of the ``null`` table
        (``null_model`` = ``topo_control_positive`` / ``topo_control_negative``).
    """
    res_sc, res_gd = results[(focus_network, "SC")], results[(focus_network, "GD")]
    _, sn_mask_cortex, other_mask_cortex = net_gradients[focus_network]
    cortex_mask = df_surf["hemisphere"].notna().to_numpy()
    g_fc_cortex = df_surf["fc_g1"].to_numpy(dtype=float)[cortex_mask]
    gd_sn_to_other = gd_cortex[np.ix_(sn_mask_cortex, other_mask_cortex)]
    P_subjects_sn = res_sc["P_subjects_sn"]

    control_rows, null_rows = [], []
    for label, synth_g in (("positive", res_sc["P_mean"]), ("negative", res_gd["P_mean"])):
        obs = group_spearman_r(synth_g, P_subjects_sn)
        topo = compute_topological_null_projection(
            modality="SC", files=None, df_yeo_surf_5k=df_surf,
            g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=synth_g,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            gd_sn_to_other=gd_sn_to_other, result={"r_group": obs}, n_rand=n_rand,
            preloaded_subjects=sc_subjects, mask_G=mask_G,
        )
        logger.info(
            f"[control | {label}] obs r_group={obs:+.3f}, p_topo={topo['p_topo']:.3e} "
            f"(expected {'reject' if label == 'positive' else 'n.s.'})"
        )
        control_rows.append({"control": label, "r_obs": obs, "p_topo": topo["p_topo"],
                             "n_perm": n_rand})
        null = topo["null_group_topo"]
        null_rows.append(pd.DataFrame({
            "network": focus_network, "measure": "SC", "null_model": f"topo_control_{label}",
            "surrogate": np.arange(null.size), "r": null,
        }))
    return pd.DataFrame(control_rows), pd.concat(null_rows, ignore_index=True)


def results_to_tables(
    results: dict, subject_ids: dict, df_surf: pd.DataFrame, n_rand: int,
) -> dict[str, pd.DataFrame]:
    """Flatten the per-network results into the figure-data tables (`CACHE_COLUMNS`).

    The projection P is stored raw (FC-gradient units); vertex membership is explicit
    (``vertex`` = fsLR-5k surface index), so the plot stage needs no other input to
    place values on the surface.
    """
    cortex_idx = np.flatnonzero(df_surf["hemisphere"].notna().to_numpy())
    hemi_all = df_surf["hemisphere"].to_numpy()
    vertex, stats, subject_r, null = [], [], [], []
    for (network, measure), res in results.items():
        idx = cortex_idx[res["sn_mask_cortex"]]
        dominant, names = compute_dominant_target_network(res)
        if dominant is None:
            dominant_name = np.full(idx.size, None, dtype=object)
        else:
            dominant_name = np.array([names[d] if d >= 0 else None for d in dominant], dtype=object)
        vertex.append(pd.DataFrame({
            "network": network, "measure": measure, "vertex": idx, "hemisphere": hemi_all[idx],
            "g_mpc": res["g_mpc_at_sn"], "P": res["P_mean"], "n_targets": res["n_targets_per_sn"],
            "dominant_network": dominant_name,
        }))
        stats.append({
            "network": network, "measure": measure, "n": res["n"],
            "n_vertices": int(np.sum(np.isfinite(res["g_mpc_at_sn"]) & np.isfinite(res["P_mean"]))),
            "r_group": res["r_group"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "t": res["t"], "p": res["p"],
            **{f"p_{null_model}": res[f"p_{null_model}"] for null_model in NULL_MODELS},
            "n_perm": n_rand,
        })
        subject_r.append(pd.DataFrame({"subject": subject_ids[measure], "network": network,
                                       "measure": measure, "r": res["r_subjects"]}))
        for null_model in NULL_MODELS:
            dist = res[f"null_group_{null_model}"]
            if dist is not None:
                null.append(pd.DataFrame({"network": network, "measure": measure,
                                          "null_model": null_model,
                                          "surrogate": np.arange(dist.size), "r": dist}))
    stats = pd.DataFrame(stats)
    return {
        "vertex": pd.concat(vertex, ignore_index=True),
        "stats": stats[list(CACHE_COLUMNS["stats"])],
        "subject_r": pd.concat(subject_r, ignore_index=True),
        "null": pd.concat(null, ignore_index=True),
    }


def run_compute_stage(paths: dict, hemisphere: str, n_rand: int) -> tuple[dict, dict]:
    """Load the inputs, run every projection and null, and return the figure-data tables.

    Returns
    -------
    tables : dict[str, pd.DataFrame]
        The tables of `CACHE_COLUMNS`, plus ``subject_r`` (per-subject r, with subject IDs).
    manifest : dict
        Settings and counts of the run, recorded next to the tables.
    """
    pni_csv = config.work_dir(GRADIENT_STEP, paths) / "subjects_fslr5k.csv"
    if not pni_csv.exists():
        raise FileNotFoundError(f"fsLR-5k subject list not found at {pni_csv}. Run step 1a first.")
    df_pni = pd.read_csv(pni_csv)

    df_surf = load_yeo_surf_5k()
    df_surf = _attach_gradient_cache(df_surf, hemisphere)
    df_surf = load_fc_gradient_5k(df_surf)
    cortex_mask = df_surf["hemisphere"].notna().to_numpy()

    # The MPC gradient is measure-independent: one per network, shared by every measure.
    net_gradients = {net: _prepare_network_gradient(df_surf, net, df_pni, hemisphere)
                     for net in NETWORKS}

    logger.info("Building Betzel distance-stratified consensus mask + loading SC subjects...")
    mask_G, sc_subjects = build_consensus_mask(
        df_pni["path_sc_5k"].tolist(), df_pni["path_sc_dist_5k"].tolist(),
        df_surf, nbins=10,
    )
    logger.info("Loading group-mean geodesic distance (Moran weights, topological-null bins)...")
    gd_cortex = load_group_mean_matrix(df_pni["path_dist_5k"].tolist(), cortex_mask)

    # Target-side spin null: rotate the oriented FC gradient once and reuse the spun field
    # across every network and measure.
    logger.info(f"Generating {n_rand} FC-gradient spin surrogates (target-side null)...")
    sphere_lh = read_surface(str(DATA_DIR / "surfaces/fsLR-5k.L.sphere.surf.gii"), itype="gii")
    sphere_rh = read_surface(str(DATA_DIR / "surfaces/fsLR-5k.R.sphere.surf.gii"), itype="gii")
    g_fc_spun_cortex = make_fc_spin_surrogates(df_surf, sphere_lh, sphere_rh, n_rand)

    results, subject_ids = compute_network_results(
        df_surf, df_pni, net_gradients, mask_G, sc_subjects, gd_cortex, g_fc_spun_cortex,
        n_rand=n_rand,
    )
    tables = results_to_tables(results, subject_ids, df_surf, n_rand)
    control, control_null = compute_topological_control(
        results, df_surf, net_gradients, sc_subjects, mask_G, gd_cortex, n_rand=n_rand,
    )
    tables["topo_control"] = control
    tables["null"] = pd.concat([tables["null"], control_null], ignore_index=True)

    manifest = {
        "schema": CACHE_SCHEMA,
        "hemisphere": hemisphere,
        "n_rand": n_rand,
        "networks": list(NETWORKS),
        "measures": list(MEASURES),
        "focus_network": FOCUS_NETWORK,
        "n_subjects": {m: len(ids) for m, ids in subject_ids.items()},
    }
    return tables, manifest


# =======================================================================================
# Figure data I/O: the only interface between the two stages
# =======================================================================================

def _cache_path(name: str, hemisphere: str) -> Path:
    """Public figure table ``source_data/step3_connectivity/<name>_<hemisphere>.tsv``."""
    return config.source_dir(STEP) / f"{name}_{hemisphere}.tsv"


def write_figure_data(tables: dict, manifest: dict, paths: dict, hemisphere: str) -> None:
    """Write the public figure tables, the private per-subject table, then the manifest.

    The manifest is removed first and written last, so an interrupted write leaves no
    manifest and the plot stage refuses the partial set.
    """
    manifest_path = config.source_dir(STEP) / "manifest.json"
    manifest_path.unlink(missing_ok=True)
    for name, columns in CACHE_COLUMNS.items():
        path = _cache_path(name, hemisphere)
        config.write_table(tables[name][list(columns)], path)
        logger.info(f"[compute] wrote {path.name} ({len(tables[name])} rows)")
    private = config.work_dir(STEP, paths) / f"subject_r_{hemisphere}.tsv"
    config.write_table(tables["subject_r"], private)
    logger.info(f"[compute] wrote {private} ({len(tables['subject_r'])} rows, private)")
    record = write_manifest(manifest_path, STEP, config.PROJECT_ROOT, **manifest)
    logger.info(f"[compute] wrote {manifest_path.name} (schema {record['schema']}, git {record['git']})")


def read_figure_data(hemisphere: str) -> tuple[dict, dict]:
    """Read and validate the figure tables written by `write_figure_data`.

    Raises
    ------
    FileNotFoundError
        If the manifest or a table is missing (run ``--stage compute`` first).
    ValueError
        If the tables were written with another schema version, for another hemisphere,
        or lack a column.
    """
    manifest_path = config.source_dir(STEP) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run '--stage compute' (or 'all') first.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError(
            f"{manifest_path.name} has schema {manifest.get('schema')}; this step reads "
            f"schema {CACHE_SCHEMA}. Re-run '--stage compute'."
        )
    if manifest.get("hemisphere") != hemisphere:
        raise ValueError(f"{manifest_path.name} records hemisphere {manifest.get('hemisphere')}, "
                         f"not {hemisphere}. Re-run '--stage compute'.")
    tables = {}
    for name, columns in CACHE_COLUMNS.items():
        path = _cache_path(name, hemisphere)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; re-run '--stage compute'.")
        table = config.read_table(path)
        missing = [c for c in columns if c not in table.columns]
        if missing:
            raise ValueError(f"{path.name} lacks columns {missing}; re-run '--stage compute'.")
        tables[name] = table
    logger.info(f"[plot] figure data read: schema {manifest['schema']}, git {manifest['git']}, "
                f"created {manifest['created']}, n_rand={manifest['n_rand']}")
    return tables, manifest


# =======================================================================================
# Plot stage: figure making only (reads the figure-data tables)
# =======================================================================================

def _network_rgb(network) -> np.ndarray:
    """Yeo colour of `network` (background colour for any other label)."""
    return yeo7_rgb[_NETWORK_INDEX.get(network, len(yeo7_rgb) - 1)]


def _dominant_colors(dominant: pd.Series, network: str) -> np.ndarray:
    """Point colours: Yeo colour of each vertex's dominant target network, or of the
    source `network` where the vertex has none."""
    return np.array([_network_rgb(d if isinstance(d, str) else network) for d in dominant])


def _stats_row(stats: pd.DataFrame, network: str, measure: str) -> pd.Series:
    rows = stats[(stats["network"] == network) & (stats["measure"] == measure)]
    if len(rows) != 1:
        raise KeyError(f"Expected one stats row for {network} x {measure}; found {len(rows)}.")
    return rows.iloc[0]


def _vertex_rows(vertex: pd.DataFrame, network: str, measure: str) -> pd.DataFrame:
    rows = vertex[(vertex["network"] == network) & (vertex["measure"] == measure)]
    if rows.empty:
        raise KeyError(f"No vertex rows for {network} x {measure}.")
    return rows


def _surface_map(vertex: pd.DataFrame, network: str, measure: str, column: str = "P_z") -> np.ndarray:
    """`column` of one network x measure on the full fsLR-5k surface (NaN elsewhere)."""
    rows = _vertex_rows(vertex, network, measure)
    out = np.full(N_TOTAL_5K, np.nan)
    out[rows["vertex"].to_numpy()] = rows[column].to_numpy(dtype=float)
    return out


def _with_display_columns(vertex: pd.DataFrame) -> pd.DataFrame:
    """Add ``P_z``: P z-scored within each network x measure (NaN-omitting, ddof=0).

    A monotone rescale for display only (SD units, shared +/-2 SD colour scale); the
    rank-based statistics in the stats table do not depend on it.
    """
    vertex = vertex.copy()
    grouped = vertex.groupby(["network", "measure"])["P"]
    vertex["P_z"] = (vertex["P"] - grouped.transform("mean")) / grouped.transform("std", ddof=0)
    return vertex


def _network_order(stats: pd.DataFrame, primary: str = MEASURES[0]) -> list[str]:
    """Networks ordered by the primary measure's signed group effect (largest first)."""
    return (stats[stats["measure"] == primary]
            .sort_values("r_group", ascending=False, kind="stable")["network"].tolist())


def _null_values(null: pd.DataFrame, network: str, measure: str,
                 null_model: str = "moran") -> np.ndarray:
    """Finite group r of every surrogate of a null (empty if absent)."""
    r = null.loc[(null["network"] == network) & (null["measure"] == measure)
                 & (null["null_model"] == null_model), "r"].to_numpy(dtype=float)
    return r[np.isfinite(r)]


def _null_band(null: pd.DataFrame, network: str, measure: str,
               null_model: str = "moran") -> tuple[float, float, float]:
    """2.5th, 50th and 97.5th percentiles of a null's group r (NaN if absent)."""
    r = _null_values(null, network, measure, null_model)
    if not r.size:
        return np.nan, np.nan, np.nan
    lo, med, hi = np.percentile(r, [2.5, 50, 97.5])
    return float(lo), float(med), float(hi)


def _plot_measure_zoom_fig2(
    values: np.ndarray, prefix: str, title: str, *, rois: list[dict], surf_lh, surf_rh,
    out_dir: Path, suffix: str, cmap: str, color_range: tuple[float, float],
    cbar_labels: tuple[str, str], cbar_title: str | None,
    outline: np.ndarray | None = None,
) -> tuple[float, float]:
    """Figure 1-style brain row with zoom insets, plus the inset row alone.

    Writes ``{prefix}_zoom_{suffix}.svg`` (brain row with marker circles, insets and
    region names, as ``figure_1a_brain_zoom.svg``) and ``{prefix}_zoom_roi_{suffix}.svg``
    (the inset row alone, titled, colour bar drawn, no region names, as the Figure 1b/2b
    zoom rows), both on the Figure 1 window (``ZOOM_KW``), with ``outline`` (the
    source-network border from ``mask_outline``) traced on the brains and insets. Mirrors
    ``step2b_ieeg_mica._plot_measure_zoom``.

    Returns
    -------
    tuple[float, float]
        Inset-row centre and height (pt) of the ``_zoom_roi`` figure, which place the
        ridgeline-plot rows level with the stacked inset rows.
    """
    kwargs = dict(ZOOM_KW, cmap=cmap, color_range=color_range, cbar_labels=cbar_labels,
                  cbar_title=cbar_title, outline=outline)
    plot_hemispheres_zoom(surf_lh, surf_rh, values, rois,
                          filename=out_dir / f"{prefix}_zoom_{suffix}.svg", **kwargs)
    fig = plot_hemispheres_zoom(surf_lh, surf_rh, values,
                                [dict(r, label=None) for r in rois], roi_only=True,
                                draw_color_bar=True, title=title,
                                filename=out_dir / f"{prefix}_zoom_roi_{suffix}.svg", **kwargs)
    geometry = (fig.inset_cy, fig.get_size_inches()[1] * 72)
    plt.close("all")
    return geometry


def plot_figure_3a(
    tables: dict, networks: list[str], df_atlas: pd.DataFrame, surf_lh, surf_rh, out_dir: Path,
    hemisphere: str,
) -> None:
    """Figure 3A (Figure 1/2 norm): per-modality projection zoom rows + aligned network lollipops.

    For the focus network, each connectivity measure is drawn as a Figure-1-style brain
    row with the ``SALIENCE_ZOOM_ROIS`` insets (aINS, dACC, PCU, IPL) and the inset row
    alone, P z-scored within the network on the shared +/-2 SD scale in
    ``CustomCmap_connectivity``, with the network border traced (``mask_outline``).
    ``plot_network_lollipop`` (``figure_3a_forest_{hemi}.svg``) then places, level with
    each measure's inset row, the group $r$ of all seven networks (in ``networks`` order,
    signs as produced) over the central 95 % of each network's Moran null, in the network
    colours, dot filled where the uncorrected $p_{moran}$ < 0.05 (as Figures 1b and 2) and
    then labelled with its $r$ and $p_{moran}$, the focus network in bold. The focus network alone is also drawn against its Moran
    null with the uncorrected $p_{moran}$ (as Figures 1b and 2b): as a ridgeline
    (``figure_3a_ridge_{hemi}.svg``) and with the null's central 95 % and median
    (``figure_3a_forest_band_{hemi}.svg``).
    """
    vertex, stats, null = tables["vertex"], tables["stats"], tables["null"]
    outline = mask_outline(surf_lh, surf_rh, compute_network_mask(df_atlas, FOCUS_NETWORK, hemisphere))
    rois = salience_zoom_rois(df_atlas, hemisphere)

    summary_rows, nulls, blocks, row_geometry = [], [], [], []
    for measure in MEASURES:
        row_geometry.append(_plot_measure_zoom_fig2(
            _surface_map(vertex, FOCUS_NETWORK, measure), f"figure_3a_{measure}",
            _MODALITY_TITLES[measure], rois=rois, surf_lh=surf_lh, surf_rh=surf_rh,
            out_dir=out_dir, suffix=hemisphere, cmap=connectivity_cmap.name,
            color_range=SALIENCE_Z_RANGE, cbar_labels=SALIENCE_Z_LABELS, cbar_title="z",
            outline=outline,
        ))
        srow = _stats_row(stats, FOCUS_NETWORK, measure)
        lo, med, hi = _null_band(null, FOCUS_NETWORK, measure)
        summary_rows.append(dict(measure=measure, r=srow["r_group"], p=srow["p_moran"],
                                 null_lo=lo, null_med=med, null_hi=hi))
        nulls.append(_null_values(null, FOCUS_NETWORK, measure))
        lines = []
        for network in networks:
            nrow = _stats_row(stats, network, measure)
            n_lo, _, n_hi = _null_band(null, network, measure)
            lines.append(dict(label=yeo7_abbrev.get(network, network), r=nrow["r_group"], p=nrow["p_moran"],
                              p_print=nrow["p_moran"], null_lo=n_lo, null_hi=n_hi, color=_network_rgb(network)))
        blocks.append(pd.DataFrame(lines))
    if len(set(row_geometry)) != 1:
        raise RuntimeError(f"Inset rows differ in geometry {set(row_geometry)}; "
                           f"the companion rows cannot be aligned.")

    summary = pd.DataFrame(summary_rows)
    for measure, block in zip(MEASURES, blocks):
        for line in block.itertuples(index=False):
            # The band is the central 95 % of the null; the test is two-sided on |r|. They
            # agree for a zero-centred symmetric null; flag any line where they do not.
            outside = line.r < line.null_lo or line.r > line.null_hi
            if outside != (line.p_print < 0.05):
                logger.warning(f"[Figure 3A] {measure} {line.label}: r={line.r:+.3f} lies "
                               f"{'outside' if outside else 'inside'} the Moran-null 95 % band "
                               f"[{line.null_lo:+.3f}, {line.null_hi:+.3f}] but p_moran={line.p_print:.3f}.")
    # Symmetric x-range covering every observed r and null band, rounded up to a 0.05
    # tick so the two ends and 0 read cleanly, never past |r| = 1.
    values = np.concatenate([b[["r", "null_lo", "null_hi"]].to_numpy().ravel() for b in blocks])
    lim = min(1.0, max(0.1, np.ceil(np.nanmax(np.abs(values)) * 1.15 * 20) / 20))
    plot_network_lollipop(blocks, *row_geometry[0], out_dir / f"figure_3a_forest_{hemisphere}.svg",
                          highlight=yeo7_abbrev.get(FOCUS_NETWORK), xlim=(-lim, lim), xlabel="Spearman r",
                          p_label="$p_{m}$")
    plot_ridge(summary, nulls, *row_geometry[0], out_dir / f"figure_3a_ridge_{hemisphere}.svg",
               xlim=(-lim, lim), xlabel="Spearman r")
    # The same rows with the null as its central 95 % band and median (forest variant).
    plot_forest(summary, *row_geometry[0], out_dir / f"figure_3a_forest_band_{hemisphere}.svg",
                xlim=(-lim, lim), xlabel="Spearman r")
    plt.close("all")
    logger.info(f"[Figure 3A] {len(MEASURES)} projection zoom rows, network lollipops and the focus "
                f"network's Moran-null ridgeline and band written (figure_3a_*_zoom_roi_{hemisphere}.svg, "
                f"figure_3a_forest_{hemisphere}.svg, figure_3a_ridge_{hemisphere}.svg, "
                f"figure_3a_forest_band_{hemisphere}.svg)")


def plot_figure_3a_scatter_supp(tables: dict, out_dir: Path, hemisphere: str) -> None:
    """Supplementary Figure 3A: per-measure vertex scatter behind the ridgeline.

    One panel per connectivity measure: the focus network's z-scored projection P (x)
    against its within-network MPC gradient (y, shared), points coloured by dominant
    target network, black regression line, and the group $r$ with $p_{moran}$,
    $p_{spin}$ and (SC, MPC, FC) $p_{topo}$ — each labelled by its null, all uncorrected.
    """
    vertex, stats = tables["vertex"], tables["stats"]
    focus = vertex[vertex["network"] == FOCUS_NETWORK]
    g_all = focus["g_mpc"].to_numpy(dtype=float)
    g_all = g_all[np.isfinite(g_all)]
    gpad = 0.05 * (np.ptp(g_all) or 1.0)
    gylim = (g_all.min() - gpad, g_all.max() + gpad)
    # x-range covers every point (the old fixed +/-5 could clip outliers silently).
    xlim = max(5.0, 1.05 * float(np.nanmax(np.abs(focus["P_z"].to_numpy(dtype=float)))))

    fig, axes = plt.subplots(1, len(MEASURES), figsize=(3.0 * len(MEASURES), 3.2),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, measure in zip(axes, MEASURES):
        rows = _vertex_rows(vertex, FOCUS_NETWORK, measure)
        srow = _stats_row(stats, FOCUS_NETWORK, measure)
        valid = rows["g_mpc"].notna() & rows["P_z"].notna()
        x, y = rows.loc[valid, "P_z"].to_numpy(), rows.loc[valid, "g_mpc"].to_numpy()
        colors = _dominant_colors(rows.loc[valid, "dominant_network"], FOCUS_NETWORK)
        ax.scatter(x, y, s=12, alpha=0.75, c=colors, edgecolor="none", rasterized=True)
        sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 2.5}, ax=ax, seed=0)
        stat_txt = (f"$r={srow['r_group']:+.2f}$\n"
                    f"$p_{{moran}}={format_pvalue(srow['p_moran'])}$")
        if np.isfinite(srow["p_topo"]):
            stat_txt += f"\n$p_{{topo}}={format_pvalue(srow['p_topo'])}$"
        if np.isfinite(srow["p_spin"]):
            stat_txt += f"\n$p_{{spin}}={format_pvalue(srow['p_spin'])}$"
        ax.text(0.05, 0.95, stat_txt, transform=ax.transAxes, va="top",
                fontweight="bold", fontsize=12)
        ax.set_xlabel(f"{measure} projection", fontsize=16)
        ax.set_xlim(-xlim, xlim)
        ax.set_xticks([-4, 0, 4])
        ax.set_ylim(*gylim)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.set_box_aspect(1)
    axes[0].set_ylabel("MPC gradient", fontsize=16)
    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(out_dir / f"figure_3a_scatter_supp_{hemisphere}.svg",
                bbox_inches="tight", transparent=True)
    plt.close(fig)
    logger.info(f"[Figure 3A] supplementary scatter grid written to "
                f"figure_3a_scatter_supp_{hemisphere}.svg")


def _sig_stars(p: float) -> str:
    """Uncorrected p-value to annotation (n.s. / * / ** / ***)."""
    if not np.isfinite(p):
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def plot_network_bubble_matrix(
    stats: pd.DataFrame, networks: list[str], out_dir: Path, hemisphere: str,
) -> None:
    """Figure 3B summary: bubble matrix of the group effect for every network × measure.

    A network (row) × measure (column) grid. Each cell is a disc whose colour encodes the
    group correlation $\\bar r$ (MPC gradient vs connectivity projection P) on a
    diverging scale and whose area encodes $|\\bar r|$; the uncorrected Moran p is the
    disc's black edge ring (p < 0.05) and the stars above it, and the signed
    value is printed below. Rows follow `networks` (the primary measure's effect order).

    Because the within-network MPC gradient is a diffusion-map eigenvector whose
    polarity is arbitrary per network, coefficients are shown exactly as produced
    (no sign manipulation): the sign is interpretable *within* a row (same g_MPC
    across measures) but not *between* rows, where only $|\\bar r|$ and significance
    are comparable — hence area, not colour, carries the cross-network message.
    """
    n_meas, n_net = len(MEASURES), len(networks)
    R = np.full((n_meas, n_net), np.nan)
    Q = np.full((n_meas, n_net), np.nan)
    for i, m in enumerate(MEASURES):
        for j, net in enumerate(networks):
            srow = _stats_row(stats, net, m)
            R[i, j], Q[i, j] = srow["r_group"], srow["p_moran"]

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
    ax.set_xticklabels(MEASURES)
    ax.set_yticks([(n_net - 1) - j for j in range(n_net)])
    ax.set_yticklabels([yeo7_abbrev.get(net, net) for net in networks])
    for tick, net in zip(ax.get_yticklabels(), networks):
        tick.set_color(_network_rgb(net))
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

    out = out_dir / f"figure_3b_network_summary_{hemisphere}.svg"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Figure 3B] bubble-matrix summary written to {out.name}")


def plot_figure_3b_test_grid(tables: dict, networks: list[str], out_dir: Path, hemisphere: str) -> None:
    """Figure 3B test grid: every network x measure group $r$ against its Moran null.

    One row per network (in `networks` order, level with the stacked 2B brain rows, each
    300 pt tall without insets) and one column per measure, on one shared $r$ axis. The
    ridge version draws each cell's Moran-null density with the observed $r$ as a dot on
    its baseline; the forest version the null's central 95 % and median. The dot is filled
    where the uncorrected $p_{moran}$ < 0.05, and that p is printed. Signs are as produced:
    the g_MPC polarity is arbitrary per network, so they compare within a row only.
    """
    stats, null = tables["stats"], tables["null"]
    cells, nulls = {}, {}
    for measure in MEASURES:
        rows, dists = [], []
        for network in networks:
            srow = _stats_row(stats, network, measure)
            lo, med, hi = _null_band(null, network, measure)
            rows.append(dict(r=srow["r_group"], p=srow["p_moran"], null_lo=lo, null_med=med, null_hi=hi))
            dists.append(_null_values(null, network, measure))
        cells[measure], nulls[measure] = pd.DataFrame(rows), dists
    # One symmetric axis for every cell, covering all r and null bands (as Figure 3A).
    values = np.concatenate([c[["r", "null_lo", "null_hi"]].to_numpy().ravel() for c in cells.values()])
    lim = min(1.0, max(0.1, np.ceil(np.nanmax(np.abs(values)) * 1.15 * 20) / 20))
    row_h = ZOOM_KW["size"][1]
    for kind in ("ridge", "forest"):
        plot_test_grid(cells, row_h / 2, row_h, out_dir / f"figure_3b_{kind}_{hemisphere}.svg",
                       nulls=nulls, kind=kind, row_labels=[yeo7_abbrev.get(n, n) for n in networks],
                       row_colors=[_network_rgb(n) for n in networks], xlim=(-lim, lim),
                       xlabel="Spearman r", p_label="$p_{m}$")
        plt.close("all")
    logger.info(f"[Figure 3B] network x measure test grids written "
                f"(figure_3b_ridge_{hemisphere}.svg, figure_3b_forest_{hemisphere}.svg)")


def plot_figure_3b_brain_maps(
    tables: dict, df_atlas: pd.DataFrame, surf_lh, surf_rh, networks: list[str],
    out_dir: Path, hemisphere: str,
) -> None:
    """Figure 3B brain rows: each network's z-scored projection P, per measure.

    Drawn as the Figure 3A brain rows (four views, ``CustomCmap_connectivity`` on the
    shared +/-2 SD scale, network border traced) without zoom insets, 1450 x 300 pt.
    The colour-bar column is kept but left empty: the scale is that of the 2A rows, and
    a bar would be cut by a thumbnail crop in a composite figure. The ``_rho_`` in the
    file names is kept from earlier layouts of results/figure3.svg, which linked them.
    """
    vertex = tables["vertex"]
    kwargs = dict(ZOOM_KW, cmap=connectivity_cmap.name, color_range=SALIENCE_Z_RANGE,
                  draw_color_bar=False)
    outlines = {net: mask_outline(surf_lh, surf_rh, compute_network_mask(df_atlas, net, hemisphere))
                for net in networks}
    for measure in MEASURES:
        for network in networks:
            plot_hemispheres_zoom(
                surf_lh, surf_rh, _surface_map(vertex, network, measure), [],
                outline=outlines[network],
                filename=out_dir / f"figure_3b_brain_{measure}_rho_{network}.svg", **kwargs)
            plt.close("all")
    logger.info(f"[Figure 3B] {len(MEASURES) * len(networks)} brain rows written "
                f"(figure_3b_brain_<measure>_rho_<network>.svg)")


def _plot_network_scatter_grid(
    tables: dict, measure: str, networks: list[str], out_dir: Path,
) -> None:
    """Per-network scatter grid (one row) for a single measure, effect-ordered.

    Numeric stats live in the bubble-matrix summary, so each panel carries only a
    colour-coded network title and the shared axes.
    """
    vertex = tables["vertex"]
    fig, axes = plt.subplots(1, len(networks), figsize=(3.0 * len(networks), 3.4),
                             sharex=True, sharey=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, network in zip(axes, networks):
        rows = _vertex_rows(vertex, network, measure)
        valid = rows["g_mpc"].notna() & rows["P_z"].notna()
        x, y = rows.loc[valid, "P_z"].to_numpy(), rows.loc[valid, "g_mpc"].to_numpy()
        colors = _dominant_colors(rows.loc[valid, "dominant_network"], network)
        ax.scatter(x, y, s=10, alpha=0.7, c=colors, edgecolor="none", rasterized=True)
        sns.regplot(x=x, y=y, scatter=False, color="black", line_kws={"linewidth": 1}, ax=ax, seed=0)
        ax.set_title(yeo7_abbrev.get(network, network), fontdict={"color": _network_rgb(network)})
        ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.1f}'))
    axes[0].set_ylabel("MPC gradient")
    fig.supxlabel(f"{measure} projection P (z)")

    sns.despine(fig=fig)
    plt.savefig(out_dir / f"figure_3b_distance_network_{measure}.svg")
    plt.close(fig)


def plot_topological_control(tables: dict, out_dir: Path) -> None:
    """Supplement: SC topological-null distributions of the two synthetic control maps."""
    control, null = tables["topo_control"], tables["null"]
    titles = {
        "positive": "Positive control\n(wiring-aligned map)",
        "negative": "Negative control\n(geometry-only map)",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, label in zip(axes, ("positive", "negative")):
        crow = control[control["control"] == label].iloc[0]
        dist = null.loc[null["null_model"] == f"topo_control_{label}", "r"].to_numpy(dtype=float)
        ax.hist(dist[np.isfinite(dist)], bins=30, color="0.7", edgecolor="white")
        ax.axvline(crow["r_obs"], color="crimson", lw=2.5,
                   label=f"observed\n$p_{{topo}}={format_pvalue(crow['p_topo'])}$")
        ax.set_title(titles[label])
        ax.set_xlabel(r"group $r$ under SC topological null")
        ax.legend(fontsize=9, loc="upper center")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0].set_ylabel("surrogate count")
    out = out_dir / "figure_3_supp_topo_control.svg"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[control] topological-null control figure written to {out.name}")


def run_plot_stage(tables: dict, hemisphere: str) -> None:
    """Draw every Figure 3 panel and supplement from the figure tables."""
    out_dir = config.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = dict(tables, vertex=_with_display_columns(tables["vertex"]))
    networks = _network_order(tables["stats"])
    df_atlas = load_yeo_surf_5k()
    surf_lh, surf_rh = load_inflated_surfaces("fsLR-5k")

    plot_figure_3a(tables, networks, df_atlas, surf_lh, surf_rh, out_dir, hemisphere)
    plot_figure_3a_scatter_supp(tables, out_dir, hemisphere)
    plot_network_bubble_matrix(tables["stats"], networks, out_dir, hemisphere)
    plot_figure_3b_test_grid(tables, networks, out_dir, hemisphere)
    plot_figure_3b_brain_maps(tables, df_atlas, surf_lh, surf_rh, networks, out_dir, hemisphere)
    for measure in MEASURES:
        _plot_network_scatter_grid(tables, measure, networks, out_dir)
    plot_topological_control(tables, out_dir)


def compute(settings: dict, paths: dict) -> None:
    """Projection, nulls and control for every network x measure; writes the figure tables."""
    hemi, n_rand = settings["hemisphere"], settings["n_permutations"]
    logger.info("Surface space  : fsLR-5k, Yeo 7-network labels")
    logger.info("Statistic      : per-network-vertex connectivity-weighted projection P[i]")
    logger.info("Group test     : Fisher-z r_s + one-sample t-test (random effects)")
    logger.info("SC weights     : SIFT2 masked by Betzel consensus (per-subject log10)")
    logger.info("GD weights     : 1/GD (within-hemisphere)")
    logger.info("MPC/FC weights : positive connections only (weighted-mean projection)")
    logger.info("Null models    : Moran spectral randomisation (map smoothness, all measures) + "
                "geometry-preserving topological null (wiring specificity, SC/MPC/FC) + "
                "target-side FC-gradient spin null (anatomical alignment, all measures)")
    tables, manifest = run_compute_stage(paths, hemi, n_rand)
    write_figure_data(tables, manifest, paths, hemi)


def plot(settings: dict) -> None:
    """Every Figure 3 panel, from source_data only."""
    tables, _ = read_figure_data(settings["hemisphere"])
    run_plot_stage(tables, settings["hemisphere"])


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 3: gradient-weighted connectivity projection (Figure 3).")

"""Connectomes and the gradient-weighted connectivity projection (Figure 3, fsLR-5k).

Loads micapipe connectomes, builds the distance-stratified consensus mask of structural
connections (Betzel et al., 2019), computes for each network vertex the
connectivity-weighted mean of a target map, and tests the projection against Moran,
topological and target-side spin null models.
"""
import logging

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp, t as t_dist

from .stats import (
    empirical_p_excess_magnitude,
    empirical_p_twosided,
    moran_surrogates_blockwise,
    rank_corr_columns,
)

logger = logging.getLogger(__name__)


N_LH_5K = 4842


N_TOTAL_5K = 9684


# Modalities that receive the geometry-preserving topological null. GD is excluded: its
# weights are 1/geodesic-distance, so a within-distance-bin target reassignment barely
# changes them and the null would be uninformative.
TOPO_MODALITIES = ("SC", "MPC", "FC")


def load_subject_matrix(path, cortex_mask: np.ndarray) -> np.ndarray:
    """Load a fsLR-5k vertex x vertex .shape.gii connectome restricted to cortex.

    micapipe stores fsLR-5k connectomes as upper triangular; the symmetrisation
    `triu(d, 1) + d.T` mirrors the upper triangle and recovers the diagonal from
    d.T. Negatives are clipped to 0; every modality uses only positive connections
    downstream, so this both removes SC numerical noise and discards the
    anticorrelated / negative-partial-correlation edges of FC and MPC.

    Parameters
    ----------
    path : str or Path
        Path to the .shape.gii connectome.
    cortex_mask : np.ndarray of bool, shape (n_vertices,)
        Boolean mask selecting cortical vertices (excludes medial wall).

    Returns
    -------
    M : np.ndarray, shape (n_cortex, n_cortex), float32
    """
    d = nib.load(path).darrays[0].data.astype(np.float32)
    d = d[np.ix_(cortex_mask, cortex_mask)]
    d = np.triu(d, 1) + d.T
    d[d < 0] = 0.0
    return d


def load_group_mean_matrix(files: list, cortex_mask: np.ndarray) -> np.ndarray:
    """Across-subject mean of per-subject cortex x cortex matrices (e.g. geodesic distance).

    Accumulates a float64 running sum one subject at a time, so peak memory is one
    accumulator plus one transient matrix rather than the whole
    (n_sub, n_cortex, n_cortex) stack.

    Parameters
    ----------
    files : list of str or Path
        Per-subject fsLR-5k .shape.gii matrices, read with `load_subject_matrix`.
    cortex_mask : np.ndarray of bool, shape (n_vertices,)
        Boolean mask selecting cortical vertices (excludes medial wall).

    Returns
    -------
    M : np.ndarray, shape (n_cortex, n_cortex), float32
    """
    if not files:
        raise ValueError("load_group_mean_matrix: no files supplied.")
    acc = None
    for f in files:
        M = load_subject_matrix(f, cortex_mask)
        acc = M.astype(np.float64) if acc is None else acc + M
    return (acc / len(files)).astype(np.float32)


def fcn_group_bins(
    adj: np.ndarray, dist: np.ndarray, hemiid: np.ndarray, nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-dependent group-representative SC thresholding (Betzel et al. 2019).

    Generates a binary group-consensus mask that preserves within- and between-
    hemisphere connection-length distributions. Tractography systematically
    over-represents short streamlines; binning by distance prevents the consensus
    from collapsing toward short-range edges.

    Parameters
    ----------
    adj : (n, n, n_sub) per-subject SC matrices.
    dist : (n, n) mean tract-distance matrix (streamline length).
    hemiid : (n,) bool, True for RH.
    nbins : int, number of distance bins.

    Returns
    -------
    G : (n, n) symmetric binary group-consensus matrix (distance-dependent).
    Gc : (n, n) symmetric binary group-consensus matrix (consistency-based).

    Betzel, R. F. et al. (2019). Network Neuroscience 3(2), 475-496.
    """
    assert adj.shape[0] == adj.shape[1], "adj must be square in its first two dims"
    if hemiid.ndim == 1:
        hemiid = hemiid[:, np.newaxis]

    n, nsub = adj.shape[0], adj.shape[-1]
    nonzero_dist = dist[np.nonzero(dist)]
    distbins = np.linspace(nonzero_dist.min(), nonzero_dist.max(), nbins + 1)
    distbins[-1] += 1

    C = np.sum(adj > 0, axis=2)
    W = np.sum(adj, axis=2) / np.where(C > 0, C, np.nan)
    W = np.nan_to_num(W, nan=0.0)

    inter_hemi_mask = np.dot(hemiid, ~hemiid.T)
    inter_hemi_mask = np.logical_or(inter_hemi_mask, inter_hemi_mask.T)

    Grp = np.zeros((n, n, 2))
    Gc_arr = np.zeros((n, n, 2))

    for j in range(2):
        inter_hemi = ~inter_hemi_mask if j else inter_hemi_mask
        m = dist * inter_hemi
        D = (adj > 0) * (dist * np.triu(inter_hemi))[..., np.newaxis]
        D = D[np.nonzero(D)]
        if len(D) == 0:
            continue
        tgt = len(D) / nsub

        G = np.zeros((n, n))
        for i_bin in range(nbins):
            mask = np.where(np.triu((m >= distbins[i_bin]) & (m < distbins[i_bin + 1]), 1))
            if len(mask[0]) == 0:
                continue
            n_D_bin = np.sum((D >= distbins[i_bin]) & (D < distbins[i_bin + 1]))
            frac = int(np.round(tgt * n_D_bin / len(D)))
            c = C[mask]
            idx = np.argsort(c)[::-1]
            G[mask[0][idx[:frac]], mask[1][idx[:frac]]] = 1
        Grp[:, :, j] = G

        I = np.where(np.triu(inter_hemi, 1))
        w = W[I]
        idx = np.argsort(w)[::-1]
        w_mat = np.zeros((n, n))
        nnz = int(G.sum())
        if nnz > 0:
            w_mat[I[0][idx[:nnz]], I[1][idx[:nnz]]] = 1
        Gc_arr[:, :, j] = w_mat

    G = np.sum(Grp, 2)
    G = G + G.T
    Gc = np.sum(Gc_arr, 2)
    Gc = Gc + Gc.T
    return G.astype(bool), Gc.astype(bool)


def build_consensus_mask(
    sc_files: list, dist_files: list, df_yeo_surf_5k: pd.DataFrame, nbins: int = 10,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build the Betzel distance-dependent consensus mask + return per-subject SCs.

    The mask is built once across all subjects; per-subject SIFT2 weights are kept
    and intended to be multiplied elementwise by the mask downstream (per-subject
    random-effects inference, mask removes spurious/non-reproducible edges).

    Parameters
    ----------
    sc_files : list of str/Path
        Per-subject SC .shape.gii paths.
    dist_files : list of str/Path
        Per-subject tract-distance .shape.gii paths (used only to build the mask).
    df_yeo_surf_5k : pd.DataFrame
        Surface DataFrame with 'hemisphere' column to derive cortex_mask + hemiid.
    nbins : int
        Distance bins for `fcn_group_bins`.

    Returns
    -------
    G : (n_cortex, n_cortex) bool, group-consensus mask.
    sc_subjects : list of np.ndarray
        Per-subject SC matrices restricted to cortex (float32, symmetric).
    """
    if not sc_files or not dist_files:
        raise FileNotFoundError("Both SC and distance file lists are required.")

    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemiid = (df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values == "RH")

    sc_subjects = [load_subject_matrix(f, cortex_mask) for f in sc_files]
    adj = np.stack(sc_subjects, axis=-1)

    dist_stack = [load_subject_matrix(f, cortex_mask) for f in dist_files]
    dist = np.mean(np.stack(dist_stack, axis=0), axis=0)
    del dist_stack

    G, _ = fcn_group_bins(adj, dist, hemiid, nbins)
    logger.info(
        f"Betzel consensus mask built: density={G.mean():.4f}, "
        f"n_edges={int(G.sum() // 2)}, nbins={nbins}"
    )
    return G, sc_subjects


def prepare_weights(
    W_raw: np.ndarray, modality: str, hemi_cortex: np.ndarray,
    sn_mask_cortex: np.ndarray, *, mask_G: np.ndarray | None = None,
) -> np.ndarray:
    """Modality-aware preprocessing of a per-subject cortex x cortex matrix.

    Always sets to NaN: diagonal, within-network edges. Cross-hemisphere is set
    NaN for GD (the spec requires within-hemisphere GD only, mirroring micapipe
    geodesic distance which is undefined across hemispheres). Every modality keeps
    only positive connections: SC log-transforms positives, GD inverts positive
    distances, and MPC/FC retain positive correlations while non-positive entries
    are dropped to NaN.

    Parameters
    ----------
    W_raw : (n_cortex, n_cortex) float
        Raw subject matrix (cortex-restricted, symmetric, non-negative after the
        load-time clip of negatives to zero).
    modality : {'SC', 'GD', 'MPC', 'FC'}
    hemi_cortex : (n_cortex,) array of {'LH','RH'}
    sn_mask_cortex : (n_cortex,) bool, True for source-network vertices.
    mask_G : (n_cortex, n_cortex) bool, optional
        Betzel consensus mask for SC.

    Returns
    -------
    W : (n_cortex, n_cortex) float
        Preprocessed weights, with NaN marking excluded entries.
    """
    n = W_raw.shape[0]
    W = W_raw.astype(np.float64, copy=True)

    if modality == "SC":
        if mask_G is not None:
            W = W * mask_G
        positives = W > 0
        if positives.any():
            eps = W[positives].min()
            log_W = np.full_like(W, np.nan, dtype=np.float64)
            log_W[positives] = np.log10(W[positives] / eps)
            W = log_W
        else:
            W = np.full_like(W, np.nan, dtype=np.float64)

    elif modality == "GD":
        same_hemi = hemi_cortex[:, None] == hemi_cortex[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where(W > 0, 1.0 / W, np.nan)
        W[~same_hemi] = np.nan

    elif modality in ("MPC", "FC"):
        # Positive connections only: keep positive partial correlations (MPC) /
        # positive correlations (FC); drop non-positive (negative/zero) edges.
        # Whole-brain, no hemisphere restriction.
        W = np.where(W > 0, W, np.nan)

    else:
        raise ValueError(f"modality must be one of 'SC', 'GD', 'MPC', 'FC'; got {modality}")

    diag_idx = np.arange(n)
    W[diag_idx, diag_idx] = np.nan
    within_sn = sn_mask_cortex[:, None] & sn_mask_cortex[None, :]
    W[within_sn] = np.nan

    return W


def weighted_mean_projection(
    W_sub: np.ndarray, g_targets: np.ndarray, *, min_valid: int = 10,
) -> np.ndarray:
    """Core weighted-mean projection on an already-sliced SN x target submatrix.

    P[i] = sum_j w_ij g_targets[j] / sum_j w_ij over positive, finite weights only
    (NaN/non-positive weights are excluded from numerator AND denominator). Rows with
    fewer than `min_valid` finite-positive targets, or a non-positive denominator,
    return NaN. Shared by `compute_projection_score` (full-matrix entry point) and the
    geometry-preserving topological null, which rewires `W_sub` directly.

    Parameters
    ----------
    W_sub : (n_sn, n_other) float, preprocessed SN->target weights (NaN = excluded).
    g_targets : (n_other,) float, FC gradient at the target vertices.
    min_valid : int

    Returns
    -------
    P : (n_sn,) float
    """
    valid = np.isfinite(W_sub) & (W_sub > 0) & np.isfinite(g_targets)[None, :]
    W_eff = np.where(valid, W_sub, 0.0)
    g_safe = np.where(np.isfinite(g_targets), g_targets, 0.0)

    num = W_eff @ g_safe
    den = W_eff.sum(axis=1)
    n_valid = valid.sum(axis=1)

    return np.where(n_valid >= min_valid, num / np.where(den > 0, den, np.nan), np.nan)


def compute_projection_score(
    W: np.ndarray, g_fc_cortex: np.ndarray,
    sn_idx_cortex: np.ndarray, other_idx_cortex: np.ndarray,
    *, min_valid: int = 10,
) -> np.ndarray:
    """Weighted-mean projection: P[i] = sum_j w_ij g_FC[j] / sum_j w_ij.

    Slices the SN x target submatrix from the full preprocessed `W` and delegates the
    arithmetic to `weighted_mean_projection`. Only positive weights contribute (NaN
    weights are excluded from numerator AND denominator). Returns NaN for rows with
    fewer than `min_valid` finite targets, or with zero/negative denominator.

    Parameters
    ----------
    W : (n_cortex, n_cortex), preprocessed weights.
    g_fc_cortex : (n_cortex,), whole-brain FC gradient on cortex.
    sn_idx_cortex : (n_cortex,) bool, source network mask.
    other_idx_cortex : (n_cortex,) bool, target set (non-network cortex).
    min_valid : int

    Returns
    -------
    P : (n_sn,) float
    """
    W_sub = W[np.ix_(sn_idx_cortex, other_idx_cortex)]
    g_targets = g_fc_cortex[other_idx_cortex]
    return weighted_mean_projection(W_sub, g_targets, min_valid=min_valid)


def _fisher_z_group(r_subjects: np.ndarray) -> dict:
    """Aggregate per-subject correlations: Fisher-z mean, t-test, 95% CI."""
    finite = np.isfinite(r_subjects)
    z = np.arctanh(np.clip(r_subjects[finite], -0.999, 0.999))
    n = z.size
    if n < 2:
        return {
            "r_group": np.nan, "t": np.nan, "p": np.nan,
            "ci_low": np.nan, "ci_high": np.nan, "n": n,
        }
    z_mean = z.mean()
    z_sd = z.std(ddof=1)
    t_stat, p_val = ttest_1samp(z, 0.0)
    t_crit = t_dist.ppf(0.975, df=n - 1)
    se = z_sd / np.sqrt(n)
    return {
        "r_group": float(np.tanh(z_mean)),
        "t": float(t_stat), "p": float(p_val),
        "ci_low": float(np.tanh(z_mean - t_crit * se)),
        "ci_high": float(np.tanh(z_mean + t_crit * se)),
        "n": int(n),
    }


def compute_projection_subjects(
    files: list, modality: str,
    g_fc_cortex: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    df_yeo_surf_5k: pd.DataFrame,
    *, mask_G: np.ndarray | None = None,
    preloaded_subjects: list[np.ndarray] | None = None,
    target_network_labels: np.ndarray | None = None,
    min_valid: int = 10,
) -> dict:
    """Per-subject projection + group inference for one modality and one network.

    Parameters
    ----------
    files : list of paths
        Per-subject connectivity files (used only if `preloaded_subjects` is None).
    modality : {'SC', 'GD', 'MPC', 'FC'}
    g_fc_cortex : (n_cortex,)
        Whole-brain FC gradient (NaN at medial wall, but cortex_mask already excludes).
    g_mpc_cortex_at_sn : (n_sn,)
        MPC gradient values at source-network vertices (already procrustes-aligned).
    sn_mask_cortex : (n_cortex,) bool
    other_mask_cortex : (n_cortex,) bool
    df_yeo_surf_5k : pd.DataFrame
        For hemi info.
    mask_G : (n_cortex, n_cortex) bool, optional
        Betzel consensus mask (SC only).
    preloaded_subjects : list of (n_cortex, n_cortex), optional
        Pre-loaded per-subject raw matrices. When supplied they are used instead of
        reading `files` from disk, so the caller can load each subject's connectome once
        and reuse it across all networks. Aligned positionally with the subject order.
    min_valid : int

    Returns
    -------
    result : dict with keys:
        P_mean : (n_sn,) group-mean projection score.
        P_subjects_full : (n_sub, n_cortex_full=9684) per-subject P_s expanded to full vertex space, NaN outside SN.
        P_subjects_sn  : (n_sub, n_sn) per-subject projection (SN-only).
        r_subjects : (n_sub,) per-subject Spearman(g_MPC, P_s).
        n_targets_per_sn : (n_sn,) mean-over-subjects count of finite-positive targets per SN vertex (sparsity diagnostic).
        target_net_weights : (n_networks, n_sn) group-mean weighted connectivity from each SN vertex to each target network (None if target_network_labels not supplied).
        target_network_names : list of str, names matching the first axis of target_net_weights.
        + Fisher-z aggregate keys: r_group, t, p, ci_low, ci_high, n.
    """
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemi_cortex = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values
    n_sn = int(sn_mask_cortex.sum())
    n_sub = len(files) if files else (len(preloaded_subjects) if preloaded_subjects else 0)
    if n_sub == 0:
        raise ValueError(
            f"[{modality}] no subjects to process: `files` is empty and no "
            f"pre-loaded `preloaded_subjects` were supplied. (For FC this means no "
            f"subject had a session-matched connectome; check `path_fc_5k`.)"
        )

    cortex_indices = np.flatnonzero(cortex_mask)
    sn_idx_full = cortex_indices[sn_mask_cortex]

    r_subjects = np.full(n_sub, np.nan)
    P_subjects_full = np.full((n_sub, N_TOTAL_5K), np.nan)
    P_subjects_sn = np.full((n_sub, n_sn), np.nan)
    # Per-subject count of finite-positive targets per SN vertex (sparsity diagnostic).
    n_targets_subjects = np.full((n_sub, n_sn), np.nan)
    g_targets_finite = np.isfinite(g_fc_cortex[other_mask_cortex])

    if target_network_labels is not None:
        target_net_names = [n for n in pd.unique(target_network_labels[other_mask_cortex])
                            if isinstance(n, str)]
        target_net_masks = {
            net: (target_network_labels == net) & other_mask_cortex
            for net in target_net_names
        }
        target_net_weights_subjects = np.full((n_sub, len(target_net_names), n_sn), np.nan)
    else:
        target_net_names = []
        target_net_masks = {}
        target_net_weights_subjects = None

    g_mpc_sn = g_mpc_cortex_at_sn.astype(np.float64)
    valid_g_mpc = np.isfinite(g_mpc_sn)

    for s in range(n_sub):
        if preloaded_subjects is not None:
            W_raw = preloaded_subjects[s]
        else:
            W_raw = load_subject_matrix(files[s], cortex_mask)

        W = prepare_weights(W_raw, modality, hemi_cortex, sn_mask_cortex, mask_G=mask_G)

        P_s = compute_projection_score(
            W, g_fc_cortex, sn_mask_cortex, other_mask_cortex, min_valid=min_valid,
        )

        P_subjects_sn[s] = P_s
        P_subjects_full[s, sn_idx_full] = P_s

        W_sub = W[np.ix_(sn_mask_cortex, other_mask_cortex)]
        n_targets_subjects[s] = (np.isfinite(W_sub) & (W_sub > 0)
                                 & g_targets_finite[None, :]).sum(axis=1)

        finite = valid_g_mpc & np.isfinite(P_s)
        if finite.sum() >= min_valid:
            r_subjects[s] = spearmanr(g_mpc_sn[finite], P_s[finite])[0]

        if target_net_weights_subjects is not None:
            W_sn_rows = W[sn_mask_cortex]
            W_pos = np.where(np.isfinite(W_sn_rows) & (W_sn_rows > 0), W_sn_rows, 0.0)
            for net_idx, net in enumerate(target_net_names):
                col_mask = target_net_masks[net]
                target_net_weights_subjects[s, net_idx] = W_pos[:, col_mask].sum(axis=1)

        if np.isnan(P_s).all():
            logger.warning(f"[{modality}] subject {s}: P_s is all-NaN.")
        elif np.nanstd(P_s) < 1e-3:
            logger.warning(
                f"[{modality}] subject {s}: P_s near-constant "
                f"(std={np.nanstd(P_s):.2e}); possible degenerate weight pattern."
            )

    agg = _fisher_z_group(r_subjects)

    P_mean = np.nanmean(P_subjects_sn, axis=0)

    if not np.isfinite(r_subjects).any():
        logger.warning(f"[{modality}] all per-subject r are NaN; group stat undefined.")
    elif np.nanstd(r_subjects) < 1e-6:
        logger.warning(
            f"[{modality}] r_subjects has near-zero variance "
            f"(std={np.nanstd(r_subjects):.2e}); subjects produced identical r."
        )

    if target_net_weights_subjects is not None:
        target_net_weights = np.nanmean(target_net_weights_subjects, axis=0)
    else:
        target_net_weights = None

    return {
        "P_mean": P_mean,
        "P_subjects_full": P_subjects_full,
        "P_subjects_sn": P_subjects_sn,
        "r_subjects": r_subjects,
        "n_targets_per_sn": np.nanmean(n_targets_subjects, axis=0),
        "target_net_weights": target_net_weights,
        "target_network_names": target_net_names,
        **agg,
    }


def group_spearman_r(g_at_sn: np.ndarray, P_subjects_sn: np.ndarray, min_valid: int = 10) -> float:
    """Fisher-z group correlation of a source-network map with each subject's projection.

    Applies the statistic of `compute_projection_subjects` to an arbitrary source map:
    per subject, the Spearman correlation over the vertices finite in both maps
    (skipped below `min_valid` vertices), then the Fisher-z mean across subjects. A
    synthetic map's observed value is therefore defined exactly as the real statistic
    and is comparable to a null's group correlations.

    Parameters
    ----------
    g_at_sn : (n_sn,)
        Source map at source-network vertices.
    P_subjects_sn : (n_sub, n_sn)
        Per-subject projection scores (``P_subjects_sn`` of `compute_projection_subjects`).
    min_valid : int
        Minimum number of finite vertex pairs per subject. Default 10.

    Returns
    -------
    r_group : float
        tanh of the mean Fisher-z per-subject correlation (NaN with fewer than 2 subjects).
    """
    rs = []
    valid_g = np.isfinite(g_at_sn)
    for P_s in P_subjects_sn:
        m = valid_g & np.isfinite(P_s)
        if m.sum() >= min_valid:
            rs.append(spearmanr(g_at_sn[m], P_s[m])[0])
    # Per-subject masks differ, so the Spearman loop cannot use `rank_corr_columns`.
    return _fisher_z_group(np.asarray(rs, dtype=float))["r_group"]


def compute_moran_null_projection(
    g_mpc_cortex_at_sn: np.ndarray, sn_mask_cortex: np.ndarray,
    gd_among_sn: np.ndarray, result: dict, n_rand: int,
    *, random_state: int = 42,
) -> dict:
    """Moran spectral randomization null for the g_MPC ↔ P alignment.

    Surrogates of g_MPC are generated to preserve its within-SN spatial
    autocorrelation (Wagner & Dray 2015; equivalent to BrainSMASH at the
    eigen-decomposition level), restricting the null entirely to the source network
    so it matches the test footprint. For each surrogate the per-subject Spearman is
    recomputed against P_s; per perm aggregated via Fisher-z mean across subjects.

    Surrogates are generated per connected component of the inverse-geodesic-distance
    graph (i.e. per hemisphere for a bilateral SN; see `moran_surrogates_blockwise`),
    so within-hemisphere autocorrelation is preserved without coupling the hemispheres.

    The empirical two-tailed p-value uses the add-one estimator
    `p = (1 + #{|null| >= |obs|}) / (1 + n_valid)`, bounded away from zero (a plain
    proportion can return 0 when the observed effect exceeds every surrogate, which is
    not a valid p-value).

    Parameters
    ----------
    g_mpc_cortex_at_sn : (n_sn,)
    sn_mask_cortex : (n_cortex,) bool
    gd_among_sn : (n_sn, n_sn) float
        Geodesic distance among SN vertices. Cross-hemisphere entries are 0 when the
        SN spans both hemispheres (yielding a 0 spatial weight, i.e. no edge in the
        Moran graph), which is what splits the graph into per-hemisphere components.
    result : dict from compute_projection_subjects.
    n_rand : int
    random_state : int

    Returns
    -------
    {null_group_moran, p_moran, null_std_moran}
    """

    with np.errstate(divide="ignore", invalid="ignore"):
        w_spatial = np.where(gd_among_sn > 0, 1.0 / gd_among_sn, 0.0)
    np.fill_diagonal(w_spatial, 0.0)
    w_spatial = 0.5 * (w_spatial + w_spatial.T)

    surrogates = moran_surrogates_blockwise(
        g_mpc_cortex_at_sn, w_spatial, n_rand, random_state,
    )  # (n_rand, n_sn)

    P_subjects_sn = result["P_subjects_sn"]
    n_sub = P_subjects_sn.shape[0]

    # Vectorised Spearman: Moran surrogates share a fixed support (g_MPC is finite at
    # every SN vertex), so for each subject the finite mask is constant across the n_rand
    # surrogates. One `rank_corr_columns` call then correlates that subject's P_s with all
    # surrogates at once, replacing an n_rand x n_sub `spearmanr` loop.
    sur_support = np.isfinite(surrogates).all(axis=0)  # vertices finite in every surrogate
    null_subjects = np.full((n_rand, n_sub), np.nan)
    for s in range(n_sub):
        P_s = P_subjects_sn[s]
        mask = sur_support & np.isfinite(P_s)
        if mask.sum() < 10:
            continue
        null_subjects[:, s] = rank_corr_columns(P_s[mask], surrogates[:, mask].T)

    with np.errstate(invalid="ignore"):
        z = np.arctanh(np.clip(null_subjects, -0.999, 0.999))
    null_group = np.tanh(np.nanmean(z, axis=1))

    p_moran = empirical_p_twosided(null_group, result["r_group"])

    null_std = float(np.nanstd(null_group))
    if null_std < 1e-6:
        logger.warning(
            f"Moran null group std={null_std:.2e}: surrogates may be degenerate."
        )

    return {"null_group_moran": null_group, "p_moran": p_moran, "null_std_moran": null_std}


def _build_distance_bins(
    gd_sn_to_other: np.ndarray, nbins: int, valid_target: np.ndarray | None = None,
) -> tuple[np.ndarray, list]:
    """Per-(SN vertex, bin) candidate target pools for distance-preserving rewiring.

    Targets are binned by their geodesic distance from each source-network vertex
    into `nbins` equal-width intra-hemisphere bins; cross-hemisphere / undefined
    targets (geodesic distance == 0) form one additional bin (index `nbins`) whose
    reassignment is restricted to the contralateral targets. Reassigning an edge
    within its bin preserves the edge-length distribution while randomising target
    identity (Roberts et al. 2016; Betzel et al. 2019).

    Parameters
    ----------
    gd_sn_to_other : (n_sn, n_other) float, geodesic distance from each SN vertex to
        each target; cross-hemisphere entries are 0.
    nbins : int, number of intra-hemisphere distance bins.
    valid_target : (n_other,) bool, optional
        Targets eligible to receive a reassigned edge. Targets with a non-finite
        projected value (e.g. NaN FC gradient) must be excluded, or the resampled
        projection numerator becomes NaN. Defaults to all targets.

    Returns
    -------
    bin_of : (n_sn, n_other) int, bin index per (row, target); the inter-hemisphere
        bin is `nbins`.
    pools : list (len n_sn) of list (len nbins+1) of int arrays
        `pools[i][b]` holds the eligible target-column indices in bin b for SN vertex i.
    """
    n_sn, n_other = gd_sn_to_other.shape
    if valid_target is None:
        valid_target = np.ones(n_other, dtype=bool)
    intra = gd_sn_to_other > 0
    bin_of = np.full(gd_sn_to_other.shape, nbins, dtype=int)  # default inter-hemi bin
    if intra.any():
        vals = gd_sn_to_other[intra]
        edges = np.linspace(vals.min(), vals.max(), nbins + 1)
        edges[-1] += 1e-6
        bin_of[intra] = np.clip(np.digitize(gd_sn_to_other[intra], edges) - 1, 0, nbins - 1)
    pools = [[np.flatnonzero((bin_of[i] == b) & valid_target) for b in range(nbins + 1)]
             for i in range(n_sn)]
    return bin_of, pools


def compute_topological_null_projection(
    modality: str, files: list, df_yeo_surf_5k: pd.DataFrame,
    g_fc_cortex: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    gd_sn_to_other: np.ndarray, result: dict, n_rand: int,
    *, preloaded_subjects: list[np.ndarray] | None = None, mask_G: np.ndarray | None = None,
    nbins: int = 10, random_state: int = 42, min_valid: int = 10, method: str = "exact",
) -> dict:
    """Geometry-preserving topological null for the g_MPC <-> projection alignment.

    Tests whether the *specific* SN->target wiring, beyond connectome geometry, drives
    the alignment. Each subject's connectivity is rewired by reassigning every
    SN->target edge to a different target **in the same geodesic-distance bin** (with
    replacement; pools are large relative to per-vertex degree), keeping the edge
    weight attached. This preserves each source vertex's degree, weight multiset, and
    edge-length distribution while randomising target identity (and hence the FC
    gradient value the edge lands on). The projection and per-subject Spearman are
    recomputed on the rewired connectome and aggregated by Fisher-z mean across
    subjects per surrogate.

    Because edge length is preserved, the null distribution of the group correlation is
    centred on the *geometry expectation* rather than zero. `p_topo` is therefore an
    **excess-magnitude** add-one estimate (`empirical_p_excess_magnitude`): it tests
    whether the observed correlation is *stronger in magnitude* than what geometry alone
    produces, isolating targeting specificity from distance dependence. The magnitude
    (rather than fixed-side) comparison is required because the group correlation
    inherits the arbitrary polarity of the source gradient `g_MPC`; the observed effect
    and the surrogates share that same fixed `g_MPC`, so both lie on the same side of
    zero and folding by `|·|` recovers the intended "alignment exceeds geometry" test
    while staying invariant to the eigenvector's arbitrary sign. (A two-sided test around
    zero would conflate the non-zero geometry offset with the effect; a fixed upper-tail
    test would spuriously report n.s. whenever `g_MPC` came out sign-negative.) Intended
    for SC, MPC and FC; GD weights are
    a pure function of distance, so a within-bin reassignment barely changes them and the
    null is uninformative there.

    Two equivalent samplers select via `method`:

    - ``"exact"`` resamples every edge's target explicitly (the brute-force null).
    - ``"clt"`` is an analytic-moment shortcut: the resampled numerator at a source vertex
      is a sum of independent per-edge draws, so its mean and variance are closed-form and
      *exactly* match the with-replacement sampler — `mean_i = Σ_b W_b·μ_b`,
      `var_i = Σ_b SW2_b·σ²_b` (`W_b`/`SW2_b` = weight-sum / squared-weight-sum of vertex i's
      edges in distance-bin `b`; `μ_b`/`σ²_b` = mean/population-variance of that bin's target
      FC-gradient pool). For dense connectomes (hundreds–thousands of edges/vertex) the
      numerator is ~Normal by the CLT, so each surrogate is drawn as `mean_i + sd_i·Z`,
      collapsing the `O(n_rand·n_edges)` per-vertex draw to `O(n_edges + n_rand)`. Intended
      for the dense measures (MPC/FC); keep ``"exact"`` for sparse SC. The two samplers
      share the same per-vertex mean and variance; the higher moments differ only by the
      normal approximation.

    Parameters
    ----------
    modality : {'SC', 'MPC', 'FC'}
    files : list of paths
        Per-subject connectivity files (used only when `preloaded_subjects` is None).
    df_yeo_surf_5k : pd.DataFrame
        Provides the cortex mask and per-vertex hemisphere labels.
    g_fc_cortex : (n_cortex,) whole-brain FC gradient on cortex.
    g_mpc_cortex_at_sn : (n_sn,) MPC gradient at source-network vertices.
    sn_mask_cortex, other_mask_cortex : (n_cortex,) bool, source-network and target masks.
    gd_sn_to_other : (n_sn, n_other) geodesic distance from SN vertices to targets
        (cross-hemisphere entries 0).
    result : dict from `compute_projection_subjects` (only `r_group` is read).
    n_rand : int, number of surrogates.
    preloaded_subjects : list of (n_cortex, n_cortex), optional pre-loaded per-subject raw
        matrices (used instead of reading `files`; load once, reuse across networks).
        Aligned positionally with the subject order.
    mask_G : (n_cortex, n_cortex) bool, optional Betzel consensus mask (SC only).
    nbins : int, intra-hemisphere distance bins. Default 10.
    random_state : int, base seed. Default 42.
    min_valid : int, minimum finite-positive targets per SN vertex. Default 10.
    method : {'exact', 'clt'}, surrogate sampler. Default 'exact'.

    Returns
    -------
    {null_group_topo, p_topo, null_std_topo}
    """
    if method not in ("exact", "clt"):
        raise ValueError(f"method must be 'exact' or 'clt'; got {method!r}")
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemi_cortex = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values
    n_sn = int(sn_mask_cortex.sum())
    g_targets = g_fc_cortex[other_mask_cortex].astype(np.float64)
    g_targets_finite = np.isfinite(g_targets)
    g_mpc_sn = g_mpc_cortex_at_sn.astype(np.float64)
    valid_g_mpc = np.isfinite(g_mpc_sn)

    # Pools exclude targets with a non-finite projected value, otherwise a resampled
    # target would inject NaN into the projection numerator.
    bin_of, pools = _build_distance_bins(gd_sn_to_other, nbins, valid_target=g_targets_finite)
    # Flatten each row's per-bin candidate FC-gradient values into one array with bin
    # start-offsets and sizes, so a surrogate resamples all of a row's edges in a single
    # vectorised draw (geometry-only; reused for every subject and surrogate).
    g_concat, bin_off, bin_sz = [], [], []
    for i in range(n_sn):
        gp = [g_targets[pools[i][b]] for b in range(nbins + 1)]
        sizes = np.array([a.size for a in gp])
        g_concat.append(np.concatenate(gp) if sizes.sum() else np.empty(0))
        bin_sz.append(sizes)
        bin_off.append(np.concatenate([[0], np.cumsum(sizes)[:-1]]))  # start of each bin

    # CLT sampler: per (vertex, bin) pool mean/variance are subject-independent, so compute
    # them once here. mean/var of an empty bin stay 0 (those bins also carry zero weight).
    if method == "clt":
        pool_mu = np.zeros((n_sn, nbins + 1))
        pool_var = np.zeros((n_sn, nbins + 1))
        for i in range(n_sn):
            for b in range(nbins + 1):
                sz = int(bin_sz[i][b])
                if sz:
                    seg = g_concat[i][bin_off[i][b]: bin_off[i][b] + sz]
                    pool_mu[i, b] = seg.mean()
                    pool_var[i, b] = seg.var()  # population variance (ddof=0)

    n_sub = len(files) if files else (len(preloaded_subjects) if preloaded_subjects else 0)
    if n_sub == 0:
        raise ValueError(f"[{modality}] topological null: no subjects to process.")

    rng = np.random.default_rng(random_state)
    null_subjects = np.full((n_rand, n_sub), np.nan)

    for s in range(n_sub):
        if preloaded_subjects is not None:
            W_raw = preloaded_subjects[s]
        else:
            W_raw = load_subject_matrix(files[s], cortex_mask)
        W = prepare_weights(W_raw, modality, hemi_cortex, sn_mask_cortex, mask_G=mask_G)
        W_sub = W[np.ix_(sn_mask_cortex, other_mask_cortex)]

        pos = np.isfinite(W_sub) & (W_sub > 0) & g_targets_finite[None, :]
        row_deg = pos.sum(axis=1)
        W_total = np.where(pos, W_sub, 0.0).sum(axis=1)

        # Denominator (total weight) is invariant under within-bin reassignment, so only
        # the numerator (sum of weight x reassigned-target FC value) is resampled.
        enough = row_deg >= min_valid
        P_surr = np.full((n_sn, n_rand), np.nan)
        if method == "exact":
            for i in range(n_sn):
                if not enough[i]:
                    continue
                cols = np.flatnonzero(pos[i])
                w_e = W_sub[i, cols].astype(np.float64)
                b_e = bin_of[i, cols]
                sz = bin_sz[i][b_e].astype(np.float64)        # pool size of each edge's bin
                off = bin_off[i][b_e]                          # pool start offset per edge
                # One draw per edge per surrogate: a uniform index into its same-dist pool.
                loc = (rng.random((n_rand, cols.size)) * sz).astype(np.intp)
                g_draw = g_concat[i][off[None, :] + loc]       # (n_rand, n_edges)
                P_surr[i] = (g_draw @ w_e) / W_total[i]
        else:  # clt: draw from the analytic per-vertex Gaussian (exact mean/var)
            Wb = np.zeros((n_sn, nbins + 1))
            SW2 = np.zeros((n_sn, nbins + 1))
            rows, jcols = np.nonzero(pos)
            w = W_sub[rows, jcols].astype(np.float64)
            bb = bin_of[rows, jcols]
            np.add.at(Wb, (rows, bb), w)
            np.add.at(SW2, (rows, bb), w * w)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_vec = (Wb * pool_mu).sum(axis=1) / W_total
                var_vec = (SW2 * pool_var).sum(axis=1) / W_total ** 2
            sd_vec = np.sqrt(np.clip(var_vec, 0.0, None))
            Z = rng.standard_normal((n_sn, n_rand))
            P_surr[enough] = mean_vec[enough, None] + sd_vec[enough, None] * Z[enough]

        mask = valid_g_mpc & enough
        if mask.sum() >= min_valid:
            null_subjects[:, s] = rank_corr_columns(g_mpc_sn[mask], P_surr[mask])

    with np.errstate(invalid="ignore"):
        z = np.arctanh(np.clip(null_subjects, -0.999, 0.999))
    null_group = np.tanh(np.nanmean(z, axis=1))

    # Excess-magnitude p: the null is centred on the geometry expectation (not zero) and
    # the group r inherits g_MPC's arbitrary polarity, so the hypothesis "specific wiring
    # *increases* alignment beyond geometry" is tested as |obs| > |null| — sign-invariant
    # (a fixed upper-tail would misfire whenever g_MPC came out sign-negative).
    p_topo = empirical_p_excess_magnitude(null_group, result["r_group"])
    null_std = float(np.nanstd(null_group))
    if null_std < 1e-6:
        logger.warning(
            f"[{modality}] topological null group std={null_std:.2e}: "
            f"surrogates may be degenerate."
        )
    logger.info(
        f"[{modality}] topological null ({method}): nbins={nbins}, n_rand={n_rand}; "
        f"null mean={np.nanmean(null_group):+.3f} (geometry expectation), "
        f"obs r_group={result['r_group']:+.3f}, p_topo={p_topo:.3e} (excess-magnitude)"
    )
    return {"null_group_topo": null_group, "p_topo": p_topo, "null_std_topo": null_std}


def make_fc_spin_surrogates(
    df_yeo_surf_5k: pd.DataFrame, sphere_lh, sphere_rh, n_rand: int,
    *, fc_col: str = "fc_g1", random_state: int = 42,
) -> np.ndarray:
    """Spherical-rotation surrogates of the FC gradient, restricted to cortex order.

    Generates `n_rand` spins of the target axis `g_FC` with `SpinPermutations`
    (Alexander-Bloch et al. 2018): each spin is a random rotation of the left sphere,
    mirrored across the midsagittal (Y-Z) plane for the right sphere, with each rotated
    vertex taking the value of its nearest original vertex. The spins preserve the
    gradient's full spatial autocorrelation while destroying its anatomical
    registration to the connectome. Medial-wall vertices carry NaN and are
    rotated like any other vertex, so a spun cortical position can receive a NaN; those
    are handled downstream (excluded from the projection's numerator and denominator per
    spin). The result is the spun field on cortex vertices only, matching the
    `g_fc_cortex` ordering the projection uses.

    Unlike the Moran null (which randomises the *source* map `g_MPC`) and the topological
    null (which rewires the *connectome*), this null acts on the *target* map and so
    applies to every modality, including GD. It complements the topological null: the
    distance-bin rewiring preserves edge length but scrambles target direction, whereas
    the spin preserves the target field's full (anisotropic) structure but removes its
    alignment to anatomy.

    Parameters
    ----------
    df_yeo_surf_5k : pd.DataFrame
        Surface table; `hemisphere` (NaN at medial wall) and `fc_col` define the field.
    sphere_lh, sphere_rh : BSPolyData
        fsLR-5k hemisphere spheres for the rotation.
    n_rand : int
        Number of spins.
    fc_col : str
        Column holding the (oriented) FC gradient. Default ``"fc_g1"``.
    random_state : int
        Seed for the rotation set.

    Returns
    -------
    g_fc_spun_cortex : (n_rand, n_cortex) float
        Spun FC gradient at cortical vertices (cortex-mask order), NaN where a spin
        rotated a medial-wall vertex into a cortical position.
    """
    from brainspace.null_models import SpinPermutations
    from brainspace.mesh.mesh_elements import get_points

    fc_full = df_yeo_surf_5k[fc_col].to_numpy(dtype=float)
    fc_lh = fc_full[:N_LH_5K]
    fc_rh = fc_full[N_LH_5K:]

    sp = SpinPermutations(n_rep=n_rand, random_state=random_state)
    sp.fit(get_points(sphere_lh), points_rh=get_points(sphere_rh))
    spun_lh, spun_rh = sp.randomize(fc_lh, x_rh=fc_rh)  # each (n_rand, n_hemi)

    spun_full = np.concatenate([spun_lh, spun_rh], axis=1)  # (n_rand, 9684)
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().to_numpy()
    return spun_full[:, cortex_mask]


def compute_spin_null_projection(
    modality: str, files: list, df_yeo_surf_5k: pd.DataFrame,
    g_fc_spun_cortex: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    result: dict, *, preloaded_subjects: list[np.ndarray] | None = None,
    mask_G: np.ndarray | None = None, min_valid: int = 10,
) -> dict:
    """Target-side spin null for the g_MPC ↔ projection alignment.

    For each spin of the FC gradient (see `make_fc_spin_surrogates`) the per-subject
    projection is recomputed on the *real* connectome — only the target axis is rotated —
    and the per-subject Spearman against the fixed `g_MPC` is aggregated across subjects
    by the Fisher-z mean. Because the rotation removes the FC gradient's alignment to
    anatomy while preserving its autocorrelation, the null is centred on zero, so the
    empirical p-value is the two-sided add-one estimator (`empirical_p_twosided`),
    already invariant to `g_MPC`'s arbitrary polarity.

    The projection's denominator is the weight sum over targets with a *finite* spun
    value, recomputed per spin (a rotated-in medial-wall NaN drops that target from both
    numerator and denominator). Per subject the Spearman is vectorised over spins on the
    rows that are finite across every spin (mirroring the Moran null's fixed-support
    handling); for the dense measures essentially all rows qualify, and for sparse SC the
    handful of threshold-crossing rows are dropped rather than biasing the statistic.

    Parameters
    ----------
    modality : {'SC', 'GD', 'MPC', 'FC'}
    files : list of paths
        Per-subject connectivity files (used only when `preloaded_subjects` is None).
    df_yeo_surf_5k : pd.DataFrame
        Provides the cortex mask and per-vertex hemisphere labels.
    g_fc_spun_cortex : (n_rand, n_cortex) float
        Spun FC gradient on cortex (from `make_fc_spin_surrogates`); shared across
        networks and measures.
    g_mpc_cortex_at_sn : (n_sn,) MPC gradient at source-network vertices.
    sn_mask_cortex, other_mask_cortex : (n_cortex,) bool, source-network and target masks.
    result : dict from `compute_projection_subjects` (only `r_group` is read).
    preloaded_subjects : list of (n_cortex, n_cortex), optional pre-loaded raw matrices.
    mask_G : (n_cortex, n_cortex) bool, optional Betzel consensus mask (SC only).
    min_valid : int, minimum finite-positive targets per SN vertex. Default 10.

    Returns
    -------
    {null_group_spin, p_spin, null_std_spin}
    """
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemi_cortex = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values
    n_rand = g_fc_spun_cortex.shape[0]

    # Spun target fields, restricted to this network's target set, split into the value
    # (NaN→0) and the finite indicator so the per-spin numerator/denominator are matmuls.
    Gs = g_fc_spun_cortex[:, other_mask_cortex].astype(np.float64)  # (n_rand, n_other)
    Gs_filled = np.nan_to_num(Gs, nan=0.0)
    Ms = np.isfinite(Gs).astype(np.float64)
    Gs_T, Ms_T = Gs_filled.T, Ms.T  # (n_other, n_rand)

    g_mpc_sn = g_mpc_cortex_at_sn.astype(np.float64)
    valid_g_mpc = np.isfinite(g_mpc_sn)

    n_sub = len(files) if files else (len(preloaded_subjects) if preloaded_subjects else 0)
    if n_sub == 0:
        raise ValueError(f"[{modality}] spin null: no subjects to process.")

    null_subjects = np.full((n_rand, n_sub), np.nan)
    for s in range(n_sub):
        W_raw = preloaded_subjects[s] if preloaded_subjects is not None \
            else load_subject_matrix(files[s], cortex_mask)
        W = prepare_weights(W_raw, modality, hemi_cortex, sn_mask_cortex, mask_G=mask_G)
        W_sub = W[np.ix_(sn_mask_cortex, other_mask_cortex)]
        W_eff = np.where(np.isfinite(W_sub) & (W_sub > 0), W_sub, 0.0)  # (n_sn, n_other)
        pos = (W_eff > 0).astype(np.float64)

        num = W_eff @ Gs_T          # (n_sn, n_rand)
        den = W_eff @ Ms_T          # weight over finite spun targets
        nval = pos @ Ms_T           # count of valid targets per (row, spin)
        with np.errstate(invalid="ignore", divide="ignore"):
            P_spin = np.where((nval >= min_valid) & (den > 0), num / den, np.nan)

        mask = valid_g_mpc & np.isfinite(P_spin).all(axis=1)
        if mask.sum() >= min_valid:
            null_subjects[:, s] = rank_corr_columns(g_mpc_sn[mask], P_spin[mask])

    with np.errstate(invalid="ignore"):
        z = np.arctanh(np.clip(null_subjects, -0.999, 0.999))
    null_group = np.tanh(np.nanmean(z, axis=1))

    p_spin = empirical_p_twosided(null_group, result["r_group"])
    null_std = float(np.nanstd(null_group))
    if null_std < 1e-6:
        logger.warning(
            f"[{modality}] spin null group std={null_std:.2e}: surrogates may be degenerate."
        )
    logger.info(
        f"[{modality}] spin null: n_rand={n_rand}; null mean={np.nanmean(null_group):+.3f} "
        f"(~0 expected), obs r_group={result['r_group']:+.3f}, p_spin={p_spin:.3e} (two-sided)"
    )
    return {"null_group_spin": null_group, "p_spin": p_spin, "null_std_spin": null_std}


def compute_projection_nulls(
    modality: str, files: list, df_yeo_surf_5k: pd.DataFrame,
    g_fc_cortex: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    gd_cortex: np.ndarray, n_rand: int,
    *, target_network_labels: np.ndarray | None = None,
    mask_G: np.ndarray | None = None,
    preloaded_subjects: list[np.ndarray] | None = None,
    g_fc_spun_cortex: np.ndarray | None = None,
) -> dict:
    """Per-subject projection plus the Moran, topological and spin nulls.

    The three nulls are complementary. The Moran spectral randomisation preserves the
    autocorrelation of g_MPC and controls for map smoothness (per hemisphere block,
    every modality). The geometry-preserving topological null (modalities in
    `TOPO_MODALITIES`: SC, MPC, FC) rewires the connectome within geodesic-distance bins
    and controls for connectome geometry, isolating wiring specificity; GD gets none
    (its weights are pure distance, so a within-bin reassignment barely changes them)
    and its `p_topo` is NaN. The dense modalities (MPC, FC) use the analytic-moment
    sampler of the topological null and the sparse SC the exact resampler. The
    target-side spin null rotates g_FC and controls for the FC gradient's
    autocorrelation independent of its anatomical position; it applies to every
    modality and `p_spin` is NaN only when `g_fc_spun_cortex` is None.

    Parameters
    ----------
    modality : {'SC', 'GD', 'MPC', 'FC'}
    files : list of paths
        Per-subject connectivity files (used only if `preloaded_subjects` is None).
    df_yeo_surf_5k : pd.DataFrame
        fsLR-5k surface table (cortex mask and hemisphere labels).
    g_fc_cortex : (n_cortex,)
        Oriented whole-brain FC gradient on cortex.
    g_mpc_cortex_at_sn : (n_sn,)
        MPC gradient at source-network vertices.
    sn_mask_cortex, other_mask_cortex : (n_cortex,) bool
        Source-network and target masks.
    gd_cortex : (n_cortex, n_cortex)
        Group-mean geodesic distance (Moran weights; topological-null distance bins).
    n_rand : int
        Number of Moran and topological surrogates.
    target_network_labels : (n_cortex,), optional
        Network label per cortex vertex (enables the dominant-target-network weights).
    mask_G : (n_cortex, n_cortex) bool, optional
        Betzel consensus mask (SC only).
    preloaded_subjects : list of (n_cortex, n_cortex), optional
        Pre-loaded per-subject raw matrices, aligned with `files`.
    g_fc_spun_cortex : (n_spin, n_cortex), optional
        Spun FC gradient from `make_fc_spin_surrogates`; the spin null is skipped if None.

    Returns
    -------
    result : dict
        The keys of `compute_projection_subjects`, plus ``null_group_moran``,
        ``p_moran``, ``null_std_moran``, ``null_group_topo``, ``p_topo``,
        ``null_std_topo``, ``null_group_spin``, ``p_spin`` and ``null_std_spin``
        (``null_group_*`` is None and ``p_*`` NaN for a null that was not run).
    """
    result = compute_projection_subjects(
        files=files, modality=modality,
        g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=g_mpc_cortex_at_sn,
        sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
        df_yeo_surf_5k=df_yeo_surf_5k,
        mask_G=mask_G, preloaded_subjects=preloaded_subjects,
        target_network_labels=target_network_labels,
    )

    gd_among_sn = gd_cortex[np.ix_(sn_mask_cortex, sn_mask_cortex)]
    moran = compute_moran_null_projection(
        g_mpc_cortex_at_sn, sn_mask_cortex, gd_among_sn, result, n_rand,
    )

    if g_fc_spun_cortex is not None:
        spin = compute_spin_null_projection(
            modality=modality, files=files, df_yeo_surf_5k=df_yeo_surf_5k,
            g_fc_spun_cortex=g_fc_spun_cortex, g_mpc_cortex_at_sn=g_mpc_cortex_at_sn,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            result=result, preloaded_subjects=preloaded_subjects, mask_G=mask_G,
        )
    else:
        spin = {"null_group_spin": None, "p_spin": np.nan, "null_std_spin": np.nan}

    if modality in TOPO_MODALITIES:
        gd_sn_to_other = gd_cortex[np.ix_(sn_mask_cortex, other_mask_cortex)]
        # Dense connectomes (MPC/FC): hundreds to thousands of edges per vertex make the
        # per-vertex numerator ~Gaussian, so the analytic-moment CLT sampler applies;
        # sparse SC keeps the exact resampler.
        topo = compute_topological_null_projection(
            modality=modality, files=files, df_yeo_surf_5k=df_yeo_surf_5k,
            g_fc_cortex=g_fc_cortex, g_mpc_cortex_at_sn=g_mpc_cortex_at_sn,
            sn_mask_cortex=sn_mask_cortex, other_mask_cortex=other_mask_cortex,
            gd_sn_to_other=gd_sn_to_other, result=result, n_rand=n_rand,
            preloaded_subjects=preloaded_subjects, mask_G=mask_G,
            method="clt" if modality in ("MPC", "FC") else "exact",
        )
    else:
        topo = {"null_group_topo": None, "p_topo": np.nan, "null_std_topo": np.nan}

    return {**result, **moran, **topo, **spin}


def compute_dominant_target_network(result: dict) -> tuple[np.ndarray, list[str]]:
    """For each SN vertex, return the index + name of the target network with
    the highest group-mean weighted connectivity.

    Parameters
    ----------
    result : dict from compute_projection_subjects
        Must contain `target_net_weights` and `target_network_names`.

    Returns
    -------
    dominant_idx : (n_sn,) int, argmax along network axis (-1 where all NaN).
    target_network_names : list of str, in the order indexed.
    """
    W = result.get("target_net_weights")
    names = result.get("target_network_names")
    if W is None or not names:
        return None, []
    valid = np.any(np.isfinite(W), axis=0)
    dominant = np.full(W.shape[1], -1, dtype=int)
    if valid.any():
        dominant[valid] = np.nanargmax(W[:, valid], axis=0)
    return dominant, names

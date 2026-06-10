"""
Connectome I/O and gradient-weighted connectivity projection at fsLR-5k.

Implements the per-network-vertex projection statistic (Park 2021, eLife; Vázquez-
Rodríguez 2019, PNAS; Suárez 2020, Trends Cogn Sci) used in figure 2: for each
network vertex i,

    P[i] = sum_{j in T_i} w_ij * g_FC[j] / sum_{j in T_i} w_ij,
    T_i  = { j : w_ij > 0, j not in network, j != i }

with per-subject inference (Fisher-z then one-sample t-test across subjects) and a
within-network Moran spectral-randomization null (spatial-autocorrelation-
preserving). Moran surrogates are generated per connected component of the
within-network spatial graph, i.e. per hemisphere (micapipe geodesic distance is
undefined across hemispheres, so the bilateral graph splits into two blocks),
preserving within-hemisphere autocorrelation while keeping both hemispheres in the
statistic.

Four modalities are supported, with modality-specific weight preprocessing. Every
modality uses only positive connections and the same weighted-mean projection:
    - SC (structural connectivity): Betzel distance-stratified consensus mask is
      applied to per-subject SIFT2 weights; weights are log10(SC*G/eps) on
      positives, NaN elsewhere. Two-stage random-effects inference (subject is
      unit of inference).
    - GD (geodesic distance): weights are 1/GD (proximity), within-hemisphere
      only. Used as a spatial-autocorrelation control reading.
    - MPC (microstructure profile covariance): positive partial correlations are
      kept as weights; non-positive (negative/zero) entries are dropped.
    - FC (functional connectivity): positive correlations are kept as weights;
      non-positive (anticorrelated/zero) entries are dropped.
"""

import logging

import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.null_models import MoranRandomization
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr, ttest_1samp, t as t_dist

logger = logging.getLogger(__name__)

N_LH_5K = 4842
N_TOTAL_5K = 9684


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Betzel distance-stratified consensus mask
# ---------------------------------------------------------------------------

def fcn_group_bins(
    adj: np.ndarray, dist: np.ndarray, hemiid: np.ndarray, nbins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Distance-dependent group-representative SC thresholding (Betzel et al. 2018).

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

    Betzel, R. F. et al. (2018). Network Neuroscience.
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


# ---------------------------------------------------------------------------
# Weight preprocessing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Projection score
# ---------------------------------------------------------------------------

def compute_projection_score(
    W: np.ndarray, g_fc_cortex: np.ndarray,
    sn_idx_cortex: np.ndarray, other_idx_cortex: np.ndarray,
    *, min_valid: int = 10,
) -> np.ndarray:
    """Weighted-mean projection: P[i] = sum_j w_ij g_FC[j] / sum_j w_ij.

    Only positive weights contribute (NaN weights are excluded from numerator
    AND denominator). Returns NaN for rows with fewer than `min_valid` finite
    targets, or with zero/negative denominator.

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

    valid = np.isfinite(W_sub) & (W_sub > 0) & np.isfinite(g_targets)[None, :]
    W_eff = np.where(valid, W_sub, 0.0)
    g_safe = np.where(np.isfinite(g_targets), g_targets, 0.0)

    num = W_eff @ g_safe
    den = W_eff.sum(axis=1)
    n_valid = valid.sum(axis=1)

    P = np.where(n_valid >= min_valid, num / np.where(den > 0, den, np.nan), np.nan)
    return P


# ---------------------------------------------------------------------------
# Per-subject orchestration
# ---------------------------------------------------------------------------

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
    sc_subjects: list[np.ndarray] | None = None,
    target_network_labels: np.ndarray | None = None,
    min_valid: int = 10,
) -> dict:
    """Per-subject projection + group inference for one modality and one network.

    Parameters
    ----------
    files : list of paths
        Per-subject connectivity files (used if `sc_subjects` is None or modality != SC).
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
    sc_subjects : list of (n_cortex, n_cortex), optional
        Pre-loaded SC matrices (avoids re-reading from disk).
    min_valid : int

    Returns
    -------
    result : dict with keys:
        P_mean : (n_sn,) group-mean projection score.
        P_subjects_full : (n_sub, n_cortex_full=9684) per-subject P_s expanded to full vertex space, NaN outside SN.
        P_subjects_sn  : (n_sub, n_sn) per-subject projection (SN-only).
        r_subjects : (n_sub,) per-subject Spearman(g_MPC, P_s).
        target_net_weights : (n_networks, n_sn) group-mean weighted connectivity from each SN vertex to each target network (None if target_network_labels not supplied).
        target_network_names : list of str, names matching the first axis of target_net_weights.
        + Fisher-z aggregate keys: r_group, t, p, ci_low, ci_high, n.
    """
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemi_cortex = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values
    n_sn = int(sn_mask_cortex.sum())
    n_sub = len(files) if files else (len(sc_subjects) if sc_subjects else 0)
    if n_sub == 0:
        raise ValueError(
            f"[{modality}] no subjects to process: `files` is empty and no "
            f"pre-loaded `sc_subjects` were supplied. (For FC this means no "
            f"subject had a session-matched connectome; check `path_fc_5k`.)"
        )

    n_cortex = int(cortex_mask.sum())
    cortex_indices = np.flatnonzero(cortex_mask)
    sn_idx_full = cortex_indices[sn_mask_cortex]

    r_subjects = np.full(n_sub, np.nan)
    P_subjects_full = np.full((n_sub, N_TOTAL_5K), np.nan)
    P_subjects_sn = np.full((n_sub, n_sn), np.nan)

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
        if modality == "SC" and sc_subjects is not None:
            W_raw = sc_subjects[s]
        else:
            W_raw = load_subject_matrix(files[s], cortex_mask)

        W = prepare_weights(W_raw, modality, hemi_cortex, sn_mask_cortex, mask_G=mask_G)

        P_s = compute_projection_score(
            W, g_fc_cortex, sn_mask_cortex, other_mask_cortex, min_valid=min_valid,
        )

        P_subjects_sn[s] = P_s
        P_subjects_full[s, sn_idx_full] = P_s

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
        "target_net_weights": target_net_weights,
        "target_network_names": target_net_names,
        **agg,
    }


# ---------------------------------------------------------------------------
# Moran spectral randomization null (within-SN, SAC-preserving)
# ---------------------------------------------------------------------------

def _moran_surrogates_blockwise(
    g: np.ndarray, w_spatial: np.ndarray, n_rand: int,
    random_state: int, min_block: int = 3,
) -> np.ndarray:
    """Generate Moran surrogates per connected component of the spatial graph.

    micapipe geodesic distance is undefined across hemispheres, so for a bilateral
    source network the inverse-distance weight matrix `w_spatial` splits into two
    disconnected blocks (one per hemisphere). Each connected component is fitted and
    randomised separately and the per-component surrogates are reassembled into the
    full vector, so within-hemisphere autocorrelation is preserved and both
    hemispheres contribute to the statistic without coupling across hemispheres. A
    single-hemisphere graph is one component.

    Components with fewer than `min_block` vertices (where spatial autocorrelation is
    not meaningfully defined) fall back to a per-permutation random shuffle of that
    block's values, which preserves the marginal distribution.

    Parameters
    ----------
    g : (n_sn,) float, the map to randomise.
    w_spatial : (n_sn, n_sn) float, symmetric non-negative spatial weights.
    n_rand : int, number of surrogates.
    random_state : int, base seed (offset per component so blocks are decorrelated).
    min_block : int, minimum component size for spectral randomisation.

    Returns
    -------
    surrogates : (n_rand, n_sn) float
    """
    n_sn = g.size
    n_comp, comp_labels = connected_components(
        csr_matrix(w_spatial > 0), directed=False
    )
    surrogates = np.empty((n_rand, n_sn), dtype=np.float64)
    sizes = []
    for c in range(n_comp):
        comp_idx = np.flatnonzero(comp_labels == c)
        sizes.append(comp_idx.size)
        sub_g = g[comp_idx].astype(np.float64)
        if comp_idx.size >= min_block:
            # brainspace.compute_mem double-centers the block weights and then
            # expects a single near-zero eigenvalue (the constant mode the centering
            # kills). It runs eigh in float32, so the true-zero eigenvalue surfaces
            # at roughly |max(ev)| * n * eps(float32) ~ 1e-4 to 1e-3 for a dense
            # inverse-distance block — above brainspace's default tol=1e-10, which
            # would spuriously reject. Each component is connected, so exactly one
            # such mode appears; raise the tolerance to recognise the float32 zero.
            moran = MoranRandomization(
                n_rep=n_rand, random_state=random_state + c,
                procedure="singleton", spectrum="nonzero", tol=1e-3,
            )
            moran.fit(w_spatial[np.ix_(comp_idx, comp_idx)])
            surrogates[:, comp_idx] = moran.randomize(sub_g)
        else:
            rng = np.random.default_rng(random_state + c)
            for k in range(n_rand):
                surrogates[k, comp_idx] = rng.permutation(sub_g)
    logger.info(
        f"Moran null: {n_comp} connected component(s) (block sizes={sizes}); "
        f"surrogates generated per component."
    )
    return surrogates


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
    graph (i.e. per hemisphere for a bilateral SN; see `_moran_surrogates_blockwise`),
    so within-hemisphere autocorrelation is preserved without coupling the hemispheres.

    The empirical two-tailed p-value uses the add-one estimator
    `p = (1 + #{|null| >= |obs|}) / (1 + n_valid)`, bounded away from zero (a plain
    proportion can return 0 when the observed effect exceeds every surrogate, which
    corrupts downstream FDR/log steps).

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
    n_sn = g_mpc_cortex_at_sn.size

    with np.errstate(divide="ignore", invalid="ignore"):
        w_spatial = np.where(gd_among_sn > 0, 1.0 / gd_among_sn, 0.0)
    np.fill_diagonal(w_spatial, 0.0)
    w_spatial = 0.5 * (w_spatial + w_spatial.T)

    surrogates = _moran_surrogates_blockwise(
        g_mpc_cortex_at_sn, w_spatial, n_rand, random_state,
    )  # (n_rand, n_sn)

    P_subjects_sn = result["P_subjects_sn"]
    n_sub = P_subjects_sn.shape[0]

    null_subjects = np.full((n_rand, n_sub), np.nan)
    for k in range(n_rand):
        sur = surrogates[k]
        sur_finite = np.isfinite(sur)
        for s in range(n_sub):
            P_s = P_subjects_sn[s]
            mask = sur_finite & np.isfinite(P_s)
            if mask.sum() < 10:
                continue
            null_subjects[k, s] = spearmanr(sur[mask], P_s[mask])[0]

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


# ---------------------------------------------------------------------------
# Dominant target network per SN vertex (for scatter colouring)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Empirical permutation p-value
# ---------------------------------------------------------------------------

def empirical_p_twosided(null: np.ndarray, observed: float) -> float:
    """Two-tailed empirical p-value via the add-one estimator `(1 + k)/(1 + n)`.

    `k` counts the finite null values whose magnitude equals or exceeds `|observed|`
    and `n` is the number of finite null values. The add-one keeps the estimate
    bounded away from zero (a plain proportion returns 0 when the observed effect
    exceeds every surrogate, which is improper and corrupts downstream FDR/log steps).
    Returns NaN if `observed` is non-finite or there are no finite null values.
    """
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or finite.size == 0:
        return np.nan
    k = int(np.sum(np.abs(finite) >= np.abs(observed)))
    return float((1 + k) / (1 + finite.size))


# ---------------------------------------------------------------------------
# BH FDR correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values). NaN-safe."""
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    pv = p[valid]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    q_full = np.empty_like(pv)
    q_full[order] = q
    out[valid] = q_full
    return out


# ---------------------------------------------------------------------------
# Tertile contrast (supplementary)
# ---------------------------------------------------------------------------

def compute_tertile_contrast(
    P_mean_sn: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
) -> pd.DataFrame:
    """Tertile SN vertices by g_MPC; report mean projection score per tertile.

    The "task-positive vs DMN" contrast on raw connectivity is more complex (needs
    target-network labels). Here we summarise the projection score itself: under
    the hypothesis, the high-g_MPC tertile should have higher P (its targets
    sit higher on the FC gradient, i.e. task-positive).
    """
    valid = np.isfinite(g_mpc_cortex_at_sn) & np.isfinite(P_mean_sn)
    g = g_mpc_cortex_at_sn[valid]
    p = P_mean_sn[valid]
    cuts = np.quantile(g, [1 / 3, 2 / 3])
    tertile = np.where(g <= cuts[0], "inferior", np.where(g >= cuts[1], "superior", "middle"))
    return pd.DataFrame({"g_mpc": g, "P": p, "tertile": tertile}).groupby("tertile").agg(
        mean_P=("P", "mean"), sd_P=("P", "std"), n=("P", "size")
    ).reset_index()

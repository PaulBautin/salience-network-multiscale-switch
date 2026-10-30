"""Spatial null models and the tests built on them, shared by every figure.

The Moran spectral randomization is fitted separately in each hemisphere, because the
inverse-distance weight graph has no edges between hemispheres. Empirical p values use
the add-one estimator (1 + k) / (1 + n).
"""
import logging

import numpy as np
import pandas as pd
from brainspace.null_models import MoranRandomization
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata, spearmanr

logger = logging.getLogger(__name__)


class HemisphereMoranRandomization:
    """Moran spectral randomisation fitted and drawn separately within each block.

    Drop-in for brainspace ``MoranRandomization`` (``fit(w)`` then ``randomize(y)``,
    so it works with `moran_spearman` and `moran_spearman_difference`). Every
    within-network Moran null of the repository uses it, with one block per
    hemisphere: distance between hemispheres is undefined, so each block gets its
    own ``MoranRandomization`` (singleton procedure, non-zero spectrum) fitted on its
    rows and columns of ``w``, and its surrogates are written back into the block's
    positions. Every surrogate therefore keeps the within-hemisphere autocorrelation
    and the distribution of the map within each hemisphere, as a spin test rotates
    each hemisphere within itself. With a single block (a one-hemisphere analysis)
    this is exactly one ``MoranRandomization`` fitted on ``w`` with seed
    ``random_state``. A block of fewer than ``min_block`` vertices, where spatial
    autocorrelation is not defined, is permuted at random instead.

    Parameters
    ----------
    blocks : (n,) array
        Block label of each vertex, e.g. its hemisphere ('LH' / 'RH'). Blocks are
        numbered in order of first appearance; block ``b`` is seeded ``random_state + b``.
    n_rep : int
        Number of surrogates.
    random_state : int
        Seed of the first block. Default 0.
    tol : float
        Eigenvalue tolerance of ``MoranRandomization``. Default 1e-6.
    min_block : int
        Minimum block size for spectral randomisation. Default 3.
    """

    def __init__(self, blocks, n_rep: int, *, random_state: int = 0, tol: float = 1e-6,
                 min_block: int = 3):
        self.blocks = np.asarray(blocks)
        self.n_rep = n_rep
        self.random_state = random_state
        self.tol = tol
        self.min_block = min_block

    def fit(self, w) -> "HemisphereMoranRandomization":
        """Fit one randomisation per block on ``w`` ((n, n) array or sparse matrix)."""
        self.fits_ = []
        for b, label in enumerate(pd.unique(self.blocks)):
            idx = np.flatnonzero(self.blocks == label)
            seed = self.random_state + b
            msr = None
            if idx.size >= self.min_block:
                msr = MoranRandomization(n_rep=self.n_rep, procedure="singleton", spectrum="nonzero",
                                         tol=self.tol, random_state=seed)
                msr.fit(w[idx][:, idx])
            self.fits_.append((idx, msr, seed))
        logger.info(f"Moran null: {len(self.fits_)} block(s) fitted separately "
                    f"(block sizes={[int(idx.size) for idx, _, _ in self.fits_]}).")
        return self

    def randomize(self, y: np.ndarray) -> np.ndarray:
        """``(n_rep, n)`` surrogates of ``y``, drawn within each block."""
        y = np.asarray(y, dtype=np.float64)
        surrogates = np.empty((self.n_rep, y.size), dtype=np.float64)
        for idx, msr, seed in self.fits_:
            if msr is not None:
                surrogates[:, idx] = msr.randomize(y[idx])
            else:
                rng = np.random.default_rng(seed)
                for k in range(self.n_rep):
                    surrogates[k, idx] = rng.permutation(y[idx])
        return surrogates


def moran_surrogates_blockwise(
    g: np.ndarray, w_spatial: np.ndarray, n_rand: int,
    random_state: int, min_block: int = 3,
) -> np.ndarray:
    """Generate Moran surrogates per connected component of the spatial graph.

    micapipe geodesic distance is undefined across hemispheres, so for a bilateral
    source network the inverse-distance weight matrix `w_spatial` splits into two
    disconnected blocks, one per hemisphere (true for every Figure 3 network). Each
    connected component is fitted and randomised separately by
    `HemisphereMoranRandomization`, the per-hemisphere null shared by every figure, and
    the per-component surrogates are reassembled into the full vector, so
    within-hemisphere autocorrelation is preserved and both hemispheres contribute to
    the statistic without coupling across hemispheres. A single-hemisphere graph is one
    component.

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
    n_comp, comp_labels = connected_components(
        csr_matrix(w_spatial > 0), directed=False
    )
    # Component labels follow first appearance, so component c is seeded random_state + c.
    # brainspace.compute_mem double-centers the block weights and then expects a single
    # near-zero eigenvalue (the constant mode the centering kills). It runs eigh in
    # float32, so the true-zero eigenvalue surfaces at roughly |max(ev)| * n *
    # eps(float32) ~ 1e-4 to 1e-3 for a dense inverse-distance block — above
    # brainspace's default tol=1e-10, which would spuriously reject. Each component is
    # connected, so exactly one such mode appears; raise the tolerance to recognise the
    # float32 zero.
    msr = HemisphereMoranRandomization(comp_labels, n_rand, random_state=random_state, tol=1e-3,
                                       min_block=min_block)
    return msr.fit(w_spatial).randomize(g)


def rank_corr_columns(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Spearman correlation of a vector with each column of a matrix.

    Ranks use tie-averaging, so Pearson on the ranks equals Spearman's rho. Vectorised
    across the K surrogate columns (the finite mask is constant across surrogates, so
    a single subset is taken upstream).

    Parameters
    ----------
    x : (M,) array
        Fixed map, finite.
    Y : (M, K) array
        One map per column (e.g. K surrogates), finite.

    Returns
    -------
    r : (K,) array
        Spearman correlation of `x` with each column of `Y`.
    """
    rx = rankdata(x).astype(np.float64)
    rx -= rx.mean()
    RY = np.empty(Y.shape, dtype=np.float64)
    for k in range(Y.shape[1]):
        RY[:, k] = rankdata(Y[:, k])
    RY -= RY.mean(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (rx @ RY) / (np.linalg.norm(rx) * np.linalg.norm(RY, axis=0))
    return r


def empirical_p_twosided(null: np.ndarray, observed: float) -> float:
    """Two-tailed empirical p-value via the add-one estimator `(1 + k)/(1 + n)`.

    `k` counts the finite null values whose magnitude equals or exceeds `|observed|`
    and `n` is the number of finite null values. The add-one keeps the estimate
    bounded away from zero (a plain proportion returns 0 when the observed effect
    exceeds every surrogate, which is not a valid p-value and breaks any later
    multiple-comparison correction).

    Parameters
    ----------
    null : (n,) array
        Null statistics; non-finite values are ignored.
    observed : float
        Observed statistic.

    Returns
    -------
    p : float
        Empirical p-value in (0, 1], or NaN if `observed` is non-finite or there are
        no finite null values.
    """
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or finite.size == 0:
        return np.nan
    k = int(np.sum(np.abs(finite) >= np.abs(observed)))
    return float((1 + k) / (1 + finite.size))


def empirical_p_excess_magnitude(null: np.ndarray, observed: float) -> float:
    """Sign-invariant excess-magnitude empirical p-value via add-one `(1 + k)/(1 + n)`.

    `k` counts the finite null values whose **absolute value** is greater than or equal
    to `|observed|`, and `n` is the number of finite null values. This tests whether the
    observed effect is *stronger in magnitude* than the null produces, regardless of
    sign — the correct one-sided test for the geometry-preserving topological null,
    whose surrogates are centred on the (non-zero) geometry expectation but whose group
    correlation inherits the arbitrary polarity of the source gradient `g_MPC`. Because
    the observed statistic and the surrogates share that same fixed `g_MPC`, both sit on
    the same side of zero, so folding by `|·|` recovers the intended "alignment exceeds
    geometry" comparison while remaining invariant to the eigenvector's arbitrary sign
    (an upper-tail p-value would report no effect whenever `g_MPC` comes out
    sign-negative). The add-one keeps the estimate bounded away from zero.

    The arithmetic is the same as `empirical_p_twosided`; the separate name records
    which question the topological null asks.

    Parameters
    ----------
    null : (n,) array
        Null statistics; non-finite values are ignored.
    observed : float
        Observed statistic.

    Returns
    -------
    p : float
        Empirical p-value in (0, 1], or NaN if `observed` is non-finite or there are
        no finite null values.
    """
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or finite.size == 0:
        return np.nan
    k = int(np.sum(np.abs(finite) >= np.abs(observed)))
    return float((1 + k) / (1 + finite.size))


def moran_spearman(x: np.ndarray, y: np.ndarray, msr) -> tuple[float, float, np.ndarray]:
    """Spearman(x, y) against a Moran spatial null on ``y``.

    ``x`` is held fixed and ``y`` is spatially randomised by the pre-fitted
    ``MoranRandomization`` ``msr``, whose graph must be restricted to the vertices
    ``x`` and ``y`` index (e.g. the finite vertices of a network). Spearman is
    rank-based, so surrogates need no rescaling.

    Parameters
    ----------
    x : (n,) array
        Map held fixed.
    y : (n,) array
        Map randomised by `msr`, on the same vertices as `x`.
    msr : HemisphereMoranRandomization or brainspace MoranRandomization
        Fitted on the spatial weights of those `n` vertices.

    Returns
    -------
    tuple[float, float, np.ndarray]
        Observed correlation, add-one two-sided empirical p-value
        (:func:`empirical_p_twosided`) and the null correlations.
    """
    r, _ = spearmanr(x, y)
    r_null = np.array([spearmanr(x, surr)[0] for surr in msr.randomize(y)])
    return float(r), float(empirical_p_twosided(r_null, r)), r_null


def moran_spearman_difference(x_a: np.ndarray, x_b: np.ndarray, y: np.ndarray, msr) -> tuple[float, float, np.ndarray]:
    """Difference of two maps' Spearman correlations with ``y``, against a Moran null.

    The observed ``r(x_a, y) - r(x_b, y)`` is compared with the differences obtained
    when the same Moran surrogates of ``y`` (from the pre-fitted ``msr``, restricted to
    the vertices the maps index) are correlated with both maps, so the null keeps the
    dependence between ``x_a`` and ``x_b`` and the spatial autocorrelation of ``y``.

    Parameters
    ----------
    x_a, x_b : (n,) array
        The two maps whose correlations with `y` are compared.
    y : (n,) array
        Map randomised by `msr`, on the same vertices.
    msr : HemisphereMoranRandomization or brainspace MoranRandomization
        Fitted on the spatial weights of those `n` vertices.

    Returns
    -------
    tuple[float, float, np.ndarray]
        Observed difference, add-one two-sided empirical p-value
        (:func:`empirical_p_twosided`) and the null differences.
    """
    surrogates = msr.randomize(y)
    d_obs = spearmanr(x_a, y)[0] - spearmanr(x_b, y)[0]
    d_null = np.array([spearmanr(x_a, s)[0] - spearmanr(x_b, s)[0] for s in surrogates])
    return float(d_obs), float(empirical_p_twosided(d_null, d_obs)), d_null

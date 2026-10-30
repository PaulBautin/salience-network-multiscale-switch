"""Spatial nulls and the tests built on them (salience_switch.stats), on synthetic data."""
import numpy as np
import pytest
from brainspace.null_models import MoranRandomization
from scipy.stats import spearmanr

from salience_switch.stats import (HemisphereMoranRandomization, empirical_p_excess_magnitude, empirical_p_twosided,
                                   moran_spearman, moran_spearman_difference, rank_corr_columns)


def chain_weights(n: int, reach: int = 4) -> np.ndarray:
    """Inverse-distance weights between points on a line, up to `reach` neighbours away."""
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    return np.where((d > 0) & (d <= reach), 1.0 / np.where(d > 0, d, 1.0), 0.0)


def smooth_field(rng: np.random.Generator, n: int, width: int = 6) -> np.ndarray:
    """A spatially autocorrelated map on the line: moving average of white noise."""
    noise = rng.standard_normal(n + width)
    return np.convolve(noise, np.ones(width) / width, mode="valid")[:n]


def test_empirical_p_add_one():
    null = np.array([0.1, -0.3, 0.5, -0.6, np.nan])
    # |null| >= 0.5: 0.5 and -0.6 -> k = 2 of n = 4 finite values
    assert empirical_p_twosided(null, 0.5) == pytest.approx(3 / 5)
    # an observed effect beyond every surrogate gives 1 / (1 + n), never 0
    assert empirical_p_twosided(null, 2.0) == pytest.approx(1 / 5)
    assert np.isnan(empirical_p_twosided(null, np.nan))
    assert np.isnan(empirical_p_twosided(np.array([np.nan]), 0.2))


def test_excess_magnitude_matches_twosided():
    rng = np.random.default_rng(0)
    null = rng.standard_normal(500)
    for obs in (-1.5, -0.1, 0.0, 0.7, 3.0):
        assert empirical_p_excess_magnitude(null, obs) == empirical_p_twosided(null, obs)


def test_rank_corr_columns_is_spearman():
    rng = np.random.default_rng(1)
    x, Y = rng.standard_normal(40), rng.standard_normal((40, 6))
    expected = [spearmanr(x, Y[:, k])[0] for k in range(Y.shape[1])]
    np.testing.assert_allclose(rank_corr_columns(x, Y), expected, rtol=1e-12)


def test_one_block_is_brainspace_moran():
    """With a single block, the null is exactly brainspace's MoranRandomization with the same seed."""
    w = chain_weights(50)
    y = smooth_field(np.random.default_rng(2), 50)
    ours = HemisphereMoranRandomization(np.full(50, "LH"), 20, random_state=7).fit(w).randomize(y)
    ref = MoranRandomization(n_rep=20, procedure="singleton", spectrum="nonzero", tol=1e-6, random_state=7)
    np.testing.assert_array_equal(ours, ref.fit(w).randomize(y))


def test_blocks_are_randomised_separately():
    """Each hemisphere is fitted on its own block; a tiny block is permuted instead."""
    w = np.zeros((53, 53))
    w[:50, :50] = chain_weights(50)
    blocks = np.array(["LH"] * 50 + ["RH"] * 3)
    y = np.arange(53, dtype=float)
    surrogates = HemisphereMoranRandomization(blocks, 10, random_state=0, min_block=4).fit(w).randomize(y)
    assert surrogates.shape == (10, 53)
    # the 3-vertex block (< min_block) keeps its own values, shuffled
    for s in surrogates:
        assert sorted(s[50:]) == [50.0, 51.0, 52.0]


def test_moran_spearman_returns_observed_r():
    rng = np.random.default_rng(3)
    x, y = smooth_field(rng, 60), smooth_field(rng, 60)
    msr = HemisphereMoranRandomization(np.full(60, "LH"), 50, random_state=0).fit(chain_weights(60))
    r, p, null = moran_spearman(x, y, msr)
    assert r == pytest.approx(spearmanr(x, y)[0])
    assert 0 < p <= 1 and null.shape == (50,)
    d, p_d, d_null = moran_spearman_difference(x, x, y, msr)
    assert d == 0 and np.all(d_null == 0) and p_d == 1


def test_moran_null_false_positive_rate():
    """Two independent but spatially smooth maps: the Moran test rejects at about the nominal rate."""
    rng = np.random.default_rng(4)
    n, n_maps = 60, 200
    msr = HemisphereMoranRandomization(np.full(n, "LH"), 99, random_state=0).fit(chain_weights(n))
    p = np.array([moran_spearman(smooth_field(rng, n), smooth_field(rng, n), msr)[1] for _ in range(n_maps)])
    # nominal 5 %; 0.12 is about 4.6 binomial standard errors above it for 200 maps
    assert np.mean(p < 0.05) <= 0.12

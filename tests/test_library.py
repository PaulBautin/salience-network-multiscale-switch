"""Projection, microstructure, iEEG, surface and provenance helpers of salience_switch, on synthetic data."""
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch
from scipy.stats import zscore

from salience_switch.colors import network_rgb, yeo7_rgb
from salience_switch.connectivity import prepare_weights, projection_masks, weighted_mean_projection
from salience_switch.ieeg import (N_LH, SPEC_BANDS, aperiodic_component, compute_spectral_parameters, covered_vertices,
                                  gaussian_channel_sensitivity, hemisphere_slice, mirror_to_cortex,
                                  project_channel_measure, welch_psd_segments)
from salience_switch.microstructure import compute_t1map, partial_corr_with_covariate
from salience_switch.provenance import git_version
from salience_switch.surface import convert_states_str2int, load_yeo_surf_5k


def test_weighted_mean_projection_by_hand():
    W = np.array([[1.0, 3.0, 0.0, -2.0],    # negative and zero weights are ignored
                  [2.0, np.nan, 2.0, 0.0],  # NaN weight ignored
                  [0.0, 0.0, 0.0, 5.0]])
    g = np.array([1.0, 2.0, np.nan, 4.0])  # a NaN target value removes that target
    P = weighted_mean_projection(W, g, min_valid=1)
    np.testing.assert_allclose(P, [(1 * 1 + 3 * 2) / 4, 1.0, 4.0])
    # fewer valid targets than min_valid gives NaN
    assert np.isnan(weighted_mean_projection(W, g, min_valid=2)).tolist() == [False, True, True]


@pytest.mark.parametrize("modality", ["SC", "GD", "MPC", "FC"])
def test_prepare_weights_block_equals_slice(modality):
    """The source x target block is the slice of the full preprocessed matrix, value for value."""
    rng = np.random.default_rng(2)
    n = 60
    hemi = np.repeat(["LH", "RH"], n // 2)
    sn = np.zeros(n, dtype=bool)
    sn[[3, 4, 5, 40, 41]] = True
    targets = ~sn
    targets[[10, 50]] = False
    W = rng.random((n, n)).astype(np.float32) - (0.2 if modality in ("MPC", "FC") else 0.0)
    W = np.triu(W, 1) + np.triu(W, 1).T
    mask_G = (rng.random((n, n)) > 0.4) if modality == "SC" else None
    full = prepare_weights(W, modality, hemi, sn, mask_G=mask_G)
    block = prepare_weights(W, modality, hemi, sn, mask_G=mask_G, targets=targets)
    np.testing.assert_array_equal(block, full[np.ix_(sn, targets)])


def test_projection_masks():
    df = pd.DataFrame({"hemisphere": ["LH", "LH", "LH", None, "RH", "RH"],
                       "network": ["A", "A", "B", "medial_wall", "A", "B"]})
    g = np.array([1.0, np.nan, 5.0, 7.0, 2.0, 3.0])
    g_sn, sn, other = projection_masks(df, g, "A", "both")
    assert g_sn.tolist() == [1.0, 2.0] and sn.tolist() == [True, False, False, True, False]
    assert other.tolist() == [False, False, True, False, True]
    _, sn_lh, other_lh = projection_masks(df, g, "A", "LH")
    assert sn_lh.tolist() == [True, False, False, False, False] and other_lh.tolist() == [False, False, True, False, False]


def test_network_rgb_follows_the_alphabetical_rows():
    np.testing.assert_array_equal(network_rgb("Cont"), yeo7_rgb[0])
    np.testing.assert_array_equal(network_rgb("Vis"), yeo7_rgb[6])
    np.testing.assert_array_equal(network_rgb("medial_wall"), yeo7_rgb[-1])
    np.testing.assert_array_equal(network_rgb(np.nan), yeo7_rgb[-1])


def test_ieeg_vertex_helpers():
    assert hemisphere_slice("LH") == slice(0, N_LH) and hemisphere_slice("RH") == slice(N_LH, 2 * N_LH)
    sens = np.array([[0.0, 1.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0]])
    covered = covered_vertices(sens, np.array([True, True, True, True]), np.array([1.0, np.nan, 3.0, 4.0]))
    assert covered.tolist() == [3]


def test_welch_psd_segments_skips_the_zero_gaps():
    """The spectrum is the mean of the window periodograms inside the segments, gaps and padding left out."""
    fs, rng = 200.0, np.random.default_rng(0)
    seg_a, seg_b, seg_c = rng.standard_normal(1400), rng.standard_normal(1000), rng.standard_normal(300)
    # segment, 2-s gap, segment, 2-s gap, segment shorter than one window, 1-s padding at the end
    x = np.concatenate([seg_a, np.zeros(400), seg_b, np.zeros(400), seg_c, np.zeros(200)])
    freq, pxx, n_windows = welch_psd_segments(x, fs, fmin=0.5, fmax=80.0)
    f_ref, p_a = welch(seg_a, fs=fs, window="hamming", nperseg=400, noverlap=200)
    _, p_b = welch(seg_b, fs=fs, window="hamming", nperseg=400, noverlap=200)
    keep = (f_ref >= 0.5) & (f_ref <= 80.0)
    expected = (6 * p_a + 4 * p_b)[keep]  # 6 and 4 windows; seg_c holds none
    np.testing.assert_array_equal(freq, f_ref[keep])
    assert n_windows.tolist() == [10]
    np.testing.assert_allclose(pxx[0], expected / expected.sum(), rtol=1e-10)


def test_welch_psd_segments_without_gaps_is_plain_welch():
    fs, x = 200.0, np.random.default_rng(1).standard_normal((2, 2000))
    x[1, 500] = 0.0  # an isolated zero sample is data, not a gap
    freq, pxx, n_windows = welch_psd_segments(x, fs, fmin=1.0, fmax=45.0)
    f_ref, p_ref = welch(x, fs=fs, window="hamming", nperseg=400, noverlap=200, axis=-1)
    keep = (f_ref >= 1.0) & (f_ref <= 45.0)
    np.testing.assert_allclose(pxx, p_ref[:, keep] / p_ref[:, keep].sum(axis=1, keepdims=True), rtol=1e-10)
    assert n_windows.tolist() == [9, 9]


def test_gaussian_channel_sensitivity_halves_at_h_and_folds():
    points = np.zeros((N_LH, 3))
    points[:, 0] = np.arange(N_LH) * 1.0  # vertices 1 mm apart on a line
    sens = gaussian_channel_sensitivity(points, points, np.array([100, N_LH + 100]), half_weight_mm=4.0,
                                        rel_thresh=0.05)
    assert sens.shape == (2, N_LH)
    np.testing.assert_array_equal(sens[0], sens[1])  # a right vertex folds onto its template index
    assert sens[0, 100] == 1.0
    np.testing.assert_allclose(sens[0, [96, 104]], 0.5)
    radius = 4.0 * np.sqrt(np.log2(20.0))  # 8.3 mm: the weight there is 5 % of the peak
    support = np.flatnonzero(sens[0])
    assert support.tolist() == list(range(100 - int(radius), 100 + int(radius) + 1))
    assert sens[0, support].min() >= 0.05
    with pytest.raises(ValueError):
        gaussian_channel_sensitivity(points[:10], points, np.array([0]))


def test_partial_corr_removes_the_covariate():
    """Two vertices that share only the covariate are uncorrelated once it is regressed out."""
    rng = np.random.default_rng(0)
    covar = rng.standard_normal(200)
    X = np.column_stack([covar + 0.1 * rng.standard_normal(200), covar + 0.1 * rng.standard_normal(200)])
    MPC = partial_corr_with_covariate(X, covar)
    assert MPC.shape == (2, 2) and np.allclose(MPC, MPC.T)
    assert abs(MPC[0, 1]) < 0.2  # Fisher z of a near-zero partial correlation


def test_t1map_is_zscored_mean():
    profiles = np.random.default_rng(1).normal(1500, 100, size=(3, 14, 50))
    np.testing.assert_allclose(compute_t1map(profiles), zscore(profiles.mean(axis=(0, 1))))


def test_aperiodic_component_formula():
    f = np.array([1.0, 10.0])
    np.testing.assert_allclose(aperiodic_component(f, np.array([2.0]), np.array([0.0]), np.array([1.5])),
                               [[2.0, 2.0 - 1.5]])


def test_specparam_recovers_exponent():
    """Pure aperiodic spectra with an alpha peak: the fitted exponent is the generating one."""
    f = np.arange(0.5, 80.5, 0.5)
    exponents = np.array([1.0, 1.8, 2.5])
    log_power = 1.0 - np.log10(f[None, :] ** exponents[:, None])
    log_power += 0.4 * np.exp(-0.5 * ((f - 10.0) / 1.5) ** 2)  # alpha peak
    spec = compute_spectral_parameters(10 ** log_power, f, bands=SPEC_BANDS)
    np.testing.assert_allclose(spec["exponent"], exponents, atol=0.05)
    assert np.all(spec["band_power"]["alpha"] > 0.2)


def test_project_channel_measure_weighted_average():
    sens = np.array([[1.0, 0.0, 0.0],
                     [3.0, 1.0, 0.0]])  # 2 channels x 3 vertices; vertex 2 has no coverage
    out = project_channel_measure(np.array([2.0, 6.0]), sens)
    np.testing.assert_allclose(out[:2], [(2 * 1 + 6 * 3) / 4, 6.0])
    assert np.isnan(out[2])


def test_mirror_to_cortex_folds_onto_both_hemispheres():
    df = pd.DataFrame({"hemisphere": ["LH", "LH", None, "RH", "RH", "RH"],
                       "network": ["A", "B", None, "A", "A", "B"]})
    out = mirror_to_cortex(np.array([1.0, 2.0, 3.0]), df, network="A")
    np.testing.assert_array_equal(np.isnan(out), [False, True, True, False, False, True])
    assert out[0] == 1.0 and out[3] == 1.0 and out[4] == 2.0


def test_convert_states_str2int():
    states, labels = convert_states_str2int(np.array(["Vis", "SomMot", "Vis"]))
    assert labels.tolist() == ["SomMot", "Vis"] and states.tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("loader, n", [(load_yeo_surf_5k, 9684)])
def test_surface_tables(loader, n):
    df = loader()
    assert len(df) == n
    assert set(df["network"].dropna()) == {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default",
                                           "medial_wall"}
    assert (df["hemisphere"].iloc[: n // 2].dropna() == "LH").all()


def test_git_version_tracks_code_only(tmp_path):
    """Outputs neither dirty the version nor change it once committed; code and new files do."""
    def git(*args):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@example.org",
                        "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *args],
                       check=True, capture_output=True)
    git("init", "-q")
    assert git_version(tmp_path) == "unknown"  # no commit yet
    for name in ("code.py", "results/panel.svg", "source_data/table.tsv"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_text("a")
    git("add", "-A")
    git("commit", "-q", "-m", "init")
    code_version = git_version(tmp_path)
    assert len(code_version) == 7 and not code_version.endswith("-dirty")
    (tmp_path / "results/panel.svg").write_text("b")
    (tmp_path / "source_data/table.tsv").write_text("b")
    assert git_version(tmp_path) == code_version
    git("commit", "-q", "-am", "outputs only")
    assert git_version(tmp_path) == code_version  # an output commit is not a code change
    (tmp_path / "new_module.py").write_text("a")
    assert git_version(tmp_path) == code_version + "-dirty"  # untracked code counts
    (tmp_path / "new_module.py").unlink()
    (tmp_path / "code.py").write_text("b")
    assert git_version(tmp_path) == code_version + "-dirty"
    git("commit", "-q", "-am", "code")
    assert git_version(tmp_path) not in (code_version, "unknown")

"""Projection, microstructure, iEEG, surface and provenance helpers of salience_switch, on synthetic data."""
import subprocess

import numpy as np
import pandas as pd
import pytest
from scipy.stats import zscore

from salience_switch.connectivity import weighted_mean_projection
from salience_switch.ieeg import (SPEC_BANDS, aperiodic_component, compute_spectral_parameters, mirror_to_cortex,
                                  project_channel_measure)
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


def test_git_version_ignores_pipeline_outputs(tmp_path):
    """Rewritten results/ and source_data/ leave the version clean; a code change makes it dirty."""
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
    head = git_version(tmp_path)
    assert head not in ("", "unknown") and not head.endswith("-dirty")
    (tmp_path / "results/panel.svg").write_text("b")
    (tmp_path / "source_data/table.tsv").write_text("b")
    assert git_version(tmp_path) == head
    (tmp_path / "code.py").write_text("b")
    assert git_version(tmp_path) == head + "-dirty"

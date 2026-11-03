"""Pipeline settings, table round trips, and the privacy of the public figure data."""
import importlib.util

import numpy as np
import pandas as pd
import pytest
import yaml

from salience_switch import config



def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, config.PROJECT_ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_all():
    spec = importlib.util.spec_from_file_location("run_all", config.PROJECT_ROOT / "pipeline" / "run_all.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_step_has_settings():
    settings = yaml.safe_load((config.CONFIG_DIR / "analysis.yaml").read_text())
    for step in _run_all().STEPS:
        assert step in settings, f"config/analysis.yaml has no section for {step}"
        assert (config.PROJECT_ROOT / "pipeline" / f"{step}.py").is_file()


def test_table_round_trip_is_exact(tmp_path):
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"x": rng.standard_normal(1000), "y": rng.uniform(-1e-12, 1e12, 1000),
                          "label": rng.choice(["a", "b"], 1000)})
    path = tmp_path / "table.tsv"
    config.write_table(table, path)
    back = config.read_table(path)
    assert back["x"].to_numpy().tobytes() == table["x"].to_numpy().tobytes()
    assert back["y"].to_numpy().tobytes() == table["y"].to_numpy().tobytes()
    assert back["label"].tolist() == table["label"].tolist()


def _source_files():
    files = sorted(p for p in config.SOURCE_DATA_DIR.rglob("*") if p.is_file())
    if not files:
        pytest.skip("source_data/ is empty")
    return files


def test_source_data_has_no_participant_information():
    """source_data/ passes the privacy check of the hook and the CI (tools/check_privacy.py)."""
    check = _load_tool("check_privacy")
    for path in _source_files():
        rel = path.relative_to(config.PROJECT_ROOT).as_posix()
        assert not check.problems(rel, path.read_bytes()), f"{rel}: {check.problems(rel, path.read_bytes())}"


def test_privacy_check_refuses_identifiers():
    check = _load_tool("check_privacy")
    fake_id = b"PNC" + b"012"  # built at run time, so this file itself passes the check
    assert check.problems("notes.md", b"subject " + fake_id + b" ran") == ["participant ID"]
    assert check.problems("results/a.svg", b"data:image/png;base64,QUPX123") == []  # inside base64 data
    assert check.problems("source_data/s/t.tsv", b"subject\tr\n1\t2\n") == ["identifying columns ['subject']"]
    assert check.problems("source_data/s/x.csv", b"a\n")
    assert check.problems("cache.pkl", b"") == ["private data location or file type"]

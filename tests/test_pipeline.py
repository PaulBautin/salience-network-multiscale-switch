"""Pipeline settings, table round trips, and the privacy of the public figure data."""
import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from salience_switch import config

# Participant identifiers of MICA-PNI, MICA-MICs and electroMICA (not inside base64 data)
ID_PATTERN = re.compile(r"(^|[^A-Za-z0-9+/])(PNC|HC|PX)[0-9]{3}([^0-9]|$)")
# Columns that would identify a participant, a recording or a session
PRIVATE_COLUMNS = {"subject", "ID_PNI", "ID_MICs", "session", "Session", "ChannelName", "ContactName1",
                   "ContactName2", "path_t1_profile", "dob", "age", "sex"}


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


def test_source_data_holds_only_tables_and_manifests():
    for path in _source_files():
        assert path.suffix in {".tsv", ".json"}, f"unexpected file type in source_data: {path}"


def test_source_data_has_no_participant_information():
    for path in _source_files():
        text = path.read_text()
        assert not ID_PATTERN.search(text), f"participant ID in {path}"
        if path.suffix == ".tsv":
            columns = set(text.split("\n", 1)[0].split("\t"))
            assert not columns & PRIVATE_COLUMNS, f"identifying columns {columns & PRIVATE_COLUMNS} in {path}"
        else:
            keys = set(json.loads(text))
            assert not keys & (PRIVATE_COLUMNS | {"subjects"}), f"identifying keys {keys & PRIVATE_COLUMNS} in {path}"

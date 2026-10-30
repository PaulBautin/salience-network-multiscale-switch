"""Where the data are, and the settings of each pipeline step.

``config/paths.yaml`` (git-ignored; copy ``config/paths.example.yaml``) gives the location
of the private datasets and of the work directory on this machine. It is needed only to
compute; drawing the figures from ``source_data/`` needs no private data.
``config/analysis.yaml`` holds the settings of each step.

Each step uses three folders:

- ``<work_dir>/<step>/``: private tables, manifest and log (local only, never in git);
- ``source_data/<step>/``: the de-identified tables its figures are drawn from (public);
- ``results/figures/``: the figure panels (public).
"""
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
SOURCE_DATA_DIR = PROJECT_ROOT / "source_data"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def load_paths(path: Path | None = None) -> dict[str, Path]:
    """Locations of the private datasets and of the work directory.

    Parameters
    ----------
    path : Path, optional
        YAML file; default ``config/paths.yaml``.

    Returns
    -------
    paths : dict
        Key -> path, for every key set in the file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = path or CONFIG_DIR / "paths.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found: copy config/paths.example.yaml to config/paths.yaml and set "
            "the paths on this machine (needed only to compute; plotting reads source_data/).")
    raw = yaml.safe_load(path.read_text()) or {}
    return {key: Path(value).expanduser() for key, value in raw.items() if value}


def step_settings(step: str, path: Path | None = None) -> dict:
    """Settings of one step from ``config/analysis.yaml``.

    Parameters
    ----------
    step : str
        Step name, e.g. ``'step3_connectivity'``.
    path : Path, optional
        YAML file; default ``config/analysis.yaml``.

    Returns
    -------
    settings : dict
    """
    path = path or CONFIG_DIR / "analysis.yaml"
    return dict((yaml.safe_load(path.read_text()) or {}).get(step) or {})


def work_dir(step: str, paths: dict[str, Path] | None = None) -> Path:
    """Private folder of a step, ``<work_dir>/<step>``, created if missing.

    Parameters
    ----------
    step : str
        Step name.
    paths : dict, optional
        Output of `load_paths`; read from ``config/paths.yaml`` if omitted.

    Returns
    -------
    folder : Path
    """
    paths = paths if paths is not None else load_paths()
    folder = paths["work_dir"] / step
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a tab-separated table written by `write_table`, floats exactly as written.

    Parameters
    ----------
    path : Path
        ``.tsv`` file.
    **kwargs
        Passed to ``pandas.read_csv``.

    Returns
    -------
    table : pd.DataFrame
    """
    return pd.read_csv(path, sep="\t", float_precision="round_trip", **kwargs)


def write_table(table: pd.DataFrame, path: Path, **kwargs) -> None:
    """Write a table as tab-separated text, without the index.

    float64 columns are written with every digit, so `read_table` returns them exactly.
    float32 columns are written in their shortest float32 form: cast them back to
    float32 after reading to recover the exact values, or convert them to float64
    before writing.

    Parameters
    ----------
    table : pd.DataFrame
    path : Path
        ``.tsv`` file.
    **kwargs
        Passed to ``pandas.DataFrame.to_csv`` (e.g. ``float_format``).
    """
    table.to_csv(path, sep="\t", index=False, **kwargs)


def source_dir(step: str) -> Path:
    """Public folder of a step's figure tables, ``source_data/<step>``, created if missing.

    Parameters
    ----------
    step : str
        Step name.

    Returns
    -------
    folder : Path
    """
    folder = SOURCE_DATA_DIR / step
    folder.mkdir(parents=True, exist_ok=True)
    return folder

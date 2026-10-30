"""Run logs and provenance records for each pipeline step.

Every step appends a run header and its log messages to ``<work_dir>/<step>/<step>.log``
and writes a ``manifest.json`` next to its tables: the step, the time, the git commit of
the code, the versions of the main packages and the settings of the run.
"""
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Packages whose versions are recorded in every manifest.
TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "nibabel", "brainspace", "vtk", "specparam",
                    "matplotlib")
# Folders the pipeline writes to; their changes do not make the code version dirty.
OUTPUT_DIRS = ("results", "source_data")


def setup_step_logger(step: str, log_dir: Path, settings: dict | None = None) -> logging.Logger:
    """Log to the console and append to ``<log_dir>/<step>.log``.

    Attaches a file handler to the root logger, so log messages from the library
    modules are captured too. Each call appends a timestamped run header.

    Parameters
    ----------
    step : str
        Step name, used for the log file name.
    log_dir : Path
        Folder of the log file; created if missing.
    settings : dict, optional
        Settings of the run, written to the header.

    Returns
    -------
    logging.Logger
        Named logger for the step.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{step}.log"

    root = logging.getLogger()

    # Console handler, unless the caller configured logging already
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(ch)
        root.setLevel(logging.INFO)

    # One step log at a time: drop the file handler of a previous step in this process
    for handler in [h for h in root.handlers if isinstance(h, logging.FileHandler)]:
        root.removeHandler(handler)
        handler.close()

    # Append-mode file handler (plain text, no level prefix for readability)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)

    logger = logging.getLogger(step)

    sep = "=" * 72
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(sep)
    logger.info(f"STEP   : {step}")
    logger.info(f"DATE   : {timestamp}")
    for k, v in (settings or {}).items():
        logger.info(f"  {k}: {v}")
    logger.info(sep)

    return logger


def git_version(project_root: Path) -> str:
    """Short commit hash of the code, with ``-dirty`` if the code has uncommitted changes.

    Changes under `OUTPUT_DIRS`, which the pipeline itself rewrites, do not count: a run
    of committed code records a clean hash even after earlier steps have rewritten their
    tables and figures.

    Parameters
    ----------
    project_root : Path
        Root of the git working tree.

    Returns
    -------
    version : str
        e.g. ``'3f2a9c1'`` or ``'3f2a9c1-dirty'``; ``'unknown'`` outside a git repository
        or before its first commit.
    """
    git = ["git", "-C", str(project_root)]
    try:
        head = subprocess.run(git + ["rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        changed = subprocess.run(git + ["diff", "--quiet", "HEAD", "--", "."]
                                 + [f":(exclude){d}" for d in OUTPUT_DIRS], capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("-dirty" if changed.returncode else "")


def package_versions() -> dict[str, str]:
    """Versions of Python and of the packages in `TRACKED_PACKAGES`.

    Returns
    -------
    versions : dict
        Package name -> version string (``'not installed'`` if missing).
    """
    versions = {"python": platform.python_version()}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def write_manifest(path: Path, step: str, project_root: Path, **fields) -> dict:
    """Write the provenance record of a step run as JSON.

    Parameters
    ----------
    path : Path
        Output ``manifest.json``.
    step : str
        Step name.
    project_root : Path
        Root of the git working tree, for the code version.
    **fields
        Settings and counts of the run (e.g. ``hemisphere``, ``n_permutations``,
        ``n_subjects``); must be JSON-serialisable.

    Returns
    -------
    manifest : dict
        The record written.
    """
    manifest = {
        "step": step,
        "created": datetime.now().isoformat(timespec="seconds"),
        "git": git_version(project_root),
        "versions": package_versions(),
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

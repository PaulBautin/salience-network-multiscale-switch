"""
Manuscript logging utility.

Adds a timestamped file handler to the root logger so that all logging
output (including from imported modules) is captured in logs/<script>.log.
Each script run appends a new block; the file is never overwritten.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_manuscript_logger(script_name: str, project_root: Path, args=None) -> logging.Logger:
    """
    Configure logging to write to both console and logs/<script_name>.log.

    Attaches a file handler to the root logger so that logging calls from
    any module (e.g. gradient_computation, ieeg_processing) are also captured.
    Each invocation appends a timestamped run header to the log file.

    Parameters
    ----------
    script_name : str
        Base name used for the log file (no extension).
    project_root : Path
        Project root directory; logs/ subdirectory is created here.
    args : argparse.Namespace, optional
        Parsed command-line arguments; key-value pairs are written to the header.

    Returns
    -------
    logging.Logger
        Named logger for the calling script.
    """
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{script_name}.log"

    root = logging.getLogger()

    # Ensure console handler exists (scripts that skip basicConfig need this)
    if not root.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        root.addHandler(ch)
        root.setLevel(logging.INFO)

    # Append-mode file handler (plain text, no level prefix for readability)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(fh)

    logger = logging.getLogger(script_name)

    sep = "=" * 72
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(sep)
    logger.info(f"SCRIPT : {script_name}")
    logger.info(f"DATE   : {timestamp}")
    if args is not None:
        for k, v in vars(args).items():
            if v is not None:
                logger.info(f"  --{k}: {v}")
    logger.info(sep)

    return logger

#!/usr/bin/env python3
"""Refuse participant-level data in the repository.

The one definition of what may not be published, used by the pre-commit hook
(``.githooks/pre-commit``), the CI (``.github/workflows/tests.yml``) and the tests
(``tests/test_pipeline.py``). A file is refused if

1. it is in a private location or of a raw-data type (`PRIVATE_PATH`);
2. a line of it holds a participant ID of MICA-PNI (PNC###), MICA-MICs (HC###) or
   electroMICA (PX###), outside base64 image data (`ID_PATTERN`);
3. it is a figure table of ``source_data/`` with an identifying column
   (`PRIVATE_COLUMNS`), a manifest with an identifying key, or any other kind of file.

Binary files are checked by path only. Standard library only, so it runs before the
package is installed.

Run      python tools/check_privacy.py --staged     # the files of the next commit (hook)
         python tools/check_privacy.py --tracked    # every tracked file (CI)
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID_PATTERN = re.compile(r"(^|[^A-Za-z0-9+/])(PNC|HC|PX)[0-9]{3}([^0-9]|$)")
PRIVATE_PATH = re.compile(r"^(data/dataframes|logs)/|\.(pkl|mat|npy|npz)$")
# Columns (and manifest keys) that would identify a participant, a recording or a session
PRIVATE_COLUMNS = frozenset({"subject", "subjects", "ID_PNI", "ID_MICs", "session", "Session", "ChannelName",
                             "ContactName1", "ContactName2", "path_t1_profile", "dob", "age", "sex"})


def problems(path: str, content: bytes) -> list[str]:
    """Why a file may not be published (empty if it may).

    Parameters
    ----------
    path : str
        Path relative to the repository root, with forward slashes.
    content : bytes
        The file's content.

    Returns
    -------
    reasons : list of str
    """
    if PRIVATE_PATH.search(path):
        return ["private data location or file type"]
    if b"\0" in content[:8000]:  # binary, as git decides
        return []
    text = content.decode("utf-8", errors="replace")
    reasons = []
    if any(ID_PATTERN.search(line) for line in text.splitlines()):
        reasons.append("participant ID")
    if path.startswith("source_data/"):
        if path.endswith(".tsv"):
            names = set(text.split("\n", 1)[0].split("\t"))
        elif path.endswith(".json"):
            names = set(json.loads(text))
        else:
            return reasons + ["source_data/ holds only .tsv tables and .json manifests"]
        if names & PRIVATE_COLUMNS:
            reasons.append(f"identifying columns {sorted(names & PRIVATE_COLUMNS)}")
    return reasons


def _git(*args: str) -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=True).stdout
    return [p for p in out.decode().split("\0") if p]


def main() -> int:
    parser = argparse.ArgumentParser(description="Refuse participant-level data in the repository.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="check the files staged for the next commit")
    mode.add_argument("--tracked", action="store_true", help="check every tracked file")
    args = parser.parse_args()

    if args.staged:
        files = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
        read = lambda p: subprocess.run(["git", "-C", str(ROOT), "show", f":{p}"],  # noqa: E731
                                        capture_output=True, check=True).stdout
    else:
        files = [p for p in _git("ls-files", "-z") if (ROOT / p).is_file()]
        read = lambda p: (ROOT / p).read_bytes()  # noqa: E731
    refused = {p: r for p in files if (r := problems(p, read(p)))}
    for path, reasons in refused.items():
        print(f"check_privacy: {path}: {'; '.join(reasons)}")
    if refused:
        print("Refused: participant-level data stays local (git restore --staged <file>).")
    return 1 if refused else 0


if __name__ == "__main__":
    sys.exit(main())

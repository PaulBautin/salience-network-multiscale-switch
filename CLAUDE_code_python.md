# CLAUDE.md — AI Assistant Instructions

## Project-Specific Rules 

---

## Non-Negotiable Rules

> Hard constraints. Never violate these, even if the user asks.

- **No `print()` in API/library code.** Use `logging` exclusively.
- **No hardcoded absolute paths.** Use `pathlib.Path` and relative references.
- **Never silently swallow exceptions.** Always re-raise or log with context and an actionable message.
- **Always preserve NIfTI affine + header** when writing modified images.
- **No functions with > 10 parameters** — suggest a dataclass or config object instead.
- **No new dependencies** without flagging them and adding to `environment.yml` / `pyproject.toml`.
- **No modifying the user's working directory** or writing files as a side-effect of a compute function.
- **Do not break BIDS directory or file naming** in existing data paths.

---

## Architecture

### Layer separation (strict)

```
CLI scripts  →  delegates to  →  API modules  →  (no imports back up)
```

- **API modules** (`<package>/`): pure logic — no CLI imports, no `argparse`, no `print()`.
- **CLI scripts**: thin wrappers — parse args, call API, handle exit codes. No business logic.
- **Tests** (`tests/`): no side-effects; always clean up created files (use `tmp_path`).

### Design priorities (in order)

1. Correctness and reproducibility over cleverness.
2. Many small, single-responsibility functions over monolithic ones.
3. Forgiving inputs where sensible (e.g., auto-reorient before failing) — **always document this behaviour**.
4. Metric / compute functions must not write to disk.

---

## 5. Code Style

### Python

- **Formatter:** `black` | **Linter:** `ruff` | **Style:** PEP 8
- Type hints on **all** public functions and methods.
- F-strings for string formatting.
- Single quotes for strings unless the string itself contains a single quote.
- `pathlib.Path` for all file paths — never `os.path` string concatenation.

---

## 6. Naming Conventions

### Python variables — prefix system

```python
path_data    # str: path to a directory          e.g. /home/user/data/
file_data    # str: filename without extension   e.g. my_file
ext_data     # str: extension                    e.g. nii.gz
fname_data   # str: full path + filename + ext
im           # object: nibabel image object
data         # array: numpy array of image data
```

Prefer prefixed names when multiple similar variables coexist:

```python
# ✓ Good — sorts logically in debugger
path_input, path_seg, path_output
im_input, im_seg, im_label

# ✗ Avoid
input_path, seg_path, output_path
```

### File names

- `snake_case` only — no camelCase, no spaces.
- **Verb before object:** `register_to_template.py` ✓ | `template_registration.py` ✗
- **Explicit over terse:** `segment_brain_t1.py` ✓ | `segment.py` ✗
- Standard anatomical vocabulary: "intervertebral disc" not "disk".

### Git

- Branches: `feature/<short-description>`, `fix/<short-description>`, `docs/<short-description>`
- Commits: imperative mood, ≤ 72 chars — e.g., `Add surface gradient computation`

---

## 7. Canonical Code Patterns

> Use these as templates. Do not improvise alternatives without good reason.

### NIfTI read / write

```python
import nibabel as nib
import numpy as np
from pathlib import Path

def load_nifti(fname: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    im = nib.load(fname)
    data = np.asarray(im.dataobj)
    return im, data

def save_nifti(data: np.ndarray, ref_im: nib.Nifti1Image, fname: Path) -> None:
    """Save array as NIfTI, preserving affine and header from ref_im."""
    out = nib.Nifti1Image(data, affine=ref_im.affine, header=ref_im.header)
    nib.save(out, fname)
```

### Logging

```python
import logging
logger = logging.getLogger(__name__)

logger.debug('Resampling image to 1 mm isotropic')
logger.info('Segmentation complete: %d voxels labelled', n_voxels)
logger.warning('No T1w found; skipping registration step')
logger.error('Could not read file: %s', fname)
```

### Wrapping external tools

```python
import os
import subprocess
from pathlib import Path

def _check_env(var: str, install_hint: str) -> None:
    if not os.environ.get(var):
        raise EnvironmentError(
            f'{var} is not set. {install_hint}'
        )

def run_bet(fname_input: Path, fname_output: Path, frac: float = 0.5) -> None:
    _check_env('FSLDIR', 'Install FSL and source $FSLDIR/etc/fslconf/fsl.sh')
    cmd = ['bet', str(fname_input), str(fname_output), '-f', str(frac)]
    logger.debug('Running: %s', ' '.join(cmd))
    subprocess.run(cmd, check=True)
```

### Public function signature + docstring

```python
def compute_metric(
    fname_input: Path,
    fname_seg: Path,
    *,
    smoothing_fwhm: float = 0.0,
    verbose: bool = False,
) -> dict[str, float]:
    """Compute <metric> from a segmentation mask.

    Parameters
    ----------
    fname_input : Path
        Full path to the input NIfTI image.
    fname_seg : Path
        Full path to the binary segmentation mask.
    smoothing_fwhm : float, optional
        FWHM of Gaussian smoothing kernel in mm (0 = no smoothing).
    verbose : bool, optional
        If True, emit DEBUG-level log messages.

    Returns
    -------
    dict[str, float]
        Mapping of metric names to computed values.

    Raises
    ------
    FileNotFoundError
        If ``fname_input`` or ``fname_seg`` does not exist.
    """
```

---

## 8. Neuroimaging-Specific Rules

- Follow **BIDS** conventions for data organization and file naming.
- Handle NIfTI files via `nibabel` — no raw binary I/O.
- Prefer `nibabel + numpy` for image manipulation; use `nilearn`, `dipy`, or `ANTsPy` for higher-level ops.
- Check required environment variables before calling external tools; raise a clear `EnvironmentError` with install instructions if missing.
- Log the exact CLI call of every external tool at `DEBUG` level for reproducibility.

---

## 9. Testing

- **Unit tests** → `tests/unit/` — required for every new function or module.
- **Functional / integration tests** → `tests/functional/` — required for major pipeline steps.
- Tests must be side-effect-free. Use `tmp_path`; clean up any created files.
- Use `@pytest.mark.parametrize` for multiple input scenarios.

```bash
pytest tests/ -v   # run before every PR
```

---

## 10. Documentation

- Every public function and class needs a **NumPy-style docstring**: one-line summary + Parameters + Returns + Raises.
- Inline comments only for **why**, never for **what** (the code shows what).
- CLI tools: use `argparse`; `--help` must be complete and accurate.
- Update `README.md` and `CHANGELOG.md` as part of every meaningful change.

---

## 11. Repository File Checklist

Flag any of the following that are missing or stale:

| File              | Required        | Notes                                      |
|-------------------|-----------------|--------------------------------------------|
| `CHANGELOG.md`    | ✅ Always        | Update on every meaningful change          |
| `LICENSE`         | ✅ Always        | MIT or Apache 2.0                          |
| `README.md`       | ✅ Always        | Purpose, install, usage, examples          |
| `environment.yml` | ✅ Always        | Pinned dependencies                        |
| `CITATION.cff`    | On release      | Update if authorship or version changes    |
| `pyproject.toml`  | If packaged     | Bump version before tagging                |
| `Dockerfile`      | Recommended     | For full environment reproducibility       |

---

## 12. Tradeoff Defaults

When facing a choice not covered by an explicit rule, apply these defaults:

| Tradeoff | Default |
|---|---|
| Correctness vs. brevity | Correctness |
| Explicit error vs. silent fallback | Explicit error |
| New dependency vs. more code | More code, unless already in the ecosystem |
| Forgiving input vs. strict validation | Forgiving — but document it |
| Unsure about BIDS naming | Ask before guessing |
| Unsure about affine handling | Preserve original; never recompute from scratch |
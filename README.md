# Salience network multiscale switch

This repository contains the data and code to reproduce the results presented in the paper:

> **Title:** THE MULTISCALE ARCHITECTURE OF THE SALIENCE NETWORK SUPPORTS A BRAIN-WIDE "SWITCH" FUNCTION
> **Authors:** Author List
> **Link:** Preprint

## Overview

Understanding how relatively static brain anatomy supports dynamic patterns of brain activity remains a fundamental challenge in neuroscience. The salience network (SN) is hypothesized to regulate critical transitions between internally and externally oriented brain states, yet the neuroanatomical principles enabling this flexibility remain elusive. Integrating in vivo 7T neuroimaging with ultra-high-resolution ex vivo histology and intracranial electrophysiology, we demonstrate that the SN possesses a distinct superior-inferior architectural profile. We find that specific patterns of laminar differentiation, connectivity and electrophysiology uniquely position the SN to bridge "task-negative" and "task-positive" systems. These results establish a structural basis for the SN's switching function, offering a mechanistic link between cortical microarchitecture and the dynamic regulation of human brain states.

## Requirements

- Python ≥ 3.10
- [micapipe](https://micapipe.readthedocs.io/) v0.2.0 — for generating T1 profiles, DWI connectomes, and cortical surfaces
- [brainspace](https://brainspace.readthedocs.io/) — for gradient computation and surface visualization
- [bctpy](https://github.com/aestrivex/bctpy) — for structural connectome communication models

See [pyproject.toml](pyproject.toml) for the full Python dependency list.

## Installation

**Option 1 — uv (recommended for pure Python environments)**

[uv](https://docs.astral.sh/uv/) is a fast Python package manager.

```bash
git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
cd salience-network-multiscale-switch
uv venv env_salience             # creates env_salience/ virtual environment
uv pip install -e . --python env_salience/bin/python
source env_salience/bin/activate
```

**Option 2 — Conda (recommended when MKL / compiled binaries are needed)**

```bash
git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
cd salience-network-multiscale-switch
conda env create -f environment.yml
conda activate env_salience
```

## Data

Raw data are not distributed within this repository due to size and privacy constraints. Instructions for obtaining and organizing the data, including links to all source datasets, are provided in [data/README.md](data/README.md).

Scripts that require external data accept derivative paths via CLI flags (see [Usage](#usage) below). Intermediate results are cached as TSV/CSV in `data/dataframes/` and reused on subsequent runs.

## Usage

All scripts are run from the **project root**. Pass `-h` to any script for the full flag reference.

| Script | Figure | Flags |
|--------|--------|-------|
| `scripts/figure_1a_t1map.py` | 1a — T1 microstructural gradient | `-pni_deriv` (req), `-mics_deriv` (req), `-hemi` (opt) |
| `scripts/figure_1b_contextualisation.py` | 1b — Multi-modal contextualization | `-hemi` (opt) |
| `scripts/figure_1c_cortical_types.py` | 1c — Cortical types | `-pni_deriv` (opt) |
| `scripts/figure_2_distance.py` | 2 — SC/GD/MPC differences at gradient extremes | `-hemi` (opt) |
| `scripts/figure_3_ieeg_mni.py` | 3 — iEEG MNI open atlas | `-pni_deriv` (req), `-ieeg_deriv` (req, `.mat` file), `-hemi` (opt) |
| `scripts/figure_3_ieeg_mica.py` | 3 — iEEG MICA dataset | `-ieeg_deriv` (req), `-hemi` (opt) |

**Example — Figure 1a**

```bash
python scripts/figure_1a_t1map.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -mics_deriv /path/to/BIDS_MICs/derivatives/micapipe_v0.2.0 \
  -hemi LH
```

Outputs are written to `results/figures/` as SVG/PNG. Run logs are saved to `logs/`.

## Repository structure

```
salience-network-multiscale-switch/
├── data/
│   ├── surfaces/        # fsLR-32k inflated and sphere GIFTIs
│   ├── parcellations/   # Schaefer-400, Von Economo, BigBrain, AHEAD profiles
│   └── dataframes/      # cached intermediate TSV/CSV results
├── docs/
│   ├── api.md           # src/ module and function reference
│   └── methods.md       # acquisition parameters and analysis methods
├── results/figures/     # SVG/PNG output figures
├── scripts/             # entry-point scripts, one per figure panel
├── src/                 # reusable processing modules
├── environment.yml      # Conda environment
└── pyproject.toml       # Python package metadata
```

See [docs/api.md](docs/api.md) for the `src/` module reference and DataFrame conventions.

## Citation

If you use this code, please cite the manuscript (link above).

# Salience Network Multiscale Switch

Code and data to reproduce the analyses from:

> **THE MULTISCALE ARCHITECTURE OF THE SALIENCE NETWORK SUPPORTS A BRAIN-WIDE "SWITCH" FUNCTION**
> Author List — *Preprint*

## What this repository does

Integrating in vivo 7T neuroimaging with ultra-high-resolution ex vivo histology and intracranial electrophysiology, this codebase reproduces all figures demonstrating that the salience network (SN) possesses a distinct superior-inferior architectural profile. Specific patterns of laminar differentiation, structural connectivity, and electrophysiology uniquely position the SN to bridge task-negative and task-positive systems.

## Quick start

```bash
git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
cd salience-network-multiscale-switch
uv venv env_salience
uv pip install -e . --python env_salience/bin/python
source env_salience/bin/activate

python scripts/figure_1a_t1map.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -mics_deriv /path/to/BIDS_MICs/derivatives/micapipe_v0.2.0
```

## Documentation

- [Usage](usage.md) — CLI reference for all figure scripts
- [API Reference](api/overview.md) — `src/` module and function documentation
- [Data Acquisition](methods/datasets.md) — MRI and iEEG acquisition parameters
- [Shared Methods](methods/shared.md) — MPC gradient and spatial statistics
- [Figure 1 Methods](methods/figure_1.md) — Microstructural heterogeneity
- [Figure 2 Methods](methods/figure_2.md) — Connectivity fingerprints
- [Figure 3 Methods](methods/figure_3.md) — iEEG frequency mapping

## Repository layout

```
salience-network-multiscale-switch/
├── data/
│   ├── surfaces/        # fsLR-32k inflated and sphere GIFTIs
│   ├── parcellations/   # Schaefer-400, Von Economo, BigBrain, AHEAD profiles
│   └── dataframes/      # cached intermediate TSV/CSV results
├── docs/                # this documentation
├── results/figures/     # SVG/PNG output figures
├── scripts/             # entry-point scripts, one per figure panel
└── src/                 # reusable processing modules
```

## Requirements

- Python ≥ 3.10
- [micapipe](https://micapipe.readthedocs.io/) v0.2.0
- [brainspace](https://brainspace.readthedocs.io/)
- [bctpy](https://github.com/aestrivex/bctpy)

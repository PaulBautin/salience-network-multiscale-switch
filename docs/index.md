# Salience Network Multiscale Switch

Code and figure data to reproduce the analyses from:

> **THE MULTISCALE ARCHITECTURE OF THE SALIENCE NETWORK SUPPORTS A BRAIN-WIDE "SWITCH" FUNCTION**
> Author List — *Preprint*

## What this repository does

Integrating in vivo 7T neuroimaging with ultra-high-resolution ex vivo histology and intracranial electrophysiology, this codebase reproduces all figures of the study of the salience network (SN): its microstructural organisation (Figure 1), its electrophysiology (Figure 2) and its connectivity (Figure 3).

The analyses run as a pipeline of steps, one per figure, in figure order. Each step computes the tables behind its figures and plots the figures from those tables alone. The tables are published without participant information in `source_data/`, so every figure can be redrawn without the participants' data.

## Quick start

```bash
git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
cd salience-network-multiscale-switch
uv venv env_salience
uv pip install -e ".[dev]" --python env_salience/bin/python
source env_salience/bin/activate

python pipeline/run_all.py --stage plot     # redraw every figure from source_data/
```

To recompute from the data, copy `config/paths.example.yaml` to `config/paths.yaml`, set the paths of the datasets on your machine, and run `python pipeline/run_all.py`. On a server without a display, prefix the commands with `xvfb-run -a`.

## Documentation

- [Usage](usage.md) — how to run each step, its inputs and outputs
- [Architecture](architecture.md) — the design of the repository, in diagrams
- [Reproducibility](reproducibility.md) — each figure panel: its step, tables and methods
- [API Reference](api/overview.md) — the `salience_switch` library, from its docstrings
- [Data Acquisition](methods/datasets.md) — MRI and iEEG acquisition parameters
- [Shared Methods](methods/shared.md) — MPC gradient and spatial statistics
- [Figure 1 Methods](methods/figure_1.md) — Microstructural heterogeneity
- [Figure 2 Methods](methods/figure_2.md) — iEEG frequency mapping
- [Figure 3 Methods](methods/figure_3.md) — Connectivity fingerprints

## Repository layout

```
salience-network-multiscale-switch/
├── pipeline/            # the steps, in run order: step1a … step3_schematic, step4; run_all.py
├── src/salience_switch/ # analysis library used by the steps
├── config/              # analysis.yaml (step settings); paths.example.yaml (private data locations)
├── source_data/         # tables behind every figure, de-identified, one folder per step
├── data/                # template surfaces, atlases and histological maps
├── results/             # composite figures; results/figures/: the panels
├── tests/               # tests on synthetic data, and the privacy check of source_data/
├── tools/               # one-off preparation of the bundled fsLR-5k maps
└── docs/                # this documentation
```

## Requirements

- Python ≥ 3.10
- [micapipe](https://micapipe.readthedocs.io/) v0.2.0 derivatives, to recompute (not run by this repository)
- [brainspace](https://brainspace.readthedocs.io/) 0.1.22 — gradients, spatial null models, surface rendering
- [specparam](https://specparam-tools.github.io/) 2.0.0rc7 — iEEG spectral parameterisation (pinned: the code uses the 2.0 API)
- VTK 9.3 (`vtk>=9.3,<9.4`) — brainspace 0.1.22 cannot draw colour bars with VTK ≥ 9.4
- seaborn, matplotlib, nibabel, numpy, scipy, pandas, pyyaml
- [Inkscape](https://inkscape.org/) ≥ 1.0 — only for step 4 (exporting the composites)

The software versions used for the manuscript are listed in [Shared Methods — Software](methods/shared.md#software).

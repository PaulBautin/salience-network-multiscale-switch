# Salience network multiscale switch

Code and figure data to reproduce the results of the paper:

> **Title:** THE MULTISCALE ARCHITECTURE OF THE SALIENCE NETWORK SUPPORTS A BRAIN-WIDE "SWITCH" FUNCTION
> **Authors:** Author List
> **Link:** Preprint

## Overview

Understanding how relatively static brain anatomy supports dynamic patterns of brain activity remains a fundamental challenge in neuroscience. The salience network (SN) is hypothesized to regulate critical transitions between internally and externally oriented brain states, yet the neuroanatomical principles enabling this flexibility remain elusive. Integrating in vivo 7T neuroimaging with ultra-high-resolution ex vivo histology and intracranial electrophysiology, we demonstrate that the SN possesses a distinct superior-inferior architectural profile. We find that specific patterns of laminar differentiation, connectivity and electrophysiology uniquely position the SN to bridge "task-negative" and "task-positive" systems. These results establish a structural basis for the SN's switching function, offering a mechanistic link between cortical microarchitecture and the dynamic regulation of human brain states.

## How it is organised

The analyses run as a pipeline of steps, one per figure, in figure order (step 1 for Figure 1 to step 3 for Figure 3; step 4 exports the composites). Each step **computes** the tables behind its figures and **plots** the figures from those tables alone. The tables are published, without any participant information, in `source_data/`, so every figure can be redrawn without access to the participants' data. [docs/architecture.md](docs/architecture.md) shows the design in diagrams.

```
salience-network-multiscale-switch/
├── pipeline/            # the steps, run in this order: step1a … step3_schematic, step4; run_all.py runs them all
├── src/salience_switch/ # analysis library used by the steps (installed with pip install -e .)
├── config/              # analysis.yaml: settings of each step; paths.example.yaml: where the private data are
├── source_data/         # tables behind every figure, de-identified, one folder per step, with its manifest
├── data/                # template surfaces, atlases and histological maps (public)
├── results/             # composite figures (figure*.svg, PNG, PDF); results/figures/: the panels
├── tests/               # tests on synthetic data, and the privacy check of source_data/
├── tools/               # one-off preparation of the bundled fsLR-5k maps
└── docs/                # documentation site: usage, architecture, API reference, methods
```

## Installation

```bash
git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
cd salience-network-multiscale-switch
```

**uv:**

```bash
uv venv env_salience
uv pip install -e ".[dev]" --python env_salience/bin/python
source env_salience/bin/activate
```

**Conda:**

```bash
conda env create -f environment.yml
conda activate env_salience
```

## Reproducing the figures

On a server without a display, prefix the commands with `xvfb-run -a` (the brain renders need an X display).

**Redraw every figure** from the published tables (no private data needed):

```bash
python pipeline/run_all.py --stage plot
```

**Recompute everything** from the data: copy `config/paths.example.yaml` to `config/paths.yaml`, set the paths of the datasets on your machine, then

```bash
python pipeline/run_all.py
```

**One step** (`--stage compute`, `plot`, or `all` by default):

```bash
python pipeline/step3_connectivity.py --stage plot
```

Steps 1b and 1c read only public inputs, so they also compute without the private data. The commands, inputs and outputs of every step are in [docs/usage.md](docs/usage.md); [docs/reproducibility.md](docs/reproducibility.md) maps each figure panel to its step, tables and methods section.

## Data availability

The participants' data are not distributed with this repository. The datasets, and what is shared here, are described in [data/README.md](data/README.md).

## Requirements

- Python ≥ 3.10
- [micapipe](https://micapipe.readthedocs.io/) v0.2.0 derivatives, to recompute (not run by this repository)
- [brainspace](https://brainspace.readthedocs.io/) 0.1.22 — gradients, spatial null models and surface renders
- [specparam](https://specparam-tools.github.io/) 2.0.0rc7 — iEEG spectral parameterisation (pinned: the code uses the 2.0 API)
- VTK 9.3 (`vtk>=9.3,<9.4`) — brainspace 0.1.22 cannot draw colour bars with VTK ≥ 9.4
- seaborn, matplotlib, nibabel, numpy, scipy, pandas, pyyaml
- [Inkscape](https://inkscape.org/) ≥ 1.0 — only for step 4 (exporting the composites)

[pyproject.toml](pyproject.toml) lists the Python dependencies, and the [Software](docs/methods/shared.md#software) section of the methods the versions used for the manuscript.

## Contributing

Enable the pre-commit check once per clone; it refuses commits that contain participant data:

```bash
git config core.hooksPath .githooks
```

Run the tests and the lint check before pushing (both also run on GitHub):

```bash
pytest
ruff check src pipeline tests
```

## Documentation

The documentation site covers usage, the architecture, the API reference (generated from the docstrings) and the methods: https://paulbautin.github.io/salience-network-multiscale-switch/

## Citation

If you use this code, please cite the manuscript (link above).

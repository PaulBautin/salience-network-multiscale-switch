# Salience network multiscale switch
<p align="justify">
This repository contains the data and code to reproduce the results presented in the paper:

> **Title:** THE MULTISCALE ARCHITECTURE OF THE SALIENCE NETWORK SUPPORTS A BRAIN-WIDE “SWITCH” FUNCTION  
> **Authors:** Author List  
> **Link:** Preprint
</p>

## Overview
<p align="justify">
Understanding how relatively static brain anatomy supports dynamic patterns of brain activity remains a fundamental challenge in neuroscience. The salience network (SN) is hypothesized to regulate critical transitions between internally and externally oriented brain states, yet the neuroanatomical principles enabling this flexibility remain elusive. Integrating in vivo 7T neuroimaging with ultra-high-resolution ex vivo histology and intracranial electrophysiology, we demonstrate that the SN possesses a distinct superior-inferior architectural profile. We find that specific patterns of laminar differentiation, connectivity and electrophysiology uniquely position the SN to bridge “task-negative” and “task-positive” systems. These results establish a structural basis for the SN’s switching function, offering a mechanistic link between cortical microarchitecture and the dynamic regulation of human brain states. 
</p>

## Repository structure
- `data/`: Contains raw and processed data.
- `results/`: Output figures and tables appearing in the manuscript.
- `scripts/`: Entry-point scripts for analyses and figures
- `src/`: Python scripts for data processing and statistical modeling.

## Installation & Requirements
<p align="justify">
Two installation paths are supported. Choose whichever best fits your workflow.

**Option 1 – uv (recommended for pure Python environments)**

[uv](https://docs.astral.sh/uv/) is a fast Python package manager.

```bash
git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
cd salience-network-multiscale-switch
uv venv env_salience             # creates env_salience/ virtual environment
uv pip install -e . --python env_salience/bin/python
source env_salience/bin/activate
```

**Option 2 – Conda (recommended when MKL / compiled binaries are needed)**

```bash
git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
conda env create -f environment.yml
conda activate env_salience
```
</p>

## Data availability
<p align="justify">
Raw data are not distributed within this repository due to size and/or privacy constraints. Instructions for obtaining and organizing the data are provided in: data/README.md

The analyses presented in this work rely on the following publicly available datasets:

- **MICA-PNI**: In vivo 7T MRI dataset (quantitative T1, DWI, resting-state fMRI) acquired at the McConnell Brain Imaging Centre, Montreal Neurological Institute. Available on [OpenNeuro](https://openneuro.org/datasets/ds005565).

- **MICA-MICs**: Multimodal imaging and connectome dataset acquired at the McConnell Brain Imaging Centre, Montreal Neurological Institute. Available on [OpenNeuro](https://openneuro.org/datasets/ds004472).

- **AHEAD**: Ultra-high-resolution multimodal 3D post-mortem human brain atlas (200 µm; Bielschowsky and Parvalbumin stainings). Available at [https://doi.org/10.21942/uva.16844500.v1](https://doi.org/10.21942/uva.16844500.v1).

- **BigBrain**: Ultra-high-resolution (20 µm) 3D histological reconstruction of a post-mortem human brain. Available at [https://bigbrain.loris.ca/main.php](https://bigbrain.loris.ca/main.php).

- **MNI open iEEG atlas**: Multicentre intracranial EEG dataset comprising recordings from 106 patients with intractable epilepsy mapped to a common stereotactic space. Available at [https://mni-open-ieegatlas.research.mcgill.ca](https://mni-open-ieegatlas.research.mcgill.ca).
</p>

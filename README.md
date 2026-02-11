# Salience network multiscale switch

This repository contains the data and code to reproduce the results presented in the paper:

> **Title:** THE MULTISCALE ARCHITECTURE OF THE SALIENCE NETWORK SUPPORTS A BRAIN-WIDE “SWITCH” FUNCTION  
> **Authors:** Author List  
> **Link:** Preprint

## Overview
<div style="text-align: justify">
Understanding how relatively static brain anatomy supports dynamic patterns of brain activity remains a fundamental challenge in neuroscience. The salience network (SN) is hypothesized to regulate critical transitions between internally and externally oriented brain states, yet the neuroanatomical principles enabling this flexibility remain elusive. Integrating in vivo 7T neuroimaging with ultra-high-resolution ex vivo histology and intracranial electrophysiology, we demonstrate that the SN possesses a distinct superior-inferior architectural profile. We find that specific patterns of laminar differentiation, connectivity and electrophysiology uniquely position the SN to bridge “task-negative” and “task-positive” systems. These results establish a structural basis for the SN’s switching function, offering a mechanistic link between cortical microarchitecture and the dynamic regulation of human brain states.
</div>

## Repository structure

- `data/`: Contains raw and processed data.
- `results/`: Output figures and tables appearing in the manuscript.
- `scripts/`: Entry-point scripts for analyses and figures
- `src/`: Python scripts for data processing and statistical modeling.

## Installation & Requirements

The recommended setup uses a Conda environment.

   ```bash
   git clone https://github.com/PaulBautin/salience-network-multiscale-switch.git
   conda env create -f environment.yml
   conda activate env_salience
   ```

## Data availability
Raw data are not distributed within this repository due to size and/or privacy constraints. Instructions for obtaining and organizing the data are provided in: data/README.md

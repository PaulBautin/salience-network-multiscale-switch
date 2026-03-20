# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Two options are available:

**uv (fast, pure-Python):**
```bash
uv venv env_salience
uv pip install -e . --python env_salience/bin/python
source env_salience/bin/activate
```

**Conda:**
```bash
conda env create -f environment.yml
conda activate env_salience
```

Key dependencies: `numpy`, `pandas`, `nibabel`, `matplotlib`, `scipy`, `brainspace`, `bctpy`.

## Running Scripts

All figure scripts are run from the **project root** and require paths to external neuroimaging derivative datasets (not included in this repo). Each script uses `argparse` and accepts `-pni_deriv` and/or `-mics_deriv` flags pointing to `micapipe_v0.2.0` derivative directories:

```bash
# Figure 1a – T1 microstructural gradient
python scripts/figure_1a_t1map.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -mics_deriv /path/to/BIDS_MICs/derivatives/micapipe_v0.2.0

# Figure 2 – Distance/connectivity analysis
python scripts/figure_2_distance.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0

# Figure 3 – iEEG (MNI and MICA datasets)
python scripts/figure_3_ieeg_mni.py
python scripts/figure_3_ieeg_mica.py
```

Scripts cache intermediate DataFrames to `data/dataframes/` (TSV/CSV) and skip recomputation if the cached file exists. Outputs are written to `results/figures/` as SVG/PNG.

## Architecture

### Data flow
1. **scripts/** — Entry points, one per figure panel. Each script loads data, calls `src/` modules for computation, then plots and saves to `results/figures/`.
2. **src/** — Reusable processing modules imported by scripts.
3. **data/** — Static inputs: parcellations (`.label.gii`), brain surfaces (`.surf.gii`), histological profiles (`.shape.gii`), and cached subject-level DataFrames.

### Core modules (`src/`)

- **`atlas_load.py`** — Loads and merges atlases onto the fsLR-32k surface. Central function is `load_yeo_atlas()` which returns `df_yeo_surf`, a per-vertex DataFrame with Schaefer-400 parcellation, Yeo 7-network labels, hemisphere, and salience network border mask. Other loaders (`load_bigbrain`, `load_ahead_biel`, `load_ahead_parva`, `load_econo_atlas`, `load_t1map`) add histology columns to this same DataFrame.

- **`gradient_computation.py`** — Computes microstructure profile covariance (MPC) gradients. `compute_t1_gradient()` takes subject T1 intensity profiles, computes per-subject partial-correlation matrices (controlling for the mean profile), fits `GradientMaps` (diffusion map, normalized angle kernel, procrustes alignment), and appends the z-scored first gradient to `df_yeo_surf`.

- **`ieeg_processing.py`** — iEEG signal processing pipeline. Loads MATLAB `.mat` files from BIDS-formatted iEEG datasets, preprocesses signals (bandpass → downsample → demean), computes Welch PSD, extracts band power (delta/theta/alpha/beta/gamma), and maps channels to fsLR-32k surface vertices via GIFTI sensitivity maps.

- **`plot_colors.py`** — Color definitions for Yeo 7-network plotting (`yeo7_rgba`, `yeo7_rgb`).

### Surface space
All analyses use **fsLR-32k** space (64,984 vertices total: 32,492 LH + 32,492 RH). Parcellation is **Schaefer-400** with Yeo 7-network labels. The primary network of interest is `'SalVentAttn'` (Salience/Ventral Attention).

### Key DataFrame conventions
- `df_yeo_surf`: one row per surface vertex, columns include `mics` (parcel ID), `network`, `hemisphere`, `label`, `salience_border`, based on schaeffer 400 parcellation. Analysis columns (e.g., `t1_gradient1_SalVentAttn`, `T1map`, `BigBrain`) are added in-place.
- iEEG DataFrames: one row per channel, with `Subject`, `Session`, `ChannelName`, and signal/PSD columns.

### External data not in repo
- PNI/MICs micapipe derivatives (T1 profiles, structural connectomes, tractography)
- BIDS iEEG dataset at `/host/verges/tank/data/BIDS_iEEG/`
- Baillarger/Intrusion MYATLAS parcellations (hardcoded paths in `atlas_load.py`)

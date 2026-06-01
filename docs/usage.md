# Usage

All scripts are run from the **project root**. Each script accepts `-h` for the full flag reference.

Scripts cache intermediate results as TSV/CSV in `data/dataframes/` and skip recomputation on subsequent runs. Outputs are written to `results/figures/` as SVG/PNG. Run logs are saved to `logs/`.

## Scripts

| Script | Figure panel | Description |
|--------|-------------|-------------|
| [`figure_1a_t1map.py`](#figure-1a) | 1a | T1 microstructural gradient within the salience network |
| [`figure_1b_contextualisation.py`](#figure-1b) | 1b | Multi-modal contextualization of the T1 gradient |
| [`figure_1c_cortical_types.py`](#figure-1c) | 1c | Von Economo cortical type distribution |
| [`figure_2_distance.py`](#figure-2) | 2 | Connectivity-weighted MPC gradient projection correlated with the FC gradient |
| [`figure_3_ieeg_mni.py`](#figure-3-mni) | 3 | iEEG spectral analysis — MNI open atlas |
| [`figure_3_ieeg_mica.py`](#figure-3-mica) | 3 | iEEG spectral analysis — MICA dataset |

---

## Figure 1a

**`scripts/figure_1a_t1map.py`** — Processes MICA-PNI micapipe derivatives to extract T1 intensity profiles, compute MPC gradients within the salience network, and visualize the relationship between T1 profiles and gradient values.

```bash
python scripts/figure_1a_t1map.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -mics_deriv /path/to/BIDS_MICs/derivatives/micapipe_v0.2.0 \
  -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-pni_deriv` | yes | — | Path to PNI micapipe derivatives directory |
| `-mics_deriv` | yes | — | Path to MICs micapipe derivatives directory |
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |

**Outputs**

- `results/figures/figure_1a_profiles.svg`
- `results/figures/figure_1a_brain.svg`
- `data/dataframes/df_1a_<hemi>.tsv` (cache)

---

## Figure 1b

**`scripts/figure_1b_contextualisation.py`** — Correlates the T1 gradient with BigBrain, AHEAD Bielschowsky, and AHEAD Parvalbumin histological profiles using spin permutation tests.

```bash
python scripts/figure_1b_contextualisation.py -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |

> No external derivatives are required. All histological profiles are bundled in `data/parcellations/`.

**Outputs**

- `results/figures/figure_1b_correlations.svg`
- `results/figures/figure_1b_correlations_circle.svg`
- `results/figures/figure_1b_brain_t1map.svg`
- `results/figures/figure_1b_brain_bigbrain.svg`
- `results/figures/figure_1b_brain_biel.svg`
- `results/figures/figure_1b_brain_parva.svg`

---

## Figure 1c

**`scripts/figure_1c_cortical_types.py`** — Maps Von Economo cortical types onto the salience network and tests for non-random type distributions using spin permutations.

```bash
python scripts/figure_1c_cortical_types.py -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-pni_deriv` | no | — | Path to PNI micapipe derivatives (for surface loading) |

**Outputs**

- `results/figures/figure_1c_brain_economo.svg`
- `results/figures/figure_1c_type_salience.svg`

---

## Figure 2

**`scripts/figure_2_distance.py`** — Computes the connectivity-weighted MPC gradient projection per other-network vertex (SC, GD, and MPC weights) and correlates the z-scored projection maps with the whole-brain FC gradient (Spearman, spin-test corrected). Panel 2A reports SC, GD, and MPC for the salience network; Figure 2B replicates the SC projection across all 7 Yeo networks. All analysis runs at fsLR-5k resolution (9,684 vertices).

Requires `figure_1a_t1map.py` to have been run first (produces `data/dataframes/figure_1a_pni_to_mics.csv`).

```bash
python scripts/figure_2_distance.py -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |

**Outputs**

- `results/figures/figure_2a_distance_metric.svg`
- `results/figures/figure_2a_brain_{SC,GD,MPC}_rho.svg`
- `results/figures/figure_2b_distance_network_SC.svg`
- `results/figures/figure_2b_brain_SC_rho_<network>.svg`
- `data/dataframes/df_2b_label_<hemi>.csv` (cache)

---

## Figure 3 MNI

**`scripts/figure_3_ieeg_mni.py`** — Computes PSD and band power from the MNI open iEEG atlas, maps channels to fsLR-32k surface vertices, and correlates band power with the T1 gradient.

```bash
python scripts/figure_3_ieeg_mni.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -ieeg_deriv /path/to/MNI_ieeg/MatlabFile.mat
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-pni_deriv` | yes | — | Path to PNI micapipe derivatives directory |
| `-ieeg_deriv` | yes | — | Path to the MNI iEEG atlas MATLAB file (`.mat`) |
| `-hemi` | no | `RH` | Hemisphere: `both`, `LH`, or `RH` |

**Outputs** — `results/figures/figure_3a_ieeg_mni_*.svg`

---

## Figure 3 MICA

**`scripts/figure_3_ieeg_mica.py`** — Same pipeline as Figure 3 MNI, applied to the MICA intracranial EEG dataset with subject-specific leadfield sensitivity maps.

```bash
python scripts/figure_3_ieeg_mica.py \
  -ieeg_deriv /path/to/BIDS_iEEG/derivatives/electroMICA \
  -hemi RH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-ieeg_deriv` | yes | — | Path to electroMICA derivatives directory |
| `-hemi` | no | `RH` | Hemisphere: `both`, `LH`, or `RH` |

**Outputs** — `results/figures/figure_3b_ieeg_mica_*.svg`

---

## External data paths

Scripts that require micapipe or iEEG derivatives expect the following directory structure:

```
micapipe_v0.2.0/
└── sub-<id>/
    └── ses-<id>/
        ├── mpc/        # T1 intensity profiles (.shape.gii)
        ├── dwi/        # tractography and connectome files
        └── surf/       # cortical surface reconstructions
```

See [data/README.md](https://github.com/PaulBautin/salience-network-multiscale-switch/blob/main/data/README.md) for download instructions.

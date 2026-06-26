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
- `data/dataframes/figure_1a_pni_to_mics{,_5k}.csv` (subject → file-path tables)
- `data/dataframes/df_1a_<hemi>.tsv`, `df_1a_<hemi>_fslr5k.tsv` (surface table + gradient cache)

---

## Figure 1b

**`scripts/figure_1b_contextualisation.py`** — Correlates the T1 gradient with BigBrain, AHEAD Bielschowsky, and AHEAD Parvalbumin histological profiles, with significance assessed against a within-network Moran spectral-randomisation null (the correlation is restricted to salience-network vertices).

```bash
python scripts/figure_1b_contextualisation.py -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |

> No external derivatives are required. All histological profiles are bundled in `data/parcellations/`.

**Outputs**

- `results/figures/figure_1b_correlations.svg`
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

**`scripts/figure_2_distance.py`** — Computes the connectivity-weighted projection of the whole-brain FC gradient across each source-network vertex's extranetwork targets, then correlates it per subject with the within-network MPC gradient (Spearman). Every modality uses only positive connections via the same weighted-mean projection. Panel 2A is laid out like Figure 1B: one row per modality (SC, GD, MPC, FC) with two columns. The left column is the salience-network scatter with the within-network MPC gradient on a single shared bottom x-axis and that measure's projection P on y (group _r_ and spatial-null _p_); the right column is a horizontal lollipop placing all 7 Yeo networks (stem length = |group _r_| on a shared bottom |_r_| axis, network-coloured, with FDR-Moran-significant networks filled and starred). Significance uses two complementary nulls: a within-network Moran spectral-randomisation null (map smoothness; all measures) and a geometry-preserving topological null that rewires the connectome within geodesic-distance bins (wiring specificity; SC/MPC/FC), each Benjamini–Hochberg-corrected across networks per measure. Figure 2B replicates the projection for all four measures across all 7 Yeo networks and summarises it as a bubble matrix (rows = networks, columns = measures; disc colour = group _r_, area = |_r_|, black ring + stars = FDR-corrected significance). The per-network projection is the single computation behind both panels, so the lollipop reuses the same numbers as the bubble matrix. All analysis runs at fsLR-5k resolution (9,684 vertices).

Requires `figure_1a_t1map.py` to have been run first: it reads the subject → file table `data/dataframes/figure_1a_pni_to_mics_5k.csv` (required) and reuses the cached fsLR-5k MPC gradient from `data/dataframes/df_1a_<hemi>_fslr5k.tsv` when present, recomputing it in-figure otherwise. The `df_1a_*.tsv` caches are tab-separated.

```bash
# full run: compute the projection + nulls, then draw every figure
python scripts/figure_2_distance.py -hemi LH
# iterate on figure aesthetics without recomputing (loads caches, seconds):
python scripts/figure_2_distance.py -hemi LH -stage plot
# redraw only Figure 2A from cache:
python scripts/figure_2_distance.py -hemi LH -stage plot -panel 2a
```

The expensive projection + Moran nulls are separated from drawing by `-stage`. `both` (default) and `compute` both run the computation over all 7 networks, write every figure-data cache below, **and draw the figures** (so a fresh compute always refreshes the figures); `plot` skips all heavy loads and redraws figures from those caches in seconds (so you can iterate on layout/colours without rerunning anything). `-stage plot` requires a prior `compute`/`both` run for the same `-hemi`. `-panel` then selects which figures are rendered.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |
| `-panel` | no | `both` | Figures to render: `both`, `2a`, or `2b` |
| `-stage` | no | `both` | Pipeline stage: `both`/`compute` (compute + write caches + draw), or `plot` (redraw from cache only) |
| `-n_rand` | no | `1000` | Surrogates for the Moran and topological nulls. Lower (e.g. `300`) for faster iteration; the add-one empirical _p_ floor is 1/(1+`n_rand`), so keep `1000` for the final run. |

**Outputs**

- `results/figures/figure_2a_distance_metric.svg` (Figure-1B-style: scatter + horizontal |_r_| lollipop, one row per modality; the scatter y-axis is the z-scored projection P)
- `results/figures/figure_2a_brain_{SC,GD,MPC,FC}_rho.svg` (salience-network projection maps; both hemispheres, colorbar on the right)
- `results/figures/figure_2b_distance_network_{SC,GD,MPC,FC}.svg` (per-measure scatter grid)
- `results/figures/figure_2b_network_summary_<hemi>.svg` (bubble matrix, all measures × networks)
- `results/figures/figure_2b_brain_{SC,GD,MPC,FC}_rho_<network>.svg`
- `results/figures/figure_2_supp_topo_control.svg` (topological-null power/specificity control: SC null distributions for a wiring-aligned and a geometry-only synthetic map, written in the compute stage)
- `data/dataframes/df_2b_label_<hemi>.csv` (vertex cache; `{network}_{measure}_P` projection columns — z-scored per network×measure for display — + `{network}_{measure}_dominant` dominant-target-network columns that drive the scatter colours on replot)
- `data/dataframes/df_2b_network_stats_<measure>_<hemi>.csv` (per-network group stats, including `p_moran`/`q_moran` and the geometry-preserving topological-null `p_topo`/`q_topo`; `p_topo`/`q_topo` are NaN for the GD measure)
- `data/dataframes/df_2b_network_subject_r_<measure>_<hemi>.csv` (per-subject _r_; row index is the subject ID, one column per network)

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

**`scripts/figure_3_ieeg_mica.py`** — Applies the iEEG spectral pipeline to the MICA intracranial EEG dataset using subject-specific leadfield sensitivity maps. Sensitivity-weighted band power is correlated with the within-network MPC gradient, and the spectral similarity between surface vertices is used as a connectivity measure to project the FC gradient (the Figure 2 gradient-weighted projection, group-level variant) and test it against the MPC gradient. Significance for both analyses is a within-network Moran spectral-randomisation null with the add-one empirical *p*.

```bash
python scripts/figure_3_ieeg_mica.py \
  -ieeg_deriv /path/to/BIDS_iEEG/derivatives/electroMICA \
  -hemi RH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-ieeg_deriv` | yes | — | Path to electroMICA derivatives directory |
| `-hemi` | no | `RH` | Hemisphere: `LH` or `RH` (the sensitivity maps are evaluated one hemisphere at a time) |
| `-network` | no | `SalVentAttn` | Yeo 7-network analysis target |

Requires `data/dataframes/df_1a_{hemi}.tsv` (run `figure_1a_t1map.py` with the matching `-hemi` first).

**Outputs** — `results/figures/figure_3b_ieeg_mica_*.svg`: the primary aperiodic-exponent (1/f slope) scatter (`*_slope_corr_{hemi}.svg`) and brain map (`*_slope_map_{hemi}.svg`); the FDR-corrected oscillatory band-power panels (`*_psd_{hemi}.svg`, `*_band_power_corr_{hemi}.svg`, `*_{band}_map_{hemi}.svg`, `*_sensitivity_map_{hemi}.svg`); and the spectral-similarity projection scatter (`*_es_scatter_{hemi}.svg`), projection brain map (`*_es_map_{hemi}.svg`), and the channel-level PSD correlation matrix (`*_corr_{hemi}.svg`).

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

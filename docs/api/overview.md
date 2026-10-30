# API Reference

The library `salience_switch` (in `src/salience_switch/`) holds the reusable analysis
code; the steps in `pipeline/` call it to make each figure. Every page of this
reference is generated from the docstrings of the code.

## Modules

| Module | Description |
|--------|-------------|
| [`surface`](surface.md) | Surface tables (Schaefer-400, Yeo 7 networks, hemisphere per vertex), atlases, histological maps and the FC gradient bundled in `data/` |
| [`microstructure`](microstructure.md) | qT1 intracortical profiles and the diffusion-map MPC gradient |
| [`connectivity`](connectivity.md) | Connectome I/O, the distance-stratified consensus mask, the gradient-weighted connectivity projection and its Moran, topological and spin null models |
| [`stats`](stats.md) | The per-hemisphere Moran spectral randomization, the Moran-null Spearman tests and the add-one empirical p-values, shared by every step |
| [`ieeg`](ieeg.md) | iEEG preprocessing, Welch spectra, spectral parameterisation (`specparam`), bipolar sensitivity maps and projection onto the template |
| [`plotting`](plotting.md) | brainspace renders composed in matplotlib: hemisphere rows with circular zoom insets and traced region outlines; ridgeline, forest, test-grid and lollipop panels aligned with them |
| [`colors`](colors.md) | Yeo 7-network, cortical-type, histology, iEEG and connectivity colours; registered colormaps |
| [`config`](config.md) | `config/paths.yaml` and `config/analysis.yaml`, the step folders, and exact table input/output |
| [`provenance`](provenance.md) | Step logs, code version, package versions and the manifest of each step |

The modules depend on one another in one direction only: `connectivity` uses `stats`;
every other module stands alone.

## Surface space

Most analyses use **fsLR-32k** space: 64,984 vertices (32,492 LH + 32,492 RH). The
Figure 3 connectivity analyses use the downsampled **fsLR-5k** space: 9,684 vertices
(4,842 LH + 4,842 RH). In both, vertices are ordered LH then RH. Parcellation is
**Schaefer-400** with Yeo 7-network labels. The primary network of interest is
`SalVentAttn` (Salience/Ventral Attention).

## Surface table

One row per surface vertex, LH then RH. Returned by `surface.load_yeo_atlas()`
(fsLR-32k) or `surface.load_yeo_surf_5k()` (fsLR-5k, without `network_int` and
`salience_border`) and extended by the loaders and steps.

| Column | Type | Description |
|--------|------|-------------|
| `mics` | `float` | Schaefer-400 parcel ID: LH parcels 1001–1200, RH parcels 2001–2200; the medial wall is 1000 (LH) and 2000 (RH). |
| `hemisphere` | `str` | `LH` or `RH`; `NaN` on the medial wall. |
| `network` | `str` | Yeo 7-network label: `Vis`, `SomMot`, `DorsAttn`, `SalVentAttn`, `Limbic`, `Cont`, `Default`, or `medial_wall`. |
| `label` | `str` | Full parcel label (e.g. `7Networks_LH_SalVentAttn_FrOperIns_1`, or `medial_wall`). |
| `network_int` | `float` | Integer code of `network` (alphabetical order, `convert_states_str2int`). |
| `salience_border` | `float` | Multiplicative border mask: `NaN` at the vertices on the `SalVentAttn` boundary (both sides), `1.0` elsewhere, so `values * salience_border` draws the border in the NaN colour. |

Analysis columns are added by the steps and loaders: `t1_gradient1_<network>` (MPC
gradient of a network, `NaN` outside it), `T1map` (`compute_t1map`), `BigBrain`,
`Bielschowsky`, `Parvalbumin` (step 1b), `surf_type` (`load_econo_atlas`), and `fc_g1` /
`fc_g1_network` (the oriented FC gradient: `load_fc_gradient_5k`, `load_fc_gradient_32k`).
The step 1a tables `source_data/step1a_mpc_gradient/surface_fslr32k_both.tsv` and
`surface_fslr5k_both.tsv` are these tables with the gradient, and every later step
reads them.

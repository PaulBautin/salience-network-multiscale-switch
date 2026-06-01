# API Reference

Reference for all public functions in the `src/` modules.

## Surface space

All analyses use **fsLR-32k** space: 64,984 vertices total (32,492 LH + 32,492 RH). Parcellation is **Schaefer-400** with Yeo 7-network labels. The primary network of interest is `SalVentAttn` (Salience/Ventral Attention).

## DataFrame conventions

### `df_yeo_surf`

One row per fsLR-32k surface vertex. Returned by `load_yeo_atlas()` and extended in-place by atlas loaders and gradient functions.

| Column | Type | Description |
|--------|------|-------------|
| `mics` | `float` | Schaefer-400 parcel ID. LH parcels: 1001–1400; RH parcels: 1801–2200. Medial wall and subcortex mapped to other ranges. |
| `network` | `str` | Yeo 7-network label: `Vis`, `SomMot`, `DorsAttn`, `SalVentAttn`, `Limbic`, `Cont`, `Default`. |
| `hemisphere` | `str` | `LH` or `RH`. |
| `label` | `str` | Full parcel label string (e.g. `LH_SalVentAttn_PFCl_1`). |
| `salience_border` | `float` | `1.0` at vertices on the `SalVentAttn` network boundary; `NaN` elsewhere. |

Analysis columns (e.g. `T1map`, `BigBrain`, `t1_gradient1`) are appended by individual loaders.

## Modules

| Module | Description |
|--------|-------------|
| [`atlas_load`](atlas_load.md) | Atlas and surface DataFrame construction; histological profile loaders |
| [`gradient_computation`](gradient_computation.md) | MPC gradient computation via diffusion map embedding |
| [`connectome_processing`](connectome_processing.md) | Connectome I/O and the gradient-weighted connectivity projection (Figure 2) |
| [`ieeg_processing`](ieeg_processing.md) | iEEG signal preprocessing, PSD estimation, and surface mapping |
| [`logging_utils`](logging_utils.md) | Timestamped file + console logging for manuscript scripts |
| [`plot_colors`](plot_colors.md) | Yeo 7-network and cortical-type color definitions; matplotlib colormap registration |

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

---

## `src/atlas_load`

### `load_yeo_atlas`

```python
load_yeo_atlas(micapipe: Path, surf_32k) -> pd.DataFrame
```

Build the base `df_yeo_surf` DataFrame for a project root.

Loads Schaefer-400 parcellation labels for both hemispheres, merges Yeo 7-network and hemisphere metadata from the lookup tables in `data/parcellations/lut/`, and computes the salience network border mask.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `micapipe` | `Path` | Project root containing `data/parcellations/`. |
| `surf_32k` | brainspace surface | Combined fsLR-32k surface object used to compute the border mask. |

**Returns** `pd.DataFrame` — base `df_yeo_surf` with columns `mics`, `network`, `hemisphere`, `label`, `network_int`, `salience_border`.

---

### `compute_network_mask`

```python
compute_network_mask(df: pd.DataFrame, network: str, hemisphere: str = 'both') -> np.ndarray
```

Return a boolean vertex mask selecting all vertices in a given network and hemisphere.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | Surface DataFrame with `network` and `hemisphere` columns. |
| `network` | `str` | Yeo 7-network label (e.g. `'SalVentAttn'`). |
| `hemisphere` | `str` | `'both'`, `'LH'`, or `'RH'`. Default `'both'`. |

**Returns** `np.ndarray` of `bool`, shape `(n_vertices,)`.

---

### `load_t1_salience_profiles`

```python
load_t1_salience_profiles(t1_files: list, mask: np.ndarray) -> np.ndarray
```

Load T1 intensity profiles for a pre-masked set of vertices across all subjects.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t1_files` | `list[Path]` | Paths to `.gii` profile files, one per subject. |
| `mask` | `np.ndarray` of `bool` | Boolean vertex mask from `compute_network_mask`. |

**Returns** `np.ndarray`, shape `(n_subjects, n_depths, n_network_vertices)`.

**Raises** `FileNotFoundError` if `t1_files` is empty. `ValueError` if `mask` is all-False.

---

### `compute_t1map`

```python
compute_t1map(t1_salience_profiles: np.ndarray) -> np.ndarray
```

Return the z-scored mean T1 profile collapsed over subjects and depths.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t1_salience_profiles` | `np.ndarray` | Shape `(n_subjects, n_depths, n_vertices)`. |

**Returns** `np.ndarray`, shape `(n_vertices,)`.

---

### `load_bigbrain`

```python
load_bigbrain(micapipe: Path, mask: np.ndarray) -> np.ndarray
```

Load BigBrain cell-staining profiles and return z-scored mean for masked vertices. Values are inverted so that high values correspond to greater cellular density.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `micapipe` | `Path` | Project root containing `data/parcellations/`. |
| `mask` | `np.ndarray` of `bool` | Boolean vertex mask. |

**Returns** `np.ndarray`, shape `(n_masked_vertices,)`.

---

### `load_ahead_biel`

```python
load_ahead_biel(micapipe: Path, mask: np.ndarray) -> np.ndarray
```

Load AHEAD Bielschowsky (nerve fiber) staining profiles and return z-scored mean for masked vertices.

**Parameters** — same as `load_bigbrain`.

**Returns** `np.ndarray`, shape `(n_masked_vertices,)`.

---

### `load_ahead_parva`

```python
load_ahead_parva(micapipe: Path, mask: np.ndarray) -> np.ndarray
```

Load AHEAD Parvalbumin (interneuron) staining profiles and return z-scored mean for masked vertices.

**Parameters** — same as `load_bigbrain`.

**Returns** `np.ndarray`, shape `(n_masked_vertices,)`.

---

### `load_econo_atlas`

```python
load_econo_atlas(micapipe: Path, df_yeo_surf: pd.DataFrame) -> pd.DataFrame
```

Append Von Economo cortical type labels (`surf_type` column) to `df_yeo_surf`.

Cortical types are mapped from Von Economo parcels to a 6-level ordinal scale (1 = koniocortex → 6 = agranular).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `micapipe` | `Path` | Project root containing `data/parcellations/`. |
| `df_yeo_surf` | `pd.DataFrame` | Existing surface DataFrame to extend. |

**Returns** `pd.DataFrame` — input DataFrame with `surf_type` column added in-place.

---

## `src/gradient_computation`

### `compute_t1_gradient`

```python
compute_t1_gradient(
    t1_salience_profiles: list | np.ndarray,
    n_components: int = 10,
    sparsity: float = 0.9,
) -> np.ndarray
```

Compute MPC (microstructure profile covariance) gradients from T1 intensity profiles and return the z-scored first component.

For each subject, the function computes a partial correlation matrix between vertex profiles while controlling for the mean profile (Fisher z-transformed). It then fits a diffusion map with a normalized angle kernel across subjects, aligns components via Procrustes rotation, and returns the mean first gradient.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `t1_salience_profiles` | `np.ndarray` | Shape `(n_subjects, n_depths, n_vertices)`. |
| `n_components` | `int` | Number of gradient components to extract. Default `10`. |
| `sparsity` | `float` | Sparsity threshold for the affinity matrix. Default `0.9`. |

**Returns** `np.ndarray`, shape `(n_vertices,)` — z-scored first gradient component.

---

### `partial_corr_with_covariate`

```python
partial_corr_with_covariate(X: np.ndarray, covar: np.ndarray) -> np.ndarray
```

Compute the Fisher z-transformed partial correlation matrix between vertices, controlling for a single covariate.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `X` | `np.ndarray` | Shape `(n_features, n_vertices)` — intensity profiles across depths. |
| `covar` | `np.ndarray` | Shape `(n_features,)` — covariate to partial out (e.g. mean profile). |

**Returns** `np.ndarray`, shape `(n_vertices, n_vertices)` — Fisher z-transformed MPC matrix.

---

## `src/ieeg_processing`

### `load_original_data_files`

```python
load_original_data_files(root: str = '/host/verges/tank/data/BIDS_iEEG/original') -> pd.DataFrame
```

Load MICA iEEG MATLAB files and return channel-level data.

Scans `root` for `*stage-W.mat` files matching `sub-PX*/ses-01/`. Each row in the returned DataFrame corresponds to one channel from one subject/session pair.

**Returns** `pd.DataFrame` with columns: `Subject`, `Session`, `ChannelName`, `SamplingRate`, `Data`, `ContactName1`, `ContactName2`.

---

### `load_channel_info`

```python
load_channel_info(root_dir: str = '/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA') -> pd.DataFrame
```

Load channel metadata and surface vertex indices from BIDS-iEEG ChannelMap TSV files.

Vertex indices are offset so that LH vertices are `0–32491` and RH vertices are `32492+`, matching the combined fsLR-32k surface ordering.

**Returns** `pd.DataFrame` with columns: `Subject`, `Session`, `ChannelName`, `ChannelNumber`, `ChannelIndices_lh`, `ChannelIndices_rh`.

---

### `load_sensitivity_info`

```python
load_sensitivity_info(
    root_dir: str = '/host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA',
    *,
    threshold: float = 0.001,
) -> pd.DataFrame
```

Load and aggregate surface-based contact sensitivity maps from leadfield `.mat` files.

Sensitivity maps are rectified, thresholded, and summed across hemispheres per contact.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `root_dir` | `str` | Root of the electroMICA derivatives. |
| `threshold` | `float` | Minimum absolute sensitivity value retained. Default `0.001`. |

**Returns** `pd.DataFrame` with columns: `Subject`, `Session`, `ContactName`, `ContactSensitivityMap`.

---

### `preprocess_and_compute_psd_ieeg`

```python
preprocess_and_compute_psd_ieeg(
    data: np.ndarray,
    fs: float,
    fmin: float = 0.5,
    fmax: float = 80.0,
    fs_target: float = 200.0,
    filter_order: int = 4,
    window_sec: float = 2.0,
    overlap_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]
```

Full iEEG preprocessing and PSD computation pipeline.

Steps: 4th-order zero-phase Butterworth bandpass filter → downsample → demean → Welch PSD (Hamming window) → frequency-range restriction → power normalization (sums to 1).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `np.ndarray` | Shape `(..., n_samples)`. Last axis is time. |
| `fs` | `float` | Original sampling frequency in Hz. |
| `fmin` / `fmax` | `float` | Bandpass and PSD frequency range in Hz. |
| `fs_target` | `float` | Target sampling rate after downsampling. Default `200.0`. |
| `filter_order` | `int` | Butterworth filter order. Default `4`. |
| `window_sec` | `float` | Welch window length in seconds. Default `2.0`. |
| `overlap_sec` | `float` | Welch overlap in seconds. Default `1.0`. |

**Returns** `(freq, pxx)` — frequency array and normalized PSD of shape `(..., n_frequencies)`.

---

### `extract_band_power`

```python
extract_band_power(
    pxx_raw: np.ndarray,
    freq: np.ndarray,
    band: tuple[float, float],
    relative: bool = True,
) -> np.ndarray
```

Integrate PSD within a frequency band using Simpson's rule and return log₁₀ power.

Canonical bands: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30–80 Hz).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pxx_raw` | `np.ndarray` | PSD array, last axis is frequency. |
| `freq` | `np.ndarray` | Frequency axis in Hz. |
| `band` | `tuple[float, float]` | `(fmin, fmax)` of the band. |
| `relative` | `bool` | Divide by total power before log transform. Default `True`. |

**Returns** `np.ndarray` — log₁₀ band power, shape matches `pxx_raw` without the last axis.

---

## `src/logging_utils`

### `setup_manuscript_logger`

```python
setup_manuscript_logger(
    script_name: str,
    project_root: Path,
    args: argparse.Namespace | None = None,
) -> logging.Logger
```

Configure logging to write to both console and `logs/<script_name>.log`.

Attaches a file handler to the root logger so that logging calls from any module are also captured. Each invocation appends a timestamped run header to the log file; the file is never overwritten.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `script_name` | `str` | Base name for the log file (no extension). |
| `project_root` | `Path` | Project root; `logs/` subdirectory is created if absent. |
| `args` | `argparse.Namespace` | Parsed CLI arguments written to the run header. Optional. |

**Returns** `logging.Logger` — named logger for the calling script.

**Example**

```python
logger = setup_manuscript_logger('figure_1a_t1map', project_root, args)
logger.info('Starting analysis')
```

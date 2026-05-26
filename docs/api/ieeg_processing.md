# `src/ieeg_processing`

### `load_original_data_files`

```python
load_original_data_files(root: str = '/host/verges/tank/data/BIDS_iEEG/original') -> pd.DataFrame
```

Load MICA iEEG MATLAB files and return bipolar channel-level data.

Scans `root` for `*stage-W.mat` files matching `sub-PX*/ses-01/`. Each row corresponds to one **bipolar channel** from one subject/session pair. In electroMICA terminology, a *channel* is a differential recording between two physical *contacts*: `ChannelName` stores the pair (e.g. `"LCi1-LCi2"`), while `ContactName1` and `ContactName2` hold the individual contact identifiers used to index sensitivity/leadfield files.

**Returns** `pd.DataFrame` with columns: `Subject`, `Session`, `ChannelName` (bipolar pair), `SamplingRate`, `Data`, `ContactName1`, `ContactName2`.

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

### `compute_psd_vectorized`

```python
compute_psd_vectorized(
    data: np.ndarray,
    fs: float,
    fmin: float = 0.5,
    fmax: float = 80.0,
) -> tuple[np.ndarray, np.ndarray]
```

Compute relative PSD for all channels simultaneously, without preprocessing.

Unlike `preprocess_and_compute_psd_ieeg`, this function skips filtering, downsampling, and demeaning — use it when the data are already preprocessed (e.g. MNI iEEG atlas data).

Uses Welch's method with a 2-second Hamming window and 1-second overlap. PSD is normalized by total power.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `np.ndarray` | Shape `(n_channels, n_times)`. |
| `fs` | `float` | Sampling frequency in Hz. |
| `fmin` / `fmax` | `float` | Frequency range to retain in Hz. Default `0.5` / `80.0`. |

**Returns** `(f_band, pxx_rel)` — frequency array shape `(n_frequencies,)` and normalized PSD shape `(n_channels, n_frequencies)`.

---

### `compute_gradient_quantiles`

```python
compute_gradient_quantiles(
    df_surf: pd.DataFrame,
    channel_indices: np.ndarray,
    gradient_col: str,
    quantiles: tuple[float, float] = (0.25, 0.75),
) -> np.ndarray
```

Assign gradient quantile labels to channels and update the surface DataFrame in-place.

Marks vertices covered by `channel_indices` as bottom-quantile (`-1`) or top-quantile (`+1`) based on their gradient value, writing results into a `'quantiles'` column on `df_surf`. Used by both MNI and MICA iEEG scripts to stratify channels by their position along the T1 microstructural gradient.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df_surf` | `pd.DataFrame` | Surface DataFrame containing `gradient_col`. Modified in-place. |
| `channel_indices` | `np.ndarray` | Integer vertex indices of each channel on the 32k surface. |
| `gradient_col` | `str` | Name of the gradient column in `df_surf`. |
| `quantiles` | `tuple[float, float]` | `(low, high)` quantile thresholds as fractions. Default `(0.25, 0.75)`. |

**Returns** `np.ndarray` — quantile label per channel (`-1`, `0`, or `1`; `NaN` where unassigned).

---

### `plot_surface_sphere`

```python
plot_surface_sphere(
    p,
    channel_position: list | np.ndarray,
    channel_color: np.ndarray,
    screenshot_path,
) -> None
```

Render iEEG electrode contacts as spheres on a VTK brain surface and save a screenshot.

Adds spheres at each channel position to both left and right hemisphere renderers (standard LH view and 180° rotated RH view). Each sphere has radius 1.5, is oriented with standard -90°X / +90°Z rotations, and is colored by `channel_color`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `p` | PyVista/VTK plotter | Plotter with two renderers (LH at index `[0][0]`, RH at index `[1][0]`). |
| `channel_position` | `list` or `np.ndarray` | 3D MNI coordinates for each contact, shape `(n_channels, 3)`. |
| `channel_color` | `np.ndarray` | RGBA colors, shape `(n_channels, 4)`. |
| `screenshot_path` | `str` or `Path` | Output path for the PNG screenshot. |

**Returns** `None` — saves screenshot to `screenshot_path` with transparent background.

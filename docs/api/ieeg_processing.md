# `src/ieeg_processing`

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

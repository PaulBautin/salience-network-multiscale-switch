# `src/ieeg_processing`

### `compute_vertex_areas`

```python
compute_vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray
```

Barycentric (hat-function) per-vertex surface areas of a triangular mesh — each vertex receives one third of the area of every triangle it belongs to. Matches electroMICA's `areasv`, used to normalise channel sensitivities to a per-area density before thresholding. MATLAB-style 1-based `faces` are detected and converted automatically.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `vertices` | `np.ndarray` | Vertex coordinates, shape `(n_vertices, 3)`, in millimetres. |
| `faces` | `np.ndarray` | Triangle vertex indices, shape `(n_faces, 3)`. |

**Returns** `np.ndarray` of per-vertex area, shape `(n_vertices,)`.

---

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
) -> tuple[pd.DataFrame, dict]
```

Load per-hemisphere **signed** contact sensitivity maps from leadfield `.mat` files, plus the surface vertex areas used to threshold them. The signed leadfield is preserved (no rectification) and the two hemispheres are kept separate, because a bipolar channel's sensitivity is the *difference* of its contacts' signed maps — deferred to [`build_bipolar_sensitivity`](#build_bipolar_sensitivity).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `root_dir` | `str` | Root of the electroMICA derivatives. |

**Returns** `tuple[pd.DataFrame, dict]` — (1) a DataFrame with columns `Subject`, `Session`, `ContactName`, `Sens_L`, `Sens_R` (signed `(32492,)` maps per hemisphere, zeros where absent); (2) `areas`, a mapping `(Subject, Session) -> {"L": areas_L, "R": areas_R}` of per-vertex surface areas.

---

### `build_bipolar_sensitivity`

```python
build_bipolar_sensitivity(
    df_channels: pd.DataFrame,
    df_sensitivity: pd.DataFrame,
    areas: dict,
    *,
    global_thresh: float = 0.001,
    rel_thresh: float = 0.05,
) -> np.ndarray
```

Assemble bipolar-channel surface sensitivities following electroMICA `ComputeFeatureMaps`: pair each channel with its two contacts' signed leadfields, per hemisphere take the **signed difference**, threshold the per-area density (absolute floor `global_thresh` and channel-relative floor `rel_thresh × second-largest density`), rectify, then fold the two hemispheres onto one fsLR-32k template by summing magnitudes (`|L1_LH − L2_LH| + |L1_RH − L2_RH|`). Channels are processed per `(Subject, Session)` so each mesh's area vector is applied once.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df_channels` | `pd.DataFrame` | One row per bipolar channel with `Subject`, `Session`, `ContactName1`, `ContactName2`. |
| `df_sensitivity` | `pd.DataFrame` | Per-contact signed maps (`Subject`, `Session`, `ContactName`, `Sens_L`, `Sens_R`) from `load_sensitivity_info`; contacts absent here contribute zero. |
| `areas` | `dict` | `(Subject, Session) -> {"L", "R"}` per-vertex surface areas, from `load_sensitivity_info`. |
| `global_thresh` | `float` | Absolute density noise floor (Vm/A). Default `0.001`. |
| `rel_thresh` | `float` | Channel-relative density floor fraction. Default `0.05`. |

**Returns** `np.ndarray` of shape `(n_channels, 32492)`, non-negative, row-aligned with `df_channels`.

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

### `compute_spectral_parameters`

```python
compute_spectral_parameters(
    pxx: np.ndarray,
    freq: np.ndarray,
    bands: dict[str, tuple[float, float]] | None = None,
    fmin: float = 1.0,
    fmax: float = 80.0,
    aperiodic_mode: str = "knee",
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.05,
) -> dict
```

Parameterise power spectra with a single `specparam` fit (FOOOF; Donoghue et al.,
2020) into an aperiodic exponent and per-band oscillatory peak power. Deriving band
power from the periodic component (peak power above the aperiodic fit) makes it
orthogonal to the exponent, so the two measures do not re-encode the same 1/f change.
Fitting in `'knee'` mode suits the broadband iEEG range, and the per-channel unit-sum
PSD normalisation only shifts the aperiodic offset, leaving both measures unchanged.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pxx` | `np.ndarray` | PSD, shape `(n_spectra, n_frequencies)` or `(n_frequencies,)`; linear power. |
| `freq` | `np.ndarray` | Frequency axis in Hz. |
| `bands` | `dict[str, tuple[float, float]] \| None` | Band name → `(fmin, fmax)` for the oscillatory readout; `None` returns only the exponent. |
| `fmin` | `float` | Lower bound of the fitting range in Hz. Default `1.0`. |
| `fmax` | `float` | Upper bound of the fitting range in Hz. Default `80.0`. |
| `aperiodic_mode` | `str` | `specparam` aperiodic mode, `'knee'` or `'fixed'`. Default `'knee'`. |
| `peak_width_limits` | `tuple[float, float]` | (min, max) Gaussian peak bandwidth in Hz. |
| `max_n_peaks` | `int` | Maximum oscillatory peaks per spectrum. Default `6`. |
| `min_peak_height` | `float` | Minimum peak height above the aperiodic fit. Default `0.05`. |

**Returns** `dict` — `'exponent'`: aperiodic exponent χ, shape `pxx.shape[:-1]` (NaN where the fit failed); `'band_power'`: `{name: array}` of oscillatory peak power per band, same shape (NaN where no peak was detected or the fit failed).

**Raises** `ImportError` — if `specparam` is not installed (a hard dependency of the iEEG spectral analysis).

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

# `src/ieeg_processing`

### `MICA_PSD_PARAMS`

`dict` — preprocessing / Welch parameters of the MICA iEEG analyses (Figures 3b and 3c), passed to [`preprocess_and_compute_psd_ieeg`](#preprocess_and_compute_psd_ieeg) and part of the channel-PSD cache key: `fmin=0.5`, `fmax=80.0`, `fs_target=200.0`, `filter_order=4`, `window_sec=2.0`, `overlap_sec=1.0`.

---

### `FREQ_BANDS`

`dict` — canonical frequency bands (Hz), spanning the 0.5–80 Hz preprocessing band: delta `(0.5, 4)`, theta `(4, 8)`, alpha `(8, 13)`, beta `(13, 30)`, gamma `(30, 80)`.

---

### `SPEC_FIT_RANGE`

`tuple` — `specparam` fit range (Hz) of the MICA iEEG analyses (Figures 3b and 3c): `(1.0, 45.0)`. It stops below the 60 Hz line-noise notch of the recordings and the transition band of the 80 Hz low-pass of [`MICA_PSD_PARAMS`](#mica_psd_params), as `specparam` advises against fitting filtered-out frequencies.

---

### `SPEC_BANDS`

`dict` — oscillatory peak-power readout bands (Hz) of Figures 3b and 3c: [`FREQ_BANDS`](#freq_bands) clipped to [`SPEC_FIT_RANGE`](#spec_fit_range), where peaks can be fitted: delta `(1.0, 4)`, theta `(4, 8)`, alpha `(8, 13)`, beta `(13, 30)`, gamma `(30, 45.0)`.

---

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

### `list_original_data_files`

```python
list_original_data_files(
    root: str = '/host/verges/tank/data/BIDS_iEEG/original',
    stage: str = 'W',
) -> list[str]
```

List the recordings of one vigilance state of the original MICA iEEG dataset, sorted so that channel order (and any cache keyed on the list) is deterministic.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `root` | `str` | Root of the iEEG recordings; files are matched as `<root>/sub-PX*/ses-01/*stage-<stage>.mat`. |
| `stage` | `str` | Vigilance state: `'W'` (wakefulness), `'N2'` or `'N3'`. Default `'W'`. |

**Returns** `list[str]` of sorted file paths.

---

### `compute_original_psd`

```python
compute_original_psd(
    root: str = '/host/verges/tank/data/BIDS_iEEG/original',
    stage: str = 'W',
    duration_s: float | None = None,
    **psd_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]
```

Welch PSD of every **bipolar channel** in the original recordings. In electroMICA terminology, a *channel* is a differential recording between two physical *contacts*: `ChannelName` stores the pair (e.g. `"LCi1-LCi2"`), while `ContactName1` and `ContactName2` hold the contact identifiers used to index the leadfield files. All recordings are truncated to the shortest recording's length, read from the MATLAB file headers (or to their first `duration_s` seconds, e.g. to match vigilance states), then streamed one subject at a time through [`preprocess_and_compute_psd_ieeg`](#preprocess_and_compute_psd_ieeg), so only one subject's raw signal is held in memory.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `root` | `str` | Root of the iEEG recordings (see [`list_original_data_files`](#list_original_data_files)). |
| `stage` | `str` | Vigilance state: `'W'`, `'N2'` or `'N3'`. Default `'W'`. |
| `duration_s` | `float \| None` | Analysed length in seconds from the start of each recording; `None` uses the shortest recording. |
| `**psd_kwargs` | | Forwarded to `preprocess_and_compute_psd_ieeg` (`fmin`, `fmax`, `fs_target`, `filter_order`, `window_sec`, `overlap_sec`). |

**Returns** `(channels, freq, pxx)` — a DataFrame with columns `Subject`, `Session`, `ChannelName` (bipolar pair), `SamplingRate`, `ContactName1`, `ContactName2`; the frequency axis in Hz; and the normalized PSD of shape `(n_channels, n_frequencies)`, row-aligned with `channels`.

**Raises** `ValueError` if no recording is found, a file's `Data` shape does not match its channel count, the recordings use more than one sampling rate (truncation is by sample count), or `duration_s` exceeds the shortest recording.

---

### `load_or_compute_channel_psd`

```python
load_or_compute_channel_psd(
    raw_root: str,
    cache_path,
    stage: str = 'W',
    duration_s: float | None = None,
    **psd_kwargs,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]
```

[`compute_original_psd`](#compute_original_psd), cached in a pickle (a few MB) whose key is every recording's relative path, size and modification time plus `stage`, `duration_s` and `psd_kwargs`; any change to the recordings or parameters invalidates it.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `raw_root` | `str` | Root of the original recordings. |
| `cache_path` | path | Pickle file of the cache. |
| `stage` | `str` | Vigilance state: `'W'`, `'N2'` or `'N3'`. Default `'W'`. |
| `duration_s` | `float \| None` | Analysed length in seconds (see `compute_original_psd`). |
| `**psd_kwargs` | | Forwarded to `preprocess_and_compute_psd_ieeg` (e.g. `MICA_PSD_PARAMS`). |

**Returns** `(channels, freq, pxx)`, as `compute_original_psd`.

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

Load per-hemisphere **signed** contact sensitivity maps from leadfield `.mat` files, plus the surface vertex areas used to threshold them. The signed leadfield is preserved (no rectification) and the two hemispheres are kept separate, because a bipolar channel's sensitivity is the *difference* of its contacts' signed maps — deferred to [`build_bipolar_sensitivity`](#build_bipolar_sensitivity). Files that cannot be parsed, read, or lack `ContactName`/`ContactSensitivityMap` are skipped with a logged warning.

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

Assemble bipolar-channel surface sensitivities following electroMICA `ComputeFeatureMaps`: pair each channel with its two contacts' signed leadfields, per hemisphere take the **signed difference**, threshold the per-area density (absolute floor `global_thresh` and channel-relative floor `rel_thresh × second-largest density`), rectify, then fold the two hemispheres onto one fsLR-32k template by summing magnitudes (`|L1_LH − L2_LH| + |L1_RH − L2_RH|`). NaN leadfield values are treated as zero before thresholding. Channels are processed per `(Subject, Session)` so each mesh's area vector is applied once; contacts with no leadfield are treated as zero and reported in a logged warning per `(Subject, Session)`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df_channels` | `pd.DataFrame` | One row per bipolar channel with `Subject`, `Session`, `ContactName1`, `ContactName2`. |
| `df_sensitivity` | `pd.DataFrame` | Per-contact signed maps (`Subject`, `Session`, `ContactName`, `Sens_L`, `Sens_R`) from `load_sensitivity_info`; contacts absent here contribute zero (logged). |
| `areas` | `dict` | `(Subject, Session) -> {"L", "R"}` per-vertex surface areas, from `load_sensitivity_info`. |
| `global_thresh` | `float` | Absolute density noise floor (Vm/A). Default `0.001`. |
| `rel_thresh` | `float` | Channel-relative density floor fraction. Default `0.05`. |

**Returns** `np.ndarray` of shape `(n_channels, 32492)`, non-negative, row-aligned with `df_channels`.

---

### `project_channel_measure`

```python
project_channel_measure(channel_vals: np.ndarray, sens: np.ndarray) -> np.ndarray
```

Sensitivity-weighted projection of per-channel values onto the surface (the electroMICA smoothed feature map): each vertex receives the sensitivity-weighted average of the channel values; vertices without sensitivity are NaN.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `channel_vals` | `np.ndarray` | Shape `(n_channels,)` for a scalar measure or `(k, n_channels)` for k values per channel (e.g. a spectrum). |
| `sens` | `np.ndarray` | Bipolar sensitivity maps, shape `(n_channels, n_vertices)` (NaN zeroed). |

**Returns** `np.ndarray` of shape `(n_vertices,)` or `(k, n_vertices)`.

---

### `mirror_to_cortex`

```python
mirror_to_cortex(template_vals: np.ndarray, df_surf: pd.DataFrame, network: str | None = None) -> np.ndarray
```

Show a folded template-hemisphere map on both fsLR-32k hemispheres: because [`build_bipolar_sensitivity`](#build_bipolar_sensitivity) folds both hemispheres onto one template, template vertex *i* stands for left vertex *i* and right vertex `N_LH + i`, and the value is placed on both, kept on labelled cortex (or on `network` vertices) and NaN elsewhere.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `template_vals` | `np.ndarray` | Per-template-vertex values, shape `(32492,)`. |
| `df_surf` | `pd.DataFrame` | Surface table over both hemispheres with `hemisphere` and `network` columns. |
| `network` | `str \| None` | Yeo network to keep; `None` keeps all labelled cortex. |

**Returns** `np.ndarray` of shape `(64984,)`, LH then RH.

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
    fmin: float = SPEC_FIT_RANGE[0],
    fmax: float = SPEC_FIT_RANGE[1],
    aperiodic_mode: str = "knee",
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.05,
    peak_threshold: float = 2.0,
    nonnegative_knee: bool = True,
    return_model: bool = False,
) -> dict
```

Parameterise power spectra with a single `specparam` fit (FOOOF; Donoghue et al.,
2020) into an aperiodic exponent and per-band oscillatory peak power. Deriving band
power from the periodic component (peak power above the aperiodic fit) makes it
orthogonal to the exponent, so the two measures do not re-encode the same 1/f change.
The settings follow the `specparam` 2.0 guidance: one model and one set of settings for
every spectrum, a fit range ([`SPEC_FIT_RANGE`](#spec_fit_range)) that excludes
filtered-out frequencies, and, in `'knee'` mode, the knee bounded to *k* ≥ 0 (the
`ap_bounds` algorithm setting), so a spectrum without a knee in the fit range reduces
to the fixed power law. The per-channel unit-sum PSD normalisation only shifts the
aperiodic offset, leaving both measures unchanged.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pxx` | `np.ndarray` | PSD, shape `(n_spectra, n_frequencies)` or `(n_frequencies,)`, linear power. |
| `freq` | `np.ndarray` | Frequency axis in Hz. |
| `bands` | `dict \| None` | `{name: (fmin, fmax)}` for the peak-power readout (e.g. [`SPEC_BANDS`](#spec_bands)); every band must lie within the fit range. `None` returns the aperiodic parameters only. |
| `fmin` | `float` | Lower bound of the fit range (Hz). Default `1.0`. |
| `fmax` | `float` | Upper bound of the fit range (Hz). Default `45.0`. |
| `aperiodic_mode` | `str` | `specparam` aperiodic mode, `'knee'` or `'fixed'`. Default `'knee'`. |
| `peak_width_limits` | `tuple[float, float]` | (min, max) Gaussian peak bandwidth in Hz; `specparam` recommends a lower limit of about twice the frequency resolution. |
| `max_n_peaks` | `int` | Maximum number of peaks per spectrum. |
| `min_peak_height` | `float` | Minimum peak height above the aperiodic fit (log10 power). |
| `peak_threshold` | `float` | Relative peak-detection threshold, in standard deviations of the flattened spectrum. Default `2.0`. |
| `nonnegative_knee` | `bool` | In `'knee'` mode, bound the knee to *k* ≥ 0. Default `True`. |
| `return_model` | `bool` | Also return the fitted `SpectralGroupModel` (for [`plot_fit_quality`](#plot_fit_quality)). Default `False`. |

**Returns** `dict` — `'exponent'`: aperiodic exponent χ, shape `pxx.shape[:-1]` (NaN where the fit failed); `'offset'`: aperiodic offset *b* (log10 power), same shape; `'knee'`: aperiodic knee *k*, same shape (0 in `'fixed'` mode, whose model is the knee model with *k* = 0); `'knee_freq'`: knee frequency *k*<sup>1/χ</sup> in Hz (`specparam`'s `compute_knee_frequency`; Gao et al., 2020), a value below `fmin` placing the knee below the fit range (NaN in `'fixed'` mode); `'r_squared'`, `'mae'`: goodness of fit and mean absolute error of each fit; `'n_peaks'`: peaks fitted per spectrum; `'band_power'`: `{name: array}` of oscillatory peak power per band, same shape (NaN where no peak was detected or the fit failed); `'report'`: `specparam`'s methods report (code version, algorithm, settings, fit range) with the fit modes and bounds, as text; `'model'`: the fitted `SpectralGroupModel`, if `return_model`. `'offset'`, `'knee'` and `'exponent'` give the fitted aperiodic component through [`aperiodic_component`](#aperiodic_component).

**Raises** `ImportError` — if `specparam` is not installed (a hard dependency of the iEEG spectral analysis). `ValueError` — if a band extends outside the fit range.

---

### `aperiodic_component`

```python
aperiodic_component(
    freq: np.ndarray,
    offset: np.ndarray,
    knee: np.ndarray,
    exponent: np.ndarray,
) -> np.ndarray
```

Evaluate the fitted `specparam` aperiodic component $L(f) = b - \log_{10}(k + f^{\chi})$
in log10 power from the parameters returned by
[`compute_spectral_parameters`](#compute_spectral_parameters). Subtracting it from the
log10 power spectrum leaves the periodic residual (the flattened spectrum).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `freq` | `np.ndarray` | Frequencies in Hz, shape `(n_frequencies,)`. |
| `offset` | `np.ndarray` | Aperiodic offset *b*, any shape `S`. |
| `knee` | `np.ndarray` | Aperiodic knee *k* (0 for `'fixed'` mode), shape `S`. |
| `exponent` | `np.ndarray` | Aperiodic exponent χ, shape `S`. |

**Returns** `np.ndarray` — aperiodic component, shape `S + (n_frequencies,)`.

---

### `fit_quality_summary`

```python
fit_quality_summary(spec: dict, fmin: float = SPEC_FIT_RANGE[0], mask: np.ndarray | None = None) -> str
```

One-line description of the fits returned by [`compute_spectral_parameters`](#compute_spectral_parameters), over the converged fits: median and interquartile range of *R*², number of fits with *R*² < 0.9 (descriptive, no exclusion), median mean absolute error, mean number of peaks per fit, and the median knee frequency with the number of knees below the fit range.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `spec` | `dict` | Output of [`compute_spectral_parameters`](#compute_spectral_parameters). |
| `fmin` | `float` | Lower bound of the fit range (Hz). |
| `mask` | `np.ndarray \| None` | Optional boolean selection of the spectra. |

**Returns** `str` — the summary.

---

### `plot_fit_quality`

```python
plot_fit_quality(model, path, n: int = 4, labels: list[str] | None = None, title: str | None = None) -> None
```

Fit check of a `specparam` group fit, following the `specparam` guidance on checking model fits: the *R*² distribution and, below it, the `n` fits with the lowest *R*² (possible under-fitting or artefactual spectra), the `n` around the median and the `n` with the highest *R*² (possible over-fitting), each drawn with `specparam`'s model plot (spectrum, full model, aperiodic fit; log-frequency axis). Failed fits are left out.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `model` | `SpectralGroupModel` | Fitted group (`compute_spectral_parameters(..., return_model=True)['model']`). |
| `path` | `str \| Path` | Output file (format from the extension). |
| `n` | `int` | Fits per row. Default `4`. |
| `labels` | `list[str] \| None` | Optional name of every spectrum, shown in the panel titles. |
| `title` | `str \| None` | Optional figure title. |

**Returns** `None`.

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

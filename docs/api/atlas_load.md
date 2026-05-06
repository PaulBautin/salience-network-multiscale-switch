# `src/atlas_load`

### `convert_states_str2int`

```python
convert_states_str2int(states_str: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]
```

Convert a list of string brain-state labels to integer codes.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `states_str` | `list` or `np.ndarray` of `str`, shape `(N,)` | State label per vertex (e.g. `['Vis', 'Vis', 'SomMot', ...]`). |

**Returns** `(states, state_labels)` — integer array of shape `(N,)` and corresponding label array of shape `(n_states,)`.

---

### `normalize_to_range`

```python
normalize_to_range(data: np.ndarray | list, target_min: float, target_max: float) -> np.ndarray
```

Min-max normalize data to a specified target range, ignoring NaN.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `data` | `np.ndarray` or `list` | Input data. |
| `target_min` | `float` | Desired minimum of the output range. |
| `target_max` | `float` | Desired maximum of the output range. |

**Returns** `np.ndarray` — normalized values in `[target_min, target_max]`. Returns the midpoint if all values are identical.

---

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

### `load_yeo_surf_5k`

```python
load_yeo_surf_5k(micapipe: Path) -> pd.DataFrame
```

Build the base surface DataFrame for the fsLR-5k downsampled surface (9,684 total vertices: 4,842 LH + 4,842 RH).

Loads Schaefer-400 parcellation labels at fsLR-5k resolution and merges Yeo 7-network metadata. Unlike `load_yeo_atlas`, this function does not compute the salience border mask and does not require a surface object.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `micapipe` | `Path` | Project root containing `data/parcellations/`. |

**Returns** `pd.DataFrame` — surface DataFrame with columns `mics`, `network`, `hemisphere`, `label`.

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

### `load_baillarger_atlas`

```python
load_baillarger_atlas(df_yeo_surf: pd.DataFrame, path_atlas: Path) -> np.ndarray
```

Load Baillarger band type labels and return a border-masked surface array.

Labels are read from a GIFTI parcellation projected from colin27 to fsLR-32k. Values 0 and 1 are collapsed to 1 (unlabeled/background). The result is masked to the salience network border vertices via `df_yeo_surf['salience_border']`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `df_yeo_surf` | `pd.DataFrame` | Surface DataFrame with `salience_border` column. |
| `path_atlas` | `Path` | Directory containing the Baillarger GIFTI files. |

**Returns** `np.ndarray`, shape `(n_vertices,)` — Baillarger type values at border vertices, `NaN` elsewhere.

---

### `load_intrusion_atlas`

```python
load_intrusion_atlas(df_yeo_surf: pd.DataFrame, path_atlas: Path) -> np.ndarray
```

Load Intrusion type labels and return a border-masked surface array.

Equivalent to `load_baillarger_atlas` but for the Intrusion parcellation.

**Parameters** — same as `load_baillarger_atlas`.

**Returns** `np.ndarray`, shape `(n_vertices,)` — Intrusion type values at border vertices, `NaN` elsewhere.

---

### `load_bigbrain_gradients`

```python
load_bigbrain_gradients() -> np.ndarray
```

Load the BigBrain histological gradient (G2) from bundled GIFTIs and return a combined bilateral surface array.

No arguments — paths are resolved relative to the package root (`data/parcellations/`).

**Returns** `np.ndarray`, shape `(64,984,)` — concatenated LH + RH BigBrain G2 gradient values.

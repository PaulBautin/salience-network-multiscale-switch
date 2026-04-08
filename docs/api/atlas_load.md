# `src/atlas_load`

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

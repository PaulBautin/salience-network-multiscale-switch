# `src/connectome_processing`

Connectome I/O and the gradient-weighted connectivity projection used in [Figure 2](../methods/figure_2.md). All matrices are handled at fsLR-5k resolution (9,684 vertices); micapipe stores fsLR-5k connectomes as upper-triangular GIFTIs, which are symmetrised on load. See [Figure 2 Methods](../methods/figure_2.md) for the statistical derivation of the projection score and the spin/Moran nulls.

### `load_subject_matrix`

```python
load_subject_matrix(path, cortex_mask: np.ndarray) -> np.ndarray
```

Load an fsLR-5k vertex × vertex `.shape.gii` connectome restricted to cortex.

The upper-triangular storage is symmetrised via `triu(d, 1) + d.T`, recovering the diagonal from the transpose. Negative entries (numerical noise in SC) are clipped to zero.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `path` | `str` or `Path` | Path to the `.shape.gii` connectome. |
| `cortex_mask` | `np.ndarray` of `bool`, shape `(n_vertices,)` | Selects cortical vertices (excludes medial wall). |

**Returns** `np.ndarray`, shape `(n_cortex, n_cortex)`, `float32` — symmetric cortex-restricted matrix.

---

### `fcn_group_bins`

```python
fcn_group_bins(adj: np.ndarray, dist: np.ndarray, hemiid: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray]
```

Distance-dependent group-representative SC thresholding (Betzel et al. 2018).

Generates a binary group-consensus mask that preserves the within- and between-hemisphere connection-length distributions. Because tractography over-represents short streamlines, binning by distance prevents the consensus from collapsing toward short-range edges.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `adj` | `np.ndarray`, shape `(n, n, n_sub)` | Per-subject SC matrices. |
| `dist` | `np.ndarray`, shape `(n, n)` | Mean tract-distance (streamline length) matrix. |
| `hemiid` | `np.ndarray`, shape `(n,)` of `bool` | `True` for RH vertices. |
| `nbins` | `int` | Number of distance bins. |

**Returns** `(G, Gc)` — distance-dependent and consistency-based symmetric binary group-consensus masks, each shape `(n, n)`.

---

### `build_consensus_mask`

```python
build_consensus_mask(
    sc_files: list, dist_files: list, df_yeo_surf_5k: pd.DataFrame, nbins: int = 10,
) -> tuple[np.ndarray, list[np.ndarray]]
```

Build the Betzel distance-dependent consensus mask once across subjects and return the per-subject SC matrices.

The mask is intended to be multiplied elementwise with per-subject SIFT2 weights downstream (per-subject random-effects inference; the mask removes spurious, non-reproducible edges).

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `sc_files` | `list[str \| Path]` | Per-subject SC `.shape.gii` paths. |
| `dist_files` | `list[str \| Path]` | Per-subject tract-distance `.shape.gii` paths (used only to build the mask). |
| `df_yeo_surf_5k` | `pd.DataFrame` | Surface DataFrame; `hemisphere` column derives the cortex mask and `hemiid`. |
| `nbins` | `int` | Distance bins passed to `fcn_group_bins`. Default `10`. |

**Returns** `(G, sc_subjects)` — group-consensus mask of shape `(n_cortex, n_cortex)` (`bool`) and a list of per-subject cortex-restricted SC matrices.

**Raises** `FileNotFoundError` if either file list is empty.

---

### `prepare_weights`

```python
prepare_weights(
    W_raw: np.ndarray, modality: str, hemi_cortex: np.ndarray,
    sn_mask_cortex: np.ndarray, *, mask_G: np.ndarray | None = None,
) -> np.ndarray
```

Modality-aware preprocessing of a per-subject cortex × cortex matrix.

Always sets to `NaN`: the diagonal and within-network edges. SC weights are masked by `mask_G`, made positive, and log-transformed; GD weights become inverse distance and are restricted to within-hemisphere; MPC zeros become `NaN`. Cross-hemisphere entries are dropped for GD only.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `W_raw` | `np.ndarray`, shape `(n_cortex, n_cortex)` | Raw subject matrix (cortex-restricted, symmetric). |
| `modality` | `str` | One of `'SC'`, `'GD'`, `'MPC'`. |
| `hemi_cortex` | `np.ndarray`, shape `(n_cortex,)` | Per-vertex `'LH'`/`'RH'` labels. |
| `sn_mask_cortex` | `np.ndarray` of `bool`, shape `(n_cortex,)` | `True` for source-network vertices. |
| `mask_G` | `np.ndarray` of `bool` | Betzel consensus mask (SC only). Optional. |

**Returns** `np.ndarray`, shape `(n_cortex, n_cortex)` — preprocessed weights with `NaN` marking excluded entries.

**Raises** `ValueError` if `modality` is not one of `'SC'`, `'GD'`, `'MPC'`.

---

### `compute_projection_score`

```python
compute_projection_score(
    W: np.ndarray, g_fc_cortex: np.ndarray,
    sn_idx_cortex: np.ndarray, other_idx_cortex: np.ndarray,
    *, min_valid: int = 10,
) -> np.ndarray
```

Weighted-mean projection score $P_i = \sum_j w_{ij}\, g^{\mathrm{FC}}_j / \sum_j w_{ij}$ for SC and GD modalities.

Only positive, finite weights contribute (to both numerator and denominator). Rows with fewer than `min_valid` finite targets, or a non-positive denominator, return `NaN`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `W` | `np.ndarray`, shape `(n_cortex, n_cortex)` | Preprocessed weights. |
| `g_fc_cortex` | `np.ndarray`, shape `(n_cortex,)` | Whole-brain FC gradient on cortex. |
| `sn_idx_cortex` | `np.ndarray` of `bool` | Source-network mask. |
| `other_idx_cortex` | `np.ndarray` of `bool` | Target set (non-network cortex). |
| `min_valid` | `int` | Minimum finite targets per vertex. Default `10`. |

**Returns** `np.ndarray`, shape `(n_sn,)` — projection score per source-network vertex.

---

### `compute_projection_score_rank`

```python
compute_projection_score_rank(
    W: np.ndarray, g_fc_cortex: np.ndarray,
    sn_idx_cortex: np.ndarray, other_idx_cortex: np.ndarray,
    *, min_valid: int = 10,
) -> np.ndarray
```

Per-source-network-vertex Spearman correlation across targets — the MPC variant, which is robust to signed weights where the weighted mean is ill-defined.

Computes $r_i = \operatorname{Spearman}_j(W_{ij},\, g^{\mathrm{FC}}_j)$ over the target set. Signature and exclusion rules match `compute_projection_score`.

**Returns** `np.ndarray`, shape `(n_sn,)` — per-vertex rank correlation.

---

### `compute_projection_subjects`

```python
compute_projection_subjects(
    files: list, modality: str,
    g_fc_cortex: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    df_yeo_surf_5k: pd.DataFrame,
    *, mask_G: np.ndarray | None = None,
    sc_subjects: list[np.ndarray] | None = None,
    target_network_labels: np.ndarray | None = None,
    min_valid: int = 10,
) -> dict
```

Per-subject projection plus group inference for one modality and one source network.

Loops over subjects, computes each subject's projection score and its Spearman alignment with the within-network MPC gradient, then aggregates across subjects with a Fisher-z transform and one-sample t-test.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `files` | `list` | Per-subject connectivity files (used when `sc_subjects` is `None` or modality ≠ SC). |
| `modality` | `str` | One of `'SC'`, `'GD'`, `'MPC'`. |
| `g_fc_cortex` | `np.ndarray`, shape `(n_cortex,)` | Whole-brain FC gradient. |
| `g_mpc_cortex_at_sn` | `np.ndarray`, shape `(n_sn,)` | Procrustes-aligned MPC gradient at source-network vertices. |
| `sn_mask_cortex` / `other_mask_cortex` | `np.ndarray` of `bool` | Source-network and target masks. |
| `df_yeo_surf_5k` | `pd.DataFrame` | Provides hemisphere info. |
| `mask_G` | `np.ndarray` of `bool` | Betzel consensus mask (SC only). Optional. |
| `sc_subjects` | `list[np.ndarray]` | Pre-loaded SC matrices, avoiding re-reads. Optional. |
| `target_network_labels` | `np.ndarray` | Enables per-target-network weight summaries. Optional. |
| `min_valid` | `int` | Minimum finite targets per vertex. Default `10`. |

**Returns** `dict` — keys include `P_mean`, `P_subjects_sn`, `P_subjects_full`, `r_subjects`, `target_net_weights`, `target_network_names`, and the Fisher-z aggregates `r_group`, `t`, `p`, `ci_low`, `ci_high`, `n`.

---

### `compute_spin_null_projection`

```python
compute_spin_null_projection(
    g_mpc_cortex_at_sn: np.ndarray, sn_mask_cortex: np.ndarray,
    cortex_mask_full: np.ndarray, result: dict,
    spin_model, n_rand: int,
) -> dict
```

Spin-test null for the per-subject MPC gradient ↔ projection alignment (Alexander-Bloch 2018, adapted to a within-network statistic).

The MPC gradient is embedded in the full 9,684-vertex space (`NaN` outside the source network) and rotated with the fitted `SpinPermutations` model. Each permutation is aggregated across subjects with the Fisher-z mean, yielding a two-tailed empirical $p_{\mathrm{spin}}$.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `g_mpc_cortex_at_sn` | `np.ndarray`, shape `(n_sn,)` | MPC gradient at source-network vertices. |
| `sn_mask_cortex` | `np.ndarray` of `bool` | Source-network mask on cortex. |
| `cortex_mask_full` | `np.ndarray` of `bool` | Cortex mask over the full 9,684-vertex space. |
| `result` | `dict` | Output of `compute_projection_subjects`. |
| `spin_model` | brainspace `SpinPermutations` | Fitted spin model. |
| `n_rand` | `int` | Number of rotations. |

**Returns** `dict` — null distribution and two-tailed empirical `p_spin`.

---

### `compute_moran_null_projection`

```python
compute_moran_null_projection(
    g_mpc_cortex_at_sn: np.ndarray, sn_mask_cortex: np.ndarray,
    gd_among_sn: np.ndarray, result: dict, n_rand: int,
    *, random_state: int = 42,
) -> dict
```

Moran spectral-randomisation null for the MPC gradient ↔ projection alignment.

Generates MPC-gradient surrogates that preserve the within-network spatial autocorrelation (Wagner & Dray 2015). Compared to the cortex-wide spin test, the null is restricted entirely to the source network — a tighter, more powerful distribution that matches the test footprint.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `g_mpc_cortex_at_sn` | `np.ndarray`, shape `(n_sn,)` | MPC gradient at source-network vertices. |
| `sn_mask_cortex` | `np.ndarray` of `bool` | Source-network mask on cortex. |
| `gd_among_sn` | `np.ndarray`, shape `(n_sn, n_sn)` | Geodesic distance among source-network vertices (cross-hemisphere entries `0`). |
| `result` | `dict` | Output of `compute_projection_subjects`. |
| `n_rand` | `int` | Number of surrogates. |
| `random_state` | `int` | Seed. Default `42`. |

**Returns** `dict` — null distribution and two-tailed empirical p-value.

---

### `compute_dominant_target_network`

```python
compute_dominant_target_network(result: dict) -> tuple[np.ndarray, list[str]]
```

For each source-network vertex, return the index and name of the target network with the highest group-mean weighted connectivity. Requires `target_net_weights` and `target_network_names` in `result`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `result` | `dict` | Output of `compute_projection_subjects` (with target-network weights). |

**Returns** `(dominant_idx, target_network_names)` — `np.ndarray` of `int`, shape `(n_sn,)` (`-1` where all `NaN`), and the ordered list of network names.

---

### `benjamini_hochberg`

```python
benjamini_hochberg(pvals: np.ndarray) -> np.ndarray
```

Benjamini–Hochberg adjusted p-values (q-values), NaN-safe.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `pvals` | `np.ndarray` | Raw p-values; `NaN` entries are ignored and preserved. |

**Returns** `np.ndarray` — adjusted q-values, same shape as `pvals`.

---

### `compute_tertile_contrast`

```python
compute_tertile_contrast(P_mean_sn: np.ndarray, g_mpc_cortex_at_sn: np.ndarray) -> pd.DataFrame
```

Supplementary summary: tertile source-network vertices by the MPC gradient and report the mean projection score per tertile. Under the directional hypothesis, the high-gradient (superior) tertile should have the highest projection score.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `P_mean_sn` | `np.ndarray`, shape `(n_sn,)` | Group-mean projection score. |
| `g_mpc_cortex_at_sn` | `np.ndarray`, shape `(n_sn,)` | MPC gradient at source-network vertices. |

**Returns** `pd.DataFrame` — one row per tertile (`inferior`, `middle`, `superior`) with `mean_P`, `sd_P`, and `n`.

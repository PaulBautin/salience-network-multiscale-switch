# `src/connectome_processing`

Connectome I/O and the gradient-weighted connectivity projection used in [Figure 2](../methods/figure_2.md). All matrices are handled at fsLR-5k resolution (9,684 vertices); micapipe stores fsLR-5k connectomes as upper-triangular GIFTIs, which are symmetrised on load. See [Figure 2 Methods](../methods/figure_2.md) for the statistical derivation of the projection score and the within-network Moran null.

### `load_subject_matrix`

```python
load_subject_matrix(path, cortex_mask: np.ndarray) -> np.ndarray
```

Load an fsLR-5k vertex × vertex `.shape.gii` connectome restricted to cortex.

The upper-triangular storage is symmetrised via `triu(d, 1) + d.T`, recovering the diagonal from the transpose. Negative entries are clipped to zero; every modality uses only positive connections downstream, so this both removes SC numerical noise and discards the anticorrelated (FC) and negative-partial-correlation (MPC) edges.

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

Always sets to `NaN`: the diagonal and within-network edges. Every modality keeps only positive connections: SC weights are masked by `mask_G` and log-transformed; GD weights become inverse distance and are restricted to within-hemisphere; MPC and FC retain positive correlations while non-positive (negative/zero) entries are dropped to `NaN`. Cross-hemisphere entries are dropped for GD only.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `W_raw` | `np.ndarray`, shape `(n_cortex, n_cortex)` | Raw subject matrix (cortex-restricted, symmetric, non-negative after the load-time clip). |
| `modality` | `str` | One of `'SC'`, `'GD'`, `'MPC'`, `'FC'`. |
| `hemi_cortex` | `np.ndarray`, shape `(n_cortex,)` | Per-vertex `'LH'`/`'RH'` labels. |
| `sn_mask_cortex` | `np.ndarray` of `bool`, shape `(n_cortex,)` | `True` for source-network vertices. |
| `mask_G` | `np.ndarray` of `bool` | Betzel consensus mask (SC only). Optional. |

**Returns** `np.ndarray`, shape `(n_cortex, n_cortex)` — preprocessed weights with `NaN` marking excluded entries.

**Raises** `ValueError` if `modality` is not one of `'SC'`, `'GD'`, `'MPC'`, `'FC'`.

---

### `compute_projection_score`

```python
compute_projection_score(
    W: np.ndarray, g_fc_cortex: np.ndarray,
    sn_idx_cortex: np.ndarray, other_idx_cortex: np.ndarray,
    *, min_valid: int = 10,
) -> np.ndarray
```

Weighted-mean projection score $P_i = \sum_j w_{ij}\, g^{\mathrm{FC}}_j / \sum_j w_{ij}$ for every modality (SC, GD, MPC, FC).

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

### `compute_projection_subjects`

```python
compute_projection_subjects(
    files: list, modality: str,
    g_fc_cortex: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    df_yeo_surf_5k: pd.DataFrame,
    *, mask_G: np.ndarray | None = None,
    preloaded_subjects: list[np.ndarray] | None = None,
    target_network_labels: np.ndarray | None = None,
    min_valid: int = 10,
) -> dict
```

Per-subject projection plus group inference for one modality and one source network.

Loops over subjects, computes each subject's projection score and its Spearman alignment with the within-network MPC gradient, then aggregates across subjects with a Fisher-z transform and one-sample t-test.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `files` | `list` | Per-subject connectivity files (used only when `preloaded_subjects` is `None`). |
| `modality` | `str` | One of `'SC'`, `'GD'`, `'MPC'`, `'FC'`. |
| `g_fc_cortex` | `np.ndarray`, shape `(n_cortex,)` | Whole-brain FC gradient. |
| `g_mpc_cortex_at_sn` | `np.ndarray`, shape `(n_sn,)` | Procrustes-aligned MPC gradient at source-network vertices. |
| `sn_mask_cortex` / `other_mask_cortex` | `np.ndarray` of `bool` | Source-network and target masks. |
| `df_yeo_surf_5k` | `pd.DataFrame` | Provides hemisphere info. |
| `mask_G` | `np.ndarray` of `bool` | Betzel consensus mask (SC only). Optional. |
| `preloaded_subjects` | `list[np.ndarray]` | Pre-loaded per-subject raw matrices (used instead of reading `files`; load once, reuse across networks). Optional. |
| `target_network_labels` | `np.ndarray` | Enables per-target-network weight summaries. Optional. |
| `min_valid` | `int` | Minimum finite targets per vertex. Default `10`. |

**Returns** `dict` — keys include `P_mean`, `P_subjects_sn`, `P_subjects_full`, `r_subjects`, `n_targets_per_sn` (mean-over-subjects finite-positive target count per source vertex, a sparsity diagnostic), `target_net_weights`, `target_network_names`, and the Fisher-z aggregates `r_group`, `t`, `p`, `ci_low`, `ci_high`, `n`.

---

### `compute_moran_null_projection`

```python
compute_moran_null_projection(
    g_mpc_cortex_at_sn: np.ndarray, sn_mask_cortex: np.ndarray,
    gd_among_sn: np.ndarray, result: dict, n_rand: int,
    *, random_state: int = 42,
) -> dict
```

Moran spectral-randomisation null for the MPC gradient ↔ projection alignment, restricted entirely to the source network to match the test footprint.

Generates MPC-gradient surrogates that preserve the within-network spatial autocorrelation (Wagner & Dray 2015). Surrogates are generated **per connected component** of the inverse-geodesic-distance graph — i.e. per hemisphere for a bilateral network, since cross-hemisphere geodesic distance is `0` — via `_moran_surrogates_blockwise`, preserving within-hemisphere autocorrelation while keeping both hemispheres in the statistic. The two-tailed empirical p-value uses the add-one estimator $p = (1+k)/(1+n)$, bounded away from zero.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `g_mpc_cortex_at_sn` | `np.ndarray`, shape `(n_sn,)` | MPC gradient at source-network vertices. |
| `sn_mask_cortex` | `np.ndarray` of `bool` | Source-network mask on cortex. |
| `gd_among_sn` | `np.ndarray`, shape `(n_sn, n_sn)` | Geodesic distance among source-network vertices; cross-hemisphere entries are `0`, which splits the spatial graph into per-hemisphere components. |
| `result` | `dict` | Output of `compute_projection_subjects`. |
| `n_rand` | `int` | Number of surrogates. |
| `random_state` | `int` | Base seed (offset per component so blocks are decorrelated). Default `42`. |

**Returns** `dict` — `null_group_moran`, `p_moran` (two-tailed, add-one), `null_std_moran`.

---

### `compute_topological_null_projection`

```python
compute_topological_null_projection(
    modality: str, files: list, df_yeo_surf_5k: pd.DataFrame,
    g_fc_cortex: np.ndarray, g_mpc_cortex_at_sn: np.ndarray,
    sn_mask_cortex: np.ndarray, other_mask_cortex: np.ndarray,
    gd_sn_to_other: np.ndarray, result: dict, n_rand: int,
    *, preloaded_subjects: list[np.ndarray] | None = None, mask_G: np.ndarray | None = None,
    nbins: int = 10, random_state: int = 42, min_valid: int = 10, method: str = "exact",
) -> dict
```

Geometry-preserving topological null for the MPC gradient ↔ projection alignment, testing whether the *specific* source→target wiring drives the effect beyond connectome geometry.

Each subject's connectome is rewired by reassigning every source→target edge to a different target **in the same geodesic-distance bin** (with replacement; pools are large relative to per-vertex degree), keeping the edge weight attached. This preserves each source vertex's degree, weight multiset, and edge-length distribution while randomising target identity. The projection and per-subject Spearman are recomputed on the rewired connectome and aggregated by the Fisher-z mean across subjects per surrogate; because edge length is preserved, the null distribution is centred on the geometry expectation rather than zero. `p_topo` is consequently an **excess-magnitude** add-one estimate (`empirical_p_excess_magnitude`) — testing whether the observed alignment is *stronger in magnitude* ($|r_{\text{obs}}| \geq |r_{\text{null}}|$) than the geometry expectation — not a two-sided test around zero. The magnitude (rather than fixed-side) comparison is required because the group correlation inherits the arbitrary polarity of the source gradient `g_MPC`; the observed effect and the surrogates share that same fixed `g_MPC`, so both lie on the same side of zero and folding by $|\cdot|$ recovers the intended test while staying invariant to the eigenvector's arbitrary sign (a fixed upper-tail test would spuriously report n.s. whenever `g_MPC` came out sign-negative). Intended for `'SC'`, `'MPC'`, `'FC'`; `'GD'` weights are a deterministic function of distance, so a within-bin reassignment leaves them essentially unchanged and the null is uninformative.

Two samplers, selected by `method`: `'exact'` resamples each edge's target explicitly; `'clt'` is an algebraically equivalent analytic-moment shortcut — the per-vertex resampled numerator has closed-form mean `Σ_b W_b·μ_b` and variance `Σ_b (Σ_e w_e²)·σ²_b` (summed over distance-bins `b`; `W_b` = bin edge-weight sum, `μ_b`/`σ²_b` = bin target-map mean/variance) and, for dense connectomes, is drawn from the matching Gaussian (central limit theorem), removing the per-surrogate resampling cost. Used for the dense `'MPC'`/`'FC'` measures; `'SC'` keeps `'exact'`. The two agree within Monte-Carlo error.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `modality` | `str` | One of `'SC'`, `'MPC'`, `'FC'`. |
| `files` | `list` | Per-subject connectivity files (used only when `preloaded_subjects` is `None`). |
| `df_yeo_surf_5k` | `pd.DataFrame` | Provides the cortex mask and per-vertex hemisphere labels. |
| `g_fc_cortex` | `np.ndarray`, shape `(n_cortex,)` | Whole-brain FC gradient on cortex. |
| `g_mpc_cortex_at_sn` | `np.ndarray`, shape `(n_sn,)` | MPC gradient at source-network vertices. |
| `sn_mask_cortex` / `other_mask_cortex` | `np.ndarray` of `bool` | Source-network and target masks. |
| `gd_sn_to_other` | `np.ndarray`, shape `(n_sn, n_other)` | Geodesic distance from source vertices to targets; cross-hemisphere entries `0` form a separate (inter-hemisphere) bin. |
| `result` | `dict` | Output of `compute_projection_subjects`; only `r_group` is read (the observed value). |
| `n_rand` | `int` | Number of surrogates. |
| `preloaded_subjects` | `list[np.ndarray]` | Pre-loaded per-subject raw matrices (used instead of reading `files`; load once, reuse across networks). Optional. |
| `mask_G` | `np.ndarray` of `bool` | Betzel consensus mask (SC only). Optional. |
| `nbins` | `int` | Intra-hemisphere distance bins. Default `10`. |
| `random_state` | `int` | Base seed. Default `42`. |
| `min_valid` | `int` | Minimum finite-positive targets per source vertex. Default `10`. |
| `method` | `str` | Surrogate sampler: `'exact'` (default) or `'clt'`. |

**Returns** `dict` — `null_group_topo`, `p_topo` (excess-magnitude, add-one), `null_std_topo`.

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

### `empirical_p_twosided`

```python
empirical_p_twosided(null: np.ndarray, observed: float) -> float
```

Two-tailed empirical permutation p-value via the add-one estimator $p = (1+k)/(1+n)$, where $k$ counts the finite null values whose magnitude is $\geq |\,\text{observed}\,|$ and $n$ is the number of finite null values. The add-one keeps the estimate bounded away from zero. Shared by `compute_moran_null_projection` and the figure 1B within-network Moran null.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `null` | `np.ndarray` | Null distribution of the statistic; non-finite entries are ignored. |
| `observed` | `float` | Observed statistic. |

**Returns** `float` — empirical p-value, or `NaN` if `observed` is non-finite or there are no finite null values.

---

### `empirical_p_upper`

```python
empirical_p_upper(null: np.ndarray, observed: float) -> float
```

One-sided **upper-tail** empirical permutation p-value via the add-one estimator $p = (1+k)/(1+n)$, where $k$ counts the finite null values $\geq \text{observed}$ (not in absolute value) and $n$ is the number of finite null values. Use for a null whose centre is **not** zero **and** whose statistic has a fixed, interpretable sign (the observed effect is expected on the upper side). When the statistic's sign is arbitrary (e.g. a correlation against an unanchored diffusion-map eigenvector) use `empirical_p_excess_magnitude` instead — a fixed-side tail test misfires when the effect comes out sign-negative. For a ~0-centred null use `empirical_p_twosided`.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `null` | `np.ndarray` | Null distribution of the statistic; non-finite entries are ignored. |
| `observed` | `float` | Observed statistic. |

**Returns** `float` — empirical p-value, or `NaN` if `observed` is non-finite or there are no finite null values.

---

### `empirical_p_excess_magnitude`

```python
empirical_p_excess_magnitude(null: np.ndarray, observed: float) -> float
```

Sign-invariant **excess-magnitude** empirical permutation p-value via the add-one estimator $p = (1+k)/(1+n)$, where $k$ counts the finite null values with $|\,\text{null}\,| \geq |\,\text{observed}\,|$ and $n$ is the number of finite null values. This is the correct one-sided test for the geometry-preserving topological null: its surrogates are centred on the non-zero geometry expectation, but the group correlation inherits the arbitrary polarity of the source gradient `g_MPC`. Because the observed statistic and the surrogates share that same fixed `g_MPC`, both lie on the same side of zero, so folding by $|\cdot|$ recovers the "alignment exceeds geometry" comparison while remaining invariant to the eigenvector's arbitrary sign (unlike `empirical_p_upper`, which silently reports n.s. when `g_MPC` comes out sign-negative). The add-one keeps the estimate bounded away from zero.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `null` | `np.ndarray` | Null distribution of the statistic; non-finite entries are ignored. |
| `observed` | `float` | Observed statistic. |

**Returns** `float` — empirical p-value, or `NaN` if `observed` is non-finite or there are no finite null values.

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

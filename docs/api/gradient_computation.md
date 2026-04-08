# `src/gradient_computation`

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

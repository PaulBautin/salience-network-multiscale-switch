# Figure 2 — Gradient-driven connectivity fingerprints

**Script:** `scripts/figure_2_distance.py`

Figure 2 tests whether the microstructural gradient within the salience network
predicts how vertices connect to the rest of the brain. All analyses run at
**fsLR-5k** resolution (9,684 vertices: 4,842 LH + 4,842 RH) to keep
whole-brain connectivity matrices in memory.

- **Figure 2A** — SC, GD, and MPC connectivity fingerprints for SalVentAttn,
  correlated with the whole-brain FC gradient.
- **Figure 2B** — MPC fingerprint replicated across all 7 Yeo networks.

The MPC gradient used to anchor the analysis is computed with the shared pipeline
in [Shared Methods — MPC gradient computation](shared.md#mpc-gradient-computation),
applied to fsLR-5k T1 profiles. Statistical testing uses spin permutations
described in [Shared Methods — Spin-test permutations](shared.md#spin-test-permutations-whole-brain).

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{V}_\mathcal{N}$ | Network vertices at fsLR-5k (e.g. SalVentAttn), $n_\mathcal{N}$ total |
| $\mathcal{V}_{\text{other}}$ | Non-target-network cortical vertices, $n_{\text{other}}$ total |
| $g_v$ | Z-scored first MPC gradient value at vertex $v \in \mathcal{V}_\mathcal{N}$ |
| $C^{(s)}_{v,j}$ | Connectivity from network vertex $v$ to target vertex $j$ for subject $s$ |
| $K = 10$ | Number of equal-quantile gradient decile bins |
| $N_S$ | Number of subjects |

---

## Connectivity metrics

Three metrics are computed at fsLR-5k vertex resolution. micapipe stores
connectivity matrices as upper-triangle GIFTI files (9,684 × 9,684); all are
symmetrised on load.

| Metric | Transform | Inter-hemispheric |
|--------|-----------|-------------------|
| **SC** — SIFT2-weighted streamline count | log1p | retained |
| **GD** — Surface geodesic path length | none | zeroed |
| **MPC** — Vertex partial-correlation (Fisher z) | none | retained |

Zero entries are set to NaN before analysis (zero = absent edge, not zero
connectivity).

---

## Group-representative SC (Betzel et al. 2018)

For SC, a group-representative binary mask $G$ is built using distance-dependent
consensus thresholding. Tractography systematically over-represents short-range
connections because short streamlines are geometrically easier to reconstruct;
a naive consistency threshold would therefore bias the group network toward
short-range edges. Distance-dependent thresholding corrects this by selecting
the most reproducible connections **within each distance stratum**, so the group
network preserves the empirical connection-length distribution rather than
collapsing toward short-range edges. This matters here because the analysis
directly tests whether MPC-gradient extremes differ in their long-range
structural connectivity.

Let $A^{(s)} \in \mathbb{R}^{n \times n}$ be the SC matrix for subject $s$ and
$D \in \mathbb{R}^{n \times n}$ the mean tract distance matrix (mean streamline
length from tractography, `path_sc_dist_5k`).

Edge consistency and mean weight across subjects:

$$C_{ij} = \sum_{s=1}^{N_S} \mathbf{1}[A^{(s)}_{ij} > 0], \qquad \bar{W}_{ij} = \frac{\sum_s A^{(s)}_{ij}}{C_{ij}}$$

Within-hemisphere (WH) and between-hemisphere (BH) edges are handled separately.
For each compartment, edges are partitioned into $B = 10$ equal-width distance
bins. The target number of group edges in bin $b$ is proportional to the
empirical edge-length distribution:

$$k_b = \text{round}\!\left(\frac{D_{\text{total}}}{N_S} \cdot \frac{D_b}{D_{\text{total}}}\right)$$

where $D_b$ is the count of present edges in bin $b$ across all subjects and
$D_{\text{total}} = \sum_b D_b$. Within each bin the $k_b$ edges with the highest
$C_{ij}$ are retained.

The binary mask $G > 0$ identifies group-consensus structural edges; vertices
with no consensus connections are connectivity-sparse by the group criterion.

<details>
<summary>Implementation — <code>fcn_group_bins</code></summary>

```python
def fcn_group_bins(adj, dist, hemiid, nbins):
    if hemiid.ndim == 1:
        hemiid = hemiid[:, np.newaxis]
    n, nsub = adj.shape[0], adj.shape[-1]
    nonzero_dist = dist[np.nonzero(dist)]
    distbins = np.linspace(nonzero_dist.min(), nonzero_dist.max(), nbins + 1)
    distbins[-1] += 1

    C = np.sum(adj > 0, axis=2)
    W = np.sum(adj, axis=2) / np.where(C > 0, C, np.nan)
    W = np.nan_to_num(W, nan=0.0)

    inter_hemi_mask = np.dot(hemiid, ~hemiid.T)
    inter_hemi_mask = np.logical_or(inter_hemi_mask, inter_hemi_mask.T)

    Grp = np.zeros((n, n, 2))
    for j in range(2):                             # j=0: between-hemi, j=1: within-hemi
        inter_hemi = ~inter_hemi_mask if j else inter_hemi_mask
        m = dist * inter_hemi
        D = (adj > 0) * (dist * np.triu(inter_hemi))[..., np.newaxis]
        D = D[np.nonzero(D)]
        if len(D) == 0:
            continue
        tgt = len(D) / nsub                        # average edges per subject

        G = np.zeros((n, n))
        for i_bin in range(nbins):
            mask = np.where(np.triu(
                (m >= distbins[i_bin]) & (m < distbins[i_bin + 1]), 1))
            if len(mask[0]) == 0:
                continue
            n_D_bin = np.sum((D >= distbins[i_bin]) & (D < distbins[i_bin + 1]))
            frac = int(np.round(tgt * n_D_bin / len(D)))
            c = C[mask]
            idx = np.argsort(c)[::-1]
            G[mask[0][idx[:frac]], mask[1][idx[:frac]]] = 1
        Grp[:, :, j] = G

    G = np.sum(Grp, 2)
    G = G + G.T
    return G
```

</details>

---

## Decile OLS regression

### Step 1 — Decile binning

Network vertices are partitioned into $K = 10$ equal-quantile bins by $g_v$:

$$q_k = \text{quantile}(g_v,\; v \in \mathcal{V}_\mathcal{N},\; k/K), \quad k = 0,\ldots,K$$

$$B_k = \{v \in \mathcal{V}_\mathcal{N} \mid q_{k-1} \leq g_v < q_k\}, \quad B_K \text{ uses } \leq q_K$$

Each bin has $|B_k| \approx n_\mathcal{N} / K$ vertices. Bin rank is 0-indexed:
$r_k = k - 1 \in \{0,\ldots,9\}$.

Binning before regression suppresses single-vertex outlier leverage that arises
in continuous-gradient OLS and avoids the need for a sparsity mask.

### Step 2 — Per-subject OLS

For each subject $s$ and target vertex $j \in \mathcal{V}_{\text{other}}$, the mean
connectivity from each gradient bin is:

$$\mu^{(s)}_{k,j} = \frac{1}{|B_k|} \sum_{v \in B_k} C^{(s)}_{v,j}, \quad k = 1,\ldots,K$$

OLS fits a line of mean connectivity against bin rank:

$$\mu^{(s)}_{\cdot,j} = \alpha^{(s)}_j + \beta^{(s)}_j\,\mathbf{r} + \varepsilon, \quad \mathbf{r} = [0,1,\ldots,9]^\top$$

The slope $\beta^{(s)}_j$ is the change in connectivity to vertex $j$ per unit
increase in gradient rank for subject $s$.

<details>
<summary>Implementation — <code>compute_decile_slope_subjects</code></summary>

```python
def compute_decile_slope_subjects(files, gradient_values_cortex,
                                  network_mask_cortex, other_idx_cortex,
                                  df_yeo_surf_5k, n_bins=10,
                                  split_hemi=False, log_transform=False):
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values
    hemi = df_yeo_surf_5k.loc[cortex_mask, "hemisphere"].values
    n_other = int(other_idx_cortex.sum())
    bin_masks = compute_decile_bins(gradient_values_cortex, network_mask_cortex, n_bins)
    x_rank = np.arange(n_bins, dtype=float)

    subject_slopes = []
    for f in files:
        data = nib.load(f).darrays[0].data.astype(float)
        data = data[np.ix_(cortex_mask, cortex_mask)]
        data = np.triu(data, 1) + data.T
        data[data == 0] = np.nan
        if split_hemi:
            same_hemi = hemi[:, None] == hemi[None, :]
            data[~same_hemi] = np.nan
        if log_transform:
            data = np.log1p(np.maximum(data, 0))

        bin_means = np.full((n_bins, n_other), np.nan)
        for k, bm in enumerate(bin_masks):
            if bm.any():
                bin_means[k] = np.nanmean(data[np.ix_(bm, other_idx_cortex)], axis=0)

        slopes = np.full(n_other, np.nan)
        for j in range(n_other):
            col = bin_means[:, j]
            valid = ~np.isnan(col)
            if valid.sum() >= 2:
                slopes[j] = np.polyfit(x_rank[valid], col[valid], 1)[0]
        subject_slopes.append(slopes)

    arr = np.stack(subject_slopes, axis=0)
    mean_slopes = np.nanmean(arr, axis=0)
    n_valid = (~np.isnan(arr)).sum(axis=0).astype(float)
    se_slopes = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(n_valid, 1))
    return mean_slopes, se_slopes
```

</details>

### Step 3 — Group summary and z-scoring

$$\bar{\beta}_j = \frac{1}{N_S}\sum_{s=1}^{N_S}\beta^{(s)}_j, \qquad \widehat{\text{SE}}(\bar{\beta}_j) = \frac{\text{std}_{N_S-1}(\beta^{(s)}_j)}{\sqrt{N_S}}$$

For brain-map visualization, $\bar{\beta}$ is z-scored across $j \in \mathcal{V}_{\text{other}}$:

$$z_j = \frac{\bar{\beta}_j - \overline{\bar{\beta}}}{\text{std}(\bar{\beta})}$$

Vertices in $\mathcal{V}_\mathcal{N}$ and medial-wall vertices are set to NaN.

---

## Correlation with FC gradient and statistical tests

The z-scored slope map $\mathbf{z}$ is correlated with the whole-brain principal
FC gradient $\mathbf{g}^{\text{FC}}$ (fsLR-5k) restricted to $\mathcal{V}_{\text{other}}$:

$$r = \text{Spearman}(\mathbf{z},\; \mathbf{g}^{\text{FC}})$$

Significance is assessed with 1000 spin permutations (see
[Shared Methods](shared.md#spin-test-permutations-whole-brain)):

$$p_{\text{spin}} = \frac{1}{1000}\left|\{p : |r_p| \geq |r|\}\right|$$

A one-sample t-test against $\mu_0 = 0$ is additionally applied to the
distribution of finite $\bar{\beta}_j$ values to assess whether the
gradient-driven connectivity pattern is systematically non-zero:

$$t = \frac{\overline{\bar{\beta}}}{\widehat{\text{SE}}(\bar{\beta})}, \quad \text{df} = n_{\text{other}} - 1$$

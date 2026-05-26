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
| $\tilde{g}_v$ | Mean-centred MPC gradient at network vertex $v$ ($\tilde{g}_v = g_v - \bar{g}$) |
| $w^{(s)}_{v,j}$ | Connectivity weight from network vertex $v$ to target vertex $j$ for subject $s$ |
| $P^{(s)}_j$ | Per-subject gradient projection at target vertex $j$ |
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

## Connectivity-weighted gradient projection

### Connectivity weights per metric

| Metric | Weight $w^{(s)}_{v,j}$ | Rationale |
|--------|------------------------|-----------|
| **SC** | log1p(SIFT2 streamlines) | structural connection strength |
| **GD** | $1 / \text{geodesic distance}_{v,j}$ | spatial proximity (within-hemisphere only) |
| **MPC** | Fisher-z partial correlation | microstructural similarity |

Zero entries are set to NaN before weighting (absent edge, not zero weight).

### Projection formula

For each subject $s$ and target vertex $j \in \mathcal{V}_{\text{other}}$:

$$P^{(s)}_j = \frac{\displaystyle\sum_{v \in \mathcal{V}_\mathcal{N}} w^{(s)}_{v,j}\,\tilde{g}_v}
                   {\displaystyle\sum_{v \in \mathcal{V}_\mathcal{N}} w^{(s)}_{v,j}}$$

$P^{(s)}_j$ is the connectivity-weighted centroid of the mean-centred MPC gradient: it quantifies
which part of the gradient axis vertex $j$ preferentially connects to within the network.

- Positive $P^{(s)}_j$: $j$ connects preferentially to the transmodal (high-gradient) end.
- Negative $P^{(s)}_j$: $j$ connects preferentially to the sensory (low-gradient) end.
- $P^{(s)}_j \approx 0$: $j$'s connections are evenly spread across the gradient.

No binning, no hyperparameter $K$. The full continuous gradient is used as a weight.

### Subject averaging and z-scoring

The group mean and SE are computed across subjects with valid projections:

$$\bar{P}_j = \frac{1}{N_S}\sum_s P^{(s)}_j, \qquad
\widehat{\text{SE}}(\bar{P}_j) = \frac{\text{std}(P^{(s)}_j)}{\sqrt{N_S}}$$

**Coverage threshold:** vertices where fewer than $0.5 \cdot N_S$ subjects yield a finite
$P^{(s)}_j$ are set to NaN.

For brain-map visualization, $\bar{P}$ is z-scored across surviving $j \in \mathcal{V}_{\text{other}}$:

$$z_j = \frac{\bar{P}_j - \overline{\bar{P}}}{\text{std}(\bar{P})}$$

<details>
<summary>Implementation — <code>compute_gradient_projection_subjects</code></summary>

```python
def compute_gradient_projection_subjects(files, gradient_values_cortex,
                                         network_mask_cortex, other_idx_cortex,
                                         df_yeo_surf_5k, split_hemi=False,
                                         log_transform=False, invert_weights=False):
    cortex_mask = df_yeo_surf_5k["hemisphere"].notna().values  # (9684,) → (n_cortex,)
    g_v = gradient_values_cortex[network_mask_cortex].astype(np.float32)
    g_v -= np.nanmean(g_v)                                # mean-centre

    subject_projs = []
    for f in files:
        data = nib.load(f).darrays[0].data.astype(np.float32)
        data = data[np.ix_(cortex_mask, cortex_mask)]
        data = np.triu(data, 1) + data.T
        data[data == 0] = np.nan
        # ... split_hemi / log_transform ...
        C_sub = data[np.ix_(network_mask_cortex, other_idx_cortex)]
        del data                                          # release full matrix

        if invert_weights:
            C_sub = np.where(C_sub > 0, 1.0 / C_sub, np.nan).astype(np.float32)

        num = np.nansum(g_v[:, None] * C_sub, axis=0)    # (n_other,)
        den = np.nansum(C_sub, axis=0)
        subject_projs.append(np.where(den > 0, num / den, np.nan))

    arr = np.stack(subject_projs, axis=0)
    mean_proj = np.nanmean(arr, axis=0)
    # ... coverage threshold, SE ...
    return mean_proj, se_proj
```

</details>

---

## Correlation with FC gradient and statistical tests

The z-scored projection map $\mathbf{z}$ is correlated with the whole-brain principal
FC gradient $\mathbf{g}^{\text{FC}}$ (fsLR-5k) restricted to $\mathcal{V}_{\text{other}}$:

$$r = \text{Spearman}(\mathbf{z},\; \mathbf{g}^{\text{FC}})$$

### Significance — spin permutations (two-tailed)

Significance is assessed with 1000 spin permutations (see
[Shared Methods](shared.md#spin-test-permutations-whole-brain)):

$$p_{\text{spin}} = \frac{1}{1000}\left|\{p : |r_p| \geq |r|\}\right|$$

A two-tailed test is used because the sign of the gradient is arbitrary (diffusion
maps do not have a canonical orientation); the test is conservative relative to a
one-tailed alternative.

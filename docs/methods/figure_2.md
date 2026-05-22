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

For SC, a group-representative binary mask $G$ is built to identify structurally
sparse vertices for visualization. Let $A^{(s)} \in \mathbb{R}^{n \times n}$ be the
SC matrix for subject $s$ and $D \in \mathbb{R}^{n \times n}$ the mean geodesic
distance matrix.

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

```
for compartment in [within-hemi, between-hemi]:
    D_total = total present edges across all subjects in compartment
    for bin b in 1..10:
        k_b = round(D_total / N_S * D_b / D_total)
        keep top-k_b edges by consistency C_ij
G = union of retained edges, symmetrised
A_group = log1p(G ⊙ W̄)   # binary mask × mean weight, then log1p
```

Vertices with $A_{\text{group},ij} = 0$ for all $j$ are flagged as connectivity-sparse
and shown in grey on brain maps.

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

```
for subject s in 1..N_S:
    load C^(s)  shape (n_cortex, n_cortex)
    for bin k in 1..K:
        μ[k, :] = mean(C^(s)[B_k, V_other], axis=0)   # (n_other,)
    β^(s) = OLS slope of μ[:,j] ~ r  for all j         # (n_other,)

mean_β = mean over subjects of β^(s)                   # (n_other,)
SE_β   = std(β^(s), ddof=1) / sqrt(N_S)               # (n_other,)
```

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

# Analysis Methods

## Microstructure profile covariance gradients

Microstructural profile covariance (MPC) was estimated by sampling intracortical signal intensities from a quantitative T1 (qT1) contrast. Fourteen equivolumetric cortical surfaces were generated between the pial and white matter boundaries, yielding per-vertex intensity profiles $\mathbf{p}_{s,v} \in \mathbb{R}^{14}$ for subject $s$ and vertex $v$. The Salience Network (SN) was defined according to the Yeo 7-network parcellation (Schaefer-400), corresponding to the ventral attention network (VAN).

**Partial correlation matrix.** For each subject, the mean profile across all network vertices is used as a covariate:

$$\bar{\mathbf{p}}_s = \frac{1}{|\mathcal{V}_\mathcal{N}|}\sum_{v \in \mathcal{V}_\mathcal{N}} \mathbf{p}_{s,v} \in \mathbb{R}^{14}$$

Profile residuals controlling for $\bar{\mathbf{p}}_s$ are obtained via OLS:

$$\mathbf{r}_{s,v} = \mathbf{p}_{s,v} - [1,\, \bar{\mathbf{p}}_s]\,\hat{\boldsymbol{\beta}}_v^{(s)}$$

The MPC matrix is the Fisher z-transformed Pearson correlation of residuals:

$$\text{MPC}^{(s)}_{ij} = \tanh^{-1}\!\left(\text{corr}(\mathbf{r}_{s,i},\, \mathbf{r}_{s,j})\right), \quad i,j \in \mathcal{V}_\mathcal{N}$$

**Gradient decomposition.** Low-dimensional representations were derived using BrainSpace. A normalized angle affinity kernel is applied to each subject's MPC matrix with sparsity threshold 0.9 (top 10% of connections retained). Diffusion map embedding extracts $n = 10$ gradient components per subject. Gradients are aligned across subjects using Procrustes rotation, averaged, and the first component is z-scored:

$$g_v = \text{zscore}\!\left(\frac{1}{N_S}\sum_{s=1}^{N_S} G^{(s,1)}_v\right), \quad v \in \mathcal{V}_\mathcal{N}$$

## Gradient extreme identification (Figure 1a)

To identify vertices at opposing poles of the SN microstructural gradient for profile visualization (Figure 1a), quantile thresholds were computed across all network-masked vertices. Vertices at or below the 25th percentile were labelled as the low-gradient pole and vertices at or above the 75th percentile as the high-gradient pole; the remaining 50% were excluded from pole comparisons. When both hemispheres were analysed jointly, thresholds were computed across all vertices and the resulting masks were combined. This stratification was used in Figure 1a to compare mean intracortical intensity profiles across the 14 equivolumetric depths.

## Gradient-driven connectivity fingerprints (Figure 2)

For Figure 2 connectivity analyses, a continuous gradient-based regression approach quantifies how connectivity to every other-network vertex scales with position along the microstructural gradient.

### Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{V}_\mathcal{N}$ | Set of $n_\mathcal{N}$ network vertices (e.g. SalVentAttn) at fsLR-5k |
| $\mathcal{V}_{\text{other}}$ | Set of $n_{\text{other}}$ non-target-network cortical vertices |
| $g_v$ | Z-scored first MPC gradient value at vertex $v \in \mathcal{V}_\mathcal{N}$ |
| $C^{(s)}_{v,j}$ | Connectivity from network vertex $v$ to other-network vertex $j$ for subject $s$ |
| $K = 10$ | Number of equal-quantile gradient bins (deciles) |
| $N_S$ | Number of subjects |

### Step 1 — Connectivity preprocessing per metric

Three metrics are computed at fsLR-5k vertex resolution:

- **SC**: SIFT2-weighted streamline count; symmetrized from upper triangle; zero entries set to NaN; log1p-transformed; inter-hemispheric connections retained.
- **GD**: Surface geodesic path length; zero entries set to NaN; no log-transform; inter-hemispheric connections zeroed (split_hemi=True).
- **MPC**: Vertex-level partial-correlation matrix (Fisher z-transformed); zero entries set to NaN; no log-transform; inter-hemispheric connections retained.

### Step 2 — Decile binning of gradient

Network vertices $\mathcal{V}_\mathcal{N}$ are partitioned into $K = 10$ equal-quantile bins $B_1, \ldots, B_K$ by the gradient $g_v$:

$$q_k = \text{quantile}(g_v,\; v \in \mathcal{V}_\mathcal{N},\; k/K), \quad k = 0,\ldots,K$$

$$B_k = \{v \in \mathcal{V}_\mathcal{N} \mid q_{k-1} \leq g_v < q_k\}, \quad k = 1,\ldots,K-1; \quad B_K \text{ uses } \leq q_K$$

Each bin has approximately equal size $|B_k| \approx n_\mathcal{N} / K$. Bin rank is 0-indexed: $r_k = k - 1 \in \{0,\ldots,9\}$.

### Step 3 — Per-subject OLS regression

For each subject $s$ and each target vertex $j \in \mathcal{V}_{\text{other}}$, compute the mean connectivity from each gradient bin:

$$\mu^{(s)}_{k,j} = \frac{1}{|B_k|} \sum_{v \in B_k} C^{(s)}_{v,j}, \quad k = 1,\ldots,K$$

Fit a linear model of mean connectivity against bin rank via OLS:

$$\mu^{(s)}_{\cdot,j} = \alpha^{(s)}_j + \beta^{(s)}_j \, \mathbf{r} + \varepsilon, \quad \mathbf{r} = [0, 1, \ldots, 9]^\top$$

The slope $\beta^{(s)}_j$ estimates how much connectivity to target vertex $j$ increases per unit rise in gradient rank for subject $s$.

```
for subject s in 1..N_S:
    load connectivity matrix C^(s)  [n_cortex × n_cortex]
    for bin k in 1..K:
        μ[k, :] = mean(C^(s)[B_k, V_other], axis=0)   # shape (n_other,)
    β^(s) = OLS slope of μ ~ r                          # shape (n_other,)

mean_β = mean(β^(s), axis=subjects)                     # shape (n_other,)
SE_β   = std(β^(s), ddof=1) / sqrt(N_S)                 # shape (n_other,)
```

### Step 4 — Group-level summary and z-scoring

$$\bar{\beta}_j = \frac{1}{N_S} \sum_{s=1}^{N_S} \beta^{(s)}_j, \qquad \widehat{\text{SE}}(\bar{\beta}_j) = \frac{\text{std}_{N_S-1}(\beta^{(s)}_j)}{\sqrt{N_S}}$$

For brain-map visualization, $\bar{\beta}$ is z-scored across $j \in \mathcal{V}_{\text{other}}$:

$$z_j = \frac{\bar{\beta}_j - \overline{\bar{\beta}}}{\text{std}(\bar{\beta})}$$

Vertices in $\mathcal{V}_\mathcal{N}$ and medial-wall vertices are set to NaN in the brain map.

### Step 5 — Correlation with the FC gradient and statistical testing

The z-scored slope map $\mathbf{z}$ is correlated with the whole-brain principal FC gradient $\mathbf{g}^{\text{FC}}$ (Margaret et al., projected to fsLR-5k) restricted to $\mathcal{V}_{\text{other}}$:

$$r = \text{Spearman}(\mathbf{z},\; \mathbf{g}^{\text{FC}})$$

**Spin-test null distribution** (1000 permutations, random_state=42): For each permutation $p$, the LH and RH FC gradient maps are independently rotated on the fsLR-5k sphere, medial-wall NaNs are restored, and a null correlation $r_p$ is computed:

$$p_{\text{spin}} = \frac{1}{1000} \left|\{p : |r_p| \geq |r|\}\right|$$

**One-sample t-test**: To assess whether gradient-driven connectivity differences are systematically non-zero across the cortex, a one-sample t-test against $\mu_0 = 0$ is applied to the distribution of finite $\bar{\beta}_j$ values:

$$t = \frac{\overline{\bar{\beta}}}{\widehat{\text{SE}}(\bar{\beta})}, \quad \text{df} = n_{\text{other}} - 1$$

## Cortical type reconstruction
Cortical types were assigned to Von Economo areas based on a recent reanalysis of Von Economo micrographs. This classification scheme was used because its criteria are (1) clearly defined, (2) applied consistently across the entire cortex, (3) align with Von Economo's original descriptions and (4) are supported by several histological samples. Criteria included 'development of layer IV, prominence (denser cellularity and larger neurons) of deep (V–VI) or superficial (II–III) layers, definition of sublayers (for example, IIIa and IIIb), sharpness of boundaries between layers and presence of large pyramids in superficial layers'. Thereby, cortical types synopsize degree of granularity, from high laminar elaboration in koniocortical areas, six identifiable layers in Eu-III to -I, poorly differentiated layers in dysgranular and absent layers in agranular.

## Structural network reconstruction

DWI pre-processing was implemented with the micapipe DWI module, which heavily relies on tools from MRtrix. Fiber orientation distributions were generated using the multi-shell, multi-tissue constrained spherical deconvolution (msmt-CSD) algorithm. 40 million streamlines were reconstructed using an anatomically constrained probabilistic tractography algorithm (ACT-iFOD2). Connectivity weights were optimized using SIFT2 by estimating a cross-section multiplier for each streamline based on apparent fiber density (AFD). A vertex-level connectivity matrix was built for each participant at fsLR-5k resolution (9,684 vertices).

Three connectivity metrics were computed for Figure 2:

- **SC**: SIFT2-weighted streamline counts, log1p-transformed.
- **GD**: Surface geodesic path length at fsLR-5k (within-hemisphere only).
- **MPC**: Vertex-level partial-correlation matrix (Fisher z-transformed), same computation as the MPC gradient section above but at fsLR-5k.

### Distance-dependent group-representative SC (Betzel et al. 2018)

For SC, a group-representative binary mask $G$ was constructed to identify structurally sparse vertices for visualization. Let $A^{(s)} \in \mathbb{R}^{n \times n}$ be the SC matrix for subject $s$ and $D \in \mathbb{R}^{n \times n}$ the mean geodesic distance matrix.

Define edge consistency and mean weight:

$$C_{ij} = \sum_{s=1}^{N_S} \mathbf{1}[A^{(s)}_{ij} > 0], \qquad \bar{W}_{ij} = \frac{\sum_s A^{(s)}_{ij}}{C_{ij}}$$

Within-hemisphere (WH) and between-hemisphere (BH) edges are processed separately. For each compartment, edges are partitioned into $B = 10$ equal-width distance bins. The target number of group edges in bin $b$ is set proportional to the empirical edge-length distribution:

$$k_b = \text{round}\!\left(\frac{D_{\text{total}}}{N_S} \cdot \frac{D_b}{D_{\text{total}}}\right)$$

where $D_b$ is the count of present edges across all subjects falling in bin $b$ and $D_{\text{total}} = \sum_b D_b$. Within each bin the $k_b$ edges with the highest $C_{ij}$ are retained in $G$.

```
for compartment in [within-hemi, between-hemi]:
    D_total = total present edges across all subjects in compartment
    for bin b in 1..10:
        candidate_edges = upper-triangle edges with dist in bin b
        k_b = round(D_total/N_S * D_b / D_total)
        keep top-k_b edges by consistency C_ij
G = union of retained edges, symmetrized
A_group = G ⊙ W̄        # hadamard: binary mask × mean weight
A_group = log1p(A_group)
```

The final weighted group matrix $A_{\text{group}} = G \odot \bar{W}$ (log1p-transformed) is used to identify connected vertices ($A_{\text{group}} > 0$). Figure 2A uses SC, GD, and MPC slopes within SalVentAttn; Figure 2B replicates the MPC slope analysis across all 7 Yeo networks. See the _Gradient-driven connectivity fingerprints_ section for the full regression pipeline.

## iEEG signal processing
iEEG signals (MNI open iEEG atlas and MICA iEEG datasets) were preprocessed using a common pipeline. Raw signals were band-pass filtered between 0.5 and 80 Hz using a 4th-order zero-phase Butterworth filter, then downsampled to 200 Hz. Signals were subsequently demeaned by subtracting the temporal mean of each channel. Power spectral density (PSD) was estimated using Welch's method with a Hamming window of 2-second segments and 1-second overlap. Band power was computed by integrating the PSD within each canonical frequency band using Simpson's rule. Relative band power was obtained by dividing each band's integral by the total power, and log₁₀-transformed (floor = 1×10⁻¹²). Frequency bands were defined as: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), and gamma (30–80 Hz).

## Brain surface feature comparisons
**Whole brain surface feature comparison** was implemented with brainspace using sphere spin permutation with 1000 permutations (random_state=42) to generate null data for hypothesis testing. Spin permutations were fitted directly on the fsLR-5k sphere surfaces and rotations were applied separately per hemisphere before concatenation. **Salience network brain surface feature comparison** was implemented with brainspace using moran spectral randomisation with 100 permutations, which uses the eigenvectors to generate null model data with similar spatial autocorrelation. The implemented procedure "singleton" matches the input data's autocorrelation more closely at the cost of fewer possible randomizations.

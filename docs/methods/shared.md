# Shared Methods

Methods used across multiple figures are documented here. Per-figure pages
reference these sections rather than repeating them.

---

## MPC gradient computation

Microstructural profile covariance (MPC) was estimated by sampling intracortical
signal intensities from a quantitative T1 (qT1) contrast. Fourteen equivolumetric
cortical surfaces were generated between the pial and white matter boundaries,
yielding per-vertex intensity profiles $\mathbf{p}_{s,v} \in \mathbb{R}^{14}$ for
subject $s$ and vertex $v$. The Salience Network (SN) was defined according to the
Yeo 7-network parcellation (Schaefer-400), corresponding to the ventral attention
network (VAN).

**Partial correlation matrix.** For each subject, the mean profile across all
network vertices is used as a covariate:

$$\bar{\mathbf{p}}_s = \frac{1}{|\mathcal{V}_\mathcal{N}|}\sum_{v \in \mathcal{V}_\mathcal{N}} \mathbf{p}_{s,v} \in \mathbb{R}^{14}$$

The covariate is the mean profile of the network's own vertices (rather than the
whole-cortex mean profile of the canonical MPC formulation; Paquola et al., 2019).
Partialling the network-common profile shape removes the signal shared across the
network, so the residual covariance — and hence the diffusion embedding — reflects
microstructural differentiation *within* the network.

Profile residuals controlling for $\bar{\mathbf{p}}_s$ are obtained via OLS:

$$\mathbf{r}_{s,v} = \mathbf{p}_{s,v} - [1,\, \bar{\mathbf{p}}_s]\,\hat{\boldsymbol{\beta}}_v^{(s)}$$

The MPC matrix is the Fisher z-transformed Pearson correlation of residuals:

$$\text{MPC}^{(s)}_{ij} = \tanh^{-1}\!\left(\text{corr}(\mathbf{r}_{s,i},\, \mathbf{r}_{s,j})\right), \quad i,j \in \mathcal{V}_\mathcal{N}$$

**Gradient decomposition.** Low-dimensional representations were derived using
BrainSpace. A normalized angle affinity kernel is applied to each subject's MPC
matrix with sparsity threshold 0.9 (top 10% of connections retained). Diffusion
map embedding extracts $n = 10$ gradient components per subject. Gradients are
aligned across subjects using Procrustes rotation, averaged, and the first
component is z-scored:

$$g_v = \text{zscore}\!\left(\frac{1}{N_S}\sum_{s=1}^{N_S} G^{(s,1)}_v\right), \quad v \in \mathcal{V}_\mathcal{N}$$

---

## Spatial statistics

### Spin-test permutations (whole-brain)

Spatial null distributions for **whole-brain** surface comparisons were generated
using sphere spin permutations (BrainSpace `SpinPermutations`, 1000 permutations,
`random_state=42`). The model is fitted on fsLR-5k sphere surfaces. For each
permutation $p$, the LH and RH feature maps are independently rotated on the
sphere, medial-wall vertices are restored to NaN, and a null statistic $r_p$ is
computed. The two-tailed spin p-value uses the add-one estimator (bounded away from
zero; $n$ is the number of valid permutations):

$$p_{\text{spin}} = \frac{1 + \left|\{p : |r_p| \geq |r_{\text{obs}}|\}\right|}{1 + n}$$

The spin test is applied to whole-brain maps (Figure 1C). Within-network statistics
use the Moran null below.

### Moran spectral randomisation (within-network)

For **within-network** comparisons (e.g. correlations restricted to SalVentAttn
vertices), spatial autocorrelation is accounted for using Moran spectral
randomisation (BrainSpace `moran`, procedure `"singleton"`; the number of
permutations is reported per analysis). Eigenvectors of the spatial weight matrix
(inverse geodesic/ring distance among the network's vertices) are used to generate
null maps with autocorrelation matched to the observed data. The `"singleton"`
procedure produces a tighter match at the cost of a smaller number of feasible
randomisations.

Because micapipe geodesic distance is undefined across hemispheres, the spatial
weight graph of a bilateral network is disconnected into one component per
hemisphere. Surrogates are therefore generated **independently within each connected
component** and reassembled into the full network vector, preserving within-hemisphere
autocorrelation while keeping both hemispheres in the statistic; a single-hemisphere
analysis forms one component. The two-tailed empirical p-value uses the add-one
estimator $p = (1 + k)/(1 + n)$.

### Geometry-preserving topological null (within-network)

The Moran null above randomises the source map and so controls for the spatial
smoothness of the two vertexwise maps, but it leaves the connectome untouched. For a
statistic built from connectivity weights, a complementary null is required to ask
whether the *specific* pattern of connections — rather than the connectome's geometry
(degree and edge length) — carries the effect. This is assessed with a
geometry-preserving topological null that rewires each subject's connectome within
geodesic-distance bins (Roberts et al., 2016; Betzel et al., 2018; Váša & Mišić,
2022).

Targets are binned by their geodesic distance from each source vertex into ten
equal-width intra-hemisphere bins; cross-hemisphere targets, for which geodesic
distance is undefined, form a single additional bin reassigned among contralateral
targets only. For each surrogate, every source→target edge is reassigned to another
target **in the same distance bin**, with the edge weight kept attached. The
reassignment preserves each source vertex's degree, its multiset of weights, and the
edge-length distribution, while randomising only the identity of the target — and
hence the value of the projected map at the target. Because edge length is preserved,
the null distribution of the group statistic is centred on the *geometry expectation*
rather than on zero, so an observed value whose **magnitude** exceeds that of the
surrogates isolates targeting specificity from distance dependence. The rewiring is
applied per subject (the subject is the unit of inference), the per-subject statistic is
recomputed on the rewired connectome and aggregated across subjects by the Fisher-z
mean. Because the surrogates are centred on the geometry expectation, the test is
directional — the alternative is that the observed alignment *exceeds* what geometry
alone produces — but it is evaluated on magnitude: the within-network microstructural
gradient is a diffusion-map eigenvector of arbitrary polarity, and since the observed
statistic and the surrogates share that same fixed gradient they lie on the same side of
zero, so the empirical p-value is the excess-magnitude add-one estimator $p = (1 +
\#\{|\text{null}| \ge |\text{obs}|\})/(1 + n)$. This is invariant to the eigenvector's
arbitrary sign (a fixed upper-tail estimator would spuriously fail whenever the gradient
came out sign-negative; a two-sided test around zero would conflate the non-zero geometry
offset with the effect). The null is defined for any
weight matrix and is therefore applied to the structural, microstructural, and
functional connectivity measures, but not to the geodesic-distance measure, whose
weights are a deterministic function of distance and are left essentially unchanged by
a within-bin reassignment.

For the structural-connectivity measure the surrogate projection is sampled explicitly
(each edge's target is redrawn). For the densely connected microstructural and functional
measures (hundreds to thousands of candidate targets per source vertex) the same null is
evaluated with an algebraically equivalent shortcut: the resampled projection numerator at
a source vertex is a sum of independent within-bin draws, so its mean and variance are
available in closed form — $\mathbb{E}=\sum_b W_b\,\mu_b$ and
$\mathrm{Var}=\sum_b (\sum_{e\in b} w_e^2)\,\sigma_b^2$, where $W_b$ is the summed edge
weight in distance-bin $b$ and $\mu_b,\sigma_b^2$ the mean and variance of that bin's target
map values. With many independent terms per vertex the numerator is Gaussian by the central
limit theorem, so each surrogate is drawn from the matching normal rather than by explicit
reassignment, leaving the test statistic unchanged while removing the per-surrogate
resampling cost. Because the two samplers share the same per-vertex mean and variance, they
agree up to Monte-Carlo error.

### Target-side spin null (FC-gradient rotation)

The Moran null randomises the *source* map and the topological null rewires the
*connectome*; a third null acts on the *target* map. The projection target axis is the
principal FC gradient $g^{\mathrm{FC}}$, which carries strong spatial autocorrelation
and is anatomically registered to the connectome. To ask whether the alignment depends
on that anatomical registration rather than on the gradient's autocorrelation alone,
$g^{\mathrm{FC}}$ was rotated on the sphere with `SpinPermutations` (Alexander-Bloch et
al., 2018), each hemisphere rotated independently. Every spin preserves the full
(anisotropic) spatial structure of the FC gradient while destroying its alignment to
anatomy. For each spin the per-subject projection was recomputed on the **real**
connectome — only the target axis is rotated — and the per-subject Spearman against the
fixed source gradient $g^{\mathrm{MPC}}$ aggregated across subjects by the Fisher-z mean;
a vertex rotated in from the medial wall carries NaN and drops from both the numerator
and denominator of that spin's projection. Because the rotation removes the gradient's
anatomical alignment, the null is centred on zero, so the empirical $p_{\mathrm{spin}}$
is the two-tailed add-one estimator $p = (1 + k)/(1 + n)$, already invariant to the
arbitrary polarity of $g^{\mathrm{MPC}}$. Being target-side, the spin null applies to
every modality, including geodesic distance, and complements the topological null, whose
within-bin reassignment scrambles target direction but not the target field's structure.

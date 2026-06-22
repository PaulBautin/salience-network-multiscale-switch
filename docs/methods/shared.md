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
rather than on zero, so an observed value in its tail isolates targeting specificity
from distance dependence. The rewiring is applied per subject (the subject is the unit
of inference), the per-subject statistic is recomputed on the rewired connectome and
aggregated across subjects by the Fisher-z mean, and the two-tailed empirical p-value
again uses the add-one estimator $p = (1 + k)/(1 + n)$. The null is defined for any
weight matrix and is therefore applied to the structural, microstructural, and
functional connectivity measures, but not to the geodesic-distance measure, whose
weights are a deterministic function of distance and are left essentially unchanged by
a within-bin reassignment.

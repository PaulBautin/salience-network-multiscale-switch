# Shared Methods

Methods used across multiple figures are documented here. Per-figure pages
reference these sections rather than repeating them.

---

## Microstructural profile covariance gradients

Intracortical microstructure was sampled from quantitative T1 (qT1) maps along 14
equivolumetric surfaces generated between the pial and white-matter boundaries with
micapipe v0.2.0 (Cruces et al., 2022), yielding an intensity profile
$\mathbf{p}_{s,v} \in \mathbb{R}^{14}$ for each participant $s$ and vertex $v$.
Profiles were sampled on the fsLR-32k surface (Figs 1 and 2) and on the fsLR-5k surface
(Fig 3). Networks were defined by the seven-network solution of Yeo and colleagues
(Yeo et al., 2011) as mapped onto the Schaefer 400-parcel atlas (Schaefer et al.,
2018); the salience network (SN) corresponds to the ventral attention network of this
solution. The gradient of a network was computed from the profiles of that network's
vertices alone, $\mathcal{V}_\mathcal{N}$, taken jointly over both hemispheres. The
left-hemisphere analyses of Fig 2 used the left-hemisphere values of this bilateral
gradient.

**Microstructural profile covariance.** For each participant, the mean profile across
the network vertices served as a covariate:

$$\bar{\mathbf{p}}_s = \frac{1}{|\mathcal{V}_\mathcal{N}|}\sum_{v \in \mathcal{V}_\mathcal{N}} \mathbf{p}_{s,v} \in \mathbb{R}^{14}.$$

This covariate differs from the whole-cortex mean profile of the original
microstructural profile covariance (MPC) formulation (Paquola et al., 2019). Removing
the profile shape shared across the network restricts the residual covariance, and
hence the gradient, to microstructural differentiation within the network. Profile
residuals were obtained by ordinary least squares,

$$\mathbf{r}_{s,v} = \mathbf{p}_{s,v} - [1,\, \bar{\mathbf{p}}_s]\,\hat{\boldsymbol{\beta}}_v^{(s)},$$

and the MPC matrix was the Fisher z-transformed Pearson correlation between residual
profiles, with non-finite entries (including the diagonal) set to zero:

$$\text{MPC}^{(s)}_{ij} = \tanh^{-1}\!\left(\text{corr}(\mathbf{r}_{s,i},\, \mathbf{r}_{s,j})\right), \quad i,j \in \mathcal{V}_\mathcal{N}.$$

**Gradient decomposition.** Gradients were derived with BrainSpace (Vos de Wael et
al., 2020). Each participant's MPC matrix was thresholded row-wise to retain the
strongest 10% of entries, converted to an affinity matrix with a normalized angle
kernel, and decomposed by diffusion map embedding (Coifman & Lafon, 2006;
$\alpha = 0.5$, diffusion time 0, 10 components, random seed 0). The embeddings were
aligned across participants by generalized Procrustes analysis (initialized on the
first participant's embedding, with up to 10 iterations aligning every participant to
the current mean), averaged across participants, and the first component was z-scored
across the network vertices:

$$g_v = \text{zscore}\!\left(\frac{1}{N_S}\sum_{s=1}^{N_S} G^{(s,1)}_v\right), \quad v \in \mathcal{V}_\mathcal{N}.$$

The polarity of a diffusion-map eigenvector is arbitrary and was not constrained;
correlations with the gradient are therefore reported with the sign produced by the
embedding.

---

## Spatial null models

Because cortical maps are spatially autocorrelated, correlations between them are
inflated relative to independent samples (Alexander-Bloch et al., 2018; Markello &
Mišić, 2021). Every correlation between maps was therefore assessed against a spatial
null model chosen according to the spatial extent of the test. Whole-cortex maps were
tested with spherical rotations (spin tests). Tests restricted to a network were
tested with Moran spectral randomization, because a rotated network overlaps its
original location only partially. Permutation p-values were computed as

$$p = \frac{1 + k}{1 + n},$$

where $n$ is the number of null values and $k$ the number of null values at least as
extreme as the observed statistic (Phipson & Smyth, 2010). The smallest attainable
value is therefore $1/1{,}001 \approx 0.001$ for 1,000 surrogates. Unless stated
otherwise, tests were two-sided, with
$k = |\{\text{null} : |\text{null}| \geq |\text{obs}|\}|$. The analysis code reports
uncorrected p-values; within each family of tests, defined with each analysis, they were
corrected for multiple comparisons with the Bonferroni procedure. The null models of
each analysis are summarized in the table below.

| Analysis | Null model | Randomized map | Spatial structure | Surrogates | Seed |
|----------|------------|----------------|-------------------|------------|------|
| Fig 1A | Moran | MPC gradient (against the FC gradient) | inverse 1-ring edge length, fsLR-32k, SN vertices | 1,000 | 0 (left), 1 (right) |
| Fig 1B | Moran | histological map | inverse 1-ring edge length, fsLR-32k, SN vertices | 1,000 | 0 (left), 1 (right) |
| Fig 1C | spin | SN mask | fsLR-32k spheres | 1,000 | 0 |
| Fig 2A | Moran | BigBrain gradient 2 | inverse Euclidean distance between channels | 1,000 | 0 |
| Figs 2B and 2C | Moran | MPC gradient | inverse 1-ring edge length, fsLR-32k, left-hemisphere SN vertices | 1,000 | 0 |
| Fig 3 | Moran | within-network MPC gradient | inverse group-mean geodesic distance, fsLR-5k | 1,000 | 42 (left), 43 (right) |
| Fig 3 | topological | each participant's connectome | geodesic-distance bins | 1,000 | 42 |
| Fig 3 | spin (target side) | FC gradient | fsLR-5k spheres | 1,000 | 42 |

### Spin permutation test

Spherical-rotation null models were generated with BrainSpace (Alexander-Bloch et al.,
2018; Vos de Wael et al., 2020). Each permutation applied a uniformly random rotation
to the left-hemisphere sphere and its mirror image, reflected across the midsagittal
plane, to the right-hemisphere sphere, so that homologous locations were rotated
symmetrically. Each rotated vertex took the value of the nearest vertex of the
original sphere. For Fig 1C, the SN mask was rotated on the fsLR-32k spheres and the
cortical-type composition was recomputed within each rotated mask
([Figure 1](figure_1.md#cortical-type-composition-fig-1c)). The target-side spin null
model of Fig 3 is described [below](#target-side-spin-null-model).

### Moran spectral randomization

Tests restricted to a network used Moran spectral randomization (Wagner & Dray, 2015),
as implemented in BrainSpace. The spatial weight matrix among the tested vertices was
doubly centered and eigendecomposed. Its eigenvectors with non-zero eigenvalues (Moran
eigenvector maps) form a basis of spatial patterns at all scales. In the singleton
procedure used here, each surrogate map inverts the sign of the observed map's
correlation with each eigenvector at random. Every surrogate therefore preserves the
magnitude of these correlations, and hence the map's Moran's I, while its spatial
arrangement is randomized. The weight matrix was always computed on exactly the
vertices entering the test (for example, the vertices at which both maps were defined),
so that the surrogates contained only observed values.

Because no spatial relation is defined between the two hemispheres, every Moran null
model was fitted and sampled separately within each hemisphere. For bilateral tests,
the surrogates of the two hemispheres were recombined into the full network map, and
the randomization of each hemisphere was seeded with the analysis seed plus the
hemisphere index (left hemisphere first). Analogous to the hemisphere-wise rotation of
the spin test, this procedure preserves the spatial autocorrelation and the
distribution of values within each hemisphere while retaining both hemispheres in the
statistic. Bilateral tests (Figs 1A, 1B and 3) therefore used two randomizations, and
single-hemisphere tests (Fig 2) used one; a hemisphere with fewer than three tested
vertices would have been permuted at random instead. In Figs 1A, 1B, 2B and 2C, the
spatial weights were the inverse edge lengths between vertices sharing a mesh edge
(1-ring neighbors) on the fsLR-32k midthickness surface. In Fig 3, they were the
inverse group-mean geodesic distances among the network's fsLR-5k vertices; geodesic
distance is undefined between hemispheres, so the two hemispheres of this graph form
its two connected components. In Fig 2A, the weights were the inverse Euclidean
distances between channel positions
([Figure 2](figure_2.md#surface-mapping-of-the-mni-atlas-fig-2a)).

### Geometry-preserving topological null model

The Moran null model randomizes the source map and thereby controls for the spatial
smoothness of the two vertexwise maps, but it leaves the connectome unchanged. A
statistic built from connectivity weights therefore requires a complementary null model
that asks whether the specific pattern of connections, rather than the geometry of the
connectome (degree and connection length), accounts for the effect. This was assessed
with a geometry-preserving topological null model that rewires each participant's
connectome within geodesic-distance bins (Roberts et al., 2016; Betzel et al., 2019;
Váša & Mišić, 2022).

Source–target pairs within a hemisphere were assigned to one of ten equal-width
distance bins. The bin edges were common to all source vertices and spanned the range
of group-mean geodesic distances over all intrahemispheric source–target pairs.
Interhemispheric pairs, for which geodesic distance is undefined, formed an eleventh
bin. Only targets at which the projected map was defined were eligible. For each
surrogate, every connection of every participant was reassigned to a target drawn
uniformly at random, with replacement, from the eligible targets in the same distance
bin of the same source vertex, retaining its weight. This procedure preserves each
source vertex's number of connections, its weights, and their distribution over
distance bins, while randomizing the identity of the targets and hence the values of
the projected map that the connections reach. The total weight of each source vertex,
which forms the denominator of the projection, is unchanged. The per-participant
statistic was recomputed on each rewired connectome and averaged across participants
after Fisher z-transformation (1,000 surrogates, seed 42).

Because connection length is preserved, the null distribution of the group statistic
is centered on the value expected from connectome geometry rather than on zero. The
alternative hypothesis is therefore directional: that the observed alignment exceeds
that produced by geometry alone. It was evaluated on magnitude, because the
within-network gradient is a diffusion-map eigenvector of arbitrary polarity; the
observed statistic and its surrogates share this gradient and therefore have the same
sign. The p-value was computed as $p = (1 + \#\{|\text{null}| \ge |\text{obs}|\})/(1 + n)$,
which is invariant to the sign of the eigenvector. A fixed upper-tail test would fail
whenever the gradient had negative polarity, and a two-sided test around zero would
confound the geometric offset with the effect. This null model applies to any weight
matrix and was used for the structural, microstructural, and functional connectivity
measures. It was not applied to geodesic distance, whose weights are a deterministic
function of distance and are essentially unchanged by reassignment within distance
bins.

For structural connectivity, surrogate projections were obtained by explicit
reassignment of each connection. The microstructural and functional connectivity
matrices are dense, with hundreds to thousands of candidate targets per source vertex.
For these measures, the same null model was sampled analytically. The resampled
numerator of the projection at a source vertex is a sum of independent within-bin
draws, with mean $\mathbb{E}=\sum_b W_b\,\mu_b$ and variance
$\mathrm{Var}=\sum_b (\sum_{e\in b} w_e^2)\,\sigma_b^2$, where $W_b$ is the summed
weight of the connections in distance bin $b$, and $\mu_b$ and $\sigma_b^2$ are the
mean and population variance of the projected map over that bin's eligible targets.
Given the large number of independent terms, the numerator is approximately normal by
the central limit theorem, and each surrogate was drawn from the corresponding normal
distribution. Both sampling schemes have the same mean and variance at each vertex.

### Target-side spin null model

The Moran null model randomizes the source map and the topological null model rewires
the connectome; a third null model acts on the target map. The target axis of the
projection is the principal functional connectivity (FC) gradient $g^{\mathrm{FC}}$,
which is strongly spatially autocorrelated and anatomically aligned with the
connectome. This null model asks whether the observed alignment depends on that
anatomical correspondence rather than on the autocorrelation of the gradient alone.
$g^{\mathrm{FC}}$ was rotated on the fsLR-5k spheres as described
[above](#spin-permutation-test) (1,000 rotations, seed 42), and the same set of
rotations was used for all networks and measures. Each rotation preserves the full,
anisotropic spatial structure of the FC gradient while removing its alignment with
anatomy. For each rotation, the per-participant projection was recomputed on the
unaltered connectome, correlated with the fixed source gradient $g^{\mathrm{MPC}}$, and
averaged across participants after Fisher z-transformation. Vertices rotated in from
the medial wall carried no value and were excluded from both the numerator and the
denominator of the projection for that rotation. Because rotation removes the
anatomical alignment of the gradient, the null distribution is centered on zero, and
$p_{\mathrm{spin}}$ was computed as a two-sided permutation p-value, which is invariant
to the polarity of $g^{\mathrm{MPC}}$. Because it acts on the target map, this null
model applies to every modality, including geodesic distance. It complements the
topological null model, whose within-bin reassignment randomizes the targets reached
by each connection but preserves the spatial structure of the target map.

---

## Software

Analyses were performed in Python 3.10 with NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3
and nibabel 5.3.2. Gradients, null models, and surface renderings used BrainSpace
0.1.22 (Vos de Wael et al., 2020) with VTK 9.3.1, spectral parameterization used
specparam 2.0.0rc7 (Donoghue et al., 2020), and figures were produced with matplotlib
3.10.8 and seaborn 0.13.2. MRI derivatives were generated with micapipe v0.2.0 (Cruces
et al., 2022). Maps were resampled from fsLR-32k to fsLR-5k with Connectome Workbench
(Marcus et al., 2011) using adaptive barycentric area resampling.

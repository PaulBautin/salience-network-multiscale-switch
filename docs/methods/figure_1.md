# Figure 1 — Microstructural heterogeneity of the salience network

**Steps:** `pipeline/step1a_mpc_gradient.py` · `pipeline/step1b_histology.py` · `pipeline/step1c_cortical_types.py`
**Modules:** [`salience_switch.microstructure`](../api/microstructure.md) · [`salience_switch.surface`](../api/surface.md) · [`salience_switch.stats`](../api/stats.md)

The microstructural organization of the salience network (SN) was characterized along
three complementary axes: in vivo quantitative T1 (qT1) microstructure (Fig 1A),
postmortem histology (Fig 1B), and cytoarchitectonic type (Fig 1C). In vivo analyses
used the 25 MICA-PNI participants of the study sample, the same participants as the
connectivity analyses of Fig 3 (see [Data Acquisition](datasets.md#mica-pni-dataset)),
and were performed on the fsLR-32k surface (32,492 vertices per hemisphere). The SN was
defined as the ventral attention network of the seven-network solution (Yeo et al.,
2011) mapped onto the Schaefer 400-parcel atlas (Schaefer et al., 2018), and all
analyses included the SN vertices of both hemispheres.

## Microstructural gradient and intracortical profiles (Fig 1A)

Intracortical qT1 intensity was sampled along 14 equivolumetric surfaces between the
pial and white-matter boundaries, yielding a depth profile at every cortical vertex.
Within the SN, the principal axis of microstructural variation was estimated as the
first microstructural profile covariance (MPC) gradient, computed jointly over the SN
vertices of both hemispheres
([Shared Methods](shared.md#microstructural-profile-covariance-gradients)). The first
gradient component was averaged across the 25 participants after Procrustes alignment
and z-scored. The proportion of variance accounted for by this component was
quantified for each participant as its eigenvalue relative to the sum of the ten
computed eigenvalues, $\lambda_1 / \sum_{k=1}^{10} \lambda_k$, and summarized across
participants.

To illustrate how intracortical laminar organization varies along this axis, SN
vertices were partitioned at the quartiles of the gradient, pooled across both
hemispheres:

$$q_{0.25} = \operatorname{quantile}(g_v,\, 0.25), \qquad q_{0.75} = \operatorname{quantile}(g_v,\, 0.75).$$

Vertices with $g_v \leq q_{0.25}$ formed the inferior pole and vertices with
$g_v \geq q_{0.75}$ the superior pole; the intervening 50% of vertices were not used
for this contrast. Mean qT1 profiles across the 14 depths were computed for each pole
and for every SN vertex. For a compact laminar summary, SN vertices were also divided
into five equal-sized bins at the quintiles of the gradient, and qT1 intensity at each
depth was averaged across participants and across the vertices of each bin.

## Relation to the functional connectivity gradient

The SN gradient was compared with the principal functional connectivity (FC) gradient
of the whole cortex, which serves as the target axis of the connectivity analyses
([Figure 3](figure_3.md)). The FC gradient was used at its native fsLR-32k resolution
and oriented with the default-mode network at its low end. The two maps were
correlated across SN vertices with Spearman's rank correlation, and significance was
assessed against the within-network Moran null model
([Shared Methods](shared.md#moran-spectral-randomization)): the MPC gradient was
randomized separately within each hemisphere (inverse 1-ring edge lengths of the
fsLR-32k surface; 1,000 surrogates; seeds 0 and 1) while the FC gradient was held
fixed. For visual comparison, the FC gradient was z-scored within the SN and displayed
on the SN vertices on the same scale as the MPC gradient.

## Histological contextualization (Fig 1B)

The in vivo gradient was compared with three postmortem histological maps sampled on
the fsLR-32k surface, which label cell bodies, nerve fibers and interneurons:

| Map | Source | Measure |
|-----|--------|---------|
| BigBrain | BigBrain (Amunts et al., 2013), 100 µm | inverted cell-body staining intensity |
| Bielschowsky | AHEAD (Alkemade et al., 2022), 200 µm | nerve-fiber staining intensity |
| Parvalbumin | AHEAD (Alkemade et al., 2022), 200 µm | parvalbumin immunoreactivity |

Acquisition and provenance of the histological volumes are described in
[Data Acquisition](datasets.md#bigbrain-dataset). Each volume was sampled as an
intensity profile at 50 cortical depths per fsLR-32k vertex. Profiles were averaged
across depths (BigBrain intensities having been inverted) and z-scored across SN
vertices. Each map was correlated with the MPC gradient across SN vertices with
Spearman's rank correlation, restricted to vertices at which both maps were defined.
Because cortical maps are spatially autocorrelated, significance was assessed against
the within-network Moran null model
([Shared Methods](shared.md#moran-spectral-randomization)) rather than against the
parametric distribution. The null model was fitted on the same vertices, with spatial
weights given by the inverse 1-ring edge lengths of the fsLR-32k surface, and was
fitted and sampled separately within each hemisphere. The histological map was
randomized (1,000 surrogates) while the gradient was held fixed, and the central 95%
of the null correlations (2.5th–97.5th percentiles) is reported alongside each observed
correlation.

## Cortical type composition (Fig 1C)

Von Economo areas were assigned to cortical types following a reanalysis of the
original micrographs of von Economo and Koskinas (García-Cabezas et al., 2020). This
classification was chosen because its criteria are explicitly defined, are applied
consistently across the entire cortex, agree with von Economo's original descriptions,
and are supported by multiple histological samples. Type assignment considered the
development of layer IV, the prominence of the deep (V–VI) and superficial (II–III)
layers, the definition of sublayers, the sharpness of laminar boundaries, and the
presence of large pyramidal neurons in the superficial layers.

The resulting ordinal scale summarizes the degree of laminar differentiation, from the
highly elaborated koniocortex, through the eulaminate types with six clearly
identifiable layers (eulaminate III to I), to the poorly differentiated dysgranular
cortex and the agranular cortex, in which layers are largely indistinct. The von
Economo atlas was mapped onto the fsLR-32k surface, and each vertex was assigned the
type of its area; vertices without a type formed a seventh, unassigned category. The
composition of the SN was expressed as the percentage of its vertices in each
category. Enrichment was evaluated against a spin permutation null model to account
for spatial autocorrelation ([Shared Methods](shared.md#spin-permutation-test)): the SN
mask was rotated on the sphere 1,000 times, and the same percentages were recomputed
over the vertices covered by each rotated mask. This procedure preserves the size and
spatial contiguity of the network while randomizing its position relative to the
cortical types. For each category, a two-sided permutation p-value compared the
observed percentage with the null distribution, both expressed as deviations from the
null mean, so that enrichment and depletion were tested alike. The seven tests formed
one family for the correction for multiple comparisons
([Shared Methods](shared.md#spatial-null-models)).

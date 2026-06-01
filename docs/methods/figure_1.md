# Figure 1 — Microstructural heterogeneity of the salience network

**Scripts:** `scripts/figure_1a_t1map.py` · `scripts/figure_1b_contextualisation.py` · `scripts/figure_1c_cortical_types.py`

Figure 1 characterises the microstructural landscape of the salience network (SN)
along three complementary axes: in vivo quantitative T1 (qT1) microstructure
(panel a), post-mortem histology (panel b), and cytoarchitectural type (panel c).
All in vivo analyses used the MICA-PNI 7 T dataset (see
[Data Acquisition](datasets.md#mica-pni-dataset)) and were performed on the
fsLR-32k surface (32,492 vertices per hemisphere). The SN was defined as the
ventral attention network of the Yeo seven-network solution mapped onto the
Schaefer-400 parcellation.

## Figure 1a — qT1 microstructural gradient and intracortical profiles

Intracortical qT1 intensity was sampled along 14 equivolumetric surfaces
generated between the pial and white-matter boundaries, yielding a depth profile
at every cortical vertex. Within the SN, the principal axis of microstructural
variation was estimated as the first microstructural profile covariance (MPC)
gradient, computed with the shared diffusion-map embedding pipeline described in
[Shared Methods — MPC gradient computation](shared.md#mpc-gradient-computation).
The first gradient component was averaged across subjects after Procrustes
alignment and z-scored.

To visualise how intracortical laminar organisation varies along this axis, SN
vertices were partitioned at the quartiles of the gradient distribution, pooled
across both hemispheres:

$$q_{0.25} = \operatorname{quantile}(g_v,\, 0.25), \qquad q_{0.75} = \operatorname{quantile}(g_v,\, 0.75).$$

Vertices with $g_v \leq q_{0.25}$ defined the inferior pole (lowest quartile) and
vertices with $g_v \geq q_{0.75}$ the superior pole (highest quartile); the
intervening 50 % of vertices were excluded from the pole contrast. Mean qT1
intensity profiles across the 14 equivolumetric depths were then computed for the
inferior pole, the superior pole, and for all individual SN vertices coloured by
their gradient value.

## Figure 1b — Histological contextualisation

The in vivo qT1 gradient was contextualised against four microstructural maps
sampled on the fsLR-32k surface, spanning in vivo and post-mortem contrasts and
both myelin- and cell-sensitive stains:

| Modality | Source | Measure |
|----------|--------|---------|
| qT1 | MICA-PNI 7 T dataset, fsLR-32k | mean qT1 intensity across subjects and depths |
| BigBrain | BigBrain open-access reconstruction (100 µm) | inverted cell-body staining intensity |
| Bielschowsky | AHEAD dataset (200 µm) | nerve-fibre staining intensity |
| Parvalbumin | AHEAD dataset (200 µm) | interneuron staining intensity |

Acquisition and provenance of the histological volumes are described in
[Data Acquisition](datasets.md#bigbrain-dataset). Each map was correlated with the
qT1 gradient vertex-wise within the SN mask using the Spearman rank coefficient.
Because cortical maps carry strong spatial autocorrelation, statistical
significance was assessed against a within-network Moran spectral randomisation
null
([Shared Methods — Moran spectral randomisation](shared.md#moran-spectral-randomisation-within-network))
rather than the parametric distribution.

## Figure 1c — Cortical type composition

Cortical types were assigned to Von Economo areas following a recent reanalysis of
the original Von Economo micrographs. This scheme was adopted because its criteria
are explicitly defined, applied consistently across the entire cortex, consistent
with Von Economo's original descriptions, and supported by multiple histological
samples. Type assignment considered the development of layer IV, the prominence of
the deep (V–VI) and superficial (II–III) laminae, the definition of sublayers, the
sharpness of laminar boundaries, and the presence of large pyramidal neurons in
the superficial layers.

The resulting ordinal scale synopsises the degree of laminar differentiation, from
the high elaboration of koniocortical (granular) areas, through the six clearly
identifiable layers of the eulaminate types (Eu-III to Eu-I), to the poorly
differentiated dysgranular cortex and the agranular cortex in which layers are
effectively absent. The proportion of each cortical type within the SN was
compared with its proportion across the remainder of the cortex, and enrichment
was evaluated against a spin-permutation null
([Shared Methods — Spin-test permutations](shared.md#spin-test-permutations-whole-brain))
to control for spatial autocorrelation.

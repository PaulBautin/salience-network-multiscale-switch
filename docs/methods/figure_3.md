# Figure 3 — Gradient-weighted connectivity projection

**Steps:** `pipeline/step3_connectivity.py` · `pipeline/step3_schematic.py`
**Modules:** [`salience_switch.connectivity`](../api/connectivity.md) · [`salience_switch.stats`](../api/stats.md)

This analysis tested whether the microstructural profile covariance (MPC) gradient
within the salience network (SN) predicts how its vertices connect to the rest of the
cortex, and whether these connections follow the sensory–transmodal axis captured by
the principal functional connectivity (FC) gradient. Fig 3A illustrates the statistic
in one participant. Fig 3B reports the result for the SN across four connectivity
modalities — structural connectivity (SC), geodesic distance (GD), microstructural
profile covariance (MPC), and functional connectivity (FC) — together with the same
test in each of the seven networks of Yeo et al. (2011). SC is derived from diffusion
MRI, a modality independent of the qT1 data from which the gradient is computed, and
was therefore considered the primary measure. GD provides a purely spatial reference.
Because the MPC weights and the gradient derive from the same qT1 profiles, the
MPC-weighted projection is not independent of the gradient and was interpreted
accordingly. Because the target axis is itself an FC gradient, the FC-weighted
projection indicates how closely functional coupling follows the gradient ordering
rather than providing an independent test.

All analyses used the 25 participants of the study sample, with the same participants
for every measure (see [Data Acquisition](datasets.md#mica-pni-dataset)). Analyses were
performed on the fsLR-5k surface (9,684 vertices; 4,842 per hemisphere), at which
whole-brain vertexwise connectivity matrices remain computationally tractable, and
included both hemispheres. The source vertices of a network were its vertices with a
defined MPC gradient; the targets were all cortical vertices outside the network,
excluding the medial wall. The MPC gradient of each network was computed jointly over
both hemispheres from the participants' fsLR-5k qT1 profiles
([Shared Methods](shared.md#microstructural-profile-covariance-gradients)). The FC
gradient $g^{\mathrm{FC}}$ was the principal FC gradient template originally derived by
Vos de Wael et al. (2018) and distributed with BrainSpace on the fsLR-32k surface (Vos
de Wael et al., 2020), a normative group-level map that is independent of the present
participants. The Schaefer parcellation and the
FC gradient were resampled from fsLR-32k to fsLR-5k with Connectome Workbench (Marcus
et al., 2011) using adaptive barycentric area resampling.

## Connectivity-weighted projection

For each source vertex $i$ and participant $s$, the projection score was defined as the
connectivity-weighted mean of the FC gradient across the vertex's targets outside the
network,

$$P^{(s)}_i \;=\; \frac{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}}, \qquad
\mathcal{T}_{i,s} = \{\, j : w_{ij} > 0,\; j \notin \mathcal{V}_\mathcal{N},\; j \neq i \,\},$$

where $\mathcal{V}_\mathcal{N}$ is the source network and $w_{ij}$ the
modality-specific connection weight between vertices $i$ and $j$. The score was left
undefined for vertices with fewer than ten targets at which the FC gradient was
defined. $P^{(s)}_i$ is the expected position of vertex $i$'s targets along the FC
gradient: high values indicate preferential coupling with the task-positive end of the
gradient and low values with the default-mode end. Because the polarity of a
diffusion-map gradient is arbitrary, $g^{\mathrm{FC}}$ was oriented before any
projection was computed: it was negated if the mean of the default-mode network
exceeded the cortical mean, placing the default-mode network at the low end and the
task-positive networks at the high end. The alignment between connectivity and
microstructure was quantified for each participant as Spearman's rank correlation
between the MPC gradient and the projection score across the source vertices with a
defined score (at least ten),
$r_s = \operatorname{Spearman}_{i \in \mathcal{V}_\mathcal{N}}\big(g^{\mathrm{MPC}}_i,\, P^{(s)}_i\big)$.
A positive $r_s$ indicates that vertices at the high end of the MPC gradient couple
preferentially with the task-positive end of the FC gradient, and vertices at the low
end with the default-mode end. The approach is in line with previous work relating
regional connectivity profiles to macroscale cortical gradients
(Vázquez-Rodríguez et al., 2019; Suárez et al., 2020; Park et al., 2021).

The schematic of the statistic (Fig 3A) was drawn from the structural connectivity of
the participant whose $r_s$ was closest to the median across participants, using the
same consensus-filtered, log-scaled weights as the analysis. The example vertex was the
left-hemisphere SN vertex closest to the 5th percentile of the MPC gradient among
vertices visible in the lateral view with at least 20 targets.

## Connectivity modalities

micapipe stores the fsLR-5k matrices as upper triangles; each matrix was symmetrized on
loading. In every modality, the diagonal, connections within the network, and
medial-wall vertices were excluded from the target set.

Structural connectivity was reconstructed from the MICA-MICs diffusion data with
micapipe, using probabilistic tractography (iFOD2, 40 million streamlines; Tournier et
al., 2010) with SIFT2 streamline weighting (Smith et al., 2015), and mapped to fsLR-5k.
Streamline weights were used without the SIFT2 proportionality coefficient ($\mu$),
node-volume scaling, or streamline-length scaling: every group-level statistic is based
on Spearman correlations, which are invariant to monotone rescaling within a
participant, so normalization across participants was not required. Positive weights
were log-transformed to reduce the skew of the SIFT2 weight distribution, as
$\log_{10}(w_{ij}/w_{\min})$, where $w_{\min}$ is the participant's smallest positive
consensus-filtered weight; the weakest connection therefore received a weight of zero
and did not enter the projection. Geodesic-distance weights were the inverse of each
participant's surface geodesic distance, $w_{ij} = 1/\mathrm{GD}_{ij}$, restricted to
intrahemispheric pairs, providing a purely spatial reference for SC. MPC weights were
the vertexwise MPC values (Fisher z-transformed partial correlations) computed by
micapipe at fsLR-5k. FC weights were the vertexwise correlation matrices of the
multi-echo resting-state acquisition of the same 7 T session. For consistency across
modalities, only positive connections were used: negative entries were set to zero on
loading, so that MPC retained only positive partial correlations and FC only positive
correlations, and non-positive connections were excluded from both the numerator and
the denominator of the projection. SC, MPC and FC included interhemispheric
connections.

To mitigate the over-representation of short streamlines in tractography and the
sparsity of vertexwise SC, structural weights were filtered with a
distance-dependent group-consensus mask (Betzel et al., 2019). The mean streamline
length across participants was divided into ten equal-width distance bins, separately
for intrahemispheric and interhemispheric connections. The mask retained as many
connections as the average participant had, distributed over the distance bins in
proportion to the pooled empirical distribution of connection lengths; within each bin,
the connections present in the largest number of participants were retained. The mask
was constructed once across all participants and applied to each participant's
weights, so that inference remained a random-effects analysis with the participant as
the unit of observation and no group-averaged weight entered the statistic.

All four modalities therefore entered the same weighted-mean projection, computed over
positive weights only. Restricting the weights to positive connections keeps the
weighted mean well defined for MPC and FC, so no modality-specific formulation was
required.

## Group inference and null models

Participant-level correlations were Fisher z-transformed (after clipping to
$|r_s| \le 0.999$) and tested against zero with a one-sample t-test across the
$N_S = 25$ participants. The group correlation $\bar r = \tanh(\bar z)$ is reported with
a 95% confidence interval back-transformed from the z scale (Student's t distribution),
together with the t statistic and its parametric p-value. This test quantifies the
consistency of the alignment across participants, with the participant as the unit of
observation. It does not account for the spatial autocorrelation of the two vertexwise
maps and is therefore reported as a measure of reliability rather than as the test of
significance.

Because vertexwise gradients are strongly autocorrelated, two smooth maps can correlate
by chance. Significance was therefore assessed against the within-network Moran null
model (Wagner & Dray, 2015; [Shared Methods](shared.md#moran-spectral-randomization)),
with spatial weights given by the inverse group-mean geodesic distances among the
source vertices. One thousand surrogate MPC gradients preserving the within-network
spatial autocorrelation were generated separately within each hemisphere (seeds 42
and 43). For each surrogate, the participant-level statistic was recomputed and
averaged across participants after Fisher z-transformation, and $p_{\mathrm{moran}}$
was the two-sided permutation p-value of the observed $\bar r$.

The Moran null model controls for the smoothness of the two maps but leaves the
connectome unchanged. It therefore cannot establish that the alignment arises from the
specific pattern of connections rather than from the geometry of the connectome: when
both maps are smooth and connectivity decays with distance, the projection can align
with the gradient through geometry alone. The specificity of the connections was
therefore tested with a geometry-preserving topological null model
([Shared Methods](shared.md#geometry-preserving-topological-null-model)), which rewired
each participant's connections within ten geodesic-distance bins (plus one
interhemispheric bin), preserving each vertex's number of connections, weights, and
distribution of connection lengths while randomizing the targets. The statistic was
recomputed on each rewired connectome and averaged across participants after Fisher
z-transformation (1,000 surrogates, seed 42). Because connection length is preserved,
this null distribution is centered on the value expected from geometry rather than on
zero, and $p_{\mathrm{topo}}$ tests whether the observed alignment exceeds this
expectation in magnitude ($|r_{\mathrm{null}}| \ge |r_{\mathrm{obs}}|$), a comparison
that is invariant to the arbitrary polarity of the gradient. The topological null
model was applied to SC, MPC and FC but not to GD, whose weights are a deterministic
function of distance. Surrogates were obtained by explicit reassignment for SC and by
the equivalent analytical sampling scheme for the dense MPC and FC matrices.

To verify the power and specificity of the topological null model, the SC null of the
SN was additionally applied to two synthetic source maps while the participants' SC
projections were held fixed: the group-mean SC projection itself, aligned with the
connections by construction, which the null model should reject, and the group-mean GD
projection, aligned with geometry only, which it should not reject. The observed value
for each map was computed with the same statistic as above.

A third null model acted on the target map. The FC gradient is itself strongly
autocorrelated and anatomically aligned with the connectome, so a target-side spin
null model ([Shared Methods](shared.md#target-side-spin-null-model)) rotated the FC
gradient on the fsLR-5k spheres (1,000 rotations, seed 42) and recomputed the
projection on the unaltered connectome for each rotation. This null model tests
whether the alignment depends on the anatomical position of the FC gradient beyond its
autocorrelation, and applies to every measure, including GD. Its null distribution is
centered on zero, and $p_{\mathrm{spin}}$ is a two-sided permutation p-value. For each
null model, the p-values of the seven networks within a measure formed one family for
the correction for multiple comparisons ([Shared Methods](shared.md#spatial-null-models)).

## Replication across networks

The analysis was repeated for each of the seven networks and each of the four
connectivity measures, with the MPC gradient computed once per network and shared
across measures. Significance was assessed with the same within-network Moran null
model, fitted separately within each hemisphere. Each connectivity measure was treated
as a separate family of tests: within each measure, the $p_{\mathrm{moran}}$,
$p_{\mathrm{topo}}$ (SC, MPC and FC) and $p_{\mathrm{spin}}$ values of the seven
networks were corrected for multiple comparisons
([Shared Methods](shared.md#spatial-null-models)). Because the polarity of each network's
MPC gradient is arbitrary and was not aligned across networks, the sign of $\bar r$ is
interpretable within a network, across measures that share its gradient, but not
between networks, for which only magnitude and significance are compared.

## Visualization

For display, the group-mean projection was z-scored across the vertices of each
network and shown on a fixed range of ±2 SD, the scale used for the network maps of
Figs 1–3; because the statistic is rank-based, this rescaling does not affect $\bar r$
or the p-values. In Fig 3, the markers of the seven networks are filled where the
uncorrected $p_{\mathrm{moran}}$ is below 0.05, and that $p_{\mathrm{moran}}$ is printed,
as in Figs 1B and 2B; for the SN alone, the observed $\bar r$ is also shown against the
kernel density of its Moran null distribution. All values shown in the figures —
vertexwise gradients and unstandardized projections, group statistics, and full null
distributions — were saved as tables, from which every panel was drawn
([Usage](../usage.md#step-3-connectivity-projection)).

# Figure 2 — Gradient-weighted connectivity projection

**Script:** `scripts/figure_2_distance.py`
**Module:** [`src/connectome_processing`](../api/connectome_processing.md)

Figure 2 tests whether the microstructural (MPC) gradient within the salience
network (SN) predicts how its vertices connect to the rest of the cortex, and
whether that connectivity tracks the whole-brain sensory–transmodal axis given by
the principal functional connectivity (FC) gradient. Panel 2A reports the result
for the SN across three connectivity modalities — structural connectivity (SC),
geodesic distance (GD), and microstructural profile covariance (MPC) — and panel
2B replicates it across all seven Yeo networks. SC, which indexes axonal
connectivity and is independent of the MPC gradient, is the primary modality; the
MPC-weighted variant is reported as a supplement because correlating an
MPC-derived gradient with the FC gradient through MPC weights partly reflects the
shared microstructural backbone of cortical hierarchy (microstructure–function
coupling) rather than network-specific connectivity. All analyses were performed
at fsLR-5k resolution (9,684 vertices: 4,842 per hemisphere), which keeps the
whole-brain connectivity matrices in memory. The anchoring within-network MPC
gradient was computed with the shared diffusion-map pipeline
([Shared Methods — MPC gradient computation](shared.md#mpc-gradient-computation))
applied to fsLR-5k qT1 profiles and Procrustes-aligned across subjects.

## The projection statistic

For each source-network vertex $i$ and subject $s$, the projection score is the
connectivity-weighted mean of the FC gradient across that vertex's extranetwork
targets,

$$P^{(s)}_i \;=\; \frac{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}}, \qquad
\mathcal{T}_{i,s} = \{\, j : w_{ij} > 0,\; j \notin \mathcal{V}_\mathcal{N},\; j \neq i \,\},$$

where $\mathcal{V}_\mathcal{N}$ is the source network and $w_{ij}$ is the
modality-dependent connectivity weight from vertex $i$ to vertex $j$. The score
$P^{(s)}_i$ is the expected FC-gradient position of vertex $i$'s targets: a high
value indicates preferential coupling to the task-positive pole of the FC
gradient, a low value coupling to the default-mode pole. Alignment between
connectivity and microstructure was quantified per subject as the Spearman rank
correlation between the within-network MPC gradient and the projection score
across SN vertices,
$r_s = \operatorname{Spearman}_{i \in \mathcal{V}_\mathcal{N}}\big(g^{\mathrm{MPC}}_i,\, P^{(s)}_i\big)$.
A positive $r_s$ supports the hypothesis that more differentiated vertices (high
$g^{\mathrm{MPC}}$) couple preferentially to task-positive systems while less
differentiated vertices couple to the default-mode network. This is the
preferred-connectivity-profile statistic (Park et al., 2021; Vázquez-Rodríguez et
al., 2019; Suárez et al., 2020) adapted to a within-network source.

## Connectivity weights

Three connectivity modalities were used as weights. micapipe stores the fsLR-5k
matrices in upper-triangular form; each was symmetrised on loading, and the
diagonal, within-network edges, and medial-wall vertices were excluded from the
target set in every modality.

Structural connectivity was taken from micapipe's SIFT2-weighted streamline
reconstruction. The streamline weights were used as produced — without the SIFT2
proportionality constant ($\mu$), inverse-node-volume scaling, or tract-length
scaling — because every group-level statistic reported here is a Spearman rank
correlation and is therefore invariant to per-subject monotone rescaling, which
removes the need for $\mu$-based inter-subject normalisation. Positive weights
were log-transformed to stabilise the heavy-tailed SIFT2 weight distribution.
Geodesic-distance weights were defined as inverse surface distance,
$w_{ij} = 1/\mathrm{GD}_{ij}$, and restricted to within-hemisphere edges, giving a
spatial-proximity reading that complements SC. MPC weights were the vertexwise
Fisher-z partial correlations.

To counter the over-representation of short streamlines in tractography and the
sparsity of fsLR-5k SC, the SC weights were filtered with a distance-stratified
group-consensus mask (Betzel et al., 2018). Across-subject mean tract length was
binned into ten distance bins, with within- and between-hemisphere edges treated
separately; within each bin the most cross-subject-consistent edges were retained
in numbers matching the empirical edge-length distribution, yielding a binary
consensus mask. The mask was constructed once across all subjects and applied as a
filter to each subject's weights, so inference remained a per-subject
random-effects analysis with the subject as the unit of inference; no
group-averaged weight entered the statistic.

Because MPC partial correlations can take negative values, the weighted mean is
ill-defined for that modality, and a rank formulation was used instead. For each
SN vertex the Spearman correlation between its MPC profile and the FC gradient was
computed across extranetwork targets,
$r^{(s)}_i = \operatorname{Spearman}_{j}\big(\mathrm{MPC}^{(s)}_{ij},\, g^{\mathrm{FC}}_j\big)$,
and this per-vertex value was then correlated with $g^{\mathrm{MPC}}$ as above,
preserving the directional interpretation without relying on the sign of the
weights.

## Gradient orientation

Diffusion-map eigenvectors are sign-arbitrary, so the polarity of the
within-network MPC gradient was fixed deterministically against an external
microstructural reference, independent of the FC test. The gradient was oriented
so that it correlated positively with mean qT1 intensity (`acq-T1map`, averaged
across subjects and the 14 intracortical depths). Because qT1 rises from
myelinated sensory and granular cortex toward agranular transmodal cortex, this
convention fixes high $g^{\mathrm{MPC}}$ as transmodal and low as
sensory/granular. The FC gradient was likewise sign-fixed so that its
transmodal/default-mode pole is high. The resulting signed correlation is directly
interpretable, while statistical significance was assessed two-tailed.

## Group inference and spatial null

Per-subject correlations were transformed to Fisher z and tested against zero with
a one-sample t-test across the $N_S = 18$ subjects. The group correlation
$\bar r = \tanh(\bar z)$ is reported with a 95 % confidence interval
back-transformed from the z-scale, together with the t statistic and parametric p
value.

Because vertexwise gradients are strongly spatially autocorrelated, the parametric
p value overstates significance and was confirmed against a spin-permutation null
([Shared Methods — Spin-test permutations](shared.md#spin-test-permutations-whole-brain);
Alexander-Bloch et al., 2018). The MPC gradient was embedded in the full
9,684-vertex fsLR-5k space (NaN outside the source network) and rotated 1,000
times; for each rotation the per-subject statistic was recomputed over the spatial
overlap of the rotated and original network footprints and aggregated across
subjects by the Fisher-z mean, giving a two-tailed empirical $p_{\mathrm{spin}}$.
A within-network Moran spectral randomisation null (Wagner & Dray, 2015), which
preserves the empirical spatial autocorrelation of the gradient and restricts the
null entirely to the source network, is also available as a tighter,
footprint-matched alternative.

As a confound check, each subject's projection score was regressed on the weighted
mean geodesic distance to its targets and its weighted degree, and the residual
was re-correlated with the MPC gradient; the partial correlation was aggregated
with the same Fisher-z one-sample t-test. This step was skipped for the
degree-invariant MPC rank variant. Group-level summaries ($\bar r$, t, p,
$p_{\mathrm{spin}}$, and the partial-correlation values) were written to
`logs/figure_2_distance.log`.

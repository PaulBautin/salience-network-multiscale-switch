# Figure 2 — Gradient-weighted connectivity projection

**Scripts:** `scripts/figure_2_distance.py`
**Module:** [`src/connectome_processing`](../api/connectome_processing.md)

Figure 2 tests whether the microstructural (MPC) gradient within the salience
network predicts how its vertices connect to the rest of the brain, and whether
that connectivity tracks the whole-brain **sensory→transmodal** axis given by the
principal functional-connectivity (FC) gradient. Panel **2A** reports the result
for SalVentAttn across three connectivity modalities (SC, GD, MPC); panel **2B**
replicates it across all seven Yeo networks. SC (axonal connectivity, independent
of the MPC gradient) is the primary modality; the MPC-weighted variant is a
supplement that partly reflects microstructure–function coupling rather than
network-specific connectivity.

All analyses run at **fsLR-5k** resolution (9,684 vertices: 4,842 LH + 4,842 RH)
so the whole-brain connectivity matrices fit in memory. The within-network MPC
gradient is computed with the shared pipeline
([Shared Methods — MPC gradient computation](shared.md#mpc-gradient-computation))
applied to fsLR-5k T1 profiles.

---

## The projection statistic

For each source-network vertex $i$ and subject $s$, the projection score is the
connectivity-weighted mean of the FC gradient across that vertex's targets:

$$P^{(s)}_i \;=\; \frac{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}}, \qquad
\mathcal{T}_{i,s} = \{\, j : w_{ij} > 0,\; j \notin \mathcal{V}_\mathcal{N},\; j \neq i \,\},$$

where $\mathcal{V}_\mathcal{N}$ is the source network and $w_{ij}$ the connectivity
weight. $P^{(s)}_i$ is the expected FC-gradient position of $i$'s targets — high
$P$ means $i$ couples to the **task-positive** pole, low $P$ to the
**default-mode** pole. Alignment with the MPC gradient is the per-subject Spearman
correlation across network vertices,
$r_s = \operatorname{Spearman}_{i}\big(g^{\mathrm{MPC}}_i,\, P^{(s)}_i\big)$; a
positive $r_s$ supports the hypothesis that more differentiated vertices (high
$g^{\mathrm{MPC}}$) couple preferentially to task-positive systems.

This is the standard "preferred connectivity profile" statistic (Park 2021,
*eLife*; Vázquez-Rodríguez 2019, *PNAS*; Suárez 2020, *Trends Cogn Sci*) adapted
to a within-network source.

---

## Connectivity weights

Three modalities are supported. The diagonal, within-network edges, and
medial-wall vertices are always excluded from the target set.

| Modality | Weight $w_{ij}$ | Notes |
|----------|-----------------|-------|
| **SC** | $\log_{10}$ of per-subject SIFT2 streamline weights | filtered by the Betzel consensus mask (below); inter-hemispheric edges retained |
| **GD** | $1/\mathrm{GD}_{ij}$ (geodesic proximity) | within-hemisphere only; a spatial-proximity complement to SC |
| **MPC** | raw partial-correlation (Fisher-z) | rank variant (below), since signed weights make the weighted mean ill-defined |

SC weights are micapipe SIFT2 outputs taken as-is — not scaled by the SIFT2
proportionality constant $\mu$, node volume, or tract length. Every group
statistic is a Spearman rank correlation, which is invariant to per-subject
monotone rescaling, so the absence of $\mu$ does not affect inference.

**Betzel consensus mask.** To counter SIFT2's over-representation of short
streamlines and the sparsity of fsLR-5k SC, a distance-stratified group-consensus
binary mask (Betzel et al. 2018, 10 distance bins, hemispheres handled separately)
is built once across subjects and applied as a filter to each subject's weights.
Inference remains per-subject random-effects; no group-averaged weight is used.

**MPC rank variant.** Because MPC partial-correlations can be negative, the
weighted mean is replaced by a per-vertex Spearman across targets,
$r^{(s)}_i = \operatorname{Spearman}_{j}\big(\mathrm{MPC}^{(s)}_{ij},\, g^{\mathrm{FC}}_j\big)$,
which is then correlated with $g^{\mathrm{MPC}}$ as above.

---

## Gradient orientation

Diffusion-map eigenvectors have arbitrary sign. To support a directional
sensory→transmodal claim, the within-network MPC gradient is flipped so it
correlates positively with mean qT1 intensity (`acq-T1map`) — an external
reference independent of the FC test. Since qT1 rises from sensory/granular toward
transmodal cortex, this fixes high $g^{\mathrm{MPC}}$ = transmodal. The FC gradient
is likewise sign-fixed so its transmodal/DMN pole is high. The signed correlation
is then interpretable, with significance assessed two-tailed.

---

## Group inference

Per-subject correlations are Fisher z-transformed and tested against zero with a
one-sample t-test across the $N_S = 18$ subjects; the group correlation
$\bar r = \tanh(\bar z)$ and back-transformed 95 % CI are reported.

Because vertex-wise gradients are spatially autocorrelated, significance is
confirmed with a spin-permutation null
([Shared Methods — Spin-test permutations](shared.md#spin-test-permutations-whole-brain),
Alexander-Bloch 2018): the MPC gradient is rotated 1,000 times and the per-subject
statistic recomputed, giving a two-tailed empirical $p_{\mathrm{spin}}$. As a
confound check, each subject's $P^{(s)}_i$ is regressed on weighted mean target
distance and weighted degree, and the residual re-correlated with the MPC gradient
(skipped for the degree-invariant MPC rank variant).

Group summaries ($\bar r$, $t$, $p$, $p_{\mathrm{spin}}$, partial-correlation
values) are written to `logs/figure_2_distance.log`.

# Figure 2 — Gradient-weighted connectivity projection

**Script:** `scripts/figure_2_distance.py`
**Module:** [`src/connectome_processing`](../api/connectome_processing.md)

Figure 2 tests whether the microstructural (MPC) gradient within the salience
network (SN) predicts how its vertices connect to the rest of the cortex, and
whether that connectivity tracks the whole-brain sensory–transmodal axis given by
the principal functional connectivity (FC) gradient. Panel 2A reports the result
for the SN across four connectivity modalities — structural connectivity (SC),
geodesic distance (GD), microstructural profile covariance (MPC), and functional
connectivity (FC) — and panel 2B replicates the test across all seven Yeo networks
for each of the same four connectivity measures. SC,
which indexes axonal connectivity and is independent of the MPC gradient, is the
primary modality and the one that fixes the network ordering shared across measures.
The MPC-weighted variant partly reflects the shared microstructural backbone of
cortical hierarchy (microstructure–function coupling) rather than network-specific
connectivity, because it correlates an MPC-derived gradient with the FC gradient
through MPC weights. The FC-weighted variant is likewise a convergence reading
rather than an independent test: because the projection target $g^{\mathrm{FC}}$ is
itself the principal FC gradient, projecting functional coupling onto it measures how
strongly resting-state connectivity recapitulates the gradient ordering. Both are
reported alongside the modality-independent SC and GD measures. All
analyses were performed
at fsLR-5k resolution (9,684 vertices: 4,842 per hemisphere), which keeps the
whole-brain connectivity matrices in memory. The within-network MPC
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
gradient, a low value coupling to the default-mode pole. Because the polarity of
the diffusion-map FC gradient is mathematically arbitrary, $g^{\mathrm{FC}}$ was
oriented from the data so that the default-mode network occupied its low pole, and
the task-positive systems its high pole, before any projection was computed (the
chosen sign is recorded in the run log). Alignment between
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

Four connectivity modalities were used as weights. micapipe stores the fsLR-5k
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
Fisher-z partial correlations. Functional-connectivity weights were the
resting-state vertex × vertex correlation matrices from the same micapipe 7T PNI
session (multi-echo `desc-me_task-rest_bold`). For consistency across modalities,
every weight matrix was restricted to positive connections: negative entries were
clipped on loading, so that MPC retained only positive partial correlations and FC
only positive (co-activating) correlations, with non-positive edges excluded from
both the numerator and denominator of the projection.

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

All four modalities therefore shared the same weighted-mean projection $P^{(s)}_i$
defined above, computed over positive weights only; no modality-specific rank or
absolute-value formulation was required, since restricting to positive connections
keeps the weighted mean well-defined for MPC and FC alike.

## Group inference and spatial null

Per-subject correlations were transformed to Fisher z and tested against zero with
a one-sample t-test across the $N_S = 18$ subjects. The group correlation
$\bar r = \tanh(\bar z)$ is reported with a 95 % confidence interval
back-transformed from the z-scale, together with the t statistic and the parametric
p value. This t-test quantifies the *subject-level reliability* of the mean
alignment — whether $\bar r$ is consistently non-zero across the random sample of
subjects — and treats the subject as the unit of inference. It does **not** account
for the spatial autocorrelation of the two vertexwise maps, so it is reported as a
reliability summary rather than the significance test.

Because vertexwise gradients are strongly spatially autocorrelated, two smooth maps
tend to correlate even by chance; spatial significance was assessed against a
within-network Moran spectral-randomisation null (Wagner & Dray, 2015;
[Shared Methods — Moran spectral randomisation](shared.md#moran-spectral-randomisation-within-network)).
Surrogates of the MPC gradient that preserve its empirical within-network spatial
autocorrelation were generated and, for each surrogate, the per-subject statistic
was recomputed and aggregated across subjects by the Fisher-z mean, giving a
two-tailed empirical $p_{\mathrm{moran}}$ via the add-one estimator
$p = (1 + k)/(1 + n_{\mathrm{perm}})$, with $k$ the number of surrogates whose
$|\,\bar r\,|$ equalled or exceeded the observed value. Because micapipe geodesic
distance is undefined across hemispheres, the inverse-distance spatial-weight graph
of a bilateral source network is disconnected into one component per hemisphere;
surrogates were therefore generated independently within each connected component
and reassembled, preserving within-hemisphere autocorrelation while keeping both
hemispheres in the statistic. A single-hemisphere analysis forms one component.

## Across-network replication (panel 2B)

The test was repeated independently for each of the seven Yeo networks and for each
of the four connectivity measures, computing a within-network MPC gradient (once per
network, shared across measures) and the per-subject projection statistic $r_s$ for
every network–measure combination. Significance was assessed with the same
within-network Moran spectral-randomisation null as panel 2A (per-hemisphere-block
surrogates, add-one empirical $p_{\mathrm{moran}}$), whose footprint-matched
construction is the appropriate per-network test. Each connectivity measure was
treated as its own inferential family: within each measure the seven networks'
$p_{\mathrm{moran}}$ values were corrected with the Benjamini–Hochberg
false-discovery-rate procedure, and the resulting $q$ values are reported alongside
the group estimates.

The replication is summarised as a bubble matrix
(`figure_2b_network_summary_{hemi}.svg`) with one row per network and one column per
connectivity measure (SC, GD, MPC, FC); the four measure columns share the width of
the Figure 2A modality panels so the two figures align when stacked. The networks are
ordered once by the primary measure's (SC) signed group effect and that order is
shared across every measure column. Each cell is a disc whose colour encodes the
group correlation $\bar r$ on a diverging scale and whose area encodes $|\bar r|$,
with FDR significance of the Moran null rendered as the disc's black edge ring and
overprinted stars and the signed value printed beneath. The within-network MPC
gradient is the first diffusion-map eigenvector, whose polarity is mathematically
arbitrary and is not anchored across networks; coefficients are therefore shown
exactly as the projection produces them, so the sign is interpretable *within* a row
(the measures share one $g_{\mathrm{MPC}}$) but not *between* rows, where only the
magnitude — carried by disc area — and significance are compared. A per-measure scatter grid
(`figure_2b_distance_network_{measure}.svg`) is laid out as a matching single row in
the same network order, carrying only the colour-coded network title and shared axes
(per-vertex MPC gradient against that measure's projection $P$).

Group-level summaries ($\bar r$, t, p, $p_{\mathrm{moran}}$, and the FDR
$q_{\mathrm{moran}}$) were written to `logs/figure_2_distance.log`, and the
per-network group statistics and per-subject coefficients were cached per measure to
`data/dataframes/df_2b_network_stats_{measure}_{hemi}.csv` and
`data/dataframes/df_2b_network_subject_r_{measure}_{hemi}.csv` so the summary
can be regenerated without recomputing the nulls. The per-subject coefficients are
keyed by subject identifier (rather than position), so the reduced FC subject set
remains traceable across measures.

# Figure 2 — Gradient-weighted connectivity projection

**Script:** `scripts/figure_2_distance.py`
**Module:** [`src/connectome_processing`](../api/connectome_processing.md)

Figure 2 tests whether the microstructural (MPC) gradient within the salience
network predicts how its vertices connect to the rest of the brain, and whether
that organisation mirrors the whole-brain **sensory→transmodal** axis captured by
the principal functional-connectivity (FC) gradient. All analyses run at
**fsLR-5k** resolution (9,684 vertices: 4,842 LH + 4,842 RH) so the whole-brain
connectivity matrices fit in memory.

- **Figure 2A** — SC, GD, and MPC projections for SalVentAttn.
- **Figure 2B** — projection replicated across all 7 Yeo networks.
  **SC is the primary modality** (axonal connectivity, independent of the MPC
  gradient). **MPC is a supplement**: an MPC-weighted analysis of the MPC
  gradient correlated with FC-G1 partly reflects the shared microstructural
  backbone of cortical hierarchy (microstructure–function coupling) rather than
  network-specific connectivity.

The MPC gradient that anchors the analysis is computed with the shared pipeline in
[Shared Methods — MPC gradient computation](shared.md#mpc-gradient-computation)
applied to fsLR-5k T1 profiles (procrustes-aligned across subjects in
`compute_t1_gradient`). Spatial null testing uses spin permutations described in
[Shared Methods — Spin-test permutations](shared.md#spin-test-permutations-whole-brain).

---

## Statistic — per-network-vertex connectivity-weighted projection

For each source-network vertex $i$ and subject $s$, the projection score is the
**connectivity-weighted mean of the FC gradient across that vertex's targets**:

$$P^{(s)}_i \;=\; \frac{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j \in \mathcal{T}_{i,s}} w_{ij}}, \qquad
\mathcal{T}_{i,s} = \{\, j : w_{ij} > 0,\; j \notin \mathcal{V}_\mathcal{N},\; j \neq i \,\}.$$

$P^{(s)}_i$ is the expected FC-gradient position of vertex $i$'s structural targets
— **high $P$** means $i$ couples preferentially with the **task-positive** end of
the FC gradient, **low $P$** with the **default-mode** end.

Per subject, alignment with the within-network MPC gradient is measured by
Spearman correlation across network vertices:

$$r_s \;=\; \operatorname{Spearman}_{i \in \mathcal{V}_\mathcal{N}}\big(g^{\mathrm{MPC}}_i,\; P^{(s)}_i\big).$$

A **positive $r_s$** supports the directional hypothesis: vertices with more
differentiated myelin profiles (high $g^{\mathrm{MPC}}$) preferentially couple to
task-positive systems; less differentiated vertices couple to DMN.

This is the standard "preferred connectivity profile" formulation (Park 2021,
*eLife*; Vázquez-Rodríguez 2019, *PNAS*; Suárez 2020, *Trends Cogn Sci*) adapted
to a within-network source.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $\mathcal{V}_\mathcal{N}$ | Network vertices at fsLR-5k (e.g. SalVentAttn) |
| $\mathcal{V}_{\text{other}}$ | Non-network cortical vertices (target set) |
| $g^{\mathrm{MPC}}_i$ | MPC gradient at network vertex $i$, oriented so high = transmodal |
| $g^{\mathrm{FC}}_j$ | Whole-brain principal FC gradient at target $j$, sign-fixed so transmodal = high |
| $w_{ij}$ | Modality-dependent connectivity weight ($i \to j$) |
| $P^{(s)}_i$ | Projection score (expected FC-gradient position of $i$'s targets) |
| $r_s$ | Per-subject Spearman across $\mathcal{V}_\mathcal{N}$ |
| $N_S = 18$ | Number of subjects |

---

## Modality routing and weight preprocessing (`src/connectome_processing.prepare_weights`)

Three connectivity types are supported. micapipe stores fsLR-5k matrices as upper
triangular GIFTI files (9,684 × 9,684); they are symmetrised on load
(`load_subject_matrix`).

| Modality | $w_{ij}$ | Inter-hemispheric | Notes |
|----------|---------|-------------------|-------|
| **SC** (SIFT2 streamline weights) | $\log_{10}\!\big((A_s \odot G)_{ij} \,/\, \varepsilon_s\big)$ on positives; NaN elsewhere | retained | $G$ is the Betzel distance-stratified consensus mask (below); $\varepsilon_s$ = smallest non-zero entry of $A_s \odot G$ |
| **GD** (surface geodesic distance) | $1 / \mathrm{GD}_{ij}$ (proximity) | dropped | spatial-proximity reading; complements the SC test |
| **MPC** (vertex partial-correlation, Fisher-z) | raw $\mathrm{MPC}_{ij}$ — **rank variant** (below) | retained | weighted-mean is ill-defined for negative MPC; use rank-Spearman |

In all cases the diagonal, within-network edges ($i, j \in \mathcal{V}_\mathcal{N}$),
and medial-wall vertices are excluded from $\mathcal{T}_{i,s}$.

### Preprocessing of micapipe SIFT2 outputs

The structural connectivity matrices come from micapipe's
[`functions/03_SC.sh`](https://github.com/MICA-MNI/micapipe/blob/master/functions/03_SC.sh),
with these verbatim commands:

```bash
tcksift2 -nthreads "$threads" "$tck" "$fod_wmN" "$weights"
tck2connectome -nthreads "$threads" "$tck" "$nodes" "${sc_file}-connectome.txt" \
               -tck_weights_in "$weights" -quiet -force
tck2connectome -nthreads "$threads" "$tck" "$nodes" "${sc_file}-edgeLengths.txt" \
               -tck_weights_in "$weights" -scale_length -stat_edge mean -quiet -force
```

These commands fix several preprocessing choices that downstream code inherits:

| Choice | What micapipe does | What we do | SOTA rationale |
|---|---|---|---|
| Matrix entries | Raw sum of SIFT2 weights per edge (no `-symmetric`, no `-zero_diagonal`, no `-scale_invnodevol`) | symmetrise via `np.triu(A,1) + A.T`, which fills the lower triangle from the upper and drops the diagonal | matches micapipe's upper-triangular storage; the diagonal is self-loops, not connectivity |
| Inter-subject scaling | `-out_mu` is **not** passed, so the SIFT2 proportionality coefficient $\mu_s$ is not saved | none possible | Smith (MRtrix author) recommends $\mu$ for inter-subject comparison, but micapipe does not write it. Every group statistic we report is a **Spearman rank correlation**, which is invariant to per-subject monotone rescalings, so this gap does not affect our inference |
| Node-volume scaling | `-scale_invnodevol` not applied | not applied | Smith: "Personally I'm not a fan" — scaling by parcel volume changes the hypothesis from "is connectivity different?" to "is connectivity different beyond volume differences?" |
| Tract-length scaling | `-scale_length` is applied **only** to the separate `*edgeLengths.txt` file (mean SIFT2-weighted tract length per edge); SC entries themselves are unscaled | length matrix is used only for the Betzel distance stratification, **not** to divide SC weights | SIFT2's cross-sectional multipliers already correct fiber-density bias; further dividing by length penalises long-range edges without clear theoretical support |
| Distribution skew | none | `log10`/`log1p` of positives | SIFT2 weights are heavy-tailed and right-skewed; log-transform stabilises variance |
| Negative values | not produced by SIFT2 | not handled (SIFT2 weights are non-negative; absent edges are exactly 0, treated as missing in `prepare_weights`) | absent edge ≠ connectivity of zero |

The de-facto field convention with micapipe (Royer et al., 2022) is "no $\mu$, no
`-scale_invnodevol`, no length scaling on SC entries" — our preprocessing matches
this convention exactly, and the rank-based inference removes the $\mu$ concern
that would otherwise apply. The edge-length matrix is consumed solely by the
Betzel consensus to stratify edges by tract length, which is the use case
`-scale_length -stat_edge mean` was designed for.

References — Smith et al., NeuroImage 2015 (SIFT2);
Royer et al., NeuroImage 2022 (micapipe);
Betzel et al., Network Neuroscience 2018 (distance-stratified consensus, already
cited below).

### SC — Betzel distance-stratified mask + per-subject SIFT2 weights

To handle the spatial bias of SIFT2 (short streamlines over-represented) and the
sparsity of fsLR-5k SC, a **Betzel et al. 2018** distance-dependent consensus
binary mask $G$ is built **once** across all $N_S$ subjects:

1. Per-subject SC stacked $\to$ cross-subject mean tract length $D$.
2. Edges binned by distance ($B = 10$ bins, within- vs. between-hemisphere
   handled separately). Within each bin the target edge count matches the
   empirical edge-length distribution and the most cross-subject-consistent
   $C_{ij}$ edges are retained.
3. $G$ is the binary mask of surviving edges.

$G$ acts only as a **filter**: per-subject SIFT2 weights $A_s$ are used as
$A_s \odot G$, then log-transformed. **Inference is per-subject random-effects**
(subject = unit of inference), in line with the analysis spec; no group-averaged
weight is used.

### MPC — rank variant

MPC partial-correlations are not non-negative, so the weighted-mean projection
is ill-defined. Instead, per network vertex $i$:

$$r^{(s)}_i \;=\; \operatorname{Spearman}_{j \in \mathcal{V}_{\text{other}}}\big(\mathrm{MPC}^{(s)}_{ij},\; g^{\mathrm{FC}}_j\big),$$

then $r_s = \operatorname{Spearman}_i\big(g^{\mathrm{MPC}}_i, r^{(s)}_i\big)$.
This preserves the directional interpretation without weight-sign hacks.

---

## Gradient orientation anchoring

Diffusion-map eigenvectors have no canonical sign, so the raw MPC gradient pole
is arbitrary. To support a *directional* sensory→transmodal claim, the
within-network gradient is oriented deterministically against an external
microstructural reference, **independent of the FC test**: it is flipped so it
correlates positively with mean qT1 intensity (`acq-T1map`, averaged over
subjects and the 14 intracortical depths). qT1 rises from myelinated
sensory/granular cortex toward agranular transmodal cortex, so this fixes
**high $g^{\mathrm{MPC}}$ = transmodal**, **low = sensory/granular**. The FC
gradient is likewise sign-fixed so the transmodal/DMN pole is high. The signed
correlation is interpretable; significance is still assessed two-tailed.

---

## Group inference

Two-stage random-effects test:

1. **Fisher z-transform** each subject's correlation:
   $z_s = \operatorname{arctanh} r_s$.
2. **One-sample t-test** of $\{z_s\}$ against zero across the $N_S$ subjects
   (`scipy.stats.ttest_1samp`). Report group correlation
   $\bar r = \tanh(\bar z)$, 95 % CI back-transformed from
   $\bar z \pm t_{0.975, N_S - 1} \, \mathrm{SE}(z)$, $t$ statistic, and
   parametric $p$.

### Spatial null — spin permutations (Alexander-Bloch 2018)

Vertex-wise gradients carry strong spatial autocorrelation, so the parametric
$p$ over-states significance. The spin null:

- Embeds $g^{\mathrm{MPC}}$ in the full 9,684-vertex fsLR-5k space (NaN outside
  the source network), then rotates with the fitted `SpinPermutations` model
  ($n_{\mathrm{rep}} = 1000$).
- Per permutation $k$, per subject $s$:
  $r^{(s)}_{k,\mathrm{null}} = \operatorname{Spearman}\!\big(g^{\mathrm{MPC}}_{\mathrm{rot},k},\, P^{(s)}\big)$
  over the spatial overlap of the rotated and original network footprints.
- Per permutation aggregate via Fisher-z mean across subjects $\to$
  $\bar r_{k,\mathrm{null}}$.
- Two-tailed empirical $p_{\mathrm{spin}} = \mathrm{mean}\big(|\bar r_{\mathrm{null}}| \geq |\bar r_{\mathrm{obs}}|\big)$.

### Confound control — partial correlation

Because SIFT2 is not length-corrected, $P^{(s)}_i$ can be biased toward nearby
targets that share FC-gradient values (spatial autocorrelation). Per subject,
$P^{(s)}_i$ is regressed on per-vertex covariates
$[\mathrm{meanGD}_i,\,\mathrm{degree}_i,\,1]$ where

$$\mathrm{meanGD}^{(s)}_i = \frac{\sum_j w_{ij} \mathrm{GD}_{ij}}{\sum_j w_{ij}},\qquad
\mathrm{degree}^{(s)}_i = \sum_j w_{ij},$$

and the residual is correlated with $g^{\mathrm{MPC}}$ (Spearman). The
partial-correlation $r_s^{\mathrm{partial}}$ is then aggregated with the same
Fisher-z + t-test as the primary stat. Reported as $r_{\mathrm{partial}}$ /
$p_{\mathrm{partial}}$ in the log. Skipped for MPC (rank statistic is
degree-invariant).

The functions implementing each step above are documented in the API reference:
[`src/connectome_processing`](../api/connectome_processing.md).

---

## Output

| File | Content |
|------|---------|
| `results/figures/figure_2a_distance_metric.svg` | Fig 2A — SalVentAttn × {SC, GD, MPC}; top row: $g^{\mathrm{MPC}} \times P$ scatter; bottom row: per-subject $r_s$ + mean ± 95 % CI |
| `results/figures/figure_2a_brain_{SC,GD,MPC}_rho.svg` | Per-modality group-mean $P$ map over SalVentAttn (NaN elsewhere) |
| `results/figures/figure_2b_distance_network_{SC,MPC}.svg` | Fig 2B — replication across 7 Yeo networks |
| `results/figures/figure_2b_brain_{measure}_rho_{network}.svg` | Per-network group-mean $P$ map |
| `data/dataframes/df_2b_label_{hemi}.csv` | Vertex-level cache: `mics, hemisphere, network, label, fc_g1, fc_g1_network, network_int, t1_gradient1_{N}, {N}_{M}_P` |

Group-level summary numbers ($r_{\mathrm{group}}$, $t$, $p$, $p_{\mathrm{spin}}$,
$r_{\mathrm{partial}}$, $p_{\mathrm{partial}}$, $n$) are written to the run log
in `logs/figure_2_distance.log`, not duplicated in the CSV.

---

## Method history

Earlier revisions of this figure used a **per-target Spearman** (rank
correlation across in-network vertices, one value per non-network target,
z-scored map correlated with FC-G1 by spin test). That statistic was binning-free
and stable but inverted the natural reading direction. The current per-SN-vertex
projection is the standard preferred-connectivity-profile statistic used in the
gradient/structure-function-coupling literature; it gives an interpretable
per-vertex score in g_FC units and a clean random-effects test (subject is the
unit of inference, in line with the project analysis spec).

A prior "centroid" variant computed the gradient-weighted mean of
$g^{\mathrm{MPC}}$ at each target (the opposite direction) and was rejected
because the weighted mean collapsed to $\approx 0$ whenever weights did not
covary with $g^{\mathrm{MPC}}$ across SN vertices. The current statistic is in
the opposite direction (averages $g^{\mathrm{FC}}$ across targets per SN vertex
weighted by SC), uses an independent weighting variable, and does not share the
collapse failure mode.

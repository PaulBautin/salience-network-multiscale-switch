# Figure 1 — Microstructural heterogeneity of the salience network

**Scripts:** `scripts/figure_1a_t1map.py` · `scripts/figure_1b_contextualisation.py` · `scripts/figure_1c_cortical_types.py`

The three panels of Figure 1 characterise the microstructural landscape of the
salience network (SN) using in-vivo qT1 MRI (panel a), post-mortem histology
(panel b), and cytoarchitectural typing (panel c). All in-vivo analyses use the
MICA-PNI 7 T dataset at fsLR-32k resolution (32,492 vertices per hemisphere).

---

## Figure 1a — T1 microstructural gradient and intracortical profiles

The MPC gradient is computed using the shared pipeline described in
[Shared Methods — MPC gradient computation](shared.md#mpc-gradient-computation).
Inputs are qT1 intensity profiles at fsLR-32k.

### Gradient extreme identification

To visualize differences in intracortical laminar organization across the
gradient, network vertices are split into two poles. Quantile thresholds are
computed across all SalVentAttn vertices (both hemispheres pooled):

$$q_{0.25} = \text{quantile}(g_v,\, 0.25), \qquad q_{0.75} = \text{quantile}(g_v,\, 0.75)$$

Vertices with $g_v \leq q_{0.25}$ form the **low pole** (bottom 25%); vertices
with $g_v \geq q_{0.75}$ form the **high pole** (top 25%); the middle 50% are
excluded from pole comparisons.

Mean intracortical intensity profiles are then plotted across the 14 equivolumetric
depths for the low-pole vertices, the high-pole vertices, and all individual
network vertices (colour-coded by gradient value).

---

## Figure 1b — Histological contextualisation

The in-vivo T1 gradient is compared with three post-mortem histological modalities
mapped to the fsLR-32k surface:

| Modality | Source | Measure |
|----------|--------|---------|
| BigBrain | BigBrain open-access (100 µm) | Inverted cell-body staining intensity |
| Bielschowsky | AHEAD dataset (200 µm) | Nerve-fibre staining intensity |
| Parvalbumin | AHEAD dataset (200 µm) | Interneuron staining intensity |

Vertex-wise Spearman correlations between the T1 gradient and each histological
map are computed within the SalVentAttn network mask. Statistical significance is
assessed with Moran spectral randomisation restricted to network vertices (see
[Shared Methods — Spatial statistics](shared.md#moran-spectral-randomisation-within-network)).

---

## Figure 1c — Cortical type composition

Cortical types were assigned to Von Economo areas based on a recent reanalysis of
Von Economo micrographs. This classification scheme was used because its criteria
are (1) clearly defined, (2) applied consistently across the entire cortex,
(3) align with Von Economo's original descriptions and (4) are supported by
several histological samples. Criteria included development of layer IV,
prominence of deep (V–VI) or superficial (II–III) layers, definition of
sublayers, sharpness of layer boundaries, and presence of large pyramids in
superficial layers.

Cortical types synopsise degree of granularity from high laminar elaboration in
koniocortical (granular) areas, through six identifiable layers in Eu-III to
Eu-I, to poorly differentiated layers in dysgranular and absent layers in
agranular cortex. The proportion of each type within SalVentAttn is compared
with its distribution across the rest of the cortex.

# `src/connectome_processing`

Connectome I/O and the gradient-weighted connectivity projection used in
[Figure 2](../methods/figure_2.md). All matrices are handled at **fsLR-5k**
resolution (9,684 vertices). micapipe stores fsLR-5k connectomes as upper
triangular GIFTI files; they are symmetrised on load.

The module implements the per-network-vertex projection statistic (Park 2021,
*eLife*; Vázquez-Rodríguez 2019, *PNAS*; Suárez 2020, *Trends Cogn Sci*): for
each source-network vertex $i$,

$$P_i \;=\; \frac{\sum_{j \in \mathcal{T}_i} w_{ij}\, g^{\mathrm{FC}}_j}{\sum_{j \in \mathcal{T}_i} w_{ij}}, \qquad
\mathcal{T}_i = \{\, j : w_{ij} > 0,\; j \notin \mathcal{V}_\mathcal{N},\; j \neq i \,\},$$

with per-subject inference (Fisher-z then one-sample t-test across subjects), a
spin-test null (Alexander-Bloch 2018) and a Moran spectral-randomisation null,
plus partial-correlation confound control (mean weighted distance to targets,
weighted degree). See [Figure 2 Methods](../methods/figure_2.md) for the full
statistical derivation.

## Functions

| Function | Role |
|----------|------|
| `load_subject_matrix(path, cortex_mask)` | Generic fsLR-5k GIFTI loader; symmetrises the upper triangle, clips negatives, restricts to cortex |
| `fcn_group_bins(adj, dist, hemiid, nbins)` | Betzel distance-stratified group-consensus thresholding; returns distance- and consistency-based binary masks |
| `build_consensus_mask(sc_files, dist_files, df, nbins=10)` | Builds the Betzel consensus mask once across subjects **and** returns the per-subject SC matrices |
| `prepare_weights(W_raw, modality, hemi_cortex, sn_mask_cortex, mask_G=None)` | Modality-aware preprocessing (`SC` / `GD` / `MPC`) → weight matrix with NaN exclusions (diagonal, within-network, modality-specific) |
| `compute_projection_score(W, g_fc, sn_idx, other_idx, min_valid=10)` | Vectorised weighted-mean projection $P_i$ (SC, GD) |
| `compute_projection_score_rank(W, g_fc, sn_idx, other_idx, min_valid=10)` | Per-row Spearman across targets (MPC variant, weight-sign safe) |
| `compute_projection_subjects(...)` | Per-subject loop + Fisher-z group aggregation for one modality/network; returns `r_subjects`, `r_group`, `t`, `p`, CI, plus per-subject `mean_GD`, `degree`, and target-network weights |
| `compute_partial_correlation_subjects(result, g_mpc)` | Regress $P_s$ on `[mean_GD, degree, 1]`; correlate residuals with $g^{\mathrm{MPC}}$ (skipped for the MPC rank variant) |
| `compute_spin_null_projection(g_mpc, sn, cortex_mask, result, spin_model, n_rand)` | Cortex-wide spin null + two-tailed empirical $p_{\mathrm{spin}}$ |
| `compute_moran_null_projection(g_mpc, sn_mask, gd_among_sn, result, n_rand)` | Within-network Moran spectral-randomisation null (tighter, SAC-matched alternative to the spin test) |
| `compute_dominant_target_network(result)` | Per SN vertex, the target network with the highest group-mean weighted connectivity |
| `benjamini_hochberg(pvals)` | Benjamini–Hochberg FDR correction across networks/modalities |
| `compute_tertile_contrast(P_mean, g_mpc)` | Supplementary tertile summary for visualisation |

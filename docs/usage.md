# Usage

All scripts are run from the **project root**. Each script accepts `-h` for the full flag reference.

Scripts cache intermediate results as TSV/CSV in `data/dataframes/` and skip recomputation on subsequent runs. Outputs are written to `results/figures/` as SVG/PNG. Run logs are saved to `logs/`.

## Scripts

| Script | Figure panel | Description |
|--------|-------------|-------------|
| [`figure_1a_t1map.py`](#figure-1a) | 1a | T1 microstructural gradient within the salience network |
| [`figure_1b_contextualisation.py`](#figure-1b) | 1b | Multi-modal contextualization of the T1 gradient |
| [`figure_1c_cortical_types.py`](#figure-1c) | 1c | Von Economo cortical type distribution |
| [`figure_2_distance.py`](#figure-2) | 2 | Connectivity-weighted MPC gradient projection correlated with the FC gradient |
| [`figure_3_ieeg_mni.py`](#figure-3-mni) | 3 | iEEG spectral analysis — MNI open atlas |
| [`figure_3_ieeg_mica.py`](#figure-3-mica) | 3b | iEEG spectral analysis — MICA dataset |
| [`figure_3c_ieeg_states.py`](#figure-3c-states) | 3c | iEEG aperiodic exponent across wakefulness, N2 and N3 sleep — MICA dataset |

---

## Figure 1a

**`scripts/figure_1a_t1map.py`** — Processes MICA-PNI micapipe derivatives to extract T1 intensity profiles, compute MPC gradients within the salience network, and visualize the relationship between T1 profiles and gradient values.

```bash
python scripts/figure_1a_t1map.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -mics_deriv /path/to/BIDS_MICs/derivatives/micapipe_v0.2.0 \
  -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-pni_deriv` | yes | — | Path to PNI micapipe derivatives directory |
| `-mics_deriv` | yes | — | Path to MICs micapipe derivatives directory |
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |

**Outputs**

- `results/figures/figure_1a_profiles.svg`
- `results/figures/figure_1a_strata.svg` — stratigraphy columns: mean qT1 at each of the 14 depths for each gradient quintile (one block per depth, coloured by raw qT1 on a linear grey scale as a qT1 map is displayed (black = lowest, white = highest plotted value); outlined columns, caps below them in the quintile's gradient colour). Same 6-in canvas, font and axes box as `figure_1a_profiles.svg`, so pial and WM sit at the same heights in both.
- `results/figures/figure_1a_brain.svg`
- `results/figures/figure_1a_brain_zoom.svg` (brain row with circular zoom insets of the `SALIENCE_ZOOM_ROIS` regions, coolwarm on the shared ±2 SD `SALIENCE_Z_RANGE`, SN border traced in black (`mask_outline`) on the brains and insets; its top 300 pt overlay `figure_1a_brain.svg` exactly, and insets are drawn for the `-hemi` regions only) and `figure_1a_brain_zoom_roi.svg` (the inset row alone, same 1450-pt width and columns, to stack 300 pt below `figure_1a_brain.svg`)
- `data/dataframes/figure_1a_pni_to_mics{,_5k}.csv` (subject → file-path tables)
- `data/dataframes/df_1a_<hemi>.tsv`, `df_1a_<hemi>_fslr5k.tsv` (surface table + gradient cache)

---

## Figure 1b

**`scripts/figure_1b_contextualisation.py`** — Correlates the T1 gradient with BigBrain, AHEAD Bielschowsky, and AHEAD Parvalbumin histological profiles, with significance assessed against a within-network Moran spectral-randomisation null (the correlation is restricted to salience-network vertices).

```bash
python scripts/figure_1b_contextualisation.py -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |

> No external derivatives are required. All histological profiles are bundled in `data/parcellations/`.

**Outputs**

- `results/figures/figure_1b_forest.svg` — context comparison (`plot_forest`): per histological map (BigBrain, Bielschowsky, Parvalbumin), the observed Spearman r (black dot, filled if the Moran p < 0.05) against the central 95% of its 1,000 Moran-null r's (grey bar, tick at the null median), on one signed r axis with r and the Moran p (labelled $p_{m}$) written above each bar. Drawn at its final size: its rows are level with the insets of the three `figure_1b_zoom_*.svg` rows stacked in that order and shown 150 mm wide (`ZOOM_ROW_WIDTH_MM`), with the forest's top aligned with the first zoom row, and its width and x-axis span copy the Figure 1a profiles panel as laid out in Figure 1, so placed at the profiles panel's x both x axes line up.
- `results/figures/figure_1b_correlations_supp.svg` — supplementary vertex scatter (gradient vs map, fit line, r / p) and Spearman |r| stem per map
- `results/figures/figure_1b_correlations_circle.svg` — half-polar bar plot of |r| per map (blue positive, red negative)
- `results/figures/figure_1b_brain_{bigbrain,t1map,biel,parva}.svg` — `plot_hemispheres` screenshots of each map's raw values on the SN (coolwarm)
- `data/dataframes/df_1b_context_stats.csv` — per map: `r`, `p`, `null_lo` / `null_med` / `null_hi` (2.5th / 50th / 97.5th percentile of the null r's), `n_vertices`, `n_perm`
- `results/figures/figure_1b_zoom_{bigbrain,biel,parva}.svg` — region zoom-ins of each histological map (inset row only, no brains, no region names), titled with the map name: the `SALIENCE_ZOOM_ROIS` regions of the `-hemi` hemisphere(s), on the same 1450-pt window and columns as `figure_1a_brain_zoom*.svg`. The maps are z-scored within the SN like the MPC gradient and share its fixed ±2 SD range (`SALIENCE_Z_RANGE`, colour bar labelled −2 / +2, z), drawn in `CustomCmap_histology` (stain colours: violet below / ochre above the SN mean): same lightness profile as the gradient's coolwarm, so a given z reads with the same weight, with different hues so histology and gradient stay distinguishable. The SN border is traced in black (`mask_outline`), as in Figures 1a and 1c

---

## Figure 1c

**`scripts/figure_1c_cortical_types.py`** — Maps Von Economo cortical types onto the salience network and tests for non-random type distributions using spin permutations.

```bash
python scripts/figure_1c_cortical_types.py -hemi both
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` (selects `df_1a_<hemi>.tsv` and the network mask) |

Requires `data/dataframes/df_1a_<hemi>.tsv` (run `figure_1a_t1map.py` with the matching `-hemi` first).

**Outputs**

- `results/figures/figure_1c_brain_economo.svg`
- `results/figures/figure_1c_brain_economo_zoom.svg` (brain row with circular zoom insets of the `SALIENCE_ZOOM_ROIS` regions, as Figure 1a; cortical types on the whole cortex with the SN border traced in black (`mask_outline`), where `figure_1c_brain_economo.svg` marks it with black border vertices; its top 300 pt overlay that file; the colour bar shows the cortical-type colours as blocks without end labels, as in that file) and `figure_1c_brain_economo_zoom_roi.svg` (the inset row alone, to stack 300 pt below `figure_1c_brain_economo.svg`)
- `results/figures/figure_1c_type_salience.svg`

---

## Figure 2

**`scripts/figure_2_distance.py`** — Computes the connectivity-weighted projection of the whole-brain FC gradient across each source-network vertex's extranetwork targets, then correlates it per subject with the within-network MPC gradient (Spearman). Every modality uses only positive connections via the same weighted-mean projection. Panel 2A follows the Figure 1/3 display convention: for the salience network, each modality (SC, GD, MPC, FC) is a brain row of its z-scored projection P on both hemispheres with the aINS/dACC/PCU/IPL zoom insets and the SN border traced (`CustomCmap_connectivity`, ±2 SD scale), accompanied by a forest plot (one row per modality: salience-network group _r_ vs its Moran null 95 % band, level with the stacked inset rows). The per-measure vertex scatters of the projection against the within-network MPC gradient move to a supplement. Significance uses two complementary nulls: a within-network Moran spectral-randomisation null (map smoothness; all measures) and a geometry-preserving topological null that rewires the connectome within geodesic-distance bins (wiring specificity; SC/MPC/FC), each Benjamini–Hochberg-corrected across networks per measure. Figure 2B replicates the projection for all four measures across all 7 Yeo networks and summarises it as a bubble matrix (rows = networks, columns = measures; disc colour = group _r_, area = |_r_|, black ring + stars = FDR-corrected significance). The per-network projection is the single computation behind both panels, so the panel 2A forest and the panel 2B bubble matrix reuse the same numbers. All analysis runs at fsLR-5k resolution (9,684 vertices).

Requires `figure_1a_t1map.py` to have been run first: it reads the subject → file table `data/dataframes/figure_1a_pni_to_mics_5k.csv` (required) and reuses the cached fsLR-5k MPC gradient from `data/dataframes/df_1a_<hemi>_fslr5k.tsv` when present, recomputing it in-figure otherwise. The `df_1a_*.tsv` caches are tab-separated.

```bash
# full run: compute the projection + nulls, then draw every figure
python scripts/figure_2_distance.py -hemi LH
# iterate on figure aesthetics without recomputing (loads caches, seconds):
python scripts/figure_2_distance.py -hemi LH -stage plot
# redraw only Figure 2A from cache:
python scripts/figure_2_distance.py -hemi LH -stage plot -panel 2a
```

The expensive projection + Moran nulls are separated from drawing by `-stage`. `both` (default) and `compute` both run the computation over all 7 networks, write every figure-data cache below, **and draw the figures** (so a fresh compute always refreshes the figures); `plot` skips all heavy loads and redraws figures from those caches in seconds (so you can iterate on layout/colours without rerunning anything). `-stage plot` requires a prior `compute`/`both` run for the same `-hemi`. `-panel` then selects which figures are rendered.

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-hemi` | no | `both` | Hemisphere: `both`, `LH`, or `RH` |
| `-panel` | no | `both` | Figures to render: `both`, `2a`, or `2b` |
| `-stage` | no | `both` | Pipeline stage: `both`/`compute` (compute + write caches + draw), or `plot` (redraw from cache only) |
| `-n_rand` | no | `1000` | Surrogates for the Moran and topological nulls. Lower (e.g. `300`) for faster iteration; the add-one empirical _p_ floor is 1/(1+`n_rand`), so keep `1000` for the final run. |

**Outputs**

- `results/figures/figure_2a_{SC,GD,MPC,FC}_zoom_<hemi>.svg` (Figure-1/3-style brain row of the z-scored salience-network projection P on both hemispheres with the aINS/dACC/PCU/IPL zoom insets and the SN border traced, in `CustomCmap_connectivity` on the ±2 SD scale)
- `results/figures/figure_2a_{SC,GD,MPC,FC}_zoom_roi_<hemi>.svg` (the titled inset row alone for each modality)
- `results/figures/figure_2a_forest_<hemi>.svg` (one row per modality: salience-network group _r_ vs its Moran null 95 % band, level with the stacked inset rows)
- `results/figures/figure_2a_scatter_supp_<hemi>.svg` (supplement: per-measure vertex scatter of the projection P against the within-network MPC gradient, points coloured by dominant target network)
- `results/figures/figure_2b_distance_network_{SC,GD,MPC,FC}.svg` (per-measure scatter grid)
- `results/figures/figure_2b_network_summary_<hemi>.svg` (bubble matrix, all measures × networks)
- `results/figures/figure_2b_brain_{SC,GD,MPC,FC}_rho_<network>.svg`
- `results/figures/figure_2_supp_topo_control.svg` (topological-null power/specificity control: SC null distributions for a wiring-aligned and a geometry-only synthetic map, written in the compute stage)
- `data/dataframes/df_2b_label_<hemi>.csv` (vertex cache; `{network}_{measure}_P` projection columns — z-scored per network×measure for display — + `{network}_{measure}_dominant` dominant-target-network columns that drive the scatter colours on replot)
- `data/dataframes/df_2b_network_stats_<measure>_<hemi>.csv` (per-network group stats, including `p_moran`/`q_moran` and the geometry-preserving topological-null `p_topo`/`q_topo`; `p_topo`/`q_topo` are NaN for the GD measure)
- `data/dataframes/df_2b_network_subject_r_<measure>_<hemi>.csv` (per-subject _r_; row index is the subject ID, one column per network)

---

## Figure 3 MNI

**`scripts/figure_3_ieeg_mni.py`** — Computes the PSD of the MNI open iEEG atlas channels, maps them to fsLR-32k vertices (projected onto the `-hemi` hemisphere), and correlates the electrophysiological-similarity difference between gradient-top and gradient-bottom target-network channels (ES<sub>top</sub> − ES<sub>bottom</sub>) with the BigBrain G2 gradient across the channels outside the target network. Significance is a Moran spectral-randomisation null on the channel layout (inverse Euclidean distances between channel positions, 1,000 surrogates, add-one *p*).

```bash
python scripts/figure_3_ieeg_mni.py \
  -ieeg_deriv /path/to/MNI_ieeg/MatlabFile.mat
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-ieeg_deriv` | yes | — | Path to the MNI iEEG atlas MATLAB file (`.mat`) |
| `-hemi` | no | `RH` | Hemisphere the channels are projected onto: `both`, `LH`, or `RH` |
| `-network` | no | `SalVentAttn` | Yeo 7-network used to stratify channels along the gradient |

Requires `data/dataframes/df_1a_{hemi}.tsv`, or `df_1a_both.tsv` restricted to the hemisphere when the hemisphere's own table is missing (as `figure_2_distance.py`).

**Outputs** — `results/figures/figure_3a_ieeg_mni_*.svg`: the channel scatter (ES difference vs BigBrain G2, r / Moran *p*) with per-network mean bars (`*_corr_scatter.svg`), and channel and smoothed surface maps of the ES difference (`*_channel_corr_diff*.svg`).

---

## Figure 3 MICA

**`scripts/figure_3_ieeg_mica.py`** — Applies the iEEG spectral pipeline to the MICA intracranial EEG dataset using subject-specific leadfield sensitivity maps. Each channel spectrum is parameterised once (`specparam` 2.0, knee mode with the knee bounded to *k* ≥ 0, `SPEC_FIT_RANGE` = 1–45 Hz, below the 60 Hz notch of the recordings and the 80 Hz low-pass transition band) into the aperiodic exponent (primary) and the oscillatory peak power of five canonical bands clipped to the fit range (`SPEC_BANDS`: delta 1–4 Hz … gamma 30–45 Hz; secondary, FDR-corrected), projected onto the surface by sensitivity weighting, and correlated with the salience-network MPC gradient on the network vertices of the `-hemi` hemisphere (left by default, as the rest of the paper). Significance is a within-network Moran spectral-randomisation null with the add-one empirical *p*. Because the sensitivities of both hemispheres are folded onto one fsLR-32k template, the maps are shown on both hemispheres, laid out as the Figure 1 zoom rows. As a supplementary analysis, the spectral similarity between surface vertices (similarity of power spectra, not connectivity) weights a projection of the FC gradient (the Figure 2 gradient-weighted projection, group-level variant), reported with its uniform- and distance-weighted baselines.

```bash
python scripts/figure_3_ieeg_mica.py \
  -ieeg_deriv /path/to/BIDS_iEEG/derivatives/electroMICA \
  -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-ieeg_deriv` | yes | — | Path to electroMICA derivatives directory |
| `-ieeg_raw` | no | `<ieeg_deriv>/../../original` | Root of the original recordings (`<root>/sub-PX*/ses-01/*stage-W.mat`) |
| `-hemi` | no | `LH` | Hemisphere of the statistics: `LH` or `RH` (maps are shown on both hemispheres) |
| `-network` | no | `SalVentAttn` | Yeo 7-network analysis target |

Requires `data/dataframes/df_1a_{hemi}.tsv` (run `figure_1a_t1map.py` with the matching `-hemi` first); as in `figure_2_distance.py`, `df_1a_both.tsv` restricted to the hemisphere is used when the hemisphere's own table is missing.

The per-channel PSD is cached in `data/dataframes/figure_3_channel_psd.pkl` (a few MB, not tracked). The cache is keyed on every recording's path, size and modification time and on the PSD parameters, so it is rebuilt automatically when the raw data change; a rebuild reads the recordings one subject at a time. The previous raw-signal cache, `figure_3_channel_data_df.pkl`, is no longer used and can be deleted.

**Outputs** (`results/figures/figure_3b_ieeg_mica_*`, `{hemi}` = hemisphere of the statistics)

All brain figures use the Figure 1 zoom-row window (1450 × 300, zoom 1.3, colour-bar column; region names at 30 pt, inset-row titles at 36 pt) with the four `SALIENCE_ZOOM_ROIS` regions. Each map comes as `*_zoom_{hemi}.svg` (brain row, marker circles, insets and region names, as `figure_1a_brain_zoom.svg`) and `*_zoom_roi_{hemi}.svg` (inset row alone, titled, colour bar drawn, no region names, as the `figure_1b_zoom_*` rows). The spectral maps show each template value on the network vertices of both hemispheres, z-scored there, on the shared ±2 SD range (`SALIENCE_Z_RANGE`, colour bar −2 / +2, z) in `CustomCmap_ieeg` (green below / plum above the SN mean). Every map traces the target-network border in black (`mask_outline`: the vertices just outside the network, i.e. the outer side of `salience_border`), on the brains and magnified in the insets.

- `*_exponent_zoom[_roi]_{hemi}.svg`, `*_{delta,theta,alpha,beta,gamma}_zoom[_roi]_{hemi}.svg` — aperiodic exponent and oscillatory peak power per band (a vertex whose channels show no peak in the band carries 0).
- `*_forest_{hemi}.svg` — test summary of the measures with an uncorrected Moran *p* < 0.05 only, one row each, in table order: the observed Spearman r (black dot, filled if *p* < 0.05) against the central 95% of its 1,000 Moran-null r's (grey bar, tick at the null median), with r and the uncorrected Moran *p* (labelled $p_{m}$) written above each bar; the band FDR *q* (across the five bands) is in the CSV. Drawn at its final size on the `figure_1b_forest.svg` canvas (same width, x-axis span and font): its rows are level with the `*_zoom_roi_{hemi}.svg` rows of those measures stacked in that order and shown 150 mm wide, with the forest's top aligned with the first row. `*_forest_supp_{hemi}.svg` (supplementary) is the complete forest of all six measures (exponent, δ, θ, α, β, γ), level with all six inset rows.
- `*_spectra_{hemi}.svg` — the inferential vertices split into quintiles of the MPC gradient: mean sensitivity-weighted spectrum per quintile (left, log-log, over the 1–45 Hz fit range) and mean flattened spectrum (right, log10 power above each channel's fitted aperiodic component), lines in the coolwarm colour of the quintile's median gradient (as the `figure_1a_strata.svg` caps), canonical bands shaded. Means only.
- `*_coverage_zoom[_roi]_{hemi}.svg`, `*_coverage_sn_zoom[_roi]_{hemi}.svg` — summed sensitivity of the fitted channels on the whole cortex and on the network vertices (both hemispheres), `Purples` over the 10th–99th percentiles (colour-bar ends labelled with those values).
- `data/dataframes/df_3b_spectral_stats_{hemi}.csv` — per measure: `r`, `p`, `q` (bands only), `null_lo` / `null_med` / `null_hi` (2.5th / 50th / 97.5th percentile of the null r's), `n_vertices`, `n_perm`.

Supplementary:

- `*_specparam_qc_supp_{hemi}.svg` — `specparam` fit check (`plot_fit_quality`): the *R*² distribution of the channel fits and the four fits with the lowest, median and highest *R*² (spectrum, full model, aperiodic fit). The log also records `specparam`'s methods report (version, algorithm, settings, fit range, modes and bounds) and a fit-quality summary (`fit_quality_summary`); no fit is excluded on these metrics.
- `*_slope_corr_supp_{hemi}.svg`, `*_band_power_corr_supp_{hemi}.svg` — vertex scatter plots of each measure against the MPC gradient (fit line, r / *p* or *q*).
- `*_es_scatter_supp_{hemi}.svg` — spectral-similarity projection against the MPC gradient (points coloured by dominant target network; r / *p*, plus the distance- and uniform-weighted baseline r's) and the per-target-network mean projection.
- `*_es_zoom[_roi]_supp_{hemi}.svg` — spectral-similarity projection on the `-hemi` network vertices (the other hemisphere is empty; insets of that hemisphere's regions), `CustomCmap_ieeg` on ±2 SD.
- `*_corr_supp_{hemi}.svg` — channel-level PSD correlation matrix, ordered by the network of each channel's peak-sensitivity vertex.

---

## Figure 3c states

**`scripts/figure_3c_ieeg_states.py`** — State dependence of the MICA iEEG aperiodic exponent along the MPC gradient. The same bipolar channels were recorded in wakefulness (stage-W), N2 and N3 sleep; the first 180 s of every recording are preprocessed and parameterised as in Figure 3b (`MICA_PSD_PARAMS`; `specparam` knee mode with *k* ≥ 0, 1–45 Hz, same settings in every state) and the exponent is projected onto the folded fsLR-32k template through the bipolar sensitivity maps. Each state's exponent is correlated with the gradient on the network vertices of the `-hemi` hemisphere (FDR across the three states), and the correlations of every pair of states are compared against paired Moran surrogates of the gradient (FDR across the three pairs); the within-channel change from wakefulness to N3 is tested as well (supplementary). Significance is the within-network Moran null with the add-one empirical *p*. Main panels: one brain row per state with the forest of the significant states level with them, and the exponent of each gradient quintile across the states with the per-state correlations.

```bash
python scripts/figure_3c_ieeg_states.py \
  -ieeg_deriv /path/to/BIDS_iEEG/derivatives/electroMICA \
  -hemi LH
```

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `-ieeg_deriv` | yes | — | Path to electroMICA derivatives directory |
| `-ieeg_raw` | no | `<ieeg_deriv>/../../original` | Root of the original recordings (`<root>/sub-PX*/ses-01/*stage-{W,N2,N3}.mat`) |
| `-hemi` | no | `LH` | Hemisphere of the statistics: `LH` or `RH` (maps are shown on both hemispheres) |
| `-network` | no | `SalVentAttn` | Yeo 7-network analysis target |

Requires `data/dataframes/df_1a_{hemi}.tsv` (or `df_1a_both.tsv`), as Figure 3 MICA. The per-state channel PSDs are cached in `data/dataframes/figure_3c_channel_psd_{W,N2,N3}.pkl` (not tracked; keyed on the recordings, state, analysed length and PSD parameters). Only channels recorded in all three states are analysed, and channels whose specparam fit fails in any state are dropped.

**Outputs** (`results/figures/figure_3c_ieeg_states_*`, `{hemi}` = hemisphere of the statistics)

Brain figures use the Figure 1 window (1450 × 300, zoom 1.3, region names at 30 pt, titles at 36 pt) with the `SALIENCE_ZOOM_ROIS` insets; maps show each template value on the network vertices of both hemispheres (folded template), z-scored there, `CustomCmap_ieeg` on ±2 SD (colour bar −2 / +2, z), with the network border traced (`mask_outline`).

- `*_exponent_{W,N2,N3}_zoom_{hemi}.svg` — one brain row with insets and region names per state (untitled; the state is in the file name).
- `*_forest_{hemi}.svg` — `plot_forest` of the per-state tests (r and uncorrected Moran *p*, labelled $p_{m}$, as the quintile panel), its rows level with the three state rows stacked in W, N2, N3 order and shown 150 mm wide; only states with *p* < 0.05 are drawn, the other slots are left empty (the FDR *q* across the three states is in the CSV).
- `*_quintiles_{hemi}.svg` — 6-in panel: mean exponent (absolute values) of each MPC-gradient quintile of the inferential vertices in wakefulness, N2 and N3, lines in the coolwarm colour of the quintile's median gradient (as the `figure_1a_strata.svg` caps). Under each state its Spearman r with the gradient and Moran *p*, labelled $p_{m}$ (uncorrected, as the forest; FDR *q* in the CSV). Means only. The between-state comparisons of the correlations are reported in the CSV and the log.
- `data/dataframes/df_3c_state_stats_{hemi}.csv` — per test: `delta_N3_W`, `exponent_W`, `exponent_N2`, `exponent_N3` (`r` = Spearman r with the gradient) and `diff_W_N2`, `diff_W_N3`, `diff_N2_N3` (`r` = difference of the two states' correlations); `p`, `q` (FDR across the three states or the three pairs), `null_lo` / `null_med` / `null_hi` (2.5th / 50th / 97.5th percentile of the null r's or differences), `n_vertices`, `n_perm`.

Supplementary:

- `*_specparam_qc_{W,N2,N3}_supp_{hemi}.svg` — `specparam` fit check of each state (`plot_fit_quality`, as Figure 3b); the log records the methods report and each state's fit-quality summary.
- `*_delta_zoom_supp_{hemi}.svg` — brain row of the change of the exponent from wakefulness to N3.
- `*_{delta_N3_W,exponent_W,exponent_N2,exponent_N3}_zoom_roi_supp_{hemi}.svg` — inset rows of the change and of each state (titled, colour bar, no region names), as the Figure 1b rows.
- `*_forest_supp_{hemi}.svg` — complete `plot_forest` of the four tests, its rows level with those four inset rows stacked in that order and shown 150 mm wide.
- `*_bands_quintiles_supp_{hemi}.svg` — every frequency band (`SPEC_BANDS`, the canonical bands clipped to the fit range): one 6-in panel per band, as the main exponent panel, with each quintile's mean oscillatory peak power (above the aperiodic fit, log10; no detected peak = 0, as Figure 3b) across the states and, under each state, its Spearman r with the gradient and Moran *p*, labelled $p_{m}$ (uncorrected; the FDR *q* across the five bands within the state is in the CSV).
- `data/dataframes/df_3c_band_state_stats_{hemi}.csv` — per band: rows `{band}_{W,N2,N3}` (`r` = Spearman r with the gradient, `q` across the five bands within the state) and `{band}_diff_{W_N2,W_N3,N2_N3}` (`r` = difference of the two states' correlations, paired Moran surrogates, `q` across the three pairs within the band); `band`, `test`, `p`, `null_lo` / `null_med` / `null_hi`, `n_vertices`, `n_perm`.

---

## External data paths

Scripts that require micapipe or iEEG derivatives expect the following directory structure:

```
micapipe_v0.2.0/
└── sub-<id>/
    └── ses-<id>/
        ├── mpc/        # T1 intensity profiles (.shape.gii)
        ├── dwi/        # tractography and connectome files
        └── surf/       # cortical surface reconstructions
```

See [data/README.md](https://github.com/PaulBautin/salience-network-multiscale-switch/blob/main/data/README.md) for download instructions.

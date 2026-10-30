# Usage

## Running the pipeline

Every command is run from the repository root. On a server without a display, prefix it
with `xvfb-run -a` (the brain renders need an X display).

| Command | What it does |
|---|---|
| `python pipeline/run_all.py` | every step in order (compute and plot), then the export |
| `python pipeline/run_all.py --stage plot` | redraw every figure from `source_data/` (no private data needed) |
| `python pipeline/run_all.py --from step3_connectivity` | that step and the ones after it |
| `python pipeline/run_all.py --only step1b_histology step1c_cortical_types` | only these steps |
| `python pipeline/<step>.py --stage compute` | one step: compute its tables (no figures) |
| `python pipeline/<step>.py --stage plot` | one step: draw its figures from `source_data/` |
| `python pipeline/<step>.py` | one step: both (`--stage all`) |
| `python pipeline/step4_export_figures.py --formats png pdf` | export the composite figures with Inkscape |

### Configuration

- **`config/paths.yaml`** — where the private data and the work directory are on your
  machine. Copy `config/paths.example.yaml` and set each path. It is git-ignored and
  needed only to compute.
- **`config/analysis.yaml`** — the settings of each step: the network, the hemisphere of
  the statistics, the number of permutations. Every other parameter is a named constant at
  the top of its step or library module.

### Where the outputs go

| Folder | Contents | In git |
|---|---|---|
| `source_data/<step>/` | the tables behind the step's figures, de-identified, and `manifest.json` (code version, package versions, settings, sample sizes) | yes |
| `results/figures/` | the figure panels (SVG) | yes |
| `results/` | the composite figures (`figure*.svg`, edited in Inkscape) and their PNG/PDF exports | yes |
| `<work_dir>/<step>/` | the step's private tables and caches, and its log (`<step>.log`, appended at every run) | no |

Tables are tab-separated text with a header row. Read them back with
`salience_switch.config.read_table`, which returns the floating-point values exactly as
written. The composite figures link their panels by file name, so the panel names below
are stable.

## Step 1a: MPC gradient

`pipeline/step1a_mpc_gradient.py` — builds the subject lists (healthy-control MICA-PNI
participants with a matching MICA-MICs scan and every input: fsLR-5k qT1 profiles, MPC,
SC and edge lengths, geodesic distance, FC, and fsLR-32k qT1 profiles), then computes the
first MPC gradient of the network at fsLR-32k and fsLR-5k on the same subjects. The
gradients are reused while the subject list is unchanged.

- **Reads:** `micapipe_pni`, `micapipe_mics`, `participants` (to build the subject lists
  and log sex and age; once the lists exist, it is no longer needed).
- **Settings:** `network`, `hemisphere`.
- **Private:** `subjects_fslr32k.csv`, `subjects_fslr5k.csv` (subject IDs and file paths;
  read by step 3), `gradient_fslr{32k,5k}_<hemi>.json` (subjects, eigenvalues, variance
  explained per subject).
- **Public:**
  - `surface_fslr32k_<hemi>.tsv`, `surface_fslr5k_<hemi>.tsv` — the surface tables
    ([API: surface table](api/overview.md#surface-table)) with `t1_gradient1_<network>`
    and `T1map`; every later step reads them.
  - `profiles_<hemi>.tsv` — per network vertex: `vertex`, `hemisphere`, the subject-mean
    qT1 (ms) at each depth `qT1_depth01` … `qT1_depth14`.
  - `gradient_fit_<hemi>.tsv` — per surface density: subjects, vertices, components,
    sparsity and the variance explained by gradient 1 (mean, SD, range).
  - `mpc_vs_fc_gradient_<hemi>.tsv` — Spearman r of the network gradient with the FC
    gradient, Moran p, null 95 % interval, vertices and permutations.
- **Figures:** `figure_1a_{profiles,strata,brain,brain_5k,brain_zoom,brain_zoom_roi}.svg`;
  supplementary `figure_1a_fc_gradient{,_sn}_zoom{,_roi}_supp.svg`.

## Step 1b: histological context

`pipeline/step1b_histology.py` — correlates the gradient with the BigBrain, Bielschowsky
and Parvalbumin maps over the network vertices, each against a Moran null fitted per
hemisphere. Its inputs are all public, so it computes without `config/paths.yaml`.

- **Settings:** `hemisphere`.
- **Public:** `histology_maps_<hemi>.tsv` (per network vertex: `vertex` and the z-scored
  mean intensity of each map), `context_stats_<hemi>.tsv` (per map: `r`, `p`, `null_lo` /
  `null_med` / `null_hi` (2.5th / 50th / 97.5th percentiles of the null), `n_vertices`,
  `n_perm`), `context_null_<hemi>.tsv` (per map and surrogate: `r`).
- **Figures:** `figure_1b_brain_{bigbrain,t1map,biel,parva}.svg`,
  `figure_1b_zoom_{bigbrain,biel,parva}.svg`, `figure_1b_forest.svg` (ridgeline of the
  Moran nulls level with the zoom rows); supplementary `figure_1b_correlations_supp.svg`,
  `figure_1b_correlations_circle.svg`.

## Step 1c: cortical types

`pipeline/step1c_cortical_types.py` — the von Economo–Koskinas cortical-type composition of
every network and, for the salience network, its enrichment test against 1,000 spin
permutations. Its inputs are all public.

- **Settings:** `hemisphere`.
- **Public:** `type_composition_<hemi>.tsv` (per network and type: `percent`),
  `type_null_SalVentAttn_<hemi>.tsv` (per spin and type: `percent`),
  `type_stats_SalVentAttn_<hemi>.tsv` (per type: `observed`, `null_mean`, `null_sd`, `z`,
  `p`, `n_spins`).
- **Figures:** `figure_1c_brain_economo.svg`, `figure_1c_brain_economo_zoom{,_roi}.svg`,
  `figure_1c_type_salience.svg`.

## Step 2a: MNI Open iEEG Atlas

`pipeline/step2a_ieeg_mni.py` — maps the atlas channels onto the fsLR-32k template, and
correlates the difference in spectral similarity with the top and bottom gradient
quarters of the network against the second BigBrain gradient (Moran null on the channel
layout).

- **Reads:** `mni_ieeg` (MatlabFile.mat), `mni_ieeg_surfaces` (`L.anat.reg.surf.gii`,
  `R.anat.reg.surf.gii`); step 1a `surface_fslr32k_both.tsv`.
- **Settings:** `network`.
- **Public:** `channels.tsv` (per channel: template `vertex`, `network`, gradient
  `quantile`, `corr_diff`, `bigbrain_g2`), `es_difference_stats.tsv` (`r`, `p_moran`, null
  95 % interval, `n_channels`, `n_permutations`).
- **Figures:** `figure_2a_ieeg_mni_corr_scatter.svg`,
  `figure_2a_ieeg_mni_channel_corr_diff{,_smooth}.svg`.

## Step 2b: MICA iEEG spectra

`pipeline/step2b_ieeg_mica.py` — parameterises the wakefulness spectrum of every bipolar
channel with specparam, projects the aperiodic exponent (primary) and the peak power of
five bands (secondary) onto the template, and tests each against the network gradient of
the `hemisphere` (Moran null). The supplementary analysis projects the FC gradient through
the spectral similarity of the vertices.

- **Reads:** `electromica` (sensitivity maps), `ieeg_raw` (recordings); step 1a
  `surface_fslr32k_both.tsv`.
- **Settings:** `network`, `hemisphere`.
- **Private:** `channel_psd_W.pkl` (channel spectra; recomputed when the recordings or the
  spectral parameters change).
- **Public:**
  - `spectral_maps_<hemi>.tsv` — per template vertex: `coverage`, `inferential` (tested
    vertex), `exponent` and the peak power of each band.
  - `spectral_stats_<hemi>.tsv`, `spectral_null_<hemi>.tsv` — per measure: `r`, `p`
    (uncorrected), null percentiles, `n_vertices`, `n_perm`; the surrogate `r`.
  - `spectral_spectra_<hemi>.tsv` — per gradient quintile and frequency: the mean spectrum
    (`psd`) and flattened spectrum (`flat`).
  - `similarity_{vertices,targets,stats}_<hemi>.tsv` — the spectral-similarity projection
    per source vertex, its mean per target network, and its statistics.
- **Figures:** `figure_2b_ieeg_mica_{exponent,delta,theta,alpha,beta,gamma}_zoom{,_roi}_<hemi>.svg`,
  `figure_2b_ieeg_mica_coverage{,_sn}_zoom{,_roi}_<hemi>.svg`,
  `figure_2b_ieeg_mica_forest_<hemi>.svg` (tests with an uncorrected p < 0.05),
  `figure_2b_ieeg_mica_spectra_<hemi>.svg`; supplementary `*_forest_supp`,
  `*_slope_corr_supp`, `*_band_power_corr_supp`, `*_es_scatter_supp`, `*_es_zoom{,_roi}_supp`.
  The figures of individual channels, `*_specparam_qc_supp` and `*_corr_supp`, are drawn
  by the compute stage.

## Step 2c: wakefulness, N2 and N3

`pipeline/step2c_ieeg_states.py` — the same channels in the three states (first 180 s of
each recording): the aperiodic exponent of each state and its change from wakefulness to
N3 against the gradient, and the difference between the correlations of every pair of
states (paired Moran surrogates); supplementary, the peak power of each band per state.

- **Reads:** as step 2b, for the three states.
- **Settings:** `network`, `hemisphere`.
- **Private:** `channel_psd_{W,N2,N3}.pkl`.
- **Public:** `state_maps_<hemi>.tsv` (per template vertex: `inferential`, `delta_N3_W`,
  `exponent_W`, `exponent_N2`, `exponent_N3`), `state_stats_<hemi>.tsv` and
  `state_null_<hemi>.tsv` (the four map tests and the pairs `diff_W_N2`, `diff_W_N3`,
  `diff_N2_N3`), `state_band_stats_<hemi>.tsv` and `state_band_values_<hemi>.tsv` (per
  band and state).
- **Figures:** `figure_2c_ieeg_states_exponent_{W,N2,N3}_zoom_<hemi>.svg`,
  `figure_2c_ieeg_states_forest_<hemi>.svg`, `figure_2c_ieeg_states_quintiles_<hemi>.svg`;
  supplementary `*_delta_zoom_supp`, `*_<measure>_zoom_roi_supp`, `*_forest_supp`,
  `*_bands_quintiles_supp`; `*_specparam_qc_<state>_supp` are drawn by the compute stage.

## Step 3: connectivity projection

`pipeline/step3_connectivity.py` — for every network and connectivity measure (SC, GD,
MPC, FC), the connectivity-weighted projection of the FC gradient for each network vertex
and subject, its per-subject Spearman r with the network MPC gradient, the group test, and
the Moran, topological and spin null models. The compute stage takes about two hours.

- **Reads:** step 1a `subjects_fslr5k.csv` (private) and `surface_fslr5k_<hemi>.tsv`; the
  connectomes listed in the subject list.
- **Settings:** `hemisphere`, `n_permutations`.
- **Private:** `subject_r_<hemi>.tsv` (per subject, network and measure: `r`).
- **Public** (`manifest.json` records the table version and refuses tables of another):
  - `vertex_<hemi>.tsv` — per network, measure and source vertex: `vertex` (fsLR-5k
    index), `hemisphere`, `g_mpc`, `P` (group-mean projection, FC-gradient units),
    `n_targets` (mean target count), `dominant_network`.
  - `stats_<hemi>.tsv` — per network and measure: `n` (subjects), `n_vertices`, `r_group`,
    `ci_low`, `ci_high`, `t`, `p` (subject-level t-test), `p_moran`, `p_topo` (NaN for GD),
    `p_spin`, `n_perm`. All p values are uncorrected.
  - `null_<hemi>.tsv` — per network, measure, null model and surrogate: the group `r`
    (null models `moran`, `topo`, `spin`, and the `topo_control_positive` /
    `topo_control_negative` controls).
  - `topo_control_<hemi>.tsv` — the two synthetic control maps of the topological null:
    `r_obs`, `p_topo`, `n_perm`.
- **Figures:** `figure_3a_{SC,GD,MPC,FC}_zoom{,_roi}_<hemi>.svg`, `figure_3a_forest_<hemi>.svg`
  (lollipops of the seven networks, filled where the uncorrected Moran p < 0.05); the
  supplements `figure_3a_ridge_<hemi>.svg`, `figure_3a_forest_band_<hemi>.svg`,
  `figure_3a_scatter_supp_<hemi>.svg`, `figure_3b_network_summary_<hemi>.svg`,
  `figure_3b_{ridge,forest}_<hemi>.svg`, `figure_3b_brain_<measure>_rho_<network>.svg`,
  `figure_3b_distance_network_<measure>.svg`, `figure_3_supp_topo_control.svg`.

## Step 3 schematic

`pipeline/step3_schematic.py` — the visual key to the projection (Figure 3, panel A), from
the subject whose structural r_s is closest to the group median.

- **Reads:** as step 3 (SC only).
- **Settings:** `hemisphere`.
- **Private:** `example_subject_<hemi>.json` (the example subject's ID, every subject's r_s).
- **Public:** `schematic_vertex_<hemi>.tsv` (per network vertex: `g_mpc`, the example
  subject's `P` and `n_targets`, the group `P_mean`), `schematic_edges_<hemi>.tsv` (the
  connections of the example vertex a that enter its projection: `source`, `target`, `w`,
  `g_fc`), `schematic_summary_<hemi>.tsv` (`example_vertex`, `r_s`, `r_s_median`,
  `n_subjects`).
- **Figures:** `figure_3_schematic_<hemi>.svg`.

## Step 4: export

`pipeline/step4_export_figures.py` — exports each composite figure (`results/figure*.svg`)
to PNG, and optionally PDF, with Inkscape, raising the resolution of the linked panels to
`--dpi` at their placed size. Only composites older than one of their panels are exported
again; `--watch` keeps doing so as the panels change.

| Flag | Default | Meaning |
|---|---|---|
| `--formats` | `png` | `png` and/or `pdf` |
| `--dpi` | 300 | PNG resolution and least resolution of the panels |
| `--figures` | every `results/*.svg` | composites to export |
| `--force` | off | export even if up to date |
| `--watch`, `--interval` | off, 2 s | keep exporting as the inputs change |

## Tests

`pytest` runs the tests in `tests/` on synthetic data (no private data), including a
calibration of the Moran test and the privacy check of `source_data/`;
`ruff check src pipeline tests` checks for undefined names and unused imports. Both run
on GitHub at every push, with a scan of every tracked file for participant data.

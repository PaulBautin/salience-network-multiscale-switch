# Reproducibility

Every figure is drawn by the plot stage of one pipeline step from the tables of
`source_data/<step>/`, which are public and hold no participant information. To redraw
all of them:

```bash
python pipeline/run_all.py --stage plot
```

Each `source_data/<step>/manifest.json` records how its tables were made: the git commit
of the code (`-dirty` if the code had uncommitted changes; changes to `results/` and
`source_data/`, which the pipeline writes, do not count), the versions of Python and the
main packages, the step settings and the sample sizes. The p values in the tables are
uncorrected for multiple comparisons; the families of tests are defined in the Methods.

## Figure panels

Each composite figure in `results/` links its panels from `results/figures/` by file name.

| Composite | Step | Panels (`results/figures/`) | Tables (`source_data/`) | Methods |
|---|---|---|---|---|
| `figure1.svg` | `step1a_mpc_gradient` | `figure_1a_brain_zoom.svg`, `figure_1a_strata.svg` | `step1a_mpc_gradient/surface_fslr32k_both.tsv`, `profiles_both.tsv`, `gradient_fit_both.tsv` | [Figure 1](methods/figure_1.md), [Shared](methods/shared.md#microstructural-profile-covariance-gradients) |
| | `step1b_histology` | `figure_1b_zoom_{bigbrain,biel,parva}.svg`, `figure_1b_forest.svg` | `step1b_histology/histology_maps_both.tsv`, `context_stats_both.tsv`, `context_null_both.tsv` | [Figure 1](methods/figure_1.md) |
| | `step1c_cortical_types` | `figure_1c_brain_economo_zoom.svg`, `figure_1c_type_salience.svg` | `step1c_cortical_types/type_*_both.tsv` | [Figure 1](methods/figure_1.md) |
| `figure_2a_mni.svg` | `step2a_ieeg_mni` | `figure_2a_ieeg_mni_*.svg` (embedded in the composite) | `step2a_ieeg_mni/channels.tsv`, `es_difference_stats.tsv` | [Figure 2](methods/figure_2.md) |
| `figure_2b_mica.svg` | `step2b_ieeg_mica` | `figure_2b_ieeg_mica_coverage_sn_zoom_LH.svg`, `figure_2b_ieeg_mica_spectra_LH.svg`, `figure_2b_ieeg_mica_{exponent,theta}_zoom_roi_LH.svg`, `figure_2b_ieeg_mica_forest_LH.svg` | `step2b_ieeg_mica/spectral_*_LH.tsv` | [Figure 2](methods/figure_2.md) |
| supplementary | `step2c_ieeg_states` | `figure_2c_ieeg_states_exponent_*_zoom_LH.svg`, `figure_2c_ieeg_states_quintiles_LH.svg`, `figure_2c_ieeg_states_bands_quintiles_supp_LH.svg`, `figure_2c_ieeg_states_forest_LH.svg` | `step2c_ieeg_states/state_*_LH.tsv` | [Figure 2](methods/figure_2.md) |
| `figure3.svg` | `step3_schematic` | `figure_3_schematic_both.svg` | `step3_schematic/schematic_*_both.tsv` | [Figure 3](methods/figure_3.md) |
| | `step3_connectivity` | `figure_3a_{SC,GD,MPC,FC}_zoom_roi_both.svg`, `figure_3a_forest_both.svg` | `step3_connectivity/vertex_both.tsv`, `stats_both.tsv`, `null_both.tsv` | [Figure 3](methods/figure_3.md) |

The other supplementary panels of each step, and every table column, are listed in
[Usage](usage.md). `pipeline/step4_export_figures.py` exports the composites to PNG and PDF.

## Recomputing

Recomputing needs the datasets of [Data Acquisition](methods/datasets.md): set their paths
in `config/paths.yaml` (copy `config/paths.example.yaml`) and run
`python pipeline/run_all.py`. Steps 1b and 1c read only public inputs and recompute
without them. The settings of each step are in `config/analysis.yaml`; random seeds and
the other parameters are named constants at the top of each step and library module, and
are listed with each analysis in the Methods.

Every spatial null model is seeded, so a rerun with the same inputs gives the same
statistics. The bootstrap confidence bands of the regression lines are seeded too, and
the SVG panels are written with a fixed element-id salt and date, so redrawing from
`source_data/` gives byte-identical panels. The pipeline was checked against the outputs of the previous version of the
code: the statistics and null distributions agree exactly, and the gradient to within
floating-point rounding (2 × 10⁻¹⁵).

## Tests

`pytest` runs the tests in `tests/` on synthetic data: the add-one p-value, the
per-hemisphere Moran null (one block reproduces brainspace's `MoranRandomization`), the
false-positive rate of the Moran-null test on independent smooth maps, the projection, the
recovery of known aperiodic exponents by the spectral fit, the exact round trip of the
tables, and the absence of participant information in `source_data/`.

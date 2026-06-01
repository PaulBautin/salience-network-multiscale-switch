# CLAUDE.md — AI Assistant Instructions

## Project Overview & Context

What this repo does: See `README.md` for full detail.
Primary language(s): Python (≥ 3.10).
Key external tools: micapipe, brainspace.
Key inputs: micapipe derivatives (MPC profiles, DWI connectomes), iEEG BIDS dataset, BigBrain dataset, AHEAD dataset. 
Key output: SVG figures in `results/figures` + intermediate TSV/CSV dataframes cached in `data/dataframes/`.
Intended users: Neuroimaging researchers. Repo is public, citable, and must be fully reproducible. Meaning that after completing any feature or significant change, update the relevant .md files in the /docs folder to reflect what changed before finishing the task".

## Environment Setup

Two options are available:

**uv (fast, pure-Python):**
```bash
uv venv env_salience
uv pip install -e . --python env_salience/bin/python
source env_salience/bin/activate
```

**Conda:**
```bash
conda env create -f environment.yml
conda activate env_salience
```

Key dependencies: `numpy`, `pandas`, `nibabel`, `matplotlib`, `scipy`, `brainspace`, `bctpy`.

## Running Scripts

Scripts are run from the **project root**. Figure 1a requires paths to external derivative datasets; downstream scripts read cached outputs produced by figure 1a:

```bash
# Figure 1a – T1 microstructural gradient (must run first)
python scripts/figure_1a_t1map.py \
  -pni_deriv /path/to/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -mics_deriv /path/to/BIDS_MICs/derivatives/micapipe_v0.2.0

# Figure 2 – SC/GD/MPC differences at gradient extremes (requires figure 1a outputs)
#   Runs at fsLR-5k resolution; no external derivatives needed
python scripts/figure_2_distance.py -hemi LH

# Figure 3 – iEEG (MNI and MICA datasets)
python scripts/figure_3_ieeg_mni.py \
  -ieeg_deriv /path/to/MNI_ieeg/MatlabFile.mat
python scripts/figure_3_ieeg_mica.py \
  -ieeg_deriv /path/to/BIDS_iEEG/derivatives/electroMICA
```

Scripts cache intermediate DataFrames to `data/dataframes/` (TSV/CSV) and skip recomputation if the cached file exists. Outputs are written to `results/figures/` as SVG/PNG.

## Architecture

### Data flow
1. **scripts/** — Entry points, one per figure panel. Each script loads data, calls `src/` modules for computation, then plots and saves to `results/figures/`.
2. **src/** — Reusable processing modules imported by scripts.
3. **data/** — Static inputs: parcellations (`.label.gii`), brain surfaces (`.surf.gii`), histological profiles (`.shape.gii`), and cached subject-level DataFrames.

### Core modules (`src/`)

- **`atlas_load.py`** — Loads and merges atlases onto the fsLR-32k surface. Central function is `load_yeo_atlas()` which returns `df_yeo_surf`, a per-vertex DataFrame with Schaefer-400 parcellation, Yeo 7-network labels, hemisphere, and salience network border mask. `load_yeo_surf_5k()` returns the same structure at fsLR-5k resolution (9,684 vertices) for figure 2 analyses. Other loaders (`load_bigbrain`, `load_ahead_biel`, `load_ahead_parva`, `load_econo_atlas`, `load_t1map`, `load_bigbrain_gradients`, `load_baillarger_atlas`, `load_intrusion_atlas`) add histology columns to the surface DataFrame. Utility functions `convert_states_str2int` and `normalize_to_range` support encoding and scaling operations.

- **`gradient_computation.py`** — Computes microstructure profile covariance (MPC) gradients. `compute_t1_gradient()` takes subject T1 intensity profiles, computes per-subject partial-correlation matrices (controlling for the mean profile), fits `GradientMaps` (diffusion map, normalized angle kernel, procrustes alignment), and returns the z-scored first gradient.

- **`connectome_processing.py`** — Connectome I/O and the gradient-weighted connectivity projection used in figure 2 (fsLR-5k). Loads/symmetrises micapipe connectomes (`load_subject_matrix`), builds the Betzel distance-stratified consensus mask (`build_consensus_mask`), preprocesses per-modality weights (`prepare_weights`), computes the per-network-vertex projection score (`compute_projection_score`, `compute_projection_score_rank`) with per-subject group inference (`compute_projection_subjects`), and runs spin/Moran nulls.

- **`ieeg_processing.py`** — iEEG signal processing pipeline. Loads MATLAB `.mat` files from BIDS-formatted iEEG datasets, preprocesses signals (bandpass → downsample → demean), computes Welch PSD (`preprocess_and_compute_psd_ieeg`), extracts band power (`extract_band_power`), and maps channels to fsLR-32k surface vertices via GIFTI sensitivity maps. `compute_psd_vectorized` computes PSD on already-preprocessed data (e.g. MNI atlas). `plot_surface_sphere` renders electrode contacts as VTK spheres on a brain surface screenshot.

- **`logging_utils.py`** — `setup_manuscript_logger()` configures dual console + file logging, appending timestamped run headers to `logs/<script_name>.log` without overwriting.

- **`plot_colors.py`** — Color definitions and registered matplotlib colormaps for Yeo 7-network (`yeo7_rgba`, `yeo7_rgb`, `CustomCmap_yeo`), Von Economo cortical types (`CustomCmap_type`, `CustomCmap_type_mw`), Baillarger bands (`CustomCmap_baillarger`), and Intrusion classes (`CustomCmap_intrusion`).

### Surface space
Most analyses use **fsLR-32k** space (64,984 vertices total: 32,492 LH + 32,492 RH). Figure 2 connectivity analyses use the downsampled **fsLR-5k** space (9,684 vertices: 4,842 LH + 4,842 RH) to reduce memory when loading whole-brain connectomes. Parcellation is **Schaefer-400** with Yeo 7-network labels. The primary network of interest is `'SalVentAttn'` (Salience/Ventral Attention).

### Key DataFrame conventions
- `df_yeo_surf`: one row per surface vertex, columns include `mics` (parcel ID), `network`, `hemisphere`, `label`, `salience_border`, based on schaeffer 400 parcellation. Analysis columns (e.g., `t1_gradient1_SalVentAttn`, `T1map`, `BigBrain`) are added in-place.
- iEEG DataFrames: one row per channel, with `Subject`, `Session`, `ChannelName`, and signal/PSD columns.

### External data not in repo
- PNI/MICs micapipe derivatives (T1 profiles, structural connectomes, tractography) at `/data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0`
- BIDS iEEG dataset at `/host/verges/tank/data/BIDS_iEEG/`
- Baillarger/Intrusion MYATLAS parcellations (hardcoded paths in `atlas_load.py`)

## Documentation conventions

Docs live in `docs/` and are built with MkDocs (`mkdocs.yml` defines the nav).
The site keeps four sections: **Home**, **Usage**, **API Reference**, **Methods**.
When changing docs, keep content in the section it belongs to (API reference for
modules, Methods for the science) and follow the patterns already in the repo:

**Source of truth.** The numpy-style docstrings in `src/` drive the API pages.
When you add or change a public function, update its docstring **and** its API
page in the same change — the API pages are hand-written mirrors and drift
otherwise. A new `src/` module gets a new `docs/api/<module>.md`, a nav entry in
`mkdocs.yml`, and a row in `docs/api/overview.md`.

**API page** (`docs/api/<module>.md`) — mirrors the source module:
- `# \`src/<module>\`` title, optional one-line module intro.
- One `### \`function_name\`` block per public function, in source order.
- Each block: a fenced ```python``` signature (verbatim from source) → one-or-two
  sentence description → **Parameters** table (`Name | Type | Description`) →
  **Returns** line → **Raises**/**Example** only when relevant.
- `---` separator between functions. No prose justification or changelogs here.

**Methods pages** (`docs/methods/*.md`) — together these form the manuscript's
**Materials and Methods**. Write them the way the "Datasets and Methods" section
of a *Nature* or *Science* paper reads: rigorous, self-contained, reproducible
prose. `datasets.md` is the model for acquisition text; the figure pages document
each analysis.
- **Voice** — formal scientific past tense, third person ("Intracortical profiles
  were sampled at 14 equivolumetric depths…"). Continuous prose, not bullet or
  step lists, except where an ordered algorithm is genuinely clearer.
- **Completeness** — report every detail a reader needs to reproduce the result:
  sequence parameters, sample sizes, thresholds, software and versions, random
  seeds, and the exact statistical test with its assumptions. State the number
  rather than gesturing at it.
- **Citations** — inline as Author et al., Year (e.g. Betzel et al., 2018).
- **Tables and equations** — tables are appropriate for acquisition parameters,
  frequency bands, or modality summaries; give key quantities a displayed
  equation (MathJax, `$...$` / `$$...$$`).
- **Page shape** — figure pages open with `**Scripts:**` / `**Module:**` links,
  then a framing paragraph and one subsection per analysis step.
- **Delegate, don't repeat** — procedures reused across figures live in
  `methods/shared.md` and are referenced, not duplicated; code-level function
  detail lives in the API pages, not here.
- **Exclude non-manuscript matter** — no module/function reference tables, no
  output-file lists (those belong in `usage.md`), no changelogs or
  "approach-rejected" rationale (git history records method evolution).

**Math** uses `$...$` / `$$...$$` (MathJax via the arithmatex extension).
Inline code identifiers use backticks; cross-page and cross-file references use
relative markdown links (e.g. `[\`src/...\`](../api/<module>.md)`). Validate with
`mkdocs build --strict` before finishing (it flags broken links and orphaned
pages).

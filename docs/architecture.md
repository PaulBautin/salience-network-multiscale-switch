# How the repository works

The analyses run as a pipeline of steps, one per figure, in figure order: step 1
(Figure 1) to step 3 (Figure 3), then step 4 exports the composite figures. Every step
has two stages: **compute** reads the data and writes the tables behind its figures to
`source_data/`; **plot** draws the figures from those tables alone. The participants'
data never enter the repository: they stay on the computing machine, with every table
that could identify a participant, in a private work directory.

Each section below is one view of the same code and can be read on its own. The
diagrams render on GitHub, on the documentation site, and in VS Code with a Mermaid
extension.

## 1. The repository at a glance

```mermaid
flowchart TD
    subgraph local ["Local only, never in git"]
        private[("Private data<br/>micapipe · electroMICA · MNI iEEG atlas")]
        paths["config/paths.yaml<br/>where the data are"]
        work[("work_dir<br/>subject lists · caches · logs")]
    end
    subgraph repo ["In the repository"]
        settings["config/analysis.yaml<br/>settings of each step"]
        templates[("data/<br/>surfaces · atlases · histology")]
        lib["src/salience_switch/<br/>analysis library"]
        steps["pipeline/<br/>one step per figure, run in order"]
        source[("source_data/<br/>de-identified figure tables")]
        figs("results/<br/>panels · composites · PNG / PDF")
        tests["tests/<br/>synthetic data, run by CI"]
        privacy["tools/check_privacy.py<br/>pre-commit hook · CI · tests"]
        docs["docs/<br/>usage · API · methods"]
    end
    private --> steps
    paths -.-> steps
    settings -.-> steps
    templates --> steps
    lib -.->|imported by| steps
    steps <-->|private tables| work
    steps -->|compute| source
    source -->|plot| figs
    lib -.->|docstrings| docs
    tests -.->|check| lib
    privacy -.->|refuses participant data| source
```

## 2. Modules

`pipeline/run_all.py` runs the steps in order; each step is one file in `pipeline/`
and shares its command line (`--stage compute|plot|all`) through `pipeline/common.py`.
Steps 2a and 2b run one analysis on two cohorts: its statistics and panels are in
`pipeline/ieeg_spectral.py`, and each step supplies its cohort's spectra and sensitivity maps.
The steps use the library in `src/salience_switch/`. The library holds the methods (the
gradient, the null models, the projection, the spectral fits) and the figure components
shared by several steps (zoom rows, ridgeline and forest panels, colormaps); the choices of
this study (network, hemisphere, file names, figure layouts) are made by the steps and
`config/analysis.yaml`. The analysis modules take their inputs as arguments and can be
reused on other data; `config` and `surface` are tied to this repository's folders
(`config/`, `source_data/`, the bundled `data/`). Within the library, only `connectivity`
imports another module (`stats`).

```mermaid
flowchart TD
    subgraph pipeline ["pipeline/"]
        run_all["run_all.py<br/>every step, in order"]
        stepfiles["step1a … step3_schematic, step4<br/>compute() and plot()"]
        common["common.py<br/>--stage compute | plot | all"]
    end
    subgraph lib ["src/salience_switch/"]
        subgraph analysis ["Analysis"]
            micro["microstructure.py<br/>qT1 profiles, MPC gradient"]
            conn["connectivity.py<br/>connectomes, projection, its nulls"]
            ieeg["ieeg.py<br/>spectra, specparam, sensitivity maps"]
            stats["stats.py<br/>Moran null, Moran-Spearman tests"]
        end
        subgraph draw ["Figures"]
            plotting["plotting.py<br/>brain rows, zoom insets, stats panels"]
            colors["colors.py<br/>colormaps"]
        end
        subgraph io ["Data and records"]
            surface["surface.py<br/>surface tables, atlases, templates"]
            config["config.py<br/>paths, settings, tables"]
            provenance["provenance.py<br/>logs, manifests"]
        end
    end
    run_all --> stepfiles
    stepfiles --> common
    stepfiles -->|imports| lib
    conn --> stats
    stats ~~~ plotting
    plotting ~~~ config
```

## 3. Steps, in the order they run

Step 1a comes first: it builds the subject list and the microstructural gradient that
every later step reads. The other steps follow in figure order.

```mermaid
flowchart TD
    s1a["Step 1a · MPC gradient<br/>Fig 1a"]
    s1b["Step 1b · histological context<br/>Fig 1b"]
    s1c["Step 1c · cortical types<br/>Fig 1c"]
    s2a["Step 2a · MNI iEEG atlas<br/>Fig 2 replication (supp.)"]
    s2b["Step 2b · MICA iEEG spectra<br/>Fig 2"]
    s2c["Step 2c · wake, N2, N3<br/>Fig 2 supplement"]
    s3["Step 3 · connectivity projection<br/>Fig 3"]
    s3s["Step 3 · projection schematic<br/>Fig 3, panel A"]
    s4["Step 4 · export<br/>composites to PNG / PDF"]
    s1a ==> s1b ==> s1c ==> s2a ==> s2b ==> s2c ==> s3 ==> s3s ==> s4
```

| Step | Reads | Private (`work_dir`) | Public (`source_data/`) |
|---|---|---|---|
| 1a | micapipe qT1 profiles; participant table | subject lists with file paths; gradient fits with subject IDs | surface tables with the gradient; subject-mean qT1 profiles; gradient fit summary; comparison with the FC gradient |
| 1b | step 1a gradient; BigBrain, AHEAD maps in `data/` | — | histological maps; statistics; Moran nulls |
| 1c | atlases in `data/` | — | type composition; spin nulls; statistics |
| 2a | MNI iEEG atlas and its registered surfaces; step 1a gradient | — | as step 2b |
| 2b | electroMICA recordings and sensitivity maps; step 1a gradient | channel spectra | template maps of each measure; statistics; nulls; quintile spectra; similarity projection |
| 2c | as step 2b, for three states | channel spectra per state | template maps per state; band powers; statistics; nulls |
| 3 | step 1a gradient and subject list; SC, GD, MPC, FC connectomes | per-subject correlations | vertex-wise group projections; statistics; nulls; control |
| 3 schematic | as step 3 | example subject's ID | that subject's projection; connections of one vertex; summary |

Steps 1b and 1c read only public inputs, so they can be recomputed without the private
data; every step can be redrawn from `source_data/` alone.

## 4. Inside a step

The two stages communicate only through files. `compute` writes a `manifest.json` next
to the tables: the code version (the last commit that changed the code, `-dirty` if it has
uncommitted changes), the package versions, the settings and the sample sizes of the run,
and no date, so an unchanged rerun rewrites it byte for byte. Figures of individual
channels (specparam fit quality, channel similarity matrix) are drawn by `compute`,
because their data are not public.

```mermaid
flowchart LR
    inputs[("inputs<br/>private data · step 1a tables · data/")]
    compute["compute(settings, paths)<br/>analysis and statistics"]
    work[("work_dir/step/<br/>private tables · log")]
    source[("source_data/step/<br/>figure tables · manifest.json")]
    plot["plot(settings)<br/>figures only"]
    figs("results/figures/<br/>panels")
    inputs --> compute
    compute --> work
    compute --> source
    source --> plot
    plot --> figs
```

## 5. Figure 1: microstructure

```mermaid
flowchart TD
    profiles[("qT1 profiles<br/>14 depths per vertex, each subject")]
    mpc["MPC matrix per subject<br/>partial correlation, mean profile removed"]
    gradient["diffusion-map gradients, Procrustes-aligned<br/>first component of their mean, z-scored"]
    panels1a["1a: profiles, strata and maps"]
    histology[("BigBrain · AHEAD maps")]
    test1b["1b: Spearman vs each map<br/>Moran null per hemisphere"]
    types[("von Economo types")]
    test1c["1c: type composition of the network<br/>spin test, 10,000 rotations, medial wall left out"]
    profiles --> mpc --> gradient
    gradient --> panels1a
    gradient --> test1b
    histology --> test1b
    types --> test1c
```

## 6. Figure 2: intracranial EEG

```mermaid
flowchart TD
    recordings[("bipolar channels<br/>MICA: W, N2, N3 · MNI atlas: W")]
    psd["Welch spectra<br/>MNI atlas: windows within the recorded segments"]
    params["specparam per channel<br/>aperiodic exponent, band peak power"]
    surface["projection onto the template<br/>MICA: contact sensitivity maps · MNI atlas: Gaussian footprints"]
    test2["steps 2a, 2b, 2c: Spearman vs the gradient<br/>Moran null, network vertices"]
    ss["steps 2a, 2b: spectral-similarity projection<br/>of the FC gradient"]
    recordings --> psd --> params --> surface --> test2
    psd --> ss
    surface --> ss
```

## 7. Figure 3: connectivity

```mermaid
flowchart TD
    connectomes[("SC · GD · MPC · FC per subject<br/>fsLR-5k")]
    weights["connection weights<br/>SC consensus mask + log; 1/GD; positive MPC, FC"]
    projection["projection of the FC gradient<br/>weighted mean over each vertex's targets"]
    corr["r per subject<br/>Spearman with the network MPC gradient"]
    group["group r<br/>Fisher z, t-test across subjects"]
    nulls["spatial tests<br/>Moran on the gradient · topological · spin of the FC gradient"]
    connectomes --> weights --> projection --> corr --> group
    corr --> nulls
```

## Where to change what

| To change | Edit |
|---|---|
| where the data are on your machine | `config/paths.yaml` (copy `config/paths.example.yaml`) |
| a step's network, hemisphere or number of permutations (step 3) | `config/analysis.yaml` |
| any other parameter | the named constant at the top of the step (`pipeline/`) or library module |
| an analysis method | its module in `src/salience_switch/`, with a test in `tests/` |
| how a figure looks | the plot functions of its step; shared panels and the zoom-row window (`ZOOM_ROW_KW`) are in `plotting.py` |
| what may not be published | `tools/check_privacy.py` (used by the hook, the CI and the tests) |
| add a figure | a new step file with `compute()` and `plot()`, its name in `pipeline/run_all.py`, its settings in `config/analysis.yaml` |
| the composite layouts | `results/figure*.svg` in Inkscape (panels are linked by file name) |
| the documentation | `docs/` (the API pages come from the docstrings) |

## What is public and what is private

| Public, in git | Private, in `work_dir` only |
|---|---|
| code, tests, documentation | participant table, subject lists, file paths |
| template surfaces, atlases and histological maps (`data/`) | per-subject and per-channel tables, spectra caches |
| figure tables without identifiers (`source_data/`) and their manifests | run logs |
| figures (`results/`) | MNI iEEG atlas files |

One script, `tools/check_privacy.py`, defines what may not be published: private file
types and locations, participant IDs, and identifying columns or keys in `source_data/`.
`.githooks/pre-commit` runs it on the staged files (enable it once per clone with
`git config core.hooksPath .githooks`), the CI on every tracked file at every push, and
`tests/test_pipeline.py` on `source_data/`.

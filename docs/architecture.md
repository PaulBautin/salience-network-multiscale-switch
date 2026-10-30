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
    tests -.->|check privacy| source
```

## 2. Modules

`pipeline/run_all.py` runs the steps in order; each step is one file in `pipeline/`
and shares its command line (`--stage compute|plot|all`) through `pipeline/common.py`.
The steps use the library in `src/salience_switch/`, which holds no figure logic, no
paths and no study constants, so it can be reused on other data. Within the library,
only `connectivity` imports another module (`stats`).

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
    s2a["Step 2a · MNI iEEG atlas<br/>Fig 2a"]
    s2b["Step 2b · MICA iEEG spectra<br/>Fig 2b"]
    s2c["Step 2c · wake, N2, N3<br/>Fig 2c"]
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
| 2a | MNI iEEG atlas and its registered surfaces; step 1a gradient | — | per channel: template vertex, network, ES difference, BigBrain G2; statistic |
| 2b | electroMICA recordings and sensitivity maps; step 1a gradient | channel spectra | template maps of each measure; statistics; nulls; quintile spectra; similarity projection |
| 2c | as step 2b, for three states | channel spectra per state | template maps per state; band powers; statistics; nulls |
| 3 | step 1a gradient and subject list; SC, GD, MPC, FC connectomes | per-subject correlations | vertex-wise group projections; statistics; nulls; control |
| 3 schematic | as step 3 | example subject's ID | that subject's projection; connections of one vertex; summary |

Steps 1b and 1c read only public inputs, so they can be recomputed without the private
data; every step can be redrawn from `source_data/` alone.

## 4. Inside a step

The two stages communicate only through files. `compute` writes a `manifest.json` next
to the tables: the code version, the package versions, the settings and the sample
sizes of the run. Figures of individual channels (specparam fit quality, channel
similarity matrix) are drawn by `compute`, because their data are not public.

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
    test1c["1c: type composition of the network<br/>spin-permutation null"]
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
    psd["Welch spectra"]
    params["specparam per channel<br/>aperiodic exponent, band peak power"]
    surface["projection onto the template<br/>contact sensitivity maps"]
    test2["2b, 2c: Spearman vs the gradient<br/>Moran null, network vertices"]
    es["2a: spectral similarity to the top vs bottom<br/>gradient quarter, vs BigBrain G2"]
    recordings --> psd --> params --> surface --> test2
    psd --> es
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
| a step's network, hemisphere or number of permutations | `config/analysis.yaml` |
| any other parameter | the named constant at the top of the step (`pipeline/`) or library module |
| an analysis method | its module in `src/salience_switch/`, with a test in `tests/` |
| how a figure looks | the plot functions of its step; shared panels are in `plotting.py` |
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

`.githooks/pre-commit` refuses commits of private file types or of participant IDs
(enable it once per clone with `git config core.hooksPath .githooks`), the CI repeats
that check on every push, and `tests/test_pipeline.py` checks that `source_data/` holds
no identifier.

# Data availability

## Datasets

The analyses rely on the following datasets. None of their participant-level data are distributed with this repository.

- **MICA-PNI**: In vivo 7T MRI dataset (quantitative T1, DWI, resting-state fMRI) acquired at the McConnell Brain Imaging Centre, Montreal Neurological Institute. Available on [OpenNeuro](https://openneuro.org/datasets/ds005565).

- **MICA-MICs**: Multimodal imaging and connectome dataset acquired at the McConnell Brain Imaging Centre, Montreal Neurological Institute. Available on [OpenNeuro](https://openneuro.org/datasets/ds004472).

- **MICA iEEG**: Intracranial EEG recordings in wakefulness, N2 and N3 sleep, with the contact sensitivity maps of electroMICA (Figures 2b and 2c). Not distributed with this repository.

- **AHEAD**: Ultra-high-resolution multimodal 3D post-mortem human brain atlas (200 µm; Bielschowsky and Parvalbumin stainings). Available at [https://doi.org/10.21942/uva.16844500.v1](https://doi.org/10.21942/uva.16844500.v1).

- **BigBrain**: Ultra-high-resolution (20 µm) 3D histological reconstruction of a post-mortem human brain. Available at [https://bigbrain.loris.ca/main.php](https://bigbrain.loris.ca/main.php).

- **MNI open iEEG atlas**: Multicentre intracranial EEG dataset comprising recordings from 106 patients with intractable epilepsy mapped to a common stereotactic space (Figure 2a). Available at [https://mni-open-ieegatlas.research.mcgill.ca](https://mni-open-ieegatlas.research.mcgill.ca); its files are not redistributed here.

## What this repository shares

- **`source_data/`**: the tables behind every figure, one folder per pipeline step, each with a `manifest.json` recording the code version, package versions, settings and sample sizes of the run. They hold group-level maps and statistics, null distributions and, where a panel draws them, values of individual vertices or channels on the template; they hold no participant identifier, demographic, date, file path or electrode name. Every figure can be redrawn from them (`python pipeline/run_all.py --stage plot`).
- **`data/`**: the template surfaces, atlases and histological maps the analyses use (below).

## What stays private

The participant table, the subject lists with their derivative paths, every per-subject or per-channel table, the iEEG spectra caches, the run logs and the MNI iEEG atlas files are kept outside the repository, in the work directory set in `config/paths.yaml` (copy `config/paths.example.yaml`). They are needed only to recompute; `.githooks/pre-commit` and the CI refuse commits that would add them.

## Bundled maps (`data/`)

| Folder | Files | Source |
|---|---|---|
| `surfaces/` | fsLR-32k and fsLR-5k inflated surfaces; fsLR-5k spheres | fsLR template (micapipe) |
| `parcellations/` | Schaefer-400 parcellation at fsLR-32k (`conte69`) and fsLR-5k, with its look-up tables | Schaefer et al., 2018 (micapipe) |
| `parcellations/` | von Economo–Koskinas areas (`economo_conte69_*`) | García-Cabezas et al., 2020 (cortical types) |
| `parcellations/` | BigBrain and AHEAD (Bielschowsky, Parvalbumin) intensity profiles at fsLR-32k | BigBrain; AHEAD |
| `parcellations/` | second BigBrain histological gradient (`tpl-fs_LR_hemi-{L,R}_den-32k_desc-Hist_G2`) | BigBrainWarp (Paquola et al., 2021) |
| `parcellations/` | principal functional-connectivity gradient (`fc_gradient_fslr-32k_*`, `fc_gradient_fslr-5k_*`) | BrainSpace's conte69 template gradient (`brainspace.datasets.load_gradient('fc', idx=0)`; Vos de Wael et al., 2018, 2020) |

The fsLR-5k Schaefer-400 parcellation and FC gradient were resampled from fsLR-32k with Connectome Workbench (`tools/surface_resampling.sh`, which reproduces the bundled files exactly).

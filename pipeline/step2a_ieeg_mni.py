#!/usr/bin/env python
"""Step 2a: iEEG spectra along the salience-network gradient, MNI Open iEEG Atlas (replication).

The analysis of step 2b (Figure 2), repeated in a second cohort for a supplementary
replication figure. The atlas holds 1 min of artifact-free wakefulness recording of 1,772
bipolar channels with normal activity from 106 patients, located in a common stereotaxic
space (https://mni-open-ieegatlas.research.mcgill.ca/). Channels in the hippocampus and amygdala,
which the neocortical surface does not represent, are left out. The atlas joins the
recorded segments of a channel with 2-s buffers of zeros and pads them to 68 s, so each
Welch spectrum is averaged over the windows within the segments. Each channel is mapped
onto the fsLR-32k template through the atlas surfaces registered to it; with no leadfield,
its spatial sensitivity is a Gaussian footprint around its template vertex (half weight at
4 mm, zero below 5 % of the peak), and the footprints of both hemispheres are folded onto
one template, as the bipolar sensitivities of step 2b. From there the analysis and its
panels are those of step 2b (``ieeg_spectral``): the aperiodic exponent (primary) and the
peak power of five bands (secondary) against the gradient of one hemisphere
(``hemisphere``), each against the within-network Moran null with the add-one empirical
p, uncorrected; and the supplementary spectral-similarity projection of the FC gradient.

Inputs   MNI Open iEEG Atlas MatlabFile.mat and its template-registered surfaces
         (config/paths.yaml); step 1a fsLR-32k gradient (source_data).
Private  <work_dir>/step2a_ieeg_mni/: log.
Public   source_data/step2a_ieeg_mni/: the template maps of every measure and of the
         coverage, the statistics and Moran nulls, the spectra averaged within gradient
         quintiles, the spectral-similarity projection and its statistics, manifest.
Figures  results/figures/figure_2a_ieeg_mni_*.svg, the panels of step 2b; the main ridgeline
         shows the tests of Figure 2 (``MAIN_MEASURES``), so the composite
         results/figure_2a_mni.svg has the layout of results/figure_2b_mica.svg. The two figures
         of individual channels (specparam fit quality, channel spectral-similarity matrix) are
         drawn by the compute stage, as in step 2b.

Run      python pipeline/step2a_ieeg_mni.py [--stage compute|plot|all]
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from brainspace.datasets import load_conte69
from brainspace.mesh.mesh_io import read_surface
from scipy.io import loadmat
from scipy.spatial import cKDTree

import common
from ieeg_spectral import (N_PERMUTATIONS, plot_spectral_measures, plot_spectral_similarity,
                           spectral_similarity_projection, spectral_statistics)
from salience_switch import config
from salience_switch.ieeg import (N_LH, PSD_PARAMS, SPEC_BANDS, SPEC_FIT_RANGE, gaussian_channel_sensitivity,
                                  welch_psd_segments)
from salience_switch.provenance import write_manifest

STEP = "step2a_ieeg_mni"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16

PREFIX = "figure_2a_ieeg_mni"
TAG = "Figure 2A"
# Atlas regions (RegionName) off the neocortical surface: their channels are left out.
EXCLUDED_REGIONS = ("Hippocampus", "Amygdala")
# Zero buffers the atlas places between the recorded segments of a channel (s).
ZERO_GAP_S = 2.0
# Gaussian footprint of a channel: half weight at 4 mm (the kernel of Gao et al., 2020, on
# this atlas), zero below 5 % of the peak (the relative threshold of the electroMICA
# sensitivities of step 2b).
KERNEL_HALF_WEIGHT_MM = 4.0
KERNEL_REL_THRESH = 0.05
# Rows of the main ridgeline: those of Figure 2 (results/figure_2b_mica.svg),
# so the replication figure shows the same tests, whatever their p here.
MAIN_MEASURES = ("exponent", "theta")


def load_mni_channels(mat_path: Path, surfaces_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Wakefulness recordings of the atlas channels and their fsLR-32k vertices.

    Channels of the ``EXCLUDED_REGIONS`` are left out. Each remaining channel is assigned
    the nearest vertex of the atlas surface (both hemispheres pooled), carried to the
    template through the atlas surfaces registered to it (``L.anat.reg.surf.gii``,
    ``R.anat.reg.surf.gii``), then to the nearest fsLR-32k (conte69) midthickness vertex.

    Parameters
    ----------
    mat_path : Path
        The atlas MatlabFile.mat.
    surfaces_dir : Path
        Folder of the template-registered atlas surfaces.

    Returns
    -------
    data : (n_channels, n_samples) array
        Wakefulness recording of each channel, as distributed (zero buffers and padding).
    vertex : (n_channels,) int array
        fsLR-32k vertex of each channel, both hemispheres joined (left first).
    patient : (n_channels,) array
        Patient index of each channel.
    fs : float
        Sampling rate in Hz.

    Raises
    ------
    ValueError
        If the recordings are not at the analysis rate ``PSD_PARAMS['fs_target']``.
    """
    mat = loadmat(mat_path, squeeze_me=True,
                  variable_names=['Data_W', 'SamplingFrequency', 'Patient', 'ChannelRegion', 'RegionName',
                                  'ChannelPosition', 'NodesLeft', 'NodesRight'])
    fs = float(mat['SamplingFrequency'])
    if fs != PSD_PARAMS['fs_target']:
        raise ValueError(f"The atlas recordings are at {fs:g} Hz, not the analysis rate {PSD_PARAMS['fs_target']:g} Hz.")
    region = np.asarray(mat['RegionName'], dtype=str)[np.asarray(mat['ChannelRegion'], dtype=int) - 1]
    keep = ~np.isin(region, EXCLUDED_REGIONS)
    logger.info("Channels excluded (off the neocortical surface): "
                + ", ".join(f"{name} {int((region == name).sum())}" for name in EXCLUDED_REGIONS)
                + f"; {int(keep.sum())}/{keep.size} channels kept")

    atlas_vertex = cKDTree(np.vstack((mat['NodesLeft'], mat['NodesRight']))).query(mat['ChannelPosition'][keep])[1]
    surf_reg = [read_surface(str(surfaces_dir / f'{h}.anat.reg.surf.gii'), itype='gii') for h in ('L', 'R')]
    position_reg = np.vstack([s.GetPoints() for s in surf_reg])[atlas_vertex]
    vertex = cKDTree(np.asarray(load_conte69(join=True).GetPoints())).query(position_reg)[1]
    data = np.asarray(mat['Data_W'], dtype=float).T[keep]
    return data, vertex, np.asarray(mat['Patient'])[keep], fs


def load_channel_spectra(paths: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-channel Welch spectra (wakefulness) and the Gaussian footprints of the channels.

    Returns the frequency bins, the (n_channels, n_freqs) PSD, the (n_channels, N_LH)
    footprints on the folded template, row-aligned, and the number of patients.
    """
    data, vertex, patient, fs = load_mni_channels(paths["mni_ieeg"], paths["mni_ieeg_surfaces"])
    logger.info(f"MNI iEEG loaded: {np.unique(patient).size} patients, {len(vertex)} bipolar channels "
                f"({int((vertex < N_LH).sum())} left, {int((vertex >= N_LH).sum())} right), "
                f"{data.shape[1] / fs:g} s at {fs:g} Hz")
    n_zero = (data == 0).sum(axis=1) / fs
    f, pxx, n_windows = welch_psd_segments(data, fs, fmin=PSD_PARAMS['fmin'], fmax=PSD_PARAMS['fmax'],
                                           window_sec=PSD_PARAMS['window_sec'], overlap_sec=PSD_PARAMS['overlap_sec'],
                                           min_gap_s=ZERO_GAP_S)
    logger.info(f"Zero buffers and padding: {n_zero.min():g}-{n_zero.max():g} s per channel; Welch windows within "
                f"the segments: min {n_windows.min()}, median {np.median(n_windows):g}, max {n_windows.max()} per channel")
    surf_lh, surf_rh = load_conte69(join=False)
    sens = gaussian_channel_sensitivity(np.asarray(surf_lh.GetPoints()), np.asarray(surf_rh.GetPoints()), vertex,
                                        half_weight_mm=KERNEL_HALF_WEIGHT_MM, rel_thresh=KERNEL_REL_THRESH)
    logger.info(f"PSD computed once: {pxx.shape[0]} channels x {pxx.shape[1]} freqs; footprint stack {sens.shape}, "
                f"median {np.median((sens > 0).sum(axis=1)):g} vertices per channel")
    return f, pxx, sens, np.unique(patient).size


def compute(settings: dict, paths: dict) -> None:
    """Spectral measures and the spectral-similarity projection, with their Moran tests."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    radius = KERNEL_HALF_WEIGHT_MM * np.sqrt(np.log2(1 / KERNEL_REL_THRESH))
    logger.info("Dataset        : MNI Open iEEG Atlas (wakefulness; 1 min of artifact-free bipolar recording per channel)")
    logger.info("Preprocessing  : as distributed (80 Hz anti-aliasing low-pass, 200 Hz, 50/60 Hz line-noise removal, "
                "segment means removed); no further filtering")
    logger.info(f"PSD            : Welch method, Hamming window 2s, overlap 1s, windows within the segments between the "
                f"{ZERO_GAP_S:g}-s zero buffers, normalized to unit sum")
    logger.info(f"Spectral model : specparam, knee mode (knee >= 0), Gaussian peaks, {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz "
                f"(below the 50/60 Hz line frequencies and the 80 Hz anti-aliasing filter); peak width 1-12 Hz, "
                f"<= 6 peaks, min height 0.05, threshold 2 SD")
    logger.info("Peak bands     : " + ", ".join(f"{b} {lo:g}-{hi:g} Hz" for b, (lo, hi) in SPEC_BANDS.items()))
    logger.info(f"Sensitivity    : Gaussian footprint around each channel's fsLR-32k vertex (Euclidean distance on its "
                f"hemisphere's midthickness), half weight at {KERNEL_HALF_WEIGHT_MM:g} mm, zero below "
                f"{KERNEL_REL_THRESH:.0%} of the peak (beyond {radius:.1f} mm) -> LH+RH fold")
    logger.info("Statistic      : aperiodic exponent (primary) + band power (secondary) vs MPC gradient, uncorrected; supplementary: spectral-similarity-weighted projection of the FC gradient (group level)")
    logger.info(f"Null model     : within-network Moran randomization (n_rep={N_PERMUTATIONS}, procedure=singleton, random_state=0), add-one empirical p")
    logger.info("Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels; maps shown on both hemispheres (LH+RH folded template)")
    logger.info(f"Analysis network: {network}; statistics on the {hemi} network vertices of the bilateral step 1a gradient")
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    f, pxx, sens, n_patients = load_channel_spectra(paths)

    spectral = spectral_statistics(f, pxx, sens, df_yeo_surf, hemi=hemi, network=network, prefix=PREFIX, tag=TAG)
    similarity = spectral_similarity_projection(pxx, sens, df_yeo_surf, hemi=hemi, network=network, prefix=PREFIX,
                                                tag=TAG)
    for name, table in spectral.items():
        config.write_table(table, source / f"spectral_{name}_{hemi}.tsv")
    for name, table in similarity.items():
        config.write_table(table, source / f"similarity_{name}_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network, hemisphere=hemi,
                   n_patients=int(n_patients), n_channels=int(pxx.shape[0]), n_permutations=N_PERMUTATIONS,
                   kernel_half_weight_mm=KERNEL_HALF_WEIGHT_MM, kernel_rel_thresh=KERNEL_REL_THRESH)


def plot(settings: dict) -> None:
    """Replication panels and supplements, from source_data only."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    spectral = {name: config.read_table(source / f"spectral_{name}_{hemi}.tsv")
                for name in ("maps", "stats", "null", "spectra")}
    similarity = {name: config.read_table(source / f"similarity_{name}_{hemi}.tsv")
                  for name in ("vertices", "targets", "stats")}
    plot_spectral_measures(spectral, df_yeo_surf, hemi, network, prefix=PREFIX, tag=TAG, main_measures=MAIN_MEASURES)
    plot_spectral_similarity(similarity, df_yeo_surf, hemi, network, prefix=PREFIX)


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 2a: iEEG spectra along the salience-network gradient, MNI Open iEEG Atlas "
                                     "(replication of Figure 2).")

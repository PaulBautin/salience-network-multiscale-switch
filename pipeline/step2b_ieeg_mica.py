#!/usr/bin/env python
"""Step 2b: iEEG spectra along the salience-network gradient, MICA dataset (Figure 2).

Wakefulness recordings of the MICA iEEG dataset (electroMICA derivatives) are cut into
bipolar channels, their Welch spectra parameterised with specparam (aperiodic exponent
and per-band oscillatory peak power from one fit per channel), and each channel measure is
projected onto the fsLR-32k template through the contact sensitivity maps. The bipolar
sensitivities of both hemispheres are folded onto one template, so the maps are shown on
both hemispheres; the statistics run on one hemisphere (``hemisphere``) against its
values of the bilateral step 1a gradient. The primary test is the aperiodic exponent
against the gradient; the five band powers are secondary; each is tested against the
within-network Moran null with the add-one empirical p, uncorrected. A supplementary
analysis projects the FC gradient through the spectral similarity of the vertices. The
analysis and its panels are shared with step 2a (``ieeg_spectral``); this step supplies the
MICA spectra and sensitivity maps.

Inputs   electroMICA derivatives and the original recordings (config/paths.yaml); step 1a
         fsLR-32k gradient (source_data).
Private  <work_dir>/step2b_ieeg_mica/: the channel spectra cache (patient IDs), log.
Public   source_data/step2b_ieeg_mica/: the template maps of every measure and of the
         coverage, the statistics and Moran nulls, the spectra averaged within gradient
         quintiles, the spectral-similarity projection and its statistics, manifest.
Figures  results/figures/figure_2b_ieeg_mica_*.svg. The two figures of individual channels
         (specparam fit quality, channel spectral-similarity matrix) are drawn by the compute
         stage, since their data are not public.

Run      python pipeline/step2b_ieeg_mica.py [--stage compute|plot|all]
"""
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common
from ieeg_spectral import (N_PERMUTATIONS, plot_spectral_measures, plot_spectral_similarity,
                           spectral_similarity_projection, spectral_statistics)
from salience_switch import config
from salience_switch.ieeg import PSD_PARAMS, SPEC_BANDS, SPEC_FIT_RANGE, load_channel_sensitivity, load_or_compute_channel_psd
from salience_switch.provenance import write_manifest

STEP = "step2b_ieeg_mica"
GRADIENT_STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

plt.rcParams['font.size'] = 16

PREFIX = "figure_2b_ieeg_mica"
TAG = "Figure 2B"


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def load_channel_spectra(paths: dict, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-channel Welch spectra (stage W) and their bipolar sensitivity maps.

    Returns the frequency bins, the (n_channels, n_freqs) PSD, the (n_channels, N_LH)
    absolute bipolar sensitivity maps (NaN zeroed), row-aligned, and the number of patients.
    """
    logger.info(f"Raw recordings : {paths['ieeg_raw']}")
    df_channel_data, f, pxx_raw = load_or_compute_channel_psd(str(paths["ieeg_raw"]), cache_dir / "channel_psd_W.pkl",
                                                              stage='W', **PSD_PARAMS)
    logger.info(f"Channel data: {df_channel_data['Subject'].nunique()} subjects, {len(df_channel_data)} bipolar channels")
    # electroMICA bipolar sensitivity, row-aligned with the channels: per-hemisphere signed
    # difference -> area-scaled thresholds -> abs -> LH+RH fold.
    sens = load_channel_sensitivity(df_channel_data, paths["electromica"])
    logger.info(f"PSD computed once: {pxx_raw.shape[0]} channels x {pxx_raw.shape[1]} freqs; sensitivity stack {sens.shape}")
    return f, pxx_raw, sens, df_channel_data['Subject'].nunique()


def compute(settings: dict, paths: dict) -> None:
    """Spectral measures and the spectral-similarity projection, with their Moran tests."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    logger.info("Dataset        : MICA iEEG (BIDS_iEEG, sub-PX*, ses-01, stage-W wakefulness)")
    logger.info("Sensitivity maps: electroMICA leadfield (signed), fsLR-32k; per-hemi bipolar difference |L1-L2| -> area-normalised density thresholds (0.001 Vm/A absolute, 5% of the channel's second-largest density) -> LH+RH fold")
    logger.info("Preprocessing  : truncated to the shortest recording, Butterworth bandpass 0.5-80 Hz (order 4, second-order sections, zero phase), downsampled to 200 Hz, demeaned")
    logger.info("PSD            : Welch method, Hamming window 2s, overlap 1s, normalized to unit sum")
    logger.info(f"Spectral model : specparam, knee mode (knee >= 0), Gaussian peaks, {SPEC_FIT_RANGE[0]:g}-{SPEC_FIT_RANGE[1]:g} Hz "
                f"(below the 60 Hz notch and the 80 Hz low-pass transition band); peak width 1-12 Hz, <= 6 peaks, "
                f"min height 0.05, threshold 2 SD")
    logger.info("Peak bands     : " + ", ".join(f"{b} {lo:g}-{hi:g} Hz" for b, (lo, hi) in SPEC_BANDS.items()))
    logger.info("Statistic      : aperiodic exponent (primary) + band power (secondary) vs MPC gradient, uncorrected; supplementary: spectral-similarity-weighted projection of the FC gradient (group level)")
    logger.info(f"Null model     : within-network Moran randomization (n_rep={N_PERMUTATIONS}, procedure=singleton, random_state=0), add-one empirical p")
    logger.info("Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels; maps shown on both hemispheres (LH+RH folded template)")
    logger.info(f"Analysis network: {network}; statistics on the {hemi} network vertices of the bilateral step 1a gradient")
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    f, pxx_raw, sens, n_patients = load_channel_spectra(paths, config.work_dir(STEP, paths))

    spectral = spectral_statistics(f, pxx_raw, sens, df_yeo_surf, hemi=hemi, network=network, prefix=PREFIX, tag=TAG)
    similarity = spectral_similarity_projection(pxx_raw, sens, df_yeo_surf, hemi=hemi, network=network, prefix=PREFIX,
                                                tag=TAG)
    for name, table in spectral.items():
        config.write_table(table, source / f"spectral_{name}_{hemi}.tsv")
    for name, table in similarity.items():
        config.write_table(table, source / f"similarity_{name}_{hemi}.tsv")
    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network, hemisphere=hemi,
                   n_patients=int(n_patients), n_channels=int(pxx_raw.shape[0]), n_permutations=N_PERMUTATIONS)


def plot(settings: dict) -> None:
    """Figure 2 panels and supplements, from source_data only."""
    hemi, network = settings["hemisphere"], settings["network"]
    source = config.source_dir(STEP)
    df_yeo_surf = config.read_table(config.source_dir(GRADIENT_STEP) / "surface_fslr32k_both.tsv")
    spectral = {name: config.read_table(source / f"spectral_{name}_{hemi}.tsv")
                for name in ("maps", "stats", "null", "spectra")}
    similarity = {name: config.read_table(source / f"similarity_{name}_{hemi}.tsv")
                  for name in ("vertices", "targets", "stats")}
    plot_spectral_measures(spectral, df_yeo_surf, hemi, network, prefix=PREFIX, tag=TAG)
    plot_spectral_similarity(similarity, df_yeo_surf, hemi, network, prefix=PREFIX)


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 2b: iEEG spectra along the salience-network gradient (Figure 2).")

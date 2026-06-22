# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 1a - Local microstructural heterogeneity of the salience network 
#
# This script processes MICA-PNI derivatives to extract T1 microstructural
# profiles and computes the diffusion-map MPC gradient within the
# Salience/Ventral Attention network, at two surface resolutions:
#   Part 1 (fsLR-32k) drives the profile and brain figures below.
#   Part 2 (fsLR-5k)  builds the per-subject file table + gradient that
#                     figure_2_distance.py consumes.
#
# Outputs:
#   results/figures/figure_1a_profiles.svg, figure_1a_brain.svg
#   data/dataframes/figure_1a_pni_to_mics{,_5k}.csv   (subject -> file paths)
#   data/dataframes/df_1a_{hemi}{,_fslr5k}.tsv         (surface table + gradient)
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1a_t1map.py \
#   -pni_deriv /data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0 \
#   -mics_deriv /data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0 \
#   -hemi LH
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24" 
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres

from src.atlas_load import load_yeo_atlas, load_yeo_surf_5k, load_t1_salience_profiles, compute_t1map, compute_network_mask
from src.gradient_computation import compute_t1_gradient
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)

# Matplotlib globals
plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'


# ---------------------------------------------------------------------------
# Per-subject input files
# ---------------------------------------------------------------------------
# Each entry is (column, derivatives_root, glob_template). `root` selects the
# derivatives tree: 'pni' = MICA-PNI (microstructure, geodesic distance, FC),
# 'mics' = MICA-MICs (diffusion connectomes). The template is formatted per
# subject row with ID_PNI / session / ID_MICs, then globbed (one file/subject).

# fsLR-32k (Part 1): only the T1 intensity profile is consumed here.
PATHS_32K = [
    ("path_t1_profile", "pni",
     "sub-{ID_PNI}/ses-{session}/mpc/acq-T1map/"
     "sub-{ID_PNI}_ses-{session}_surf-fsLR-32k_desc-intensity_profiles.shape.gii"),
]
REQUIRED_32K = ["path_t1_profile"]

# fsLR-5k (Part 2): consumed downstream by figure_2_distance.py. FC is optional
# (a session-matched rest run may be absent), so it is not in REQUIRED_5K and a
# missing FC leaves NaN rather than dropping the subject from the other modalities.
PATHS_5K = [
    ("path_t1_profile_5k", "pni",
     "sub-{ID_PNI}/ses-{session}/mpc/acq-T1map/"
     "sub-{ID_PNI}_ses-{session}_surf-fsLR-5k_desc-intensity_profiles.shape.gii"),
    ("path_mpc_5k", "pni",
     "sub-{ID_PNI}/ses-{session}/mpc/acq-T1map/"
     "sub-{ID_PNI}_ses-{session}_surf-fsLR-5k_desc-MPC.shape.gii"),
    ("path_sc_5k", "mics",
     "sub-{ID_MICs}/ses-01/dwi/connectomes/"
     "sub-{ID_MICs}_ses-01_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-connectome.shape.gii"),
    ("path_sc_dist_5k", "mics",
     "sub-{ID_MICs}/ses-01/dwi/connectomes/"
     "sub-{ID_MICs}_ses-01_surf-fsLR-5k_desc-iFOD2-40M-SIFT2_full-edgeLengths.shape.gii"),
    ("path_dist_5k", "pni",
     "sub-{ID_PNI}/ses-{session}/dist/"
     "sub-{ID_PNI}_ses-{session}_surf-fsLR-5k_GD.shape.gii"),
    ("path_fc_5k", "pni",
     "sub-{ID_PNI}/ses-{session}/func/desc-me_task-rest_bold/surf/"
     "sub-{ID_PNI}_ses-{session}_surf-fsLR-5k_desc-FC.shape.gii"),
]
REQUIRED_5K = ["path_t1_profile_5k", "path_mpc_5k", "path_sc_5k",
               "path_sc_dist_5k", "path_dist_5k"]  # FC optional


def get_parser():
    """Configure and return the argument parser."""
    parser = argparse.ArgumentParser(
        description="Process PNI derivatives and surfaces for T1 microstructural profiles.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    mandatory = parser.add_argument_group("MANDATORY ARGUMENTS")
    mandatory.add_argument(
        "-pni_deriv", 
        type=str, 
        required=True,
        help="Absolute path to the PNI derivatives folder (e.g., /data/mica/mica3/...)"
    )
    mandatory.add_argument(
        "-mics_deriv",
        type=str,
        required=True,
        help="Absolute path to the MICs derivatives folder (e.g., /data/mica/mica3/...)"
    )
    optional = parser.add_argument_group("OPTIONAL ARGUMENTS")
    optional.add_argument(
        "-hemi",
        type=str,
        default="both",
        choices=["both", "LH", "RH"],
        help="Hemisphere for gradient computation: 'both', 'LH', or 'RH' (default: both)"
    )
    return parser


def plot_gradient_profiles(
    df_yeo_surf: pd.DataFrame,
    t1_salience_profiles: np.ndarray,
    screenshot_path: Path,
    network: str = 'SalVentAttn',
    hemisphere: str = 'both',
) -> None:
    """Plot per-vertex T1 profiles within `network`, coloured by the MPC gradient.

    Each network vertex's intracortical T1 profile is drawn faint and coloured by
    its first MPC-gradient value (diverging scale, clipped to [-3, 3]); the mean
    profiles of the bottom- and top-quartile gradient vertices are overlaid. The
    vertex axis of `t1_salience_profiles` is in the same order as the masked rows
    of `df_yeo_surf` (both follow vertex index over the network mask), so colours
    and quartile masks align with the profiles. Does not mutate `df_yeo_surf`.

    Parameters
    ----------
    df_yeo_surf : pd.DataFrame
        Surface table carrying the `t1_gradient1_{network}` column.
    t1_salience_profiles : np.ndarray, shape (n_subjects, n_depths, n_network_vertices)
        Per-subject T1 profiles for the network vertices.
    screenshot_path : Path
        Output SVG path.
    network : str
        Yeo 7-network label whose gradient colours the profiles.
    hemisphere : {'both', 'LH', 'RH'}
        Restricts the vertices (and the quartile thresholds) to one hemisphere.
    """
    grad_col = f"t1_gradient1_{network}"
    net_mask = compute_network_mask(df_yeo_surf, network, hemisphere)
    grad_sn = df_yeo_surf.loc[net_mask, grad_col].to_numpy()

    # Bottom/top gradient quartiles, computed within the plotted vertex set.
    low_q, high_q = np.nanquantile(grad_sn, [0.25, 0.75])
    bottom_mask = grad_sn <= low_q
    top_mask = grad_sn >= high_q
    profiles = np.mean(t1_salience_profiles, axis=0)
    bottom_profiles = np.mean(t1_salience_profiles[:, :, bottom_mask], axis=0)
    top_profiles = np.mean(t1_salience_profiles[:, :, top_mask], axis=0)

    # Plotting setup
    fig, ax = plt.subplots(figsize=(6, 6))
    custom_cmap = plt.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=-3, vmax=3)
    colors = custom_cmap(norm(grad_sn))

    y_axis = np.linspace(0, 1, profiles.shape[0])

    # Plot individual profiles (Consider LineCollection here in the future if this loop is slow)
    for i, col in enumerate(colors):
        ax.plot(profiles[:, i] / 1000, y_axis, color=col, alpha=0.1, rasterized=True)

    ax.plot(np.mean(bottom_profiles, axis=1) / 1000, y_axis, color='b', alpha=0.8, label='bottom 25%')
    ax.plot(np.mean(top_profiles, axis=1) / 1000, y_axis, color='r', alpha=0.8, label='top 25%')
    
    # Aesthetics
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.legend(frameon=False, loc='lower right', bbox_to_anchor=(1, 0.1))
    plt.xlim(1.4, 2.5)
    plt.ylabel("Intracortical depth")
    plt.xlabel("Long. relaxation time (s)")
    plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
    plt.axhline(y=1, color='k', linestyle='--', linewidth=1)
    plt.yticks([0, 1], ['pial', 'WM'])
    plt.gca().invert_yaxis()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(screenshot_path)


def _filtered_pni_subjects(mica_pni_csv: Path) -> pd.DataFrame:
    """Healthy-control PNI subjects (PNC*, ses-a1) matched to a MICA-MICs scan."""
    df = pd.read_csv(mica_pni_csv)[["ID_PNI", "session", "ID_MICs"]].drop_duplicates()
    return df[df["ID_PNI"].str.contains("PNC", na=False)
              & df["session"].str.contains("a1", na=False)
              & df["ID_MICs"].str.contains("HC", na=False)]


def _glob_subject_paths(
    df: pd.DataFrame, pni_deriv: Path, mics_deriv: Path, path_specs: list,
) -> pd.DataFrame:
    """Add one path column per (column, root, template) spec by globbing derivatives.

    Each cell is the list of files the template resolves to for that subject row
    (usually one; empty when the file is absent).
    """
    roots = {"pni": pni_deriv, "mics": mics_deriv}
    df = df.copy()
    for col, root, template in path_specs:
        base = roots[root]
        # base/template bound as defaults so each lambda captures this iteration's spec.
        df[col] = df.apply(
            lambda row, base=base, template=template: list(base.glob(template.format(
                ID_PNI=row["ID_PNI"], session=row["session"], ID_MICs=row["ID_MICs"]))),
            axis=1,
        )
    return df


def build_or_load_subject_table(
    csv_path: Path, mica_pni_csv: Path, pni_deriv: Path, mics_deriv: Path,
    path_specs: list, required: list, log_lines: list,
) -> pd.DataFrame:
    """Return the cached subject->file table, or build it by globbing derivatives.

    One row per subject with one file-path column per `path_specs` entry. Rows
    missing any `required` path are dropped; other columns may be NaN (e.g. the
    optional FC run), so a missing optional input never drops the subject from
    the required modalities.
    """
    if csv_path.exists():
        logger.info(f"Found existing subject table at {csv_path}. Loading...")
        return pd.read_csv(csv_path)

    df = _filtered_pni_subjects(mica_pni_csv)
    df = _glob_subject_paths(df, pni_deriv, mics_deriv, path_specs)
    # Each cell is the list of glob matches (expected 0 or 1 file per subject).
    # Take the first match per column independently (NaN when none) rather than
    # exploding each list column in turn, which would cross-join any column with
    # >1 match into a cartesian product and silently duplicate subjects.
    for col, _root, _template in path_specs:
        df[col] = df[col].apply(lambda paths: str(paths[0]) if paths else np.nan)
    df = df.dropna(subset=required)

    logger.info(f"Participants   : N={len(df)} (MICA-PNI, ses-a1, healthy controls matched to MICA-MICs)")
    for line in log_lines:
        logger.info(line)
    df.to_csv(csv_path, index=False)
    return df


def gradient_and_profiles(
    df_yeo: pd.DataFrame, df_pni: pd.DataFrame, t1_col: str,
    cache_path: Path, hemisphere: str, network: str = "SalVentAttn",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Attach the network T1 gradient + mean T1map to `df_yeo`; return masked profiles.

    The network mask and per-subject T1 profiles are loaded either way (the
    profiles drive the profile figure). When `cache_path` exists the surface
    table (with the gradient columns) is read back from it; otherwise the
    diffusion-map MPC gradient is computed and the table is cached. The profile
    vertex axis is in the same order as the masked rows of `df_yeo`.
    """
    net_mask = compute_network_mask(df_yeo, network, hemisphere)
    profiles = load_t1_salience_profiles(df_pni[t1_col].tolist(), net_mask)
    if cache_path.exists():
        logger.info(f"Found existing gradient table at {cache_path}. Loading...")
        df_yeo = pd.read_csv(cache_path, sep="\t")
    else:
        df_yeo.loc[net_mask, f"t1_gradient1_{network}"] = compute_t1_gradient(profiles)
        df_yeo.loc[net_mask, "T1map"] = compute_t1map(profiles)
        df_yeo.to_csv(cache_path, sep="\t", index=False)
    return df_yeo, profiles


def main():
    parser = get_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    pni_deriv = Path(args.pni_deriv)
    mics_deriv = Path(args.mics_deriv)

    logger = setup_manuscript_logger("figure_1a_t1map", project_root, args)
    logger.info("Surface space  : fsLR-32k (Part 1) and fsLR-5k (Part 2)")
    logger.info("Parcellation   : Schaefer-400 with Yeo 7-network labels")
    logger.info("Network        : SalVentAttn (Salience/Ventral Attention)")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"MICA-PNI derivatives: {pni_deriv}")
    logger.info(f"MICA-MICs derivatives: {mics_deriv}")

    mica_pni_csv = project_root / "data/dataframes/MICA_PNI.csv"

    # surfaces (fsLR-32k inflated for screenshots; conte69 for atlas borders)
    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf5k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-5k.L.inflated.surf.gii', itype='gii')
    surf5k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-5k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)

    ######### Part 1 -- fsLR-32k T1 gradient (drives the profile + brain figures)
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)
    df_pni = build_or_load_subject_table(
        project_root / "data/dataframes/figure_1a_pni_to_mics.csv",
        mica_pni_csv, pni_deriv, mics_deriv, PATHS_32K, REQUIRED_32K,
        log_lines=[
            "T1 profiles    : acq-T1map, fsLR-32k surface, 14 intracortical depths",
            "Gradient       : diffusion maps, normalized angle kernel, sparsity=0.9, n_components=10, procrustes alignment",
        ],
    )
    df_yeo_surf, t1_salience_profiles = gradient_and_profiles(
        df_yeo_surf, df_pni, "path_t1_profile",
        project_root / f"data/dataframes/df_1a_{args.hemi}.tsv", args.hemi,
    )

    ######### Part 2 -- fsLR-5k T1 gradient (subject table consumed by figure_2_distance.py)
    df_yeo_surf_5k = load_yeo_surf_5k(project_root)
    df_pni_5k = build_or_load_subject_table(
        project_root / "data/dataframes/figure_1a_pni_to_mics_5k.csv",
        mica_pni_csv, pni_deriv, mics_deriv, PATHS_5K, REQUIRED_5K,
        log_lines=[
            "T1 profiles    : acq-T1map, fsLR-5k surface, 14 intracortical depths",
            "MPC            : fsLR-5k vertex-level MPC matrix",
            "Connectomes    : iFOD2 40M streamlines, SIFT2-weighted, fsLR-5k",
            "FC             : resting-state (desc-me_task-rest_bold), fsLR-5k (optional)",
            "Gradient       : diffusion maps, normalized angle kernel, sparsity=0.9, n_components=10, procrustes alignment",
        ],
    )
    df_yeo_surf_5k, _ = gradient_and_profiles(
        df_yeo_surf_5k, df_pni_5k, "path_t1_profile_5k",
        project_root / f"data/dataframes/df_1a_{args.hemi}_fslr5k.tsv", args.hemi,
    )

    ######### Figures (fsLR-32k)
    screenshot_path = project_root / "results/figures/figure_1a_profiles.svg"
    logger.info(f"Generating qt1 profiles figure at {screenshot_path}")
    plot_gradient_profiles(df_yeo_surf, t1_salience_profiles, screenshot_path, network='SalVentAttn', hemisphere=args.hemi)

    screenshot_path = project_root / "results/figures/figure_1a_brain.svg"
    logger.info(f"Generating brain hemispheres screenshot at {screenshot_path}")
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range=(-3,3), transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)

    screenshot_path = project_root / "results/figures/figure_1a_brain_5k.svg"
    logger.info(f"Generating brain hemispheres screenshot at {screenshot_path}")
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range=(-3,3), transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)


if __name__ == "__main__":
    main()
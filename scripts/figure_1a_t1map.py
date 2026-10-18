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
from src.surface_plotting import plot_hemispheres_zoom, salience_zoom_rois, mask_outline, SALIENCE_Z_RANGE, SALIENCE_Z_LABELS

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

# Axes box (pt from the top-left of the 432-pt, 6-in canvas) shared by the profiles and
# strata panels, so pial and WM sit at the same heights in both. It is the box
# tight_layout gave the profiles panel at 21 pt; figure_1b_contextualisation's
# PROFILES_AXIS_PT copies its x span for the forest plot.
PANEL_CANVAS_PT = 432.0
PANEL_AXES_PT = dict(left=92.72, right=386.01, top=22.68, bottom=358.0)
# Depth axis of both panels (0 = pial, 1 = WM, pial at the top), with the 5 % margin
# matplotlib's autoscale gave the profiles panel, so pial/WM sit at the same heights.
PANEL_YLIM = (1.05, -0.05)


def _apply_panel_axes(fig) -> None:
    """Place a 6-in panel's axes on ``PANEL_AXES_PT`` (instead of tight_layout)."""
    c = PANEL_CANVAS_PT
    fig.subplots_adjust(left=PANEL_AXES_PT['left'] / c, right=PANEL_AXES_PT['right'] / c,
                        top=1 - PANEL_AXES_PT['top'] / c, bottom=1 - PANEL_AXES_PT['bottom'] / c)


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

    # Font: the 6-in panel is shown at ~0.41x in the Figure 1 layout, so 21 pt reads
    # like the ~8.8-pt text of the zoom rows (30 pt shown at ~0.29x).
    with plt.rc_context({'font.size': 21}):
        # Plotting setup
        fig, ax = plt.subplots(figsize=(6, 6))
        custom_cmap = plt.get_cmap("coolwarm")
        norm = mpl.colors.Normalize(*SALIENCE_Z_RANGE)
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
        plt.ylim(*PANEL_YLIM)
        plt.grid(False)
        _apply_panel_axes(fig)
        plt.savefig(screenshot_path)
        plt.close(fig)


def plot_gradient_strata(
    df_yeo_surf: pd.DataFrame,
    t1_salience_profiles: np.ndarray,
    screenshot_path: Path,
    network: str = 'SalVentAttn',
    hemisphere: str = 'both',
    n_bins: int = 5,
) -> None:
    """Stratigraphy columns: mean qT1 at each depth for each gradient quantile bin.

    Network vertices are split into ``n_bins`` equal-count bins of the MPC gradient
    (quintiles by default) and each bin is drawn as one column of blocks, one block
    per sampled depth (pial at the top, white matter at the bottom). Each block is
    coloured by the raw qT1 averaged over subjects and the bin's vertices on a linear
    grey scale, as a qT1 map is displayed: black at the lowest and white at the
    highest plotted value (the window spans exactly the blocks' range, which maximises
    the contrast between columns without clipping any block). Each column is outlined
    so its brightest block stays visible on a white page, and a cap below it takes the
    coolwarm colour of the bin's median gradient value (same scale as the brain map).
    Blocks are vector rectangles, so the sampling stays visible in the SVG. The
    vertex axis of `t1_salience_profiles` follows the masked rows of `df_yeo_surf`,
    as in :func:`plot_gradient_profiles`.

    Parameters
    ----------
    df_yeo_surf : pd.DataFrame
        Surface table carrying the `t1_gradient1_{network}` column.
    t1_salience_profiles : np.ndarray, shape (n_subjects, n_depths, n_network_vertices)
        Per-subject T1 profiles (ms) for the network vertices.
    screenshot_path : Path
        Output SVG path.
    network : str
        Yeo 7-network label whose gradient defines the bins.
    hemisphere : {'both', 'LH', 'RH'}
        Restricts the vertices (and the bin edges) to one hemisphere.
    n_bins : int
        Number of equal-count gradient bins (columns).
    """
    grad_col = f"t1_gradient1_{network}"
    net_mask = compute_network_mask(df_yeo_surf, network, hemisphere)
    grad_sn = df_yeo_surf.loc[net_mask, grad_col].to_numpy()
    bins = pd.qcut(grad_sn, n_bins, labels=False)
    mean_profiles = np.mean(t1_salience_profiles, axis=0) / 1000  # (n_depths, n_vertices), seconds
    strata = np.stack([mean_profiles[:, bins == k].mean(axis=1) for k in range(n_bins)])  # (n_bins, n_depths)
    medians = [np.median(grad_sn[bins == k]) for k in range(n_bins)]
    for k in range(n_bins):
        logger.info(f"[Figure 1A] strata bin {k + 1}/{n_bins}: {int((bins == k).sum())} vertices, "
                    f"gradient median {medians[k]:+.2f}")

    n_depths = strata.shape[1]
    # Linear grey scale windowed on the blocks' own range (black = lowest, white = highest).
    t1_norm = mpl.colors.Normalize(vmin=strata.min(), vmax=strata.max())
    t1_cmap = plt.get_cmap('gray')
    grad_cmap, grad_norm = plt.get_cmap('coolwarm'), mpl.colors.Normalize(*SALIENCE_Z_RANGE)
    col_w, band_h = 0.8, 1.0 / n_depths  # wide columns: abutting greys compare better

    # Same canvas, font and axes box as figure_1a_profiles.svg (6-in panel shown at
    # ~0.41x in the Figure 1 layout, so 21 pt reads like the ~8.8-pt zoom-row text).
    with plt.rc_context({'font.size': 21}):
        fig, ax = plt.subplots(figsize=(6, 6))
        for k in range(n_bins):
            for d in range(n_depths):
                color = t1_cmap(t1_norm(strata[k, d]))
                # Edge in the face colour hides antialiasing seams between blocks.
                ax.add_patch(mpl.patches.Rectangle((k - col_w / 2, d * band_h), col_w, band_h,
                                                   facecolor=color, edgecolor=color, linewidth=0.3))
            ax.add_patch(mpl.patches.Rectangle((k - col_w / 2, 0), col_w, 1, fill=False,
                                               edgecolor='0.4', linewidth=0.8))
            ax.add_patch(mpl.patches.Rectangle((k - col_w / 2, 1.01), col_w, 0.03, clip_on=False,
                                               facecolor=grad_cmap(grad_norm(medians[k])), edgecolor='none'))
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax.axhline(y=1, color='k', linestyle='--', linewidth=1)
        ax.set_xlim(-0.6, n_bins + 0.2)  # free space on the right for the colour bar
        ax.set_ylim(*PANEL_YLIM)  # pial at the top, same heights as the profiles panel
        ax.set_yticks([0, 1], ['pial', 'WM'])
        ax.set_ylabel("Intracortical depth")
        ax.set_xticks(range(n_bins), [f"Q{k + 1}" for k in range(n_bins)])
        ax.tick_params(axis='x', length=0)
        ax.set_xlabel("MPC gradient (quintiles)" if n_bins == 5 else f"MPC gradient ({n_bins} bins)")
        for side in ('top', 'right', 'bottom'):
            ax.spines[side].set_visible(False)
        _apply_panel_axes(fig)
        # Colour bar in the free right end of the axes box (middle 60 % of its height),
        # its tick labels and title in the right margin.
        c, box = PANEL_CANVAS_PT, PANEL_AXES_PT
        h = box['bottom'] - box['top']
        cax = fig.add_axes([352 / c, 1 - (box['bottom'] - 0.2 * h) / c, 9 / c, 0.6 * h / c])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=t1_norm, cmap=t1_cmap), cax=cax)
        cb.ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
        cb.ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
        cb.set_label("qT1 (s)")
        cb.outline.set_edgecolor('0.4')
        cb.outline.set_linewidth(0.8)
        plt.savefig(screenshot_path)
        plt.close(fig)


def plot_brain_zoom(
    df_yeo_surf: pd.DataFrame,
    surf_lh,
    surf_rh,
    out_dir: Path,
    hemisphere: str = 'both',
) -> None:
    """Brain row of the SalVentAttn MPC gradient with zoom insets of ``SALIENCE_ZOOM_ROIS``.

    Writes ``figure_1a_brain_zoom.svg`` (brain row, marker circles, lines, insets,
    colour bar) and ``figure_1a_brain_zoom_roi.svg`` (the inset row alone). Both use
    the window geometry of ``figure_1a_brain.svg`` (1450 x 300, zoom 1.3, colour
    bar right): its brain row overlays that file exactly, and the inset row placed
    300 pt below it reproduces the full figure. Insets are drawn only for regions in
    the analysed ``hemisphere``, since the gradient is undefined on the other one. The
    SalVentAttn border (``mask_outline``) is traced on the brains and insets.

    Parameters
    ----------
    df_yeo_surf : pd.DataFrame
        fsLR-32k surface table with ``label`` and ``t1_gradient1_SalVentAttn`` columns.
    surf_lh, surf_rh : BSPolyData
        Inflated fsLR-32k hemispheres.
    out_dir : Path
        Output directory.
    hemisphere : {'both', 'LH', 'RH'}
        Hemisphere(s) the gradient was computed on.
    """
    rois = salience_zoom_rois(df_yeo_surf, hemisphere)
    values = df_yeo_surf['t1_gradient1_SalVentAttn'].to_numpy()
    # Same window as the plot_hemispheres call that writes figure_1a_brain.svg. Labels at
    # 30 pt: the 1450-pt rows are shown at ~0.29x in the Figure 1 layout (~8.8 pt, as 1b).
    kwargs = dict(size=(1450, 300), zoom=1.3, color_bar=True, cmap='coolwarm', color_range=SALIENCE_Z_RANGE,
                  nan_color=(220, 220, 220, 1), label_fontsize=30,
                  outline=mask_outline(surf_lh, surf_rh, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()))
    plot_hemispheres_zoom(surf_lh, surf_rh, values, rois, cbar_labels=SALIENCE_Z_LABELS, cbar_title='z',
                          filename=out_dir / "figure_1a_brain_zoom.svg", **kwargs)
    plot_hemispheres_zoom(surf_lh, surf_rh, values, rois, roi_only=True,
                          filename=out_dir / "figure_1a_brain_zoom_roi.svg", **kwargs)
    plt.close('all')


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
    need_profiles: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Attach the network T1 gradient + mean T1map to `df_yeo`; return masked profiles.

    When `cache_path` exists the surface table (with the gradient columns) is read
    back from it; otherwise the diffusion-map MPC gradient is computed from the
    per-subject T1 profiles and the table is cached. The per-subject profiles are
    only loaded when they are actually consumed: always on a cache-miss (they drive
    the gradient) and on a cache-hit only when `need_profiles` is set (the caller
    plots them). When they are not loaded the second return value is ``None``. The
    profile vertex axis is in the same order as the masked rows of `df_yeo`.
    """
    net_mask = compute_network_mask(df_yeo, network, hemisphere)
    if cache_path.exists():
        logger.info(f"Found existing gradient table at {cache_path}. Loading...")
        profiles = (load_t1_salience_profiles(df_pni[t1_col].tolist(), net_mask)
                    if need_profiles else None)
        df_yeo = pd.read_csv(cache_path, sep="\t")
    else:
        profiles = load_t1_salience_profiles(df_pni[t1_col].tolist(), net_mask)
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
        need_profiles=False,  # Part 2 only needs the gradient table, not the profiles
    )

    ######### Figures (fsLR-32k)
    screenshot_path = project_root / "results/figures/figure_1a_profiles.svg"
    logger.info(f"Generating qt1 profiles figure at {screenshot_path}")
    plot_gradient_profiles(df_yeo_surf, t1_salience_profiles, screenshot_path, network='SalVentAttn', hemisphere=args.hemi)

    screenshot_path = project_root / "results/figures/figure_1a_strata.svg"
    logger.info(f"Generating qt1 stratigraphy columns at {screenshot_path}")
    plot_gradient_strata(df_yeo_surf, t1_salience_profiles, screenshot_path, network='SalVentAttn', hemisphere=args.hemi)

    screenshot_path = project_root / "results/figures/figure_1a_brain.svg"
    logger.info(f"Generating brain hemispheres screenshot at {screenshot_path}")
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range=SALIENCE_Z_RANGE, transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)

    logger.info("Generating brain zoom-inset figures (figure_1a_brain_zoom*.svg)")
    plot_brain_zoom(df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, project_root / "results/figures", hemisphere=args.hemi)

    screenshot_path = project_root / "results/figures/figure_1a_brain_5k.svg"
    logger.info(f"Generating brain hemispheres screenshot at {screenshot_path}")
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k['t1_gradient1_SalVentAttn'].values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(220, 220, 220, 1), cmap='coolwarm', color_range=SALIENCE_Z_RANGE, transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)


if __name__ == "__main__":
    main()
#!/usr/bin/env python
"""Step 1a: microstructural gradient of the salience network (Figure 1a).

Computes the first diffusion-map gradient of microstructure profile covariance (MPC)
within the network from each subject's intracortical qT1 profiles, at fsLR-32k (Figures
1 and 2) and fsLR-5k (Figure 3). Only subjects with every input are analysed (fsLR-5k
profiles, MPC, structural connectome and edge lengths, geodesic distance and functional
connectivity, and fsLR-32k profiles), so both gradients use the same subjects.

Inputs   micapipe derivatives of MICA-PNI and MICA-MICs; the participant table, only to
         build the subject lists (config/paths.yaml).
Private  <work_dir>/step1a_mpc_gradient/: subject lists with file paths, gradient fit
         summaries (with subject IDs), log.
Public   source_data/step1a_mpc_gradient/: the surface tables with the gradient (read by
         every later step), the subject-mean qT1 profile of each network vertex, the
         gradient fit summary, the comparison with the FC gradient, manifest.
Figures  results/figures/figure_1a_*.svg

Run      python pipeline/step1a_mpc_gradient.py [--stage compute|plot|all]
"""
import json
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainspace.datasets import load_conte69
from brainspace.mesh import mesh_elements
from brainspace.plotting import plot_hemispheres

import common
from salience_switch import config
from salience_switch.colors import connectivity_cmap
from salience_switch.microstructure import compute_t1_gradient, compute_t1map, load_t1_salience_profiles
from salience_switch.plotting import (SALIENCE_Z_LABELS, SALIENCE_Z_RANGE, mask_outline, plot_hemispheres_zoom,
                                      salience_zoom_rois)
from salience_switch.provenance import write_manifest
from salience_switch.stats import HemisphereMoranRandomization, moran_spearman
from salience_switch.surface import (compute_network_mask, load_fc_gradient_32k, load_inflated_surfaces,
                                     load_yeo_atlas, load_yeo_surf_5k)

STEP = "step1a_mpc_gradient"
logger = logging.getLogger(STEP)

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

# fsLR-32k: only the T1 intensity profile is consumed here.
PATHS_32K = [
    ("path_t1_profile", "pni",
     "sub-{ID_PNI}/ses-{session}/mpc/acq-T1map/"
     "sub-{ID_PNI}_ses-{session}_surf-fsLR-32k_desc-intensity_profiles.shape.gii"),
]
REQUIRED_32K = ["path_t1_profile"]

# fsLR-5k: consumed downstream by step 3. Every input is required, FC included: only
# subjects with all data enter the study sample, so every figure and every connectivity
# measure uses the same participants.
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
               "path_sc_dist_5k", "path_dist_5k", "path_fc_5k"]

# Moran surrogates of the comparison with the FC gradient
N_PERMUTATIONS = 1000

# Axes box (pt from the top-left of the 432-pt, 6-in canvas) shared by the profiles and
# strata panels, so pial and WM sit at the same heights in both. It is the box
# tight_layout gave the profiles panel at 21 pt; salience_switch.plotting's
# _FOREST_AXIS_PT copies its x span for the ridgeline and forest plots.
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


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
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
        Per-subject T1 profiles for the network vertices (or their subject mean, with
        a leading axis of length 1).
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
        Per-subject T1 profiles (ms) for the network vertices (or their subject mean,
        with a leading axis of length 1).
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


def plot_fc_gradient_zoom(df_fc: pd.DataFrame, surf_lh, surf_rh, out_dir: Path, network: str = 'SalVentAttn') -> None:
    """Supplementary: the FC gradient with zoom insets of ``SALIENCE_ZOOM_ROIS``.

    The Figure 3 target axis (`load_fc_gradient_32k`) in ``CustomCmap_connectivity`` on
    the shared +/-2 SD scale, on the Figure 1 window with the four salience insets and the
    network border traced (``mask_outline``), in two versions: z-scored over the whole
    cortex (``figure_1a_fc_gradient_*``, the context, as the Figure 3 schematic) and
    z-scored within the network and shown on its vertices only
    (``figure_1a_fc_gradient_sn_*``), the footprint and scale of the MPC-gradient map, so
    the two read side by side. Each version is written as ``*_zoom_supp.svg`` (brain row,
    insets, region names) and ``*_zoom_roi_supp.svg`` (titled inset row alone).
    """
    fc = df_fc["fc_g1"].to_numpy(dtype=float)
    net = df_fc['network'].eq(network).to_numpy()
    in_net = np.where(net, fc, np.nan)
    versions = {
        "figure_1a_fc_gradient": ((fc - np.nanmean(fc)) / np.nanstd(fc),
                                  "Functional connectivity gradient (whole brain)"),
        "figure_1a_fc_gradient_sn": ((in_net - np.nanmean(in_net)) / np.nanstd(in_net),
                                     "Functional connectivity gradient (salience network)"),
    }
    rois = salience_zoom_rois(df_fc, 'both')
    kwargs = dict(size=(1450, 300), zoom=1.3, color_bar=True, cmap=connectivity_cmap.name, color_range=SALIENCE_Z_RANGE,
                  nan_color=(220, 220, 220, 1), label_fontsize=30, title_fontsize=36, cbar_labels=SALIENCE_Z_LABELS,
                  cbar_title='z', outline=mask_outline(surf_lh, surf_rh, net))
    for stem, (values, title) in versions.items():
        plot_hemispheres_zoom(surf_lh, surf_rh, values, rois, filename=out_dir / f"{stem}_zoom_supp.svg", **kwargs)
        plot_hemispheres_zoom(surf_lh, surf_rh, values, [dict(r, label=None) for r in rois], roi_only=True,
                              draw_color_bar=True, title=title, filename=out_dir / f"{stem}_zoom_roi_supp.svg", **kwargs)
        plt.close('all')


# ---------------------------------------------------------------------------
# Subject lists
# ---------------------------------------------------------------------------
def _read_participant_table(mica_pni_csv: Path, **kwargs) -> pd.DataFrame:
    """Read the MICA-PNI participant table, as UTF-8 or, failing that, Windows-1252.

    The table is edited in spreadsheet software, which saves accented characters in
    Windows-1252 (e.g. Excel's CSV export).
    """
    try:
        return pd.read_csv(mica_pni_csv, **kwargs)
    except UnicodeDecodeError:
        logger.info(f"{mica_pni_csv.name} is not UTF-8; reading it as Windows-1252.")
        return pd.read_csv(mica_pni_csv, encoding="cp1252", **kwargs)


def _filtered_pni_subjects(mica_pni_csv: Path) -> pd.DataFrame:
    """Healthy-control PNI subjects (PNC*, ses-a1) matched to a MICA-MICs scan."""
    df = _read_participant_table(mica_pni_csv)[["ID_PNI", "session", "ID_MICs"]].drop_duplicates()
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
    missing any `required` path are dropped; a column not listed in `required` may
    be NaN without dropping the subject.
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


def restrict_to_subjects(df_pni: pd.DataFrame, df_sample: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Restrict a subject table to the subjects of `df_sample`, in `df_sample`'s order.

    The fsLR-32k table (Figure 1) is cut down to the fsLR-5k table (Figure 3 sample),
    so both gradients are computed on the same participants with the same Procrustes
    reference (the first subject). `csv_path` is rewritten when the table changes, so
    the cached table always lists the analysed sample.
    """
    out = df_sample[["ID_PNI"]].merge(df_pni, on="ID_PNI", how="inner")
    missing = sorted(set(df_sample["ID_PNI"]) - set(out["ID_PNI"]))
    if missing:
        logger.warning(f"{len(missing)} subjects of the fsLR-5k sample have no fsLR-32k T1 profile and are "
                       f"missing from Figure 1: {missing}")
    if out["ID_PNI"].tolist() != df_pni["ID_PNI"].tolist():
        dropped = sorted(set(df_pni["ID_PNI"]) - set(out["ID_PNI"]))
        logger.info(f"fsLR-32k subject table restricted to the {len(out)} subjects of the fsLR-5k sample "
                    f"(Figure 3 connectomes); dropped {len(dropped)}: {dropped}")
        out.to_csv(csv_path, index=False)
    return out


def log_participants(df_pni: pd.DataFrame, mica_pni_csv: Path | None, label: str) -> None:
    """Log the sample size, sex counts and age of the subjects of a subject table.

    Sex and age come from the MICA-PNI participant table (matched on ``ID_PNI`` and
    ``session``; private). Age is summarised over the subjects with a recorded age. Only
    aggregate values are logged.
    """
    if mica_pni_csv is None or not mica_pni_csv.exists():
        logger.warning(f"Participants   : {label}: N={len(df_pni)}; participant table not found, "
                       f"sex and age not logged.")
        return
    info = _read_participant_table(mica_pni_csv, usecols=["ID_PNI", "session", "sex", "age"], dtype={"session": str})
    info = info.drop_duplicates(["ID_PNI", "session"])
    sub = df_pni[["ID_PNI", "session"]].astype(str).merge(info, on=["ID_PNI", "session"], how="left")
    sex = sub["sex"].fillna("unknown").value_counts()
    age = pd.to_numeric(sub["age"], errors="coerce").dropna()
    sex_txt = ", ".join(f"{k}={v}" for k, v in sex.items())
    age_txt = (f"age mean {age.mean():.1f} (SD {age.std(ddof=1):.1f}, range {age.min():g}-{age.max():g}) "
               f"years, recorded for {age.size}/{len(sub)}") if age.size else "age not recorded"
    logger.info(f"Participants   : {label}: N={len(sub)}; sex {sex_txt}; {age_txt}")


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------
def write_gradient_summary(summary_path: Path, df_pni: pd.DataFrame, network: str, hemisphere: str,
                           n_vertices: int, lambdas: np.ndarray) -> None:
    """Write the JSON summary of a gradient fit (see `gradient_and_profiles`)."""
    summary = {
        "network": network, "hemisphere": hemisphere, "subjects": df_pni["ID_PNI"].tolist(),
        "n_vertices": int(n_vertices), "n_components": int(lambdas.shape[1]),
        "sparsity": 0.9, "random_state": 0,
        "mean_lambdas": lambdas.mean(axis=0).tolist(),
        "variance_explained_g1": (lambdas[:, 0] / lambdas.sum(axis=1)).tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=1) + "\n")


def _log_gradient_summary(summary: dict) -> None:
    share = np.asarray(summary["variance_explained_g1"], dtype=float)
    logger.info(f"Gradient 1 variance explained (lambda_1 / sum of {summary['n_components']} lambdas): "
                f"mean {share.mean():.1%} (SD {share.std(ddof=1) if share.size > 1 else 0:.1%}, "
                f"range {share.min():.1%}-{share.max():.1%}) across {share.size} subjects")


def gradient_and_profiles(
    df_yeo: pd.DataFrame, df_pni: pd.DataFrame, t1_col: str,
    table_path: Path, summary_path: Path, hemisphere: str, network: str = "SalVentAttn",
    need_profiles: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Attach the network T1 gradient + mean T1map to `df_yeo`; return masked profiles.

    The diffusion-map MPC gradient is computed from the per-subject T1 profiles and
    written to `table_path`, with a JSON summary at `summary_path` (subjects, settings,
    mean lambdas and every subject's lambda_1 / sum(lambda)). The table is reused only
    if the summary lists exactly the subjects of `df_pni`, in the same order (the
    Procrustes reference is the first subject); otherwise the gradient is recomputed
    and both files are rewritten. The variance explained is logged either way. The
    per-subject profiles are only loaded when they are actually consumed: always on a
    recompute (they drive the gradient) and on reuse only when `need_profiles` is set.
    When they are not loaded the second return value is ``None``. The profile vertex
    axis is in the same order as the masked rows of `df_yeo`.
    """
    net_mask = compute_network_mask(df_yeo, network, hemisphere)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    subjects = df_pni["ID_PNI"].tolist()
    if table_path.exists() and summary is not None and summary["subjects"] == subjects:
        logger.info(f"Found existing gradient table at {table_path} ({len(subjects)} subjects). Loading...")
        profiles = (load_t1_salience_profiles(df_pni[t1_col].tolist(), net_mask)
                    if need_profiles else None)
        df_yeo = config.read_table(table_path)
    else:
        if table_path.exists():
            reason = ("has no summary" if summary is None
                      else f"was computed on {len(summary['subjects'])} other/differently ordered subjects")
            logger.info(f"Gradient table {table_path.name} {reason}; recomputing on the "
                        f"{len(subjects)} subjects of the subject table.")
        profiles = load_t1_salience_profiles(df_pni[t1_col].tolist(), net_mask)
        gradient, lambdas = compute_t1_gradient(profiles, return_lambdas=True)
        df_yeo.loc[net_mask, f"t1_gradient1_{network}"] = gradient
        df_yeo.loc[net_mask, "T1map"] = compute_t1map(profiles)
        config.write_table(df_yeo, table_path)
        write_gradient_summary(summary_path, df_pni, network, hemisphere, net_mask.sum(), lambdas)
        summary = json.loads(summary_path.read_text())
    _log_gradient_summary(summary)
    return df_yeo, profiles


def gradient_fit_row(summary: dict, density: str) -> dict:
    """De-identified summary of one gradient fit, for the public source data."""
    share = np.asarray(summary["variance_explained_g1"], dtype=float)
    return {"density": density, "network": summary["network"], "hemisphere": summary["hemisphere"],
            "n_subjects": share.size, "n_vertices": summary["n_vertices"],
            "n_components": summary["n_components"], "sparsity": summary["sparsity"],
            "variance_explained_g1_mean": share.mean(), "variance_explained_g1_sd": share.std(ddof=1),
            "variance_explained_g1_min": share.min(), "variance_explained_g1_max": share.max()}


def compare_with_fc_gradient(
    df_fc: pd.DataFrame, surf_32k, hemisphere: str = 'both', network: str = 'SalVentAttn',
    n_rep: int = N_PERMUTATIONS,
) -> dict:
    """Spearman correlation of the network MPC gradient with the whole-brain FC gradient.

    The FC gradient is the Figure 3 target axis at fsLR-32k (`load_fc_gradient_32k`,
    default-mode network low). The correlation runs over the network vertices of
    ``hemisphere`` where both maps are finite, against the per-hemisphere Moran null of
    the other figures (``HemisphereMoranRandomization``; inverse 1-ring edge lengths on
    the fsLR-32k midthickness surface ``surf_32k``; seeds 0 and 1). The MPC gradient is
    randomised and the FC gradient held fixed (add-one two-sided p). Signs are as produced.

    Returns
    -------
    result : dict
        ``r``, ``p_moran``, ``null_lo`` and ``null_hi`` (2.5th and 97.5th percentiles of
        the null), ``n_vertices`` and ``n_permutations``.
    """
    grad = df_fc[f"t1_gradient1_{network}"].to_numpy(dtype=float)
    fc = df_fc["fc_g1"].to_numpy(dtype=float)
    tested = compute_network_mask(df_fc, network, hemisphere) & np.isfinite(grad) & np.isfinite(fc)
    w = mesh_elements.get_ring_distance(surf_32k, n_ring=1, mask=tested)
    w.data **= -1
    msr = HemisphereMoranRandomization(df_fc["hemisphere"].to_numpy()[tested], n_rep, random_state=0).fit(w)
    r, p, r_null = moran_spearman(fc[tested], grad[tested], msr)
    lo, hi = np.percentile(r_null, [2.5, 97.5])
    logger.info(f"[Figure 1A] {network} MPC gradient vs whole-brain FC gradient (Figure 3 target axis, fsLR-32k, "
                f"default-mode network low) | Spearman r={r:+.3f}, Moran permutation p={p:.3e} (MPC gradient "
                f"randomised per hemisphere, n_perm={n_rep}, n_vertices={int(tested.sum())}; "
                f"null r 95% [{lo:+.3f}, {hi:+.3f}])")
    return {"network": network, "hemisphere": hemisphere, "r": r, "p_moran": p, "null_lo": lo, "null_hi": hi,
            "n_vertices": int(tested.sum()), "n_permutations": n_rep}


def write_mean_profiles(path: Path, df_yeo_surf: pd.DataFrame, profiles: np.ndarray,
                        network: str, hemisphere: str) -> None:
    """Subject-mean qT1 profile (ms) of each network vertex, one row per vertex."""
    net_mask = compute_network_mask(df_yeo_surf, network, hemisphere)
    mean_profiles = np.mean(profiles, axis=0)  # (n_depths, n_vertices)
    table = pd.DataFrame({"vertex": np.flatnonzero(net_mask),
                          "hemisphere": df_yeo_surf.loc[net_mask, "hemisphere"].to_numpy()})
    for d in range(mean_profiles.shape[0]):
        table[f"qT1_depth{d + 1:02d}"] = mean_profiles[d]
    config.write_table(table, path)


def read_mean_profiles(path: Path, df_yeo_surf: pd.DataFrame, network: str, hemisphere: str) -> np.ndarray:
    """Read `write_mean_profiles` back as an (n_depths, n_vertices) float32 array.

    micapipe stores the profiles as float32 and their means are computed in float32, so
    they are read back in float32: the panels then match those drawn from the profiles.
    """
    table = config.read_table(path)
    expected = np.flatnonzero(compute_network_mask(df_yeo_surf, network, hemisphere))
    if not np.array_equal(table["vertex"].to_numpy(), expected):
        raise ValueError(f"{path.name} does not list the {network} vertices of the surface table in order.")
    return table.filter(like="qT1_depth").to_numpy(dtype=np.float32).T


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------
def compute(settings: dict, paths: dict) -> None:
    """Subject lists, both gradients and their statistics."""
    network, hemi = settings["network"], settings["hemisphere"]
    work, source = config.work_dir(STEP, paths), config.source_dir(STEP)
    pni_deriv, mics_deriv = paths["micapipe_pni"], paths["micapipe_mics"]
    participants = paths.get("participants")
    logger.info("Surface space  : fsLR-32k (Figures 1 and 2) and fsLR-5k (Figure 3)")
    logger.info("Parcellation   : Schaefer-400 with Yeo 7-network labels")
    logger.info(f"Network        : {network}")
    logger.info(f"MICA-PNI derivatives: {pni_deriv}")
    logger.info(f"MICA-MICs derivatives: {mics_deriv}")

    # Subject lists. The fsLR-5k list (complete Figure 3 derivatives) defines the sample
    # of every analysis: the fsLR-32k list is restricted to its subjects, so the Figure 1
    # gradient is computed on the same participants as the Figure 3 connectomes.
    df_pni_5k = build_or_load_subject_table(
        work / "subjects_fslr5k.csv", participants, pni_deriv, mics_deriv, PATHS_5K, REQUIRED_5K,
        log_lines=[
            "T1 profiles    : acq-T1map, fsLR-5k surface, 14 intracortical depths",
            "MPC            : fsLR-5k vertex-level MPC matrix",
            "Connectomes    : iFOD2 40M streamlines, SIFT2-weighted, fsLR-5k",
            "FC             : resting-state (desc-me_task-rest_bold), fsLR-5k",
            "Gradient       : diffusion maps, normalized angle kernel, sparsity=0.9, n_components=10, procrustes alignment",
        ],
    )
    subjects_32k = work / "subjects_fslr32k.csv"
    df_pni = build_or_load_subject_table(
        subjects_32k, participants, pni_deriv, mics_deriv, PATHS_32K, REQUIRED_32K,
        log_lines=[
            "T1 profiles    : acq-T1map, fsLR-32k surface, 14 intracortical depths",
            "Gradient       : diffusion maps, normalized angle kernel, sparsity=0.9, n_components=10, procrustes alignment",
        ],
    )
    df_pni = restrict_to_subjects(df_pni, df_pni_5k, subjects_32k)
    log_participants(df_pni, participants, "study sample, subjects with all data (Figures 1-3)")

    # fsLR-32k gradient (Figures 1 and 2)
    surf_32k = load_conte69(join=True)
    df_yeo_surf, t1_salience_profiles = gradient_and_profiles(
        load_yeo_atlas(surf_32k), df_pni, "path_t1_profile",
        source / f"surface_fslr32k_{hemi}.tsv", work / f"gradient_fslr32k_{hemi}.json", hemi, network,
    )
    write_mean_profiles(source / f"profiles_{hemi}.tsv", df_yeo_surf, t1_salience_profiles, network, hemi)

    # fsLR-5k gradient (Figure 3)
    gradient_and_profiles(
        load_yeo_surf_5k(), df_pni_5k, "path_t1_profile_5k",
        source / f"surface_fslr5k_{hemi}.tsv", work / f"gradient_fslr5k_{hemi}.json", hemi, network,
        need_profiles=False,  # only the gradient table is needed
    )
    fits = [gradient_fit_row(json.loads((work / f"gradient_{d}_{hemi}.json").read_text()), d)
            for d in ("fslr32k", "fslr5k")]
    config.write_table(pd.DataFrame(fits), source / f"gradient_fit_{hemi}.tsv")

    # The network gradient against the whole-brain FC gradient that Figure 3 projects onto
    comparison = compare_with_fc_gradient(load_fc_gradient_32k(df_yeo_surf), surf_32k, hemisphere=hemi,
                                          network=network)
    config.write_table(pd.DataFrame([comparison]), source / f"mpc_vs_fc_gradient_{hemi}.tsv")

    write_manifest(source / "manifest.json", STEP, config.PROJECT_ROOT, network=network, hemisphere=hemi,
                   n_subjects=len(df_pni), n_permutations=N_PERMUTATIONS)


def plot(settings: dict) -> None:
    """Figure 1a panels and the supplementary FC-gradient maps, from source_data only."""
    network, hemi = settings["network"], settings["hemisphere"]
    source, out = config.source_dir(STEP), config.FIGURES_DIR
    df_yeo_surf = config.read_table(source / f"surface_fslr32k_{hemi}.tsv")
    df_yeo_surf_5k = config.read_table(source / f"surface_fslr5k_{hemi}.tsv")
    # Subject-mean profiles with a leading axis of length 1, as the plot functions expect
    mean_profiles = read_mean_profiles(source / f"profiles_{hemi}.tsv", df_yeo_surf, network, hemi)[None]
    surf32k_lh_infl, surf32k_rh_infl = load_inflated_surfaces("fsLR-32k")
    surf5k_lh_infl, surf5k_rh_infl = load_inflated_surfaces("fsLR-5k")

    logger.info("Generating the qT1 profiles and stratigraphy panels")
    plot_gradient_profiles(df_yeo_surf, mean_profiles, out / "figure_1a_profiles.svg", network=network, hemisphere=hemi)
    plot_gradient_strata(df_yeo_surf, mean_profiles, out / "figure_1a_strata.svg", network=network, hemisphere=hemi)

    logger.info("Generating the brain maps of the gradient")
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=df_yeo_surf[f't1_gradient1_{network}'].values,
                     size=(1450, 300), zoom=1.3, color_bar='right', share='both', nan_color=(220, 220, 220, 1),
                     cmap='coolwarm', color_range=SALIENCE_Z_RANGE, transparent_bg=True, screenshot=True,
                     filename=out / "figure_1a_brain.svg", cb__numberOfLabels=0)
    plot_brain_zoom(df_yeo_surf, surf32k_lh_infl, surf32k_rh_infl, out, hemisphere=hemi)
    plot_hemispheres(surf5k_lh_infl, surf5k_rh_infl, array_name=df_yeo_surf_5k[f't1_gradient1_{network}'].values,
                     size=(1450, 300), zoom=1.3, color_bar='right', share='both', nan_color=(220, 220, 220, 1),
                     cmap='coolwarm', color_range=SALIENCE_Z_RANGE, transparent_bg=True, screenshot=True,
                     filename=out / "figure_1a_brain_5k.svg", cb__numberOfLabels=0)

    logger.info("Generating the supplementary FC-gradient maps")
    plot_fc_gradient_zoom(load_fc_gradient_32k(df_yeo_surf), surf32k_lh_infl, surf32k_rh_infl, out, network=network)


if __name__ == "__main__":
    common.main(STEP, compute, plot, "Step 1a: microstructural (MPC) gradient of the salience network (Figure 1a).")

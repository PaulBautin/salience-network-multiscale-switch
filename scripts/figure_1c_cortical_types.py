# !/usr/bin/env python
# -*- coding: utf-8
#########################################################################################
#
# Figure 1C - Local cortical type heterogeneity of the salience network
#
# Loads the cached gradient dataframe produced by figure_1a_t1map.py, overlays
# von Economo-Koskinas cortical types, and tests whether each Yeo network is
# enriched or depleted for each type relative to a spin-permutation null.
#
# Requires figure_1a_t1map.py to have been run first (produces df_1a_<hemi>.tsv).
#
# example:
# python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1c_cortical_types.py \
#   -hemi LH
#
# If working on remote server add before command: xvfb-run -s "-screen 0 1920x1080x24"
# ---------------------------------------------------------------------------------------
# Authors: Paul Bautin
#
# About the license: see the file LICENSE
#########################################################################################

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from brainspace.plotting import plot_hemispheres
from brainspace.mesh.mesh_io import read_surface
from brainspace.datasets import load_conte69
from brainspace.null_models import SpinPermutations

import logging

from src.atlas_load import load_yeo_atlas, load_econo_atlas, convert_states_str2int
from src.connectome_processing import empirical_p_twosided, benjamini_hochberg
from src.plot_colors import cmap_types, cmap_types_mw
from src.logging_utils import setup_manuscript_logger

logger = logging.getLogger(__name__)


plt.rcParams['font.size'] = 16
plt.rcParams['svg.fonttype'] = 'none'

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cortical type composition per Yeo network vs spin-permutation null (Figure 1C).",
        formatter_class=argparse.RawTextHelpFormatter,
        prog=os.path.basename(__file__).strip(".py")
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


def cortical_type_analysis(df_yeo_surf: pd.DataFrame, project_root: Path, hemisphere: str = 'both') -> None:
    # Define type labels
    type_labels = ['Kon', 'Eu-III', 'Eu-II', 'Eu-I', 'Dys', 'Ag', 'Other']
    label_map = dict(zip(range(1, 8), type_labels))
    sphere32k_lh, sphere32k_rh = load_conte69(as_sphere=True)

    # Always fit on the full surface so spin indices match sphere size (32492 per hemi).
    # Hemisphere selection is applied to the per-network mask, not to df_yeo_surf itself,
    # because medial-wall vertices are absent from the 'hemisphere' column and filtering
    # the DataFrame would shrink it below 32492 rows, causing out-of-bounds spin indices.
    # 1000 spins matches figures 2 and 3 and keeps the add-one p floor at ~1e-3.
    n_rand = 1000
    sp = SpinPermutations(n_rep=n_rand, random_state=0)
    sp.fit(sphere32k_lh, points_rh=sphere32k_rh)

    all_data = {}
    real_data = {}

    # surf_type == 0 means unassigned; map to the "Other" bin (index 7) so it
    # doesn't silently inflate zero-counts when building the distribution.
    df_yeo_surf.loc[df_yeo_surf.surf_type == 0, 'surf_type'] = 7
    state, state_name = convert_states_str2int(df_yeo_surf['network'].values)

    for net_idx, net_name in enumerate(state_name):
        mask = (state == net_idx)
        if hemisphere in ('LH', 'RH'):
            mask = mask & (df_yeo_surf['hemisphere'] == hemisphere).values
        mask_lh, mask_rh = mask[:32492], mask[32492:]

        expected_types = np.arange(1, 8)
        comp = df_yeo_surf.surf_type.values[mask] * mask[mask]
        observed_types, counts = np.unique(comp, return_counts=True)
        counts_dict = dict(zip(observed_types, counts))
        full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
        percentages = (full_counts / len(comp)) * 100
        real_data[net_name] = dict(zip(expected_types, percentages))
        type_summary = ", ".join(f"{label_map[t]}={percentages[t-1]:.1f}%" for t in expected_types)
        logger.info(f"[Figure 1C] {net_name}: cortical type composition | {type_summary}")

        # Spin-permutation null distribution
        net_rot = np.hstack(sp.randomize(mask_lh, mask_rh))
        comp_dict = {val: [] for val in df_yeo_surf.surf_type.unique()}
        for n in range(n_rand):
            comp = df_yeo_surf.surf_type.values[net_rot[n]] * net_rot[n][net_rot[n]]
            u, c = np.unique(comp, return_counts=True)
            counts_dict = dict(zip(u, c))
            full_counts = np.array([counts_dict.get(t, 0) for t in expected_types])
            perc = (full_counts / len(comp)) * 100
            for val in comp_dict:
                comp_dict[val].append(dict(zip(expected_types, perc)).get(val, 0))
        df = pd.DataFrame(comp_dict)
        df.rename(columns={k: label_map.get(k, k) for k in df.columns}, inplace=True)
        all_data[net_name] = df

    logger.info(f"[Figure 1C] Spin permutations: n_rep={n_rand}, random_state=0")

    # Spin-permutation enrichment test for the focus network (SalVentAttn): per cortical
    # type, the two-sided add-one empirical p comparing the observed composition against
    # the spin-null distribution (centred on the null mean so it tests deviation in either
    # direction), Benjamini-Hochberg-corrected across the seven types. Reported alongside
    # the descriptive z-scores below.
    sal_null = all_data["SalVentAttn"]          # (n_rand, n_types), columns = type labels
    sal_obs = {label_map[t]: real_data["SalVentAttn"][t] for t in range(1, 8)}
    sal_p = []
    for lbl in type_labels:
        null_vals = sal_null[lbl].to_numpy()
        null_mean = null_vals.mean()
        sal_p.append(empirical_p_twosided(null_vals - null_mean, sal_obs[lbl] - null_mean))
    sal_q = benjamini_hochberg(np.array(sal_p))
    for lbl, p_val, q_val in zip(type_labels, sal_p, sal_q):
        logger.info(f"[Figure 1C] SalVentAttn {lbl}: spin enrichment p={p_val:.3e}, FDR q={q_val:.3e}")

    n_cols = 4
    sal_idx = np.where(state_name == "SalVentAttn")[0][0]
    other_names = [n for i, n in enumerate(state_name)
                   if i != sal_idx and n != "medial_wall"]
    n_rows = int(np.ceil(len(other_names) / (n_cols - 1)))

    # Standalone SalVentAttn panel saved separately for the manuscript figure
    fig, ax_sal = plt.subplots(figsize=(6, 6))
    df = all_data["SalVentAttn"][type_labels]
    sns.barplot(data=df, ax=ax_sal, color='lightgrey')
    rdict = {label_map.get(k, k): v for k, v in real_data["SalVentAttn"].items()}
    sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, edgecolors='none', ax=ax_sal)
    ax_sal.set_ylim(0, 60)
    ax_sal.tick_params(axis='x', labelrotation=90)
    ax_sal.set_ylabel("Salience percentage (%)")
    ax_sal.spines['top'].set_visible(False)
    ax_sal.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(project_root / "results/figures/figure_1c_type_salience.svg")

    # Overview panel: SalVentAttn spans the full left column, other networks fill the grid
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(n_rows, n_cols, wspace=0.4, hspace=0.6)
    ax_sal = fig.add_subplot(gs[:, 0])
    df = all_data["SalVentAttn"]
    null_means = df[type_labels].mean()
    null_stds = df[type_labels].std()
    rdict = {label_map.get(k, k): v for k, v in real_data["SalVentAttn"].items()}
    for lbl in type_labels:
        obs = rdict.get(lbl, np.nan)
        z = (obs - null_means[lbl]) / (null_stds[lbl] + 1e-12) if null_stds[lbl] > 0 else np.nan
        logger.info(f"[Figure 1C] SalVentAttn {lbl}: observed={obs:.1f}%, spin null mean={null_means[lbl]:.1f}% ± {null_stds[lbl]:.1f}%, z={z:.2f}")
    sns.barplot(data=df, ax=ax_sal, color='lightgrey')
    sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax_sal)
    ax_sal.set_title("SalVentAttn")
    ax_sal.set_ylim(0, 60)
    ax_sal.tick_params(axis='x', labelrotation=90)

    for i, net_name in enumerate(other_names):
        row, col = divmod(i, n_cols - 1)
        ax = fig.add_subplot(gs[row, col + 1])
        df = all_data[net_name]
        sns.barplot(data=df, ax=ax, color='lightgrey')
        rdict = {label_map.get(k, k): v for k, v in real_data[net_name].items()}
        sns.scatterplot(x=list(rdict.keys()), y=list(rdict.values()), color=cmap_types_mw.colors, s=100, ax=ax)
        ax.set_title(net_name)
        ax.set_ylim(0, 60)
        ax.tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    logger = setup_manuscript_logger("figure_1c_cortical_types", project_root, args)
    logger.info(f"Surface space  : fsLR-32k, Schaefer-400, Yeo 7-network labels")
    logger.info(f"Hemisphere     : {args.hemi}")
    logger.info(f"Cortical types : von Economo-Koskinas atlas (7 types: Kon, Eu-III, Eu-II, Eu-I, Dys, Ag, Other)")
    logger.info(f"Analysis       : cortical type composition per network vs spin permutation null")
    logger.info(f"Null model     : SpinPermutations (n_rep=1000, random_state=0)")
    logger.info(f"Script path: {script_path}")
    logger.info(f"Project root: {project_root}")

    surf32k_lh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.L.inflated.surf.gii', itype='gii')
    surf32k_rh_infl = read_surface(project_root / 'data/surfaces/fsLR-32k.R.inflated.surf.gii', itype='gii')
    surf_32k = load_conte69(join=True)
    df_yeo_surf = load_yeo_atlas(micapipe=project_root, surf_32k=surf_32k)

    # Load cached gradient dataframe written by figure_1a_t1map.py
    path_df_1a = project_root / f'data/dataframes/df_1a_{args.hemi}.tsv'
    if not path_df_1a.exists():
        raise FileNotFoundError(f"Gradient dataframe not found at {path_df_1a}. Run figure_1a_t1map.py with -hemi {args.hemi} first.")
    logger.info(f"Loading gradient dataframe from {path_df_1a}")
    df_yeo_surf = pd.read_csv(path_df_1a, sep="\t")

    df_yeo_surf = load_econo_atlas(project_root, df_yeo_surf)
    screenshot_path = project_root / "results/figures/figure_1c_brain_economo.svg"
    plt_values = df_yeo_surf['surf_type'].values * df_yeo_surf['salience_border'].values
    plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
        nan_color=(0, 0, 0, 1), cmap=cmap_types, transparent_bg=True, screenshot=True, filename=screenshot_path, cb__numberOfLabels=0)

    cortical_type_analysis(df_yeo_surf, project_root, hemisphere=args.hemi)



if __name__ == "__main__":
    main()
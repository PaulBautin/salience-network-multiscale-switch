"""Surface tables, atlases and the template maps bundled in `data/`.

Every table has one row per fsLR vertex (fsLR-32k: 64,984 rows; fsLR-5k: 9,684 rows; left
hemisphere first) with the Schaefer-400 parcel, its Yeo 7-network label and hemisphere.
The loaders add columns from the bundled atlases and histological maps.
"""
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from brainspace.mesh.array_operations import get_labeling_border
from brainspace.mesh.mesh_io import read_surface
from brainspace.utils.parcellation import relabel
from scipy.stats import zscore

logger = logging.getLogger(__name__)

# Template surfaces, parcellations and histological maps shipped with the repository.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_inflated_surfaces(density: str = "fsLR-32k") -> tuple:
    """Left and right inflated surfaces bundled in ``data/surfaces``.

    Parameters
    ----------
    density : {'fsLR-32k', 'fsLR-5k'}
        Surface resolution.

    Returns
    -------
    surf_lh, surf_rh : brainspace BSPolyData
    """
    return tuple(read_surface(str(DATA_DIR / f"surfaces/{density}.{h}.inflated.surf.gii"), itype="gii")
                 for h in ("L", "R"))


def convert_states_str2int(states_str: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer code of each label, in the alphabetical order of the distinct labels.

    Parameters
    ----------
    states_str : (n,) list or array of str
        One label per vertex, e.g. ``['Vis', 'Vis', 'SomMot']``.

    Returns
    -------
    states : (n,) float array
        Code of each label: its index in `state_labels`.
    state_labels : (n_states,) array
        The distinct labels, sorted; ``states == k`` selects ``state_labels[k]``.
    """
    state_labels, states = np.unique(states_str, return_inverse=True)
    return states.astype(float), state_labels


def compute_network_mask(df: pd.DataFrame, network: str, hemisphere: str = 'both') -> np.ndarray:
    """
    Compute a boolean vertex mask for a given network and hemisphere.

    Parameters
    ----------
    df : pd.DataFrame
        Surface DataFrame with 'network' and 'hemisphere' columns.
    network : str
        Yeo 7-network label (e.g. 'SalVentAttn').
    hemisphere : str
        'both', 'LH', or 'RH'.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_vertices,)
    """
    mask = df['network'].eq(network)
    if hemisphere in ('LH', 'RH'):
        mask = mask & df['hemisphere'].eq(hemisphere)
    return mask.to_numpy()


def _schaefer_table(density: str) -> pd.DataFrame:
    """Schaefer-400 parcel (``mics`` ID), Yeo 7 network, hemisphere and label per vertex."""
    lh = nib.load(DATA_DIR / f'parcellations/schaefer-400_{density}_lh.label.gii').darrays[0].data + 1000
    rh = nib.load(DATA_DIR / f'parcellations/schaefer-400_{density}_rh.label.gii').darrays[0].data + 1800
    rh[rh == 1800] = 2000  # right medial wall
    df = pd.DataFrame({'mics': np.concatenate([lh, rh]).astype(float)})
    lut = pd.concat([pd.read_csv(DATA_DIR / 'parcellations/lut/lut_subcortical-cerebellum_mics.csv'),
                     pd.read_csv(DATA_DIR / 'parcellations/lut/lut_schaefer-400_mics.csv')])
    lut['network'] = lut['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    lut['hemisphere'] = lut['label'].str.extract(r'(LH|RH)')
    return df.merge(lut[['mics', 'hemisphere', 'network', 'label']], on='mics', validate="many_to_one", how='left')


def load_yeo_atlas(surf_32k, border_network: str = 'SalVentAttn') -> pd.DataFrame:
    """fsLR-32k surface table: Schaefer-400 parcel, Yeo 7 network and hemisphere per vertex.

    Parameters
    ----------
    surf_32k : brainspace BSPolyData
        Both fsLR-32k hemispheres combined (left first), used to trace the network border.
    border_network : str
        Network whose border the ``salience_border`` column traces. Default
        ``'SalVentAttn'``.

    Returns
    -------
    df_yeo_surf : pd.DataFrame
        64,984 rows (left hemisphere first) with `mics` (parcel ID), `hemisphere`,
        `network`, `label`, `network_int` (integer network code of
        `convert_states_str2int`) and `salience_border` (NaN on the border of
        `border_network`, 1 elsewhere, for drawing the border on maps).
    """
    df_yeo_surf = _schaefer_table('conte69')
    df_yeo_surf['network_int'] = convert_states_str2int(df_yeo_surf['network'].values)[0]
    df_yeo_surf['salience_border'] = get_labeling_border(surf_32k, df_yeo_surf['network'].eq(border_network).to_numpy())
    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 1, 'salience_border'] = np.nan
    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 0, 'salience_border'] = 1
    return df_yeo_surf


def load_yeo_surf_5k() -> pd.DataFrame:
    """fsLR-5k surface table: Schaefer-400 parcel, Yeo 7 network and hemisphere per vertex.

    The parcellation is the fsLR-32k Schaefer-400 atlas resampled to fsLR-5k
    (``tools/surface_resampling.sh``).

    Returns
    -------
    df_yeo_surf_5k : pd.DataFrame
        9,684 rows (left hemisphere first) with `mics` (parcel ID), `hemisphere`,
        `network` and `label`.
    """
    return _schaefer_table('fslr-5k')


def load_fc_gradient_5k(df_yeo_surf_5k: pd.DataFrame) -> pd.DataFrame:
    """
    Add the principal fsLR-5k FC gradient, oriented with the default-mode network low.

    The gradient is BrainSpace's template first FC gradient
    (``brainspace.datasets.load_gradient('fc', idx=0)``), stored at fsLR-32k and resampled
    to fsLR-5k with ``wb_command -metric-resample`` (``tools/surface_resampling.sh``).
    The diffusion-map eigenvector polarity is arbitrary, so the gradient is oriented by
    anatomy rather than a hardcoded sign: it is negated if the default-mode network's
    mean exceeds the cortical mean, which puts the default-mode network at the low end
    and the task-positive networks at the high end. The chosen sign is logged, so a
    change in the source GIFTI's polarity shows in the logs rather than silently
    inverting results.

    Parameters
    ----------
    df_yeo_surf_5k : pd.DataFrame
        Output of `load_yeo_surf_5k` (9,684 rows, LH then RH).

    Returns
    -------
    df : pd.DataFrame
        A copy of `df_yeo_surf_5k` with `fc_g1` (oriented gradient, NaN off cortex).
    """
    return _add_fc_gradient(df_yeo_surf_5k, "fslr-5k")


def load_fc_gradient_32k(df_yeo_surf: pd.DataFrame) -> pd.DataFrame:
    """
    Add the principal FC gradient at fsLR-32k, oriented with the default-mode network low.

    The same map and orientation rule as `load_fc_gradient_5k`, at the native resolution
    of the bundled gradient (`data/parcellations/fc_gradient_fslr-32k_{lh,rh}.shape.gii`,
    BrainSpace's template first FC gradient), from which the fsLR-5k copy was resampled.

    Parameters
    ----------
    df_yeo_surf : pd.DataFrame
        fsLR-32k surface table (64,984 rows, LH then RH), e.g. from `load_yeo_atlas`.

    Returns
    -------
    df : pd.DataFrame
        A copy of `df_yeo_surf` with `fc_g1` (oriented gradient, NaN off cortex).
    """
    return _add_fc_gradient(df_yeo_surf, "fslr-32k")


def _add_fc_gradient(df_surf: pd.DataFrame, density: str) -> pd.DataFrame:
    """Copy of `df_surf` with the oriented FC gradient (`fc_g1`)."""
    fc_lh = nib.load(DATA_DIR / f'parcellations/fc_gradient_{density}_lh.shape.gii').darrays[0].data
    fc_rh = nib.load(DATA_DIR / f'parcellations/fc_gradient_{density}_rh.shape.gii').darrays[0].data
    df = df_surf.copy()
    df['fc_g1'] = np.concatenate([fc_lh, fc_rh]).astype(float)
    df.loc[df['hemisphere'].isna(), 'fc_g1'] = np.nan

    cortical = df['hemisphere'].notna()
    default_mean = df.loc[cortical & (df['network'] == 'Default'), 'fc_g1'].mean()
    cortical_mean = df.loc[cortical, 'fc_g1'].mean()
    if default_mean > cortical_mean:
        df['fc_g1'] = -df['fc_g1']
        logger.info(f"[FC gradient {density}] flipped sign so the default-mode network sits at the low end.")
    else:
        logger.info(f"[FC gradient {density}] sign kept as loaded (default-mode network already at the low end).")
    return df


def load_econo_atlas(df_yeo_surf: pd.DataFrame) -> pd.DataFrame:
    """Add the cortical type of each fsLR-32k vertex from the von Economo-Koskinas atlas.

    Each von Economo area is assigned the cortical type of García-Cabezas et al. (2020).

    Parameters
    ----------
    df_yeo_surf : pd.DataFrame
        fsLR-32k surface table from `load_yeo_atlas`; modified in place.

    Returns
    -------
    df_yeo_surf : pd.DataFrame
        The same table with `surf_type` (cortical type code per vertex).
    """
    #### load econo atlas Hardcoded based on table data in Garcia-Cabezas et al. (2020, Front Neuroanat)
    econo_surf_lh = nib.load(DATA_DIR / 'parcellations/economo_conte69_lh.label.gii').darrays[0].data
    econo_surf_rh = nib.load(DATA_DIR / 'parcellations/economo_conte69_rh.label.gii').darrays[0].data
    econo_surf = np.concatenate((econo_surf_lh, econo_surf_rh), axis=0).astype(float)
    econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])[[0] + list(range(2, 45))]
    df_yeo_surf['surf_type'] = relabel(econo_surf, econ_ctb).astype(float)
    return df_yeo_surf


def _histology_mean(name: str, mask: np.ndarray, invert: bool = False) -> np.ndarray:
    """Depth-averaged fsLR-32k intensity profile of a bundled histology map, z-scored over `mask`."""
    profiles = nib.load(DATA_DIR / f'parcellations/sub-{name}_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data[:, mask]
    if invert:
        profiles = -profiles
    return zscore(np.mean(profiles, axis=0), nan_policy='omit')


def load_bigbrain(mask: np.ndarray) -> np.ndarray:
    """Load BigBrain intensity profiles and return z-scored mean for masked vertices.

    Inverts values so high values correspond to more staining.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask (e.g. from compute_network_mask).

    Returns
    -------
    np.ndarray, shape (n_masked_vertices,)
    """
    return _histology_mean('BigBrain', mask, invert=True)


def load_ahead_biel(mask: np.ndarray) -> np.ndarray:
    """Load AHEAD Bielschowsky profiles and return z-scored mean for masked vertices.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask (e.g. from compute_network_mask).

    Returns
    -------
    np.ndarray, shape (n_masked_vertices,)
    """
    return _histology_mean('Ahead-Bielschowsky', mask)


def load_ahead_parva(mask: np.ndarray) -> np.ndarray:
    """Load AHEAD Parvalbumin profiles and return z-scored mean for masked vertices.

    Parameters
    ----------
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask (e.g. from compute_network_mask).

    Returns
    -------
    np.ndarray, shape (n_masked_vertices,)
    """
    return _histology_mean('Ahead-Parvalbumin', mask)

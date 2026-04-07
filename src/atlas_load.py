import logging
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore

from brainspace.mesh.array_operations import get_labeling_border
from brainspace.utils.parcellation import relabel

logger = logging.getLogger(__name__)


def convert_states_str2int(states_str: list | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """This function takes a list of strings that designate a distinct set of binary brain states and returns
    a numpy array of integers encoding those states alongside a list of keys for those integers.

    Args:
        states_str (N, list): a list of strings that designate which regions belong to which states.
            For example, states = ['Vis', 'Vis', 'Vis', 'SomMot', 'SomMot', 'SomMot']

    Returns:
        states (N, numpy array): array of integers denoting which node belongs to which state.
        state_labels (n_states, list): list of keys corresponding to integers.
            For example, if state_labels[1] = 'SomMot' then the integer 1 in `states` corresponds to 'SomMot'.
            Together, a binary state can be extracted like so: x0 = states == state_labels.index('SomMot')

    """
    n_states = len(states_str)
    state_labels = np.unique(states_str)

    states = np.zeros(n_states)
    for i, state in enumerate(state_labels):
        for j in np.arange(n_states):
            if state == states_str[j]:
                states[j] = i

    return states.astype(float), state_labels


def normalize_to_range(data: np.ndarray | list, target_min: float, target_max: float) -> np.ndarray:
    """
    Normalizes a NumPy array or list of numerical data to a specified target range.

    Args:
        data (np.array or list): The input numerical data.
        target_min (float): The desired minimum value of the normalized range.
        target_max (float): The desired maximum value of the normalized range.

    Returns:
        np.array: The normalized data within the target range.
    """
    data = np.array(data) # Ensure data is a NumPy array for min/max operations
    
    original_min = np.nanmin(data)
    original_max = np.nanmax(data)

    if original_min == original_max: # Handle cases where all values are the same
        return np.full_like(data, (target_min + target_max) / 2)

    # Normalize to 0-1 range first
    normalized_0_1 = (data - original_min) / (original_max - original_min)

    # Scale to the target range
    scaled_data = target_min + (normalized_0_1 * (target_max - target_min))
    return scaled_data


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


def load_t1_salience_profiles(t1_files: list, mask: np.ndarray) -> np.ndarray:
    """
    Load T1 intensity profiles for a pre-masked set of vertices across all subjects.

    Parameters
    ----------
    t1_files : list
        List of paths to .gii profile files, one per subject.
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask selecting the vertices to load (e.g. from compute_network_mask).

    Returns
    -------
    t1_stack : np.ndarray
        Stack of profiles with shape (n_subjects, n_depths, n_network_vertices).
    """
    n_files = len(t1_files)
    if n_files == 0:
        raise FileNotFoundError("No files found")
    if not np.any(mask):
        raise ValueError("mask is all-False: no vertices selected.")
    logger.info(f"Loading profiles for {n_files} subjects...")
    t1_salience_profiles = np.stack([nib.load(f).darrays[0].data[:, mask] for f in t1_files])
    logger.info(f"Final array shape: {t1_salience_profiles.shape}")
    return t1_salience_profiles


def load_yeo_atlas(micapipe: Path, surf_32k) -> pd.DataFrame:
    # Yeo 7-network atlas (Schaefer-400)
    atlas_yeo_lh = nib.load(micapipe / 'data/parcellations/schaefer-400_conte69_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh = nib.load(micapipe / 'data/parcellations/schaefer-400_conte69_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh[atlas_yeo_rh == 1800] = 2000
    df_yeo_surf = pd.DataFrame({'mics': np.concatenate([atlas_yeo_lh, atlas_yeo_rh]).astype(float)})

    #### load yeo atlas 7 network information
    df_label = pd.read_csv(micapipe / 'data/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label_sub = pd.read_csv(micapipe / 'data/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf = df_yeo_surf.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    df_yeo_surf['network_int'] = convert_states_str2int(df_yeo_surf['network'].values)[0]
    df_yeo_surf['salience_border'] = get_labeling_border(surf_32k, df_yeo_surf['network'].eq('SalVentAttn').to_numpy())
    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 1, 'salience_border'] = np.nan
    df_yeo_surf.loc[df_yeo_surf['salience_border'].values == 0, 'salience_border'] = 1
    return df_yeo_surf


def load_yeo_surf_5k(micapipe: str) -> pd.DataFrame:
    #### load yeo atlas 7 network fslr5k
    atlas_yeo_lh_5k = nib.load(micapipe + '/parcellations/schaefer-400_fslr-5k_lh.label.gii').darrays[0].data + 1000
    atlas_yeo_rh_5k = nib.load(micapipe + '/parcellations/schaefer-400_fslr-5k_rh.label.gii').darrays[0].data + 1800
    atlas_yeo_rh_5k[atlas_yeo_rh_5k == 1800] = 2000
    yeo_surf_5k = np.concatenate((atlas_yeo_lh_5k, atlas_yeo_rh_5k), axis=0).astype(float)
    df_yeo_surf_5k = pd.DataFrame(data={'mics': yeo_surf_5k})

    df_label = pd.read_csv(micapipe + '/parcellations/lut/lut_schaefer-400_mics.csv')
    df_label_sub = pd.read_csv(micapipe + '/parcellations/lut/lut_subcortical-cerebellum_mics.csv')
    df_label = pd.concat([df_label_sub, df_label])
    df_label['network'] = df_label['label'].str.extract(r'(Vis|Default|Cont|DorsAttn|Limbic|SalVentAttn|SomMot|medial_wall)')
    df_label['hemisphere'] = df_label['label'].str.extract(r'(LH|RH)')
    df_yeo_surf_5k = df_yeo_surf_5k.merge(df_label[['mics', 'hemisphere','network', 'label']], on='mics', validate="many_to_one", how='left')
    return df_yeo_surf_5k


def load_econo_atlas(micapipe: Path, df_yeo_surf: pd.DataFrame) -> pd.DataFrame:
    #### load econo atlas Hardcoded based on table data in Garcia-Cabezas (2021)
    econo_surf_lh = nib.load(micapipe / 'data/parcellations/economo_conte69_lh.label.gii').darrays[0].data
    econo_surf_rh = nib.load(micapipe / 'data/parcellations/economo_conte69_rh.label.gii').darrays[0].data
    econo_surf = np.concatenate((econo_surf_lh, econo_surf_rh), axis=0).astype(float)
    econ_ctb = np.array([0, 0, 2, 3, 4, 3, 3, 3, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 5, 4, 6, 6, 4, 4, 6, 6, 6, 2, 1, 1, 2, 1, 2, 3, 2, 3, 4, 3, 3, 2, 1, 1, 2, 4, 5])[[0] + list(range(2, 45))]
    df_yeo_surf['surf_type'] = relabel(econo_surf, econ_ctb).astype(float)
    # plt_values = df_yeo_surf['surf_type'].values * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #             nan_color=(0, 0, 0, 1), cmap='CustomCmap_type', transparent_bg=True, interactive=False)
    return df_yeo_surf


def load_baillarger_atlas(df_yeo_surf: pd.DataFrame, path_atlas: Path) -> np.ndarray:
    #### Baillarger type
    baillarger_surf_lh = nib.load(path_atlas / 'Baillarger_type_parcellation_from_colin27_to_conte69_32k_lh.label.gii').darrays[0].data
    baillarger_surf_rh = nib.load(path_atlas / 'Baillarger_type_parcellation_from_colin27_to_conte69_32k_rh.label.gii').darrays[0].data
    baillarger_surf = np.concatenate((baillarger_surf_lh, baillarger_surf_rh), axis=0).astype(float)
    baillarger_surf[(baillarger_surf == 0) | (baillarger_surf == 1)] = 1
    logger.debug('Baillarger unique values: %s', np.unique(baillarger_surf))
    baillarger_surf = baillarger_surf * df_yeo_surf['salience_border'].values
    return baillarger_surf


def load_intrusion_atlas(df_yeo_surf: pd.DataFrame, path_atlas: Path) -> np.ndarray:
    #### Intrusion type
    intrusion_surf_lh = nib.load(path_atlas / 'Intrusion_type_parcellation_from_colin27_to_conte69_32k_lh.label.gii').darrays[0].data
    intrusion_surf_rh = nib.load(path_atlas / 'Intrusion_type_parcellation_from_colin27_to_conte69_32k_rh.label.gii').darrays[0].data
    intrusion_surf = np.concatenate((intrusion_surf_lh, intrusion_surf_rh), axis=0).astype(float)
    intrusion_surf[(intrusion_surf == 0) | (intrusion_surf == 1)] = 1
    logger.debug('Intrusion unique values: %s', np.unique(intrusion_surf))
    intrusion_surf = intrusion_surf * df_yeo_surf['salience_border'].values
    return intrusion_surf


def compute_t1map(t1_salience_profiles: np.ndarray) -> np.ndarray:
    """Return the z-scored mean profile collapsed over subjects and depths.

    Parameters
    ----------
    t1_salience_profiles : np.ndarray, shape (n_subjects, n_depths, n_vertices)
        Pre-masked T1 profiles for the network of interest.

    Returns
    -------
    np.ndarray, shape (n_vertices,)
    """
    return zscore(np.mean(t1_salience_profiles, axis=(0, 1)), nan_policy='omit')


def load_bigbrain(micapipe: Path, mask: np.ndarray) -> np.ndarray:
    """Load BigBrain intensity profiles and return z-scored mean for masked vertices.

    Inverts values so high values correspond to more staining.

    Parameters
    ----------
    micapipe : Path
        Project root containing data/parcellations/.
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask (e.g. from compute_network_mask).

    Returns
    -------
    np.ndarray, shape (n_masked_vertices,)
    """
    data_bigbrain = nib.load(micapipe / 'data/parcellations/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_bigbrain = -data_bigbrain[:, mask]
    return zscore(np.mean(salience_bigbrain, axis=0), nan_policy='omit')


def load_bigbrain_gradients() -> np.ndarray:
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    gradient_lh = nib.load(project_root / 'data/parcellations/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    gradient_rh = nib.load(project_root / 'data/parcellations/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    gradient = np.concatenate((gradient_lh, gradient_rh), axis=0)
    return gradient   


def load_ahead_biel(micapipe: Path, mask: np.ndarray) -> np.ndarray:
    """Load AHEAD Bielschowsky profiles and return z-scored mean for masked vertices.

    Parameters
    ----------
    micapipe : Path
        Project root containing data/parcellations/.
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask (e.g. from compute_network_mask).

    Returns
    -------
    np.ndarray, shape (n_masked_vertices,)
    """
    data_biel = nib.load(micapipe / 'data/parcellations/sub-Ahead-Bielschowsky_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_biel = data_biel[:, mask]
    return zscore(np.mean(salience_biel, axis=0), nan_policy='omit')


def load_ahead_parva(micapipe: Path, mask: np.ndarray) -> np.ndarray:
    """Load AHEAD Parvalbumin profiles and return z-scored mean for masked vertices.

    Parameters
    ----------
    micapipe : Path
        Project root containing data/parcellations/.
    mask : np.ndarray of bool, shape (n_vertices,)
        Boolean vertex mask (e.g. from compute_network_mask).

    Returns
    -------
    np.ndarray, shape (n_masked_vertices,)
    """
    data_parva = nib.load(micapipe / 'data/parcellations/sub-Ahead-Parvalbumin_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_parva = data_parva[:, mask]
    return zscore(np.mean(salience_parva, axis=0), nan_policy='omit')


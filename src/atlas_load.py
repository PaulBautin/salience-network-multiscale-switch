from pathlib import Path
from typing import Tuple
import glob

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore

from brainspace.mesh.array_operations import get_labeling_border
from brainspace.utils.parcellation import relabel


def convert_states_str2int(states_str):
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


def normalize_to_range(data, target_min, target_max):
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


def load_t1_salience_profiles(path, df_yeo_surf, network='SalVentAttn'):
    ## t1 profiles (n_subject, n_features, n_vertices)
    t1_files = glob.glob(path)
    print("number of files/subjects: {}".format(len(t1_files)))
    t1_salience_profiles = np.stack([nib.load(f).darrays[0].data[:, df_yeo_surf['network'].eq(network).to_numpy()] for f in t1_files[:]])
    return t1_salience_profiles


def load_yeo_atlas(micapipe, surf_32k):
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
    # plt_values = df_yeo_surf['network_int'].values * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=plt_values, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_yeo', transparent_bg=True)
    return df_yeo_surf


def load_yeo_surf_5k(micapipe):
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


def load_econo_atlas(micapipe, df_yeo_surf):
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


def load_baillarger_atlas(df_yeo_surf):
        #### Baillarger type
    baillarger_surf_lh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Baillarger_type_parcellation_from_colin27_to_conte69_32k_lh.label.gii').darrays[0].data
    baillarger_surf_rh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Baillarger_type_parcellation_from_colin27_to_conte69_32k_rh.label.gii').darrays[0].data
    baillarger_surf = np.concatenate((baillarger_surf_lh, baillarger_surf_rh), axis=0).astype(float)
    baillarger_surf[(baillarger_surf == 0) | (baillarger_surf == 1)] = 1
    print(np.unique(baillarger_surf))
    print(np.array(sns.color_palette('Set2', 5)) * 255)
    baillarger_surf = baillarger_surf * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=baillarger_surf, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_baillarger', transparent_bg=True)


def load_intrusion_atlas(df_yeo_surf):
    #### Intrusion type
    intrusion_surf_lh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Intrusion_type_parcellation_from_colin27_to_conte69_32k_lh.label.gii').darrays[0].data
    intrusion_surf_rh = nib.load('/local_raid/data/pbautin/downloads/MYATLAS_package/MYATLAS_package_new/maps/Surface/HCP_conte69/conte69_32k/gii/parcellation/Intrusion_type_parcellation_from_colin27_to_conte69_32k_rh.label.gii').darrays[0].data
    intrusion_surf = np.concatenate((intrusion_surf_lh, intrusion_surf_rh), axis=0).astype(float)
    intrusion_surf[(intrusion_surf == 0) | (intrusion_surf == 1)] = 1
    print(np.unique(intrusion_surf))
    print(np.array(sns.color_palette('Set2', 5)) * 255)
    intrusion_surf = intrusion_surf * df_yeo_surf['salience_border'].values
    # plot_hemispheres(surf32k_lh_infl, surf32k_rh_infl, array_name=intrusion_surf, size=(1450, 300), zoom=1.3, color_bar='right', share='both',
    #         nan_color=(0, 0, 0, 1), cmap='CustomCmap_intrusion', transparent_bg=True)


def load_t1map(df_yeo_surf, t1_salience_profiles):
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'T1map'] = zscore(np.mean(t1_salience_profiles, axis=(0, 1)), nan_policy='omit')
    return df_yeo_surf


def load_bigbrain(micapipe, df_yeo_surf):
    ### Load the data from BigBrain (Invert values so high values ~ more staining)
    data_bigbrain = nib.load(micapipe / 'data/parcellations/sub-BigBrain_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_bigbrain = -data_bigbrain[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'BigBrain'] = zscore(np.mean(salience_bigbrain, axis=0), nan_policy='omit')
    return df_yeo_surf


def load_bigbrain_gradients():
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    gradient_lh = nib.load(project_root / 'data/parcellations/tpl-fs_LR_hemi-L_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    gradient_rh = nib.load(project_root / 'data/parcellations/tpl-fs_LR_hemi-R_den-32k_desc-Hist_G2.shape.gii').darrays[0].data
    gradient = np.concatenate((gradient_lh, gradient_rh), axis=0)
    return gradient   


def load_ahead_biel(micapipe, df_yeo_surf):
    ### Load the data from AHEAD
    data_biel = nib.load(micapipe / 'data/parcellations/sub-Ahead-Bielschowsky_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_biel = data_biel[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'Bielschowsky'] = zscore(np.mean(salience_biel, axis=0), nan_policy='omit')
    return df_yeo_surf


def load_ahead_parva(micapipe, df_yeo_surf):
    data_parva = nib.load(micapipe / 'data/parcellations/sub-Ahead-Parvalbumin_surf-fsLR-32k_desc-intensity_profiles.shape.gii').darrays[0].data
    salience_parva = data_parva[:, df_yeo_surf['network'].eq('SalVentAttn').to_numpy()]
    df_yeo_surf.loc[df_yeo_surf['network'].eq('SalVentAttn'), 'Parvalbumin'] = zscore(np.mean(salience_parva, axis=0), nan_policy='omit')
    return df_yeo_surf
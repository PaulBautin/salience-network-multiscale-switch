# Figure 1a - Local microstructural heterogeneity of the salience network 
#
# This script processes MICA-PNI derivatives to extract T1 microstructural profiles, 
# computes gradients within the Salience/Ventral Attention network, and visualizes 
# the relationship between T1 profiles and gradient values.
#
# example:
condash
conda activate env_salience

python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1a_t1map.py \
  -pni_deriv /data/mica/mica3/BIDS_PNI/derivatives/micapipe_v0.2.0 \
  -mics_deriv /data/mica/mica3/BIDS_MICs/derivatives/micapipe_v0.2.0 \
  -hemi LH

# Figure 1b - Contextualisation of local microstructural heterogeneity of the salience network
# using BigBrain and Ahead datasets
# 
#
# example:
python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1b_contextualisation.py \
  -hemi LH

# Figure 1C - Local cortical type heterogeneity of the salience network
#
# Loads the cached gradient dataframe produced by figure_1a_t1map.py, overlays
# von Economo-Koskinas cortical types, and tests whether each Yeo network is
# enriched or depleted for each type relative to a spin-permutation null.
#
# Requires figure_1a_t1map.py to have been run first (produces df_1a_<hemi>.tsv).
#
# example:
python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_1c_cortical_types.py \
  -hemi LH

# Figure 2 - Structural connectivity at MPC gradient extremes
#
# Tests SC/geodesic distance/MPC differences between gradient-high and gradient-low
# vertices within each Yeo network. Requires df_1a_<hemi>.tsv from figure_1a_t1map.py.
#
# example:
python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_2_distance.py \
  -hemi LH

# Figure 3a - MNI open iEEG atlas
#
# Maps electrophysiological similarity differences between gradient-top and
# gradient-bottom channels onto the cortical surface and correlates with BigBrain G2.
# Requires df_1a_<hemi>.tsv produced by figure_1a_t1map.py.
#
# example:
python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3_ieeg_mni.py \
  -ieeg_deriv /local_raid/data/pbautin/downloads/MNI_ieeg/MatlabFile.mat \
  -hemi RH \
  -network SalVentAttn

# Figure 3b - MICA iEEG dataset
#
# example:
python /local_raid/data/pbautin/software/salience-network-multiscale-switch/scripts/figure_3_ieeg_mica.py \
  -ieeg_deriv /host/verges/tank/data/BIDS_iEEG/derivatives/electroMICA \
  -hemi RH
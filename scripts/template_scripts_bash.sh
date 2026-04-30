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
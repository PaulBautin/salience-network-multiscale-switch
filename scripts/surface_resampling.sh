#!/usr/bin/env bash
set -euo pipefail

MICAPIPE=/local_raid/data/pbautin/software/micapipe
PARC_DIR=${MICAPIPE}/parcellations
SURF_DIR=${MICAPIPE}/surfaces

# # Schaefer resolutions available in micapipe
# RESOLUTIONS=(400)

# for N in "${RESOLUTIONS[@]}"; do
#   for HEMI in L R; do
#     hemi_lower=$(echo "${HEMI}" | tr '[:upper:]' '[:lower:]')

#     INPUT=${PARC_DIR}/schaefer-${N}_conte69_${hemi_lower}h.label.gii
#     OUTPUT=/local_raid/data/pbautin/software/salience-network-multiscale-switch/data/parcellations/schaefer-${N}_fslr-5k_${hemi_lower}h.label.gii

#     # Skip if output already exists
#     [[ -f "${OUTPUT}" ]] && { echo "Exists, skipping: ${OUTPUT}"; continue; }

#     echo "Resampling schaefer-${N} ${HEMI}..."
#     wb_command -label-resample \
#       "${INPUT}" \
#       "${SURF_DIR}/fsLR-32k.${HEMI}.sphere.reg.surf.gii" \
#       "${SURF_DIR}/fsLR-5k.${HEMI}.sphere.reg.surf.gii" \
#       ADAP_BARY_AREA \
#       "${OUTPUT}" \
#       -area-surfs \
#         "${SURF_DIR}/fsLR-32k.${HEMI}.midthickness.surf.gii" \
#         "${SURF_DIR}/fsLR-5k.${HEMI}.surf.gii"
#   done
# done

# Resample fc_gradient from fsLR-32k to fsLR-5k
DATA_DIR=/local_raid/data/pbautin/software/salience-network-multiscale-switch/data/parcellations
for HEMI in L R; do
  hemi_lower=$(echo "${HEMI}" | tr '[:upper:]' '[:lower:]')

  INPUT=${DATA_DIR}/fc_gradient_fslr-32k_${hemi_lower}h.shape.gii
  OUTPUT=${DATA_DIR}/fc_gradient_fslr-5k_${hemi_lower}h.shape.gii

  [[ -f "${OUTPUT}" ]] && { echo "Exists, skipping: ${OUTPUT}"; continue; }

  echo "Resampling fc_gradient ${HEMI}..."
  wb_command -metric-resample \
    "${INPUT}" \
    "${SURF_DIR}/fsLR-32k.${HEMI}.sphere.reg.surf.gii" \
    "${SURF_DIR}/fsLR-5k.${HEMI}.sphere.reg.surf.gii" \
    ADAP_BARY_AREA \
    "${OUTPUT}" \
    -area-surfs \
      "${SURF_DIR}/fsLR-32k.${HEMI}.midthickness.surf.gii" \
      "${SURF_DIR}/fsLR-5k.${HEMI}.surf.gii"
done

echo "Done."

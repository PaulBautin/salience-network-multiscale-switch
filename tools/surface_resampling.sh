#!/usr/bin/env bash
# Resample the bundled fsLR-32k maps to fsLR-5k with Connectome Workbench (one-off).
#
# This is how data/parcellations/schaefer-400_fslr-5k_{lh,rh}.label.gii and
# data/parcellations/fc_gradient_fslr-5k_{lh,rh}.shape.gii were made from their fsLR-32k
# versions. Existing outputs are kept.
#
# usage: tools/surface_resampling.sh /path/to/micapipe
#   micapipe's parcellations/ and surfaces/ folders hold the fsLR-32k Schaefer-400
#   parcellation and the fsLR-32k / fsLR-5k spheres and midthickness surfaces.
set -euo pipefail

MICAPIPE=${1:?usage: $0 /path/to/micapipe}
PARC_DIR=${MICAPIPE}/parcellations
SURF_DIR=${MICAPIPE}/surfaces
DATA_DIR=$(cd "$(dirname "$0")/../data/parcellations" && pwd)

for HEMI in L R; do
  hemi_lower=$(echo "${HEMI}" | tr '[:upper:]' '[:lower:]')
  area_surfs=(-area-surfs "${SURF_DIR}/fsLR-32k.${HEMI}.midthickness.surf.gii" "${SURF_DIR}/fsLR-5k.${HEMI}.surf.gii")
  spheres=("${SURF_DIR}/fsLR-32k.${HEMI}.sphere.reg.surf.gii" "${SURF_DIR}/fsLR-5k.${HEMI}.sphere.reg.surf.gii")

  # Schaefer-400 parcellation (labels)
  OUTPUT=${DATA_DIR}/schaefer-400_fslr-5k_${hemi_lower}h.label.gii
  if [[ -f "${OUTPUT}" ]]; then
    echo "Exists, skipping: ${OUTPUT}"
  else
    echo "Resampling schaefer-400 ${HEMI}..."
    wb_command -label-resample "${PARC_DIR}/schaefer-400_conte69_${hemi_lower}h.label.gii" "${spheres[@]}" \
      ADAP_BARY_AREA "${OUTPUT}" "${area_surfs[@]}"
  fi

  # Principal FC gradient (metric)
  OUTPUT=${DATA_DIR}/fc_gradient_fslr-5k_${hemi_lower}h.shape.gii
  if [[ -f "${OUTPUT}" ]]; then
    echo "Exists, skipping: ${OUTPUT}"
  else
    echo "Resampling fc_gradient ${HEMI}..."
    wb_command -metric-resample "${DATA_DIR}/fc_gradient_fslr-32k_${hemi_lower}h.shape.gii" "${spheres[@]}" \
      ADAP_BARY_AREA "${OUTPUT}" "${area_surfs[@]}"
  fi
done

echo "Done."

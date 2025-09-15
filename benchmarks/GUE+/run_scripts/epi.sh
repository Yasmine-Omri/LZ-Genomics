#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/../outputs"
GUE_PLUS_PATH="${SCRIPT_DIR}/../data/GUE_v2"
PY_SCRIPT="${SCRIPT_DIR}/../python_scripts/epi.py"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH-}"

CELL_TYPES=(GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK)
GAMMA_SET="{0.1, 0.33, 0.5, 0.75, 1, 3, 5}"
MAX_DEPTH_SET="{8, 16}"
C_SET="{0.03, 0.1, 0.3, 1, 3, 10}"
NUM_THREADS=64

fail_count=0
success_count=0
failed_cells=()

for x in "${CELL_TYPES[@]}"; do
  DATASET_FOLDER="${GUE_PLUS_PATH}/EPI/${x}"

  if [[ ! -d "${DATASET_FOLDER}" ]]; then
    echo "[$(date '+%F %T')] Skipping ${x}: folder not found: ${DATASET_FOLDER}" | tee -a "${OUTPUT_DIR}/missing.log"
    failed_cells+=("${x}")
    ((fail_count++)) || true
    continue
  fi

  log="${OUTPUT_DIR}/${x}.txt"
  echo "[$(date '+%F %T')] Starting ${x} …" | tee "${log}"

  # Guard with 'if'; under set -e this won't abort the script.
  if python "${PY_SCRIPT}" \
      -dataset_folder "${DATASET_FOLDER}" \
      --gamma "${GAMMA_SET}" \
      --max_depth "${MAX_DEPTH_SET}" \
      --clf_C "${C_SET}" \
      --num_threads "${NUM_THREADS}" \
      >> "${log}" 2>&1; then
    echo "[$(date '+%F %T')] Finished ${x} (OK)" | tee -a "${log}"
    ((success_count++)) || true
  else
    rc=$?
    echo "[$(date '+%F %T')] FAILED ${x} (exit ${rc})" | tee -a "${log}" | tee -a "${OUTPUT_DIR}/failures.log"
    failed_cells+=("${x}")
    ((fail_count++)) || true
    # continue to next cell type
  fi
done

echo "----- SUMMARY -----"
echo "Successes: ${success_count}"
echo "Failures : ${fail_count}"
if (( fail_count > 0 )); then
  echo "Failed cell types: ${failed_cells[*]}"
  # Optional: make the script return non-zero if any failed
  # exit 1
fi

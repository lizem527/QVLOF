#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASE_LIST=(
  "base_vectors"
  "f1_AG5_sorted"
  "f2_zorder_sorted"
  "f3_hilbert_sorted"
  "f4_idistance_sorted"
)

for base in "${BASE_LIST[@]}"; do
  echo "============================================================"
  echo "Running with base: $base"
  echo "============================================================"

  python det_lsh_re.py "$base" 100

done

echo "All runs finished."

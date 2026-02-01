#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

BASE_LIST=(
  "f2_zorder_sorted"
  "f3_hilbert_sorted"
  "f4_idistance_sorted"
)

for base in "${BASE_LIST[@]}"; do
  echo "============================================================"
  echo "Running with base: $base"
  echo "============================================================"

  taskset -c 0 numactl --cpunodebind=0 --membind=0 ./dblsh \
    "$base" 2.0 100 24 14 0.15

done

echo "All runs finished."

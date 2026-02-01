#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

DEMO="./demos/demo_sift1M"
BASE_DIR="/hd1/vec_index/Data"
QUERY_CSV="/hd1/vec_index/Query/selected_query.csv"
OUT_DIR="/hd1/vec_index/Results/IVFPQ"

declare -A BASE_FILES=(
  ["base"]="base_vectors.csv"
  ["f1_AG4"]="f1_AG4_sorted.csv"
  ["f1_AG5"]="f1_AG5_sorted.csv"
  ["f2"]="f2_zorder_sorted.csv"
  ["f3"]="f3_hilbert_sorted.csv"
  ["f4"]="f4_idistance_sorted.csv"
  ["random"]="random_perm_sorted.csv"
)

mkdir -p "$OUT_DIR"

for key in "${!BASE_FILES[@]}"; do
  base_csv="${BASE_DIR}/${BASE_FILES[$key]}"
  csv_path="${OUT_DIR}/ivfpq_${key}.csv"
  ids_csv_path="${OUT_DIR}/ivfpq_ids_${key}.csv"

  echo "============================================================"
  echo "Running: ${key}"
  echo "  base_csv     = ${base_csv}"
  echo "  csv_path     = ${csv_path}"
  echo "  ids_csv_path = ${ids_csv_path}"
  echo "============================================================"

  taskset -c 0 numactl --cpunodebind=0 --membind=0 "$DEMO" \
    --base_csv "$base_csv" \
    --query_csv "$QUERY_CSV" \
    --csv_skip_header 1 \
    --csv_drop_first_col 1 \
    --rounds 3 \
	--nlist 4096 --m 32 --nbits 8 \
	--train_size 200000 \
	--nprobe_list 256 \
	--k_list 100 \
	--refine_r 8 \
    --csv_path "$csv_path" \
	--hits_csv  "$ids_csv_path"  \
	--hits_write_round 0
done

echo "All runs finished."

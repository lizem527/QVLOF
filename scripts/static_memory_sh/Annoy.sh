#!/usr/bin/env bash
set -euo pipefail

# 线程相关环境变量（对每次执行都生效）
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATA_DIR="/hd1/vec_index/Data"
POINT_CSV="/hd1/vec_index/Query/selected_query.csv"
LOG_DIR="/hd1/vec_index/Results/Annoy"

# 文件列表
csv_list=(
  "base_vectors.csv"
  "f1_AG5_sorted.csv"
  "f2_zorder_sorted.csv"
  "f3_hilbert_sorted.csv"
  "f4_idistance_sorted.csv"
)

for csv_name in "${csv_list[@]}"; do
  loadcsv="${DATA_DIR}/${csv_name}"
  echo "=== Running with --loadcsv ${loadcsv} ==="

  python run_annoy_index_search.py \
    --loadcsv "${loadcsv}" \
    --point_csv "${POINT_CSV}" \
    --log_dir "${LOG_DIR}" \
    --topKs 100 \
    --metric euclidean \
    --n_trees 300 \
    --n_jobs 1
done
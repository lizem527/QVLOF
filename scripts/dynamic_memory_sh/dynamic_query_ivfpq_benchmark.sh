#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 固定环境（每次都写的那三行）
# -----------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# -----------------------------
# 二进制 & 固定参数
# -----------------------------
BIN="./demos/demo_sift1M"

CSV_SKIP_HEADER=1
CSV_DROP_FIRST_COL=1

K_LIST="100"
NPROBE_LIST="256"

NLIST=4096
M=32
NBITS=8
TRAIN_SIZE=200000

ROUNDS=3
REFINE_R=8

DYNAMIC=1
WINDOW_SIZE=200
DYNAMIC_K=100

HITS_WRITE_ROUND=0

# -----------------------------
# base_csv（5组）
# -----------------------------
declare -A BASES
BASES["base"]="/hd1/student/lzm/FANNS/Chiristmas/result/test/base_vectors.csv"
BASES["f1_ag5"]="/hd1/student/lzm/FANNS/Chiristmas/result/test/f1_AG5_sorted.csv"
BASES["f2"]="/hd1/student/lzm/FANNS/Chiristmas/result/test/f2_zorder_sorted.csv"
BASES["f3"]="/hd1/student/lzm/FANNS/Chiristmas/result/test/f3_hilbert_sorted.csv"
BASES["f4"]="/hd1/student/lzm/FANNS/Chiristmas/result/test/f4_idistance_sorted.csv"

# -----------------------------
# query_csv（2组）
# -----------------------------
declare -A QUERIES
QUERIES["q250"]="/hd1/workspace/mirageCode/dynamic_query/base_vectors_q250.csv"
QUERIES["q500"]="/hd1/workspace/mirageCode/dynamic_query/base_vectors_q500.csv"

# -----------------------------
# 输出根目录（注意：ivfpq_123）
# -----------------------------
OUT_ROOT="/hd1/workspace/mirageCode/results_dynamic_123/ivfpq_123"

# -----------------------------
# CPU/NUMA 绑定（保持一致）
# -----------------------------
PIN_PREFIX=(taskset -c 0 numactl --cpunodebind=0 --membind=0)

require_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] file not found: $f" >&2
    exit 1
  fi
}

run_one() {
  local base_tag="$1"
  local base_csv="$2"
  local q_tag="$3"
  local query_csv="$4"

  local out_dir="${OUT_ROOT}/${q_tag}"
  mkdir -p "$out_dir"

  local per_window_csv="${out_dir}/ivfpq_${base_tag}.csv"
  local ids_csv="${out_dir}/ivfpq_${base_tag}_ids.csv"

  echo "============================================================"
  echo "[RUN] base=${base_tag}  query=${q_tag}"
  echo "      base_csv=${base_csv}"
  echo "      query_csv=${query_csv}"
  echo "      per_window_csv=${per_window_csv}"
  echo "      ids_csv=${ids_csv}"
  echo "============================================================"

  "${PIN_PREFIX[@]}" "$BIN" \
    --base_csv "$base_csv" \
    --query_csv "$query_csv" \
    --csv_skip_header "$CSV_SKIP_HEADER" \
    --csv_drop_first_col "$CSV_DROP_FIRST_COL" \
    --k_list "$K_LIST" \
    --nprobe_list "$NPROBE_LIST" \
    --nlist "$NLIST" --m "$M" --nbits "$NBITS" --train_size "$TRAIN_SIZE" \
    --rounds "$ROUNDS" \
    --refine_r "$REFINE_R" \
    --dynamic "$DYNAMIC" --window_size "$WINDOW_SIZE" --dynamic_k "$DYNAMIC_K" \
    --per_window_csv_path "$per_window_csv" \
    --ids_csv_path "$ids_csv" \
    --hits_write_round "$HITS_WRITE_ROUND"
}

main() {
  require_file "$BIN"

  for q_tag in q250 q500; do
    require_file "${QUERIES[$q_tag]}"
    mkdir -p "${OUT_ROOT}/${q_tag}"

    for base_tag in base f1_ag5 f2 f3 f4; do
      require_file "${BASES[$base_tag]}"
      run_one "$base_tag" "${BASES[$base_tag]}" "$q_tag" "${QUERIES[$q_tag]}"
    done
  done

  echo "[DONE] All IVFPQ runs finished."
}

main "$@"

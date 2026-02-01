#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 0) 路径配置（按实际路径）
# -----------------------------
PYTHON_BIN="python"
SCRIPT="window_batch.py"   # 如果文件名其实叫 window_batch.py，就改成 window_batch.py

QUERY_Q250="/hd1/workspace/mirageCode/dynamic_query/base_vectors_q250.csv"
QUERY_Q500="/hd1/workspace/mirageCode/dynamic_query/base_vectors_q500.csv"

BASE_BASE="/hd1/student/lzm/FANNS/Chiristmas/result/test/base_vectors.csv"
BASE_F1="/hd1/student/lzm/FANNS/Chiristmas/result/test/f1_AG5_sorted.csv"
BASE_F2="/hd1/student/lzm/FANNS/Chiristmas/result/test/f2_zorder_sorted.csv"
BASE_F3="/hd1/student/lzm/FANNS/Chiristmas/result/test/f3_hilbert_sorted.csv"
BASE_F4="/hd1/student/lzm/FANNS/Chiristmas/result/test/f4_idistance_sorted.csv"

OUT_ROOT="/hd1/workspace/AnnoyIndex/results_dynamic_123"

# -----------------------------
# 1) 单线程环境
# -----------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# CPU/NUMA bind
PIN_PREFIX=(taskset -c 0 numactl --cpunodebind=0 --membind=0)

# -----------------------------
# 2) Annoy 参数
#   注意：为了“单核口径”更一致，n_jobs 默认=1（你也可改回 -1）
# -----------------------------
TOPKS="100"
METRIC="euclidean"
N_TREES=300
N_JOBS=1          # 想多核就改回 -1
WINDOW_SIZE=200

# -----------------------------
# 3) 是否为每次运行加时间戳子目录（避免 csv 追加污染）
#   - 空字符串：严格复刻你原先的 log_dir（可能追加）
#   - 非空：每次 run 都落在独立目录
# -----------------------------
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
# RUN_TAG=""

# 可选：每次跑之前清理 log_dir 下旧 csv（谨慎使用）
CLEAN_OLD_CSV=0

die() { echo "[ERROR] $*" >&2; exit 1; }

run_one() {
  local q_tag="$1"        # q250 / q500
  local base_tag="$2"     # base / f1_ag5 / f2 / f3 / f4
  local loadcsv="$3"
  local point_csv="$4"

  local log_dir="${OUT_ROOT}/${q_tag}/annoy_${base_tag}_${q_tag}"
  if [[ -n "$RUN_TAG" ]]; then
    log_dir="${log_dir}/runs/${RUN_TAG}"
  fi
  mkdir -p "$log_dir"

  if [[ "$CLEAN_OLD_CSV" -eq 1 ]]; then
    rm -f "${log_dir}"/*.csv 2>/dev/null || true
  fi

  echo "============================================================"
  echo "[RUN] ${q_tag}  ${base_tag}"
  echo "      loadcsv=${loadcsv}"
  echo "      point_csv=${point_csv}"
  echo "      log_dir=${log_dir}"
  echo "      n_trees=${N_TREES} n_jobs=${N_JOBS} topKs=${TOPKS} window_size=${WINDOW_SIZE}"
  echo "============================================================"

  "${PIN_PREFIX[@]}" "$PYTHON_BIN" "$SCRIPT" \
    --loadcsv "$loadcsv" \
    --point_csv "$point_csv" \
    --log_dir "$log_dir" \
    --topKs "$TOPKS" \
    --metric "$METRIC" \
    --n_trees "$N_TREES" \
    --n_jobs "$N_JOBS" \
    --window_size "$WINDOW_SIZE" \
    > "${log_dir}/run.stdout.txt" 2> "${log_dir}/run.stderr.txt"

  echo "[OK] outputs in: ${log_dir}"
}

main() {
  [[ -f "$SCRIPT" ]] || die "Cannot find $SCRIPT in current dir. (cd to its dir or set SCRIPT to absolute path.)"

  # q250
  run_one "q250" "base"   "$BASE_BASE" "$QUERY_Q250"
  run_one "q250" "f1_ag5" "$BASE_F1"   "$QUERY_Q250"
  run_one "q250" "f2"     "$BASE_F2"   "$QUERY_Q250"
  run_one "q250" "f3"     "$BASE_F3"   "$QUERY_Q250"
  run_one "q250" "f4"     "$BASE_F4"   "$QUERY_Q250"

  # q500
  run_one "q500" "base"   "$BASE_BASE" "$QUERY_Q500"
  run_one "q500" "f1_ag5" "$BASE_F1"   "$QUERY_Q500"
  run_one "q500" "f2"     "$BASE_F2"   "$QUERY_Q500"
  run_one "q500" "f3"     "$BASE_F3"   "$QUERY_Q500"
  run_one "q500" "f4"     "$BASE_F4"   "$QUERY_Q500"

  echo "[DONE] Annoy batch finished."
}

main "$@"

#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 0) 定位脚本目录 & 二进制绝对路径
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/lgo"

# -----------------------------
# 1) 固定环境（单线程）
# -----------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# -----------------------------
# 2) CPU/NUMA 绑定（保持原口径）
# -----------------------------
PIN_PREFIX=(taskset -c 0 numactl --cpunodebind=0 --membind=0)

# -----------------------------
# 3) 参数（严格对齐 main.cpp argv 含义）
#   ./lgo <datasetName> <isbuilt> <k_list> <L> <K> <T> <efC> <pC> <pQ> <lsh_UB>
# -----------------------------
ISBUILT=0          # 当前命令是 0；以后想复用索引可改 1
K_LIST="100"       # 支持 "100" 或 "10,50,100"
L=4
K=16
T=32
EFC=200            # 注意：这是 efC；window_size 在 main.cpp 里写死 200
PC=0.95
PQ=0.90
LSH_UB=0

# -----------------------------
# 4) Case 列表（datasetName）
# -----------------------------
CASES_Q250=(100w_base_q250 100w_f1_ag5_q250 100w_f2_q250 100w_f3_q250 100w_f4_q250)
CASES_Q500=(100w_base_q500 100w_f1_ag5_q500 100w_f2_q500 100w_f3_q500 100w_f4_q500)

# -----------------------------
# 5) 输出根目录
#    每个 case 固定一个 work 目录，支持 isbuilt=1 复用 ./indexes
# -----------------------------
OUT_ROOT="/hd1/workspace/mirageCode/results_dynamic_123/lsh_apg_123"

# 可选：每次运行打 tag（避免 result.txt 追加混在一起）
# - 为空：写到固定目录（会覆盖 stdout/stderr，但 result.txt 会 append，取决于程序）
# - 非空：写到独立 run 子目录
RUN_TAG=""

die() { echo "[ERROR] $*" >&2; exit 1; }

check_bin() {
  [[ -f "$BIN_PATH" ]] || die "Binary not found: $BIN_PATH"
  [[ -x "$BIN_PATH" ]] || die "Binary not executable (chmod +x?): $BIN_PATH"
}

run_one_case() {
  local q_tag="$1"
  local datasetName="$2"

  local base_dir="${OUT_ROOT}/${q_tag}/${datasetName}"
  local out_dir="$base_dir"
  if [[ -n "$RUN_TAG" ]]; then
    out_dir="${base_dir}/runs/${RUN_TAG}"
  fi
  mkdir -p "$out_dir"

  # 固定 workdir：让 ./indexes 持久化，isbuilt=1 才能复用
  local work_dir="${base_dir}/work"
  mkdir -p "$work_dir"

  echo "============================================================"
  echo "[RUN] group=${q_tag}  datasetName=${datasetName}"
  echo "      BIN=${BIN_PATH}"
  echo "      work_dir=${work_dir}"
  echo "      out_dir=${out_dir}"
  echo "============================================================"

  # meta 追溯
  {
    echo "datasetName=${datasetName}"
    echo "group=${q_tag}"
    echo "BIN_PATH=${BIN_PATH}"
    echo "work_dir=${work_dir}"
    echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
    echo "MKL_NUM_THREADS=${MKL_NUM_THREADS}"
    echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
    echo "PIN_PREFIX=${PIN_PREFIX[*]}"
    echo "ARGS=${ISBUILT} ${K_LIST} ${L} ${K} ${T} ${EFC} ${PC} ${PQ} ${LSH_UB}"
    echo "timestamp=$(date -Iseconds)"
  } > "${out_dir}/meta.txt"

  (
    cd "$work_dir"
    "${PIN_PREFIX[@]}" "$BIN_PATH" "$datasetName" \
      "$ISBUILT" "$K_LIST" "$L" "$K" "$T" "$EFC" "$PC" "$PQ" "$LSH_UB" \
      > "${out_dir}/run.stdout.txt" 2> "${out_dir}/run.stderr.txt"
  )

  # 把 work_dir 下新增/变化的产物也保留一份快照到 out_dir（可选但很有用）
  # - 这里不做 rm -rf work_dir，因为可能想复用 indexes
  # - 只把 indexes/result.txt 等留在 work_dir 本身即可；out_dir 已有 stdout/stderr/meta
  echo "[OK] done. indexes/results are under: ${work_dir}"
}

main() {
  check_bin
  mkdir -p "${OUT_ROOT}/q250" "${OUT_ROOT}/q500"

  for c in "${CASES_Q250[@]}"; do
    run_one_case "q250" "$c"
  done
  for c in "${CASES_Q500[@]}"; do
    run_one_case "q500" "$c"
  done

  echo "[DONE] All LSH-APG runs finished."
}

main "$@"

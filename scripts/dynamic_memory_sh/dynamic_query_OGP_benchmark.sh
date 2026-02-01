#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# 0) 可配置区：按实际路径改这里即可
# -----------------------------
# 可执行文件（如果就在当前目录，可保持相对路径）
PARTITION_BIN="./Partition"
SSQ_BIN="./SmallScaleQueries"

# 数据
BASE_FBIN="/hd1/workspace/gp-ann/data/my_dataset/base_vectors.fbin"

Q250_FBIN="/hd1/workspace/gp-ann/data/my_dataset/queries.q250.fbin"
GT250_BIN="/hd1/workspace/gp-ann/data/my_dataset/ground_truth_k100.bin"

Q500_FBIN="/hd1/workspace/gp-ann/data/my_dataset/queries.q500.fbin"
GT500_BIN="/hd1/workspace/gp-ann/data/my_dataset/ground_truth_k100_q500.bin"

# Partition 输出前缀（注意：Partition 实际会生成带参数后缀的文件名）
PART_PREFIX="/hd1/workspace/gp-ann/data/my_dataset/base.partition"

# OGP partition 参数
KPART=20
PART_METHOD="OGP"
PART_MODE="default"
O_PARAM="0.2"

# SmallScaleQueries 的 K（num-neighbors）
TOPK=100

# SmallScaleQueries 输出前缀
OUT250_PREFIX="/hd1/workspace/gp-ann/exp_outputs/q250/my_dataset.OGP.k=20.o=0.2/base"
OUT500_PREFIX="/hd1/workspace/gp-ann/exp_outputs/q500/my_dataset.OGP.k=20.o=0.2/base"


# 是否强制重跑 partition（0=如已存在则复用，1=强制重建）
FORCE_PARTITION=0

# -----------------------------
# 1) 固定线程环境（与你命令一致）
# -----------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PARLAY_NUM_THREADS=1
export PARLAY_NUM_WORKERS=1

# CPU/NUMA bind（与你命令一致）
PIN_PREFIX=(taskset -c 0 numactl --cpunodebind=0 --membind=0)

# -----------------------------
# 2) 工具函数
# -----------------------------
die() { echo "[ERROR] $*" >&2; exit 1; }

need_file() {
  local f="$1"
  [[ -f "$f" ]] || die "File not found: $f"
}

need_exec() {
  local f="$1"
  [[ -f "$f" ]] || die "Binary not found: $f"
  [[ -x "$f" ]] || die "Binary not executable (chmod +x?): $f"
}

log() { echo "[LOG] $*"; }

# Partition 的产物文件名：示例里是：
#   base.partition.k=20.OGP.o=0.2
# 这里按命名规则直接拼出来
partition_out_file() {
  echo "${PART_PREFIX}.k=${KPART}.${PART_METHOD}.o=${O_PARAM}"
}

run_partition() {
  local part_file
  part_file="$(partition_out_file)"

  if [[ "$FORCE_PARTITION" -eq 0 && -f "$part_file" ]]; then
    log "Partition exists, reuse: $part_file"
    return 0
  fi

  log "Run Partition -> $part_file"
  "${PIN_PREFIX[@]}" "$PARTITION_BIN" \
    "$BASE_FBIN" \
    "$PART_PREFIX" \
    "$KPART" \
    "$PART_METHOD" \
    "$PART_MODE" \
    "$O_PARAM" \
    > "${part_file}.partition.stdout.txt" 2> "${part_file}.partition.stderr.txt"

  [[ -f "$part_file" ]] || die "Partition finished but output file not found: $part_file"
  log "Partition done: $part_file"
}

run_ssq() {
  local q_tag="$1"
  local query_fbin="$2"
  local gt_bin="$3"
  local out_prefix="$4"

  local part_file
  part_file="$(partition_out_file)"
  need_file "$part_file"

  log "Run SmallScaleQueries (${q_tag})"
  log "  query=$query_fbin"
  log "  gt=$gt_bin"
  log "  out_prefix=$out_prefix"

  "${PIN_PREFIX[@]}" "$SSQ_BIN" \
    "$BASE_FBIN" \
    "$query_fbin" \
    "$gt_bin" \
    "$TOPK" \
    "$part_file" \
    "$PART_METHOD" \
    "$out_prefix" \
    > "${out_prefix}.stdout.txt" 2> "${out_prefix}.stderr.txt"

  log "SmallScaleQueries done: ${q_tag}"
}

# -----------------------------
# 3) main
# -----------------------------
main() {
  need_exec "$PARTITION_BIN"
  need_exec "$SSQ_BIN"

  need_file "$BASE_FBIN"

  need_file "$Q250_FBIN"
  need_file "$GT250_BIN"

  need_file "$Q500_FBIN"
  need_file "$GT500_BIN"

  # 1) partition（只跑一次）
  run_partition

  # 2) 两组 query（各跑一次）
  run_ssq "q250" "$Q250_FBIN" "$GT250_BIN" "$OUT250_PREFIX"
  run_ssq "q500" "$Q500_FBIN" "$GT500_BIN" "$OUT500_PREFIX"

  log "ALL DONE."
}

main "$@"

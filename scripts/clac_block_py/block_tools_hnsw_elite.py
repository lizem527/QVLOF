import csv
import os
import sys
from typing import List, Iterable, Set, Optional

# hnsw 静态 埋点
# 静态有两个 HNSW对应两个计算跨块。

def count_blocks(indices: Iterable[int], n: int) -> int:
    if n <= 0:
        raise ValueError("N must be a positive integer")

    blocks: Set[int] = set()
    for i in indices:
        if not isinstance(i, int):
            raise TypeError(f"index must be int, got {type(i)}: {i}")
        if i < 0:
            raise ValueError(f"index must be non-negative, got {i}")
        blocks.add(i // n)
    return len(blocks)


def parse_space_separated_ints(s: str) -> List[int]:
    """
    Parse strings like:
      '1 2 3'
      '"1 2 3"'
      '' / None -> []
    """
    if s is None:
        return []
    s = str(s).strip()

    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        s = s[1:-1].strip()

    if not s:
        return []

    parts = s.split()
    nums: List[int] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        nums.append(int(p))
    return nums


def _fmt_avg(avg: float) -> str:
    return f"{avg:.6f}".rstrip("0").rstrip(".")


def build_output_filename(base_name: str, block_size: int, avgs: List[float]) -> str:
    """
    文件名后缀：_{block_size}_{avg1}[_{avg2}]
    """
    avg_part = "_".join(_fmt_avg(a) for a in avgs)
    suffix = f"_{block_size}_{avg_part}" if avg_part else f"_{block_size}"

    root, ext = os.path.splitext(base_name)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{base_name}{suffix}"


def main(in_name: str,
         block_size: int,
         ef_value: str,
         k_value: str,
         col1: str,
         col2: Optional[str] = None) -> int:
    """
    col1: 必填，第一列列名
    col2: 可选，第二列列名；不传则只处理 col1
    """

    in_path = os.path.join(os.getcwd(), in_name)
    if not os.path.isfile(in_path):
        print(f"Input file not found in current directory: {in_name}", file=sys.stderr)
        return 1

    rows_out: List[List[str]] = []

    # 每个目标列对应一份 block_nums
    block_nums_list: List[List[int]] = []

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        header = next(reader, None)
        if header is None:
            print("CSV is empty (missing header).", file=sys.stderr)
            return 1

        # ef/k 列索引（用于筛选）
        try:
            ef_idx = header.index("ef")
            k_idx = header.index("k")
        except ValueError:
            print("CSV must contain columns named 'ef' and 'k'.", file=sys.stderr)
            return 1

        # 目标列列表（1列或2列）
        target_cols: List[str] = [col1]
        if col2:
            target_cols.append(col2)

        # 目标列索引
        target_idxs: List[int] = []
        for c in target_cols:
            try:
                target_idxs.append(header.index(c))
            except ValueError:
                print(f"CSV must contain column named '{c}'.", file=sys.stderr)
                return 1

        # 为每个目标列准备一个 block_nums list
        block_nums_list = [[] for _ in target_cols]

        # 输出表头：新增 block_num_<col>
        header_out = list(header)
        out_cols = [f"block_num_{c}" for c in target_cols]
        header_out.extend(out_cols)
        rows_out.append(header_out)

        # 只处理 ef==X 且 k==Y 的行，并且输出也只输出这些行
        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue

            if row[ef_idx] != ef_value:
                continue
            if row[k_idx] != k_value:
                continue

            # 对每个目标列计算 block_num
            block_nums_this_row: List[int] = []
            for j, idx in enumerate(target_idxs):
                v = row[idx] if idx < len(row) else ""
                nums = parse_space_separated_ints(v)
                b = count_blocks(nums, block_size)
                block_nums_list[j].append(b)
                block_nums_this_row.append(b)

            row_new = list(row)
            row_new.extend([str(x) for x in block_nums_this_row])
            rows_out.append(row_new)

    matched = len(block_nums_list[0]) if block_nums_list else 0
    avgs: List[float] = []
    for nums in block_nums_list:
        avg = (sum(nums) / matched) if matched else 0.0
        avgs.append(avg)

    out_name_final = build_output_filename(in_name, block_size, avgs)
    out_path = os.path.join(os.getcwd(), out_name_final)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    # 打印统计：1列就打印一份，2列就打印两份
    for i, c in enumerate([col1] + ([col2] if col2 else [])):
        nums = block_nums_list[i]
        print(f"===== Column {i + 1} =====")
        print(f"col={c}")
        print(f"total_block={sum(nums)}")
        print(f"block_size={block_size}, block_num_avg={avgs[i]:.6f}")
        print(f"matched_rows={matched}")

    print(f"Done. Wrote output to: {out_name_final}")
    return 0


if __name__ == "__main__":

    main(r"C:\Users\Launcher\Desktop\静态向量索引实验_V6\GIST\hnsw_ids_f4_k100.csv", 10000, "256", "100", "topk_id_list", "dist_id_list")

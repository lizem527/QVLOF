import csv
import os
import sys
from typing import List
from typing import Iterable, Set

import pandas as pd


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


def parse_last_col_to_int_list(s: str) -> List[int]:
    if s is None:
        return []
    s = s.strip()

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


def build_output_filename(base_name: str, block_size: int, avg: float) -> str:
    avg_str = f"{avg:.6f}".rstrip("0").rstrip(".")
    suffix = f"_{block_size}_{avg_str}"

    root, ext = os.path.splitext(base_name)
    if ext:
        return f"{root}{suffix}{ext}"
    return f"{base_name}{suffix}"


def main(in_name, block_size, ef_value, k_value) -> int:

    # in_name = sys.argv[1]
    out_name_base = in_name    # 输出文件名基于输入文件名（延续原来的逻辑）
    # block_size_str = sys.argv[2]
    # ef_value = sys.argv[3]
    # k_value = sys.argv[4]

    in_path = os.path.join(os.getcwd(), in_name)
    if not os.path.isfile(in_path):
        print(f"Input file not found in current directory: {in_name}", file=sys.stderr)
        return 1

    rows_out: List[List[str]] = []
    block_nums: List[int] = []

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        # 读表头
        header = next(reader, None)
        if header is None:
            print("CSV is empty (missing header).", file=sys.stderr)
            return 1

        # 找到 ef/k 列索引（用于筛选）
        try:
            ef_idx = header.index("ef")
            k_idx = header.index("k")
        except ValueError:
            print("CSV must contain columns named 'ef' and 'k'.", file=sys.stderr)
            return 1

        # 输出表头（新增 block_num）
        header_out = list(header)
        header_out.append("block_num")
        rows_out.append(header_out)

        # 只处理 ef==X 且 k==Y 的行，并且输出也只输出这些行
        for row_idx, row in enumerate(reader, start=2):
            if not row:
                continue

            if row[ef_idx] != ef_value:
                continue
            if row[k_idx] != k_value:
                continue

            last_val = row[-1]
            nums = parse_last_col_to_int_list(last_val)

            block_num = count_blocks(nums, block_size)
            block_nums.append(block_num)

            row_new = list(row)
            row_new.append(str(block_num))
            rows_out.append(row_new)

    # 平均数只对匹配行计算
    avg = (sum(block_nums) / len(block_nums)) if block_nums else 0.0

    # 输出文件名：基于输入文件名 + block_size + avg
    out_name_final = build_output_filename(out_name_base, block_size, avg)

    out_path = os.path.join(os.getcwd(), out_name_final)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    print(f"total_block={sum(block_nums)}")
    print(f"block_size={block_size}, block_num_avg={avg:.6f}")
    print(f"matched_rows={len(block_nums)}")
    print(f"Done. Wrote output to: {out_name_final}")
    return 0


if __name__ == "__main__":

    main(r'C:\Users\Launcher\Desktop\静态向量索引实验_V6\DEEP\hnsw_ids_f1_k100.csv', 10000, '256', '100')

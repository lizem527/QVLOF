import csv
import os
import sys
from typing import Iterable, List, Set


def count_blocks(indices: Iterable[int], block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer")

    blocks: Set[int] = set()
    for i in indices:
        if not isinstance(i, int):
            raise TypeError(f"index must be int, got {type(i)}: {i}")
        if i < 0:
            raise ValueError(f"index must be non-negative, got {i}")
        blocks.add(i // block_size)
    return len(blocks)


def parse_last_col_to_int_list(s: str) -> List[int]:
    if s is None:
        return []
    s = s.strip()

    # strip wrapping quotes if present
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


def build_block_output_path(in_path: str) -> str:
    """
    output in same directory as input,
    filename: <input_root>_block<input_ext>
    """
    dir_name = os.path.dirname(in_path)
    base = os.path.basename(in_path)
    root, ext = os.path.splitext(base)
    out_base = f"{root}_block{ext}"
    return os.path.join(dir_name if dir_name else os.getcwd(), out_base)


def main(in_path: str, block_size, row_size) -> int:

    if not os.path.isfile(in_path):
        print(f"Input file not found: {in_path}", file=sys.stderr)
        return 1

    # 逐行计算 block_num（跨块数），只在内存里用来做 window 聚合
    block_nums: List[int] = []

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        header = next(reader, None)
        if header is None:
            print("CSV is empty (missing header).", file=sys.stderr)
            return 1

        for row in reader:
            if not row:
                continue
            nums = parse_last_col_to_int_list(row[-1])
            block_num = count_blocks(nums, block_size)
            block_nums.append(block_num)

    total_rows = len(block_nums)
    full_windows = total_rows // row_size
    remainder = total_rows % row_size

    out_path = build_block_output_path(in_path)

    # 写 window 统计结果到 CSV
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_id", "rows_in_window", "avg_block", "total_block"])

        for win in range(full_windows):
            start = win * row_size
            end = start + row_size
            chunk = block_nums[start:end]
            total_block = sum(chunk)
            avg_block = total_block / row_size
            w.writerow([win + 1, row_size, f"{avg_block:.6f}", total_block])

    print(f"Input: {in_path}")
    print(f"block_size={block_size}, row_size={row_size}")
    print(f"data_rows={total_rows} (excluding header)")
    print(f"windows_written={full_windows}, remainder_rows_ignored={remainder}")
    print(f"Done. Wrote window summary to: {out_path}")
    return 0


if __name__ == "__main__":
    main(
        r"",
        100,
        200
    )

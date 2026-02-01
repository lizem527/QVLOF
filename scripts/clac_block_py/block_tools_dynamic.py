import csv
import os
import sys
from typing import List, Iterable, Set

# 动态 
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


def main(in_name: str, block_size: int, windows_num: int, shard_num: int, windows_size: int, shard_seat: int) -> int:
    """
    输出（打印）每个 window_id 一行：
      window_id, shard_seat, sum_blocks(num_probes in [1..shard_seat]), sum_blocks(num_probes == shard_seat)
    行数固定为 windows_num（window_id 从 0 到 windows_num-1）。
    """

    if block_size <= 0 or windows_num <= 0 or shard_num <= 0 or windows_size <= 0 or shard_seat <= 0:
        print("block_size/windows_num/shard_num/windows_size/shard_seat must be positive integers.", file=sys.stderr)
        return 2

    out_name_base = in_name
    in_path = os.path.join(os.getcwd(), in_name)
    if not os.path.isfile(in_path):
        print(f"Input file not found in current directory: {in_name}", file=sys.stderr)
        return 1

    rows_out: List[List[str]] = []

    # 你要的两个统计：范围 [1..shard_seat] 和 精确 == shard_seat
    window_sum_blocks_1_to_seat: List[int] = [0 for _ in range(windows_num)]
    window_sum_blocks_eq_seat: List[int] = [0 for _ in range(windows_num)]

    # （可选）保留原来的 row_counts，调试用；你如果不需要可以删掉
    window_row_counts: List[int] = [0 for _ in range(windows_num)]

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)

        header = next(reader, None)
        if header is None:
            print("CSV is empty (missing header).", file=sys.stderr)
            return 1

        # 找列
        try:
            window_id_idx = header.index("window_id")
        except ValueError:
            print("CSV must contain a column named 'window_id'.", file=sys.stderr)
            return 1

        try:
            num_probes_idx = header.index("num_probes")
        except ValueError:
            print("CSV must contain a column named 'num_probes'.", file=sys.stderr)
            return 1

        # 仍然输出一个带 block_num 的CSV（如果你不需要这块，下面这一段和后面写文件的部分都可以删掉）
        header_out = list(header)
        header_out.append("block_num")
        rows_out.append(header_out)

        for row in reader:
            if not row:
                continue

            # window_id
            try:
                window_id = int(str(row[window_id_idx]).strip())
            except (ValueError, TypeError):
                continue
            if window_id < 0 or window_id >= windows_num:
                continue

            # num_probes
            try:
                num_probes = int(str(row[num_probes_idx]).strip())
            except (ValueError, TypeError):
                continue

            # block_num from last column
            nums = parse_last_col_to_int_list(row[-1])
            block_num = count_blocks(nums, block_size)

            # 累计你要的两类统计
            if 1 <= num_probes <= shard_seat:
                window_sum_blocks_1_to_seat[window_id] += block_num
            if num_probes == shard_seat:
                window_sum_blocks_eq_seat[window_id] += block_num

            window_row_counts[window_id] += 1

            # （可选）写入输出CSV
            row_new = list(row)
            row_new.append(str(block_num))
            rows_out.append(row_new)

    # ---- 打印：只输出你要的四个字段（windows_num 行）----
    # 你如果不想要表头，把下一行注释掉即可
    headers = [
        "window_id",
        "shard_seat",
        "sum_blocks(num_probes=1..seat)",
        "sum_blocks(num_probes=seat)",
    ]

    rows = []
    for wid in range(windows_num):
        rows.append([
            str(wid),
            str(shard_seat),
            str(window_sum_blocks_1_to_seat[wid]),
            str(window_sum_blocks_eq_seat[wid]),
        ])

    # 计算每列宽度
    col_widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows) if rows else 0)
        for i in range(len(headers))
    ]

    def fmt_row(cols):
        return " | ".join(cols[i].rjust(col_widths[i]) for i in range(len(cols)))

    sep = "-+-".join("-" * w for w in col_widths)

    print(fmt_row(headers))
    print(sep)
    for r in rows:
        print(fmt_row(r))

    # ---- （可选）输出CSV文件名：沿用你之前的方式（overall_avg 仍按旧口径算，用于命名）----
    total_blocks_all = sum(window_sum_blocks_1_to_seat)  # 用范围统计作为 overall 口径
    denom_all = windows_num * shard_num * windows_size
    overall_avg = (total_blocks_all / denom_all) if denom_all > 0 else 0.0
    out_name_final = build_output_filename(out_name_base, block_size, overall_avg)

    out_path = os.path.join(os.getcwd(), out_name_final)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_out)

    return 0


if __name__ == "__main__":
    # 示例：新增 shard_seat 参数
    main(
        r"C:\\Users\\Launcher\\Desktop\\base.per_query.csv",
        100,   # block_size
        15,    # windows_num
        23,    # shard_num（用于可选的文件命名 avg 口径）
        200,   # windows_size（用于可选的文件命名 avg 口径）
        3      # shard_seat（新增的参数） 取前三个块去查，3之后就不查了
    )

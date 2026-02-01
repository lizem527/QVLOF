import csv
from collections import Counter

def check_duplicate_ids_in_csv(
        csv_path,
        id_col="id_list",
        ignore_id=-1,
        max_print=10,
):
    """
    读取 ids_csv_path，检查 id_list 中是否存在除 ignore_id 之外的重复 id

    参数:
      csv_path   : ids_csv_path
      id_col     : id 列名（默认 id_list）
      ignore_id  : 忽略的 id（默认 -1）
      max_print  : 最多打印多少条异常记录

    返回:
      bad_rows: list[dict]  每个 dict 是一条有重复 id 的记录
    """
    bad_rows = []
    total_rows = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        assert id_col in reader.fieldnames, f"Column '{id_col}' not found in CSV"

        for row in reader:
            total_rows += 1

            # 解析 id_list
            raw = row[id_col].strip()
            if not raw:
                continue

            ids = [int(x) for x in raw.split()]
            ids = [x for x in ids if x != ignore_id]

            if len(ids) <= 1:
                continue

            cnt = Counter(ids)
            dup_ids = [i for i, c in cnt.items() if c > 1]

            if dup_ids:
                info = {
                    "ef": row.get("ef"),
                    "k": row.get("k"),
                    "round": row.get("round"),
                    "query": row.get("query"),
                    "dup_ids": dup_ids,
                    "full_id_list": ids,
                }
                bad_rows.append(info)

                if len(bad_rows) <= max_print:
                    print(" DUPLICATE ID FOUND")
                    print(f"  ef={info['ef']} k={info['k']} round={info['round']} query={info['query']}")
                    print(f"  duplicate ids = {dup_ids}")
                    print(f"  id_list = {ids}")
                    print("-" * 60)

    print(f"\nChecked {total_rows} rows")
    print(f"Found {len(bad_rows)} rows with duplicate ids (excluding {ignore_id})")

    return bad_rows

if __name__ == '__main__':
    bad = check_duplicate_ids_in_csv(
            "/hd1/workspace/LSH-APG/LSH-APG-main/dataset/ANN/5w_f7.per_query.csv",
        ignore_id=-1,
    )

    if not bad:
        print(" 所有 query 的 id_list 都是唯一的（除了 -1）")
    else:
        print(" 发现重复 idx，请重点排查这些 query")



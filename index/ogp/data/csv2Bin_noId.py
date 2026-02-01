import csv
import struct
import numpy as np

CSV_IN  = "/hd1/student/lxb/query/dynamic/base_vectors_q250.csv"          # v0..v127
FBIN_OUT = "/hd1/workspace/gp-ann/data/my_dataset/queries.q250.fbin"  # 输出给 SmallScaleQueries 用

def csv_to_fbin_no_id(csv_in, fbin_out):
    rows = []
    with open(csv_in, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        dim = len(header)
        print("detected dim =", dim)

        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r])

    X = np.asarray(rows, dtype=np.float32)
    n, d = X.shape
    assert d == dim

    with open(fbin_out, "wb") as f:
        f.write(struct.pack("<ii", n, d))   # int32 n, int32 d
        f.write(X.tobytes(order="C"))

    print(f"Wrote {n} queries, dim={d} → {fbin_out}")

csv_to_fbin_no_id(CSV_IN, FBIN_OUT)


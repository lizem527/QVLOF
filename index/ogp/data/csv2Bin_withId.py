import csv
import struct
import numpy as np

CSV_IN  = "/hd1/student/lzm/FANNS/Chiristmas/result/test/f6_noA2_sorted.csv"
FBIN_OUT = "/hd1/workspace/gp-ann/data/my_dataset/data_f6.fbin"

def csv_to_fbin_drop_id(csv_in, fbin_out):
    rows = []
    with open(csv_in, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header[0].lower() == "id"
        dim = len(header) - 1
        print("detected dim =", dim)

        for r in reader:
            if not r:
                continue
            rows.append([float(x) for x in r[1:]])

    X = np.asarray(rows, dtype=np.float32)
    n, d = X.shape
    assert d == dim

    with open(fbin_out, "wb") as f:
        f.write(struct.pack("<ii", n, d))
        f.write(X.tobytes(order="C"))

    print(f"Wrote {n} base vectors, dim={d} → {fbin_out}")

csv_to_fbin_drop_id(CSV_IN, FBIN_OUT)



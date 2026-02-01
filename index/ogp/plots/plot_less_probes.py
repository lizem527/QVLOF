import pandas as pd
import matplotlib.pyplot as plt

PER_Q = "/hd1/workspace/gp-ann/exp_outputs/my_dataset.OGP.k=20.o=0.2.csv.per_query.csv"
TARGET_RECALL = 0.9
FIXED_PROBES = 20

df = pd.read_csv(PER_Q)

# -------- dynamic：最小 probes 达到 target recall --------
dyn = (
    df[df["recall"] >= TARGET_RECALL]
        .sort_values(["query_id", "num_probes"])
        .groupby("query_id")
        .first()
        .reset_index()
)

# -------- fixed probes baseline --------
fixed = (
    df[df["num_probes"] == FIXED_PROBES]
        .groupby("query_id")
        .mean(numeric_only=True)
        .reset_index()
)

merged = dyn.merge(
    fixed,
    on="query_id",
    suffixes=("_dyn", "_fixed")
)

# -------- 画延迟分布 --------
plt.figure(figsize=(6,4))
plt.hist(merged["query_lat_ms_fixed"], bins=50, alpha=0.6, label=f"fixed probes={FIXED_PROBES}")
plt.hist(merged["query_lat_ms_dyn"], bins=50, alpha=0.6, label="dynamic (min probes)")
plt.xlabel("query latency (ms)")
plt.ylabel("count")
plt.title(f"Latency @ recall≥{TARGET_RECALL}")
plt.legend()
plt.tight_layout()
plt.show()

print(merged[["query_lat_ms_dyn","query_lat_ms_fixed"]].describe())


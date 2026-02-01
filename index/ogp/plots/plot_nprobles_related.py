import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PER_Q = "/hd1/workspace/gp-ann/exp_outputs/my_dataset.OGP.k=20.o=0.2_q500.csv.per_query.csv"

WINDOW_SIZE = 200
TARGET_RECALL = 0.90

df = pd.read_csv(PER_Q)

# 确保类型正确
df["num_probes"] = df["num_probes"].astype(int)
df["query_id"] = df["query_id"].astype(int)
df["window_id"] = df["window_id"].astype(int)

# -----------------------------
# 1) 每条 query：找达到 TARGET_RECALL 的最小 probes（dynamic/min probes）
# -----------------------------
ok = df[df["recall"] >= TARGET_RECALL].copy()

# 对每个 query_id 取最小 num_probes 的那一行（也就是该 probes 截止点的 latency/recall）
idx = ok.groupby("query_id")["num_probes"].idxmin()
minp = ok.loc[idx].copy()

# 有些 query 可能永远达不到阈值（比如 recall<0.9），会缺失
# 你可以看一下缺失比例：
missing = df["query_id"].nunique() - minp["query_id"].nunique()
print("queries total =", df["query_id"].nunique(), "missing(threshold not reached) =", missing)

# 补一个 x 轴：用 window 的起始 query_id（你说不想用 window_id，就用这个）
minp["window_begin_qid"] = (minp["window_id"] * WINDOW_SIZE).astype(int)

# -----------------------------
# 2) window 粒度聚合：avg_min_probes / avg_latency_at_min_probes / avg_recall_at_min_probes
# -----------------------------
w = (minp.groupby("window_id")
     .agg(window_begin_qid=("window_begin_qid", "first"),
          avg_min_probes=("num_probes", "mean"),
          p50_min_probes=("num_probes", "median"),
          avg_total_lat_ms=("total_lat_ms", "mean"),
          avg_query_lat_ms=("query_lat_ms", "mean"),
          avg_recall=("recall", "mean"),
          cnt=("query_id","count"))
     .reset_index()
     .sort_values("window_id")
     )

# -----------------------------
# 3) 图：横轴=window_begin_qid（每 200 一个点）
# -----------------------------

# 图 A：window vs avg_min_probes（最关键：体现 nprobes）
plt.figure(figsize=(10,4))
plt.plot(w["window_begin_qid"], w["avg_min_probes"], marker="o")
plt.xlabel(f"window begin query_id (every {WINDOW_SIZE} queries)")
plt.ylabel(f"avg min probes for recall >= {TARGET_RECALL}")
plt.title("Window-wise difficulty: probes needed to reach target recall")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 图 B：window vs avg_total_lat_ms（在“达标最小 probes”处的平均总延迟）
plt.figure(figsize=(10,4))
plt.plot(w["window_begin_qid"], w["avg_total_lat_ms"], marker="o")
plt.xlabel(f"window begin query_id (every {WINDOW_SIZE} queries)")
plt.ylabel("avg total latency (ms) at min probes")
plt.title("Window-wise latency at dynamic(min probes) cutoff")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# （可选）图 C：window vs avg_recall（理论上应该 >=TARGET_RECALL，但可能略高）
plt.figure(figsize=(10,4))
plt.plot(w["window_begin_qid"], w["avg_recall"], marker="o")
plt.xlabel(f"window begin query_id (every {WINDOW_SIZE} queries)")
plt.ylabel("avg recall@k at min probes")
plt.title("Window-wise recall at dynamic(min probes) cutoff")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


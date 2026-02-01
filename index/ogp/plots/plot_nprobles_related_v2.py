import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PER_Q = "/hd1/workspace/gp-ann/exp_outputs/my_dataset.OGP.k=20.o=0.2_q500.csv.per_query.csv"

WINDOW_SIZE = 200
TARGET_RECALL = 0.95

df = pd.read_csv(PER_Q)

# 确保类型正确
df["num_probes"] = df["num_probes"].astype(int)
df["query_id"] = df["query_id"].astype(int)
df["window_id"] = df["window_id"].astype(int)

# -----------------------------
# 1) 每条 query：找达到 TARGET_RECALL 的最小 probes（dynamic/min probes）
# -----------------------------
ok = df[df["recall"] >= TARGET_RECALL].copy()

# 对每个 query_id 取最小 num_probes 的那一行（达到阈值的最小 probes）
idx = ok.groupby("query_id")["num_probes"].idxmin()
minp = ok.loc[idx].copy()

missing = df["query_id"].nunique() - minp["query_id"].nunique()
print("queries total =", df["query_id"].nunique(), "missing(threshold not reached) =", missing)

# window 起止 query_id（你想用 query_id 做横轴）
minp["window_begin_qid"] = (minp["window_id"] * WINDOW_SIZE).astype(int)
minp["window_end_qid"] = (minp["window_begin_qid"] + WINDOW_SIZE).astype(int)

# 如果最后一个 window 不满 200，修正 end
max_qid = int(df["query_id"].max()) + 1
minp["window_end_qid"] = minp["window_end_qid"].clip(upper=max_qid)

# -----------------------------
# 2) window 粒度聚合：avg_min_probes / avg_latency / avg_recall
# -----------------------------
w = (minp.groupby("window_id")
     .agg(
    window_begin_qid=("window_begin_qid", "first"),
    window_end_qid=("window_end_qid", "first"),
    avg_min_probes=("num_probes", "mean"),
    p50_min_probes=("num_probes", "median"),
    avg_total_lat_ms=("total_lat_ms", "mean"),
    avg_query_lat_ms=("query_lat_ms", "mean"),
    avg_recall=("recall", "mean"),
    cnt=("query_id", "count"),
)
     .reset_index()
     .sort_values("window_id")
     )

# -----------------------------
# 3) 生成 step 序列：每个 window 一段水平线
#    用 where="post"：x[i] 到 x[i+1] 是 y[i]
#    所以我们给 x = [begin0, begin1, ..., beginN, end_last]
# -----------------------------
x_edges = w["window_begin_qid"].to_numpy()
x_edges = np.append(x_edges, w["window_end_qid"].iloc[-1])   # 最后补一个 end

def plot_step(y, ylabel, title):
    plt.figure(figsize=(10,4))
    plt.step(x_edges, np.append(y, y.iloc[-1]), where="post", linewidth=2.2)
    # 画一下 window 边界竖线（可选，能更像“窗口段”）
    for xb in w["window_begin_qid"].to_numpy():
        plt.axvline(xb, alpha=0.08)
    plt.xlabel(f"query_id (windows of {WINDOW_SIZE})")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()

# 图 A：window-wise avg_min_probes（阶梯线）
plot_step(
    w["avg_min_probes"],
    ylabel=f"avg min probes (recall >= {TARGET_RECALL})",
    title="Dynamic(min-probes) per window: avg_min_probes as step segments"
)

# 图 B：window-wise avg_total_lat_ms（达标最小 probes 处的 total latency，阶梯线）
plot_step(
    w["avg_total_lat_ms"],
    ylabel="avg total latency (ms) at min probes",
    title="Dynamic(min-probes) per window: avg_total_lat_ms as step segments"
)

# 图 C：window-wise avg_recall（阶梯线）
plot_step(
    w["avg_recall"],
    ylabel="avg recall@k at min probes",
    title="Dynamic(min-probes) per window: avg_recall as step segments"
)


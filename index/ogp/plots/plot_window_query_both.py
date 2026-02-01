import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PER_Q = "/hd1/workspace/gp-ann/exp_outputs/my_dataset.OGP.k=20.o=0.2_q500.csv.per_query.csv"
PER_W = "/hd1/workspace/gp-ann/exp_outputs/my_dataset.OGP.k=20.o=0.2_q500.csv.per_window.csv"

TARGET_PROBES = 20
WINDOW_SIZE = 200

q = pd.read_csv(PER_Q)
w = pd.read_csv(PER_W)

# 只取某个 num_probes 的“动态累计结果”
q = q[q["num_probes"] == TARGET_PROBES].copy()
w = w[w["num_probes"] == TARGET_PROBES].copy()

# 你要的 qps：建议先用 query_lat_ms（不含 routing），也可改成 total_lat_ms（含 routing）
q["qps"] = 1000.0 / q["query_lat_ms"].clip(lower=1e-9)      # 每条 query 的“等效 QPS”
# q["qps"] = 1000.0 / q["total_lat_ms"].clip(lower=1e-9)    # 如果你想把 routing 也算进去

# window 均值 qps（用 per_window 的 avg_query_lat_ms 反推）
w = w.sort_values("window_id")
w["q_begin"] = w["window_id"] * WINDOW_SIZE
w["avg_qps"] = 1000.0 / w["avg_query_lat_ms"].clip(lower=1e-9)
# w["avg_qps"] = 1000.0 / w["avg_total_lat_ms"].clip(lower=1e-9)  # 如果含 routing

q = q.sort_values("query_id")

def shade_windows(ax, n_queries, window_size):
    # 交替底色，突出 window 分段
    n_windows = int((n_queries + window_size - 1) // window_size)
    for wid in range(n_windows):
        x0 = wid * window_size
        x1 = min((wid + 1) * window_size, n_queries)
        if wid % 2 == 0:
            ax.axvspan(x0, x1, alpha=0.08)

nq = int(q["query_id"].max()) + 1

# ---------------- 图1：recall ----------------
plt.figure(figsize=(10,4))
ax = plt.gca()
shade_windows(ax, nq, WINDOW_SIZE)

ax.scatter(q["query_id"], q["recall"], s=6, alpha=0.35, label="per-query recall")
# window avg 画成阶梯线（更像“每 200 一段”）
ax.step(w["q_begin"], w["avg_recall"], where="post", linewidth=2.0, label="window avg recall")

ax.set_xlabel("query_id (grouped by windows of 200)")
ax.set_ylabel("recall@k")
ax.set_title(f"Dynamic Query (num_probes={TARGET_PROBES}): per-query recall + window average")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()

# ---------------- 图2：qps ----------------
plt.figure(figsize=(10,4))
ax = plt.gca()
shade_windows(ax, nq, WINDOW_SIZE)

ax.scatter(q["query_id"], q["qps"], s=6, alpha=0.35, label="per-query qps")
ax.step(w["q_begin"], w["avg_qps"], where="post", linewidth=2.0, label="window avg qps")

ax.set_xlabel("query_id (grouped by windows of 200)")
ax.set_ylabel("QPS (approx = 1000 / latency_ms)")
ax.set_title(f"Dynamic Query (num_probes={TARGET_PROBES}): per-query QPS + window average")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()


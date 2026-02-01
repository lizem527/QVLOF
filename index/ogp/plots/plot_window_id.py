import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/hd1/workspace/gp-ann/exp_outputs/my_dataset.OGP.k=20.o=0.2.csv.per_window.csv")

# 固定一个 num_probes
TARGET_PROBES = 20
sub = df[df["num_probes"] == TARGET_PROBES].sort_values("window_id")

# -------- 图 1：window vs recall --------
plt.figure(figsize=(6,4))
plt.plot(sub["window_id"], sub["avg_recall"], marker='o')
plt.xlabel("window_id")
plt.ylabel("avg_recall@k")
plt.title(f"Window-wise avg recall (num_probes={TARGET_PROBES})")
plt.grid(True)
plt.tight_layout()
plt.show()

# -------- 图 2：window vs latency --------
plt.figure(figsize=(6,4))
plt.plot(sub["window_id"], sub["avg_query_lat_ms"], marker='o')
plt.xlabel("window_id")
plt.ylabel("avg_query_lat_ms")
plt.title(f"Window-wise avg query latency (num_probes={TARGET_PROBES})")
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

FEATURES = ["requests_per_min", "error_rate", "p95_latency_ms", "cpu_utilization", "mem_utilization"]

df = pd.read_csv("data/metrics.csv", parse_dates=["timestamp"])

scaler = joblib.load(Path("artifacts/scaler.pkl"))
model  = joblib.load(Path("artifacts/iforest.pkl"))

Xs = scaler.transform(df[FEATURES].values)
scores = model.decision_function(Xs)            # smaller = more anomalous
preds  = (model.predict(Xs) == -1).astype(int)  # 1 = anamaly

df["if_score"] = scores
df["if_pred"]  = preds

# Plot p95 latency and mark anomalies in red
plt.figure(figsize=(12,4))
plt.plot(df["timestamp"], df["p95_latency_ms"], label="p95 latency (ms)")
anom = df[df["if_pred"]==1]
plt.scatter(anom["timestamp"], anom["p95_latency_ms"], s=12, label="anomaly", zorder=3)
plt.legend()
plt.title("p95 latency with anomalies")
plt.xlabel("time"); plt.ylabel("ms")
plt.tight_layout()
plt.show()

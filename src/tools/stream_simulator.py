import os, time, json, requests, joblib
import pandas as pd
import numpy as np
from pathlib import Path

API = os.environ.get("PREDICT_API", "http://127.0.0.1:8000/predict")
SLEEP_SEC = float(os.environ.get("STREAM_INTERVAL", "1.0"))

def row_to_payload(row):
    return {
        "requests_per_min": float(row["requests_per_min"]),
        "error_rate": float(row["error_rate"]),
        "p95_latency_ms": float(row["p95_latency_ms"]),
        "cpu_utilization": float(row["cpu_utilization"]),
        "mem_utilization": float(row["mem_utilization"]),
    }

def main():
    df = pd.read_csv("data/metrics.csv")
    print(f"Streaming {len(df)} rows to {API} every {SLEEP_SEC}s ... Ctrl+C to stop")
    for i, row in df.iterrows():
        payload = row_to_payload(row)
        r = requests.post(API, json=payload, timeout=5)
        r.raise_for_status()
        out = r.json()
        tag = "🚨" if out.get("is_anomaly") else "✅"
        print(f"{tag} t={i:04d} resp={json.dumps(out)}")
        time.sleep(SLEEP_SEC)

if __name__ == "__main__":
    main()

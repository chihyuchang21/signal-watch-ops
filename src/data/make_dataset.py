import os
from pathlib import Path
import numpy as np
import pandas as pd

def generate_series(n_minutes=24*60, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_minutes, freq="T")

    # Base signals
    req = rng.normal(320, 30, size=n_minutes).clip(100, 800)
    err = np.clip(rng.normal(0.012, 0.004, size=n_minutes), 0.0, 0.08)
    p95 = np.clip(rng.normal(150, 25, size=n_minutes), 60, 400)
    cpu = np.clip(rng.normal(50, 8, size=n_minutes), 10, 90)
    mem = np.clip(rng.normal(62, 6, size=n_minutes), 20, 95)

    # Inject anomalies (spikes in error or latency, or resource saturation)
    labels = np.zeros(n_minutes, dtype=int)
    anomaly_idxs = rng.choice(n_minutes, size=18, replace=False)
    for i in anomaly_idxs:
        choice = rng.choice(["error_spike","latency_spike","cpu_mem_spike"])
        if choice == "error_spike":
            err[i:i+3] = np.clip(err[i:i+3] + rng.uniform(0.15, 0.4), 0, 0.9)
        elif choice == "latency_spike":
            p95[i:i+5] = np.clip(p95[i:i+5] + rng.uniform(500, 1000), 60, 2000)
        else:
            cpu[i:i+4] = np.clip(cpu[i:i+4] + rng.uniform(30, 45), 10, 100)
            mem[i:i+4] = np.clip(mem[i:i+4] + rng.uniform(25, 35), 20, 100)
        labels[i] = 1  # mark the starting minute as anomalous

    df = pd.DataFrame({
        "timestamp": idx,
        "requests_per_min": req,
        "error_rate": err,
        "p95_latency_ms": p95,
        "cpu_utilization": cpu,
        "mem_utilization": mem,
        "label": labels
    })
    return df

def main():
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True, parents=True)
    df = generate_series()
    csv_path = out_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} with shape {df.shape}")

if __name__ == "__main__":
    main()

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from pathlib import Path
import numpy as np

from src.utils.features import to_feature_vector

app = FastAPI(title="Anomaly Detection API", version="0.1.0")
SCALER_PATH = Path("artifacts/scaler.pkl")
MODEL_PATH = Path("artifacts/iforest.pkl")

scaler = None
model = None

class MetricsIn(BaseModel):
    requests_per_min: float = Field(..., ge=0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    cpu_utilization: float = Field(..., ge=0.0, le=100.0)
    mem_utilization: float = Field(..., ge=0.0, le=100.0)

@app.on_event("startup")
def load_artifacts():
    global scaler, model
    if not SCALER_PATH.exists() or not MODEL_PATH.exists():
        # Lazy message to help local dev
        print("WARNING: artifacts missing. Run training: python -m src.models.train_isolation_forest")
        scaler = None
        model = None
    else:
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)
        print("Artifacts loaded. Ready to score.")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

def quick_reason(m):
    reasons = []
    if m["error_rate"] > 0.1:
        reasons.append("High error_rate")
    if m["p95_latency_ms"] > 800:
        reasons.append("High latency")
    if m["cpu_utilization"] > 85:
        reasons.append("CPU saturation")
    if m["mem_utilization"] > 85:
        reasons.append("Memory saturation")
    return reasons or ["Pattern anomaly"]

@app.post("/predict")
def predict(payload: MetricsIn):
    m = payload.model_dump()
    feats = np.array([to_feature_vector(m)])
    if scaler is not None and model is not None:
        Xs = scaler.transform(feats)
        score = model.decision_function(Xs)[0]
        pred = int(model.predict(Xs)[0] == -1)
    else:
        # Fallback: simple rule-based if model not trained yet
        score = 0.0
        pred = int(any(r in ["High error_rate","High latency","CPU saturation","Memory saturation"] for r in quick_reason(m)))

    return {
        "is_anomaly": bool(pred),
        "score": float(score),  # lower is more anomalous in IsolationForest
        "reasons": quick_reason(m)
    }

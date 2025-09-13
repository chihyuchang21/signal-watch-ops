# AI-powered Observability (Anomaly Detection + LLM-ready API)

This project builds a small observability tool that detects anomalies in system metrics (requests, error rate, p95 latency, CPU, memory).
It includes:
- Synthetic data generation
- Isolation Forest anomaly detection
- FastAPI microservice for real-time predictions

## Quickstart

### 1) Create a virtual environment and install deps
Windows (PowerShell):
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux (bash/zsh):
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate synthetic dataset
```
python -m src.data.make_dataset
```

### 3) Train the Isolation Forest model
```
python -m src.models.train_isolation_forest
```

### 4) Run the API
```
uvicorn src.api.main:app --reload
```

### 5) Try a prediction
```
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample.json
```

Where `sample.json`:
```json
{
  "requests_per_min": 320,
  "error_rate": 0.015,
  "p95_latency_ms": 140,
  "cpu_utilization": 48.0,
  "mem_utilization": 62.0
}
```

## Project Layout
```
src/
  api/main.py                # FastAPI app
  data/make_dataset.py       # Generate synthetic metrics with injected anomalies
  models/train_isolation_forest.py  # Train and export model + scaler
  utils/features.py          # (placeholder) feature helpers
  utils/parsing.py           # (placeholder) parsing helpers for logs
artifacts/                   # Saved model/scaler and metrics
data/                        # Generated datasets
tests/                       # (placeholder) tests
```

## Next steps
- Add streaming ingestion (Kafka) + online scoring
- Hook to Slack/Jira for alerting and incident tickets
- Add LLM agent service for root-cause suggestions

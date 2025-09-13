from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

FEATURES = ["requests_per_min", "error_rate", "p95_latency_ms", "cpu_utilization", "mem_utilization"]

def main():
    data_path = Path("data/metrics.csv")
    if not data_path.exists():
        raise FileNotFoundError("data/metrics.csv not found. Run: python -m src.data.make_dataset")
    df = pd.read_csv(data_path)
    X = df[FEATURES].values
    y = df["label"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Approximate anomaly proportion (tune if needed)
    model = IsolationForest(random_state=42, contamination=0.05, n_estimators=300)
    model.fit(Xs)

    # Map IsolationForest outputs to 0/1 (1=anomaly)
    y_pred = (model.predict(Xs) == -1).astype(int)

    report = classification_report(y, y_pred, digits=4)
    print(report)

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True, parents=True)
    joblib.dump(scaler, artifacts / "scaler.pkl")
    joblib.dump(model, artifacts / "iforest.pkl")
    (artifacts / "metrics.txt").write_text(report)
    print("Saved artifacts/scaler.pkl and artifacts/iforest.pkl")

if __name__ == "__main__":
    main()

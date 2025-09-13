def to_feature_vector(d: dict):
    """Select and order features for the model from a raw dict."""
    keys = ["requests_per_min", "error_rate", "p95_latency_ms", "cpu_utilization", "mem_utilization"]
    return [float(d[k]) for k in keys]

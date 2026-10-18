"""Log one training run to MLflow (only when a tracking server is set).

Set MLFLOW_TRACKING_URI (e.g. http://mlflow:5000 inside Docker) to turn
tracking on. Without it, training works exactly as before.
"""
import os

EXPERIMENT = "grid-anomaly-detection"


def flat_metrics(metrics: dict) -> dict:
    """{'mstl': {'recall': 0.7}} -> {'mstl_recall': 0.7}"""
    return {f"{detector}_{name}": float(value)
            for detector, values in metrics.items()
            for name, value in values.items()}


def log_run(meta: dict, model_dir: str = "models") -> str | None:
    """Log params, metrics and model files. Returns the run id or None."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("MLflow: MLFLOW_TRACKING_URI not set, skipping tracking")
        return None

    import mlflow  # imported here so tests do not need a server

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=meta["model_version"]) as run:
        mlflow.log_params({
            "k": meta["k"],
            "data_start": meta["data_start"],
            "data_end": meta["data_end"],
            "n_points": meta["n_points"],
            "n_events": meta["n_events"],
        })
        mlflow.log_metrics(flat_metrics(meta["metrics"]))
        for name in ("model.joblib", "metadata.json", "anomaly_days.csv"):
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                mlflow.log_artifact(path)
        print("MLflow: logged run", run.info.run_id)
        return run.info.run_id

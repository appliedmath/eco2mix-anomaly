"""Log one training run to MLflow (only when a tracking server is set).

Set MLFLOW_TRACKING_URI (e.g. http://mlflow:5000 inside Docker) to turn
tracking on. Without it, training works exactly as before.

Each run is tagged with the promote-or-reject decision. A promoted model
is registered as a new version of MODEL_NAME and gets the alias
"champion"; a rejected one is logged but not registered.
"""
import os

EXPERIMENT = "grid-anomaly-detection"
MODEL_NAME = "grid-anomaly-detector"
CHAMPION_ALIAS = "champion"


def flat_metrics(metrics: dict) -> dict:
    """{'mstl': {'recall': 0.7}} -> {'mstl_recall': 0.7}"""
    return {f"{detector}_{name}": float(value)
            for detector, values in metrics.items()
            for name, value in values.items()}


def run_tags(meta: dict) -> dict:
    """MLflow tags for the promote-or-reject decision."""
    promotion = meta.get("promotion", {})
    return {"decision": promotion.get("decision", "promoted"),
            "reason": promotion.get("reason", ""),
            "compared_with": str(promotion.get("champion_version", "none"))}


def champion_metrics(meta: dict) -> dict:
    """The champion's scores on the same data, as champion_<name> metrics."""
    scores = meta.get("promotion", {}).get("champion_scores") or {}
    return {f"champion_{name}": float(value) for name, value in scores.items()
            if isinstance(value, (int, float))}


def register_champion(model_dir: str) -> str:
    """Log the model as a pyfunc, register it and move the alias to it."""
    import mlflow
    from mlflow import MlflowClient

    from src.models.pyfunc_model import AnomalyModel

    info = mlflow.pyfunc.log_model(
        name="model",
        python_model=AnomalyModel(),
        artifacts={"model": os.path.join(model_dir, "model.joblib")},
        code_paths=["src"],
        pip_requirements=["pandas", "joblib"],
        registered_model_name=MODEL_NAME,
    )
    version = str(info.registered_model_version)
    MlflowClient().set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version)
    print(f"MLflow: registered {MODEL_NAME} version {version} "
          f"with alias '{CHAMPION_ALIAS}'")
    return version


def log_run(meta: dict, model_dir: str = "models",
            register: bool = False) -> str | None:
    """Log params, metrics, decision and model files. Returns the run id or None."""
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
        mlflow.log_metrics(champion_metrics(meta))
        mlflow.set_tags(run_tags(meta))
        for name in ("model.joblib", "metadata.json", "anomaly_days.csv"):
            path = os.path.join(model_dir, name)
            if os.path.exists(path):
                mlflow.log_artifact(path)
        if register:
            register_champion(model_dir)
        print("MLflow: logged run", run.info.run_id)
        return run.info.run_id

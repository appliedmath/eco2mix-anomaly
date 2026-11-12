"""The champion model comes from the MLflow Model Registry (alias @champion).

Why the registry and not the files in models/:
  - integrity: MLflow knows which model file belongs to which training run,
    nobody can quietly swap models/model.joblib for another file;
  - it works when trainer, MLflow and the jobs run on different machines
    (they only need the MLflow URL, no shared folder);
  - every version stays available: moving the alias back to an older
    version in the MLflow UI is a rollback, the next job uses it.

The files in models/ are kept as a local copy and as the fallback when no
tracking server is set (tests, local runs) or MLflow cannot be reached.
"""
import json
import os
import tempfile
from pathlib import Path

import joblib

from src.models.model import load_model, save_model
from src.models.tracking import CHAMPION_ALIAS, MODEL_NAME

CHAMPION_URI = f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}"


def load_from_registry() -> tuple[dict, dict, str]:
    """(model, meta, version) of the version that has the alias @champion."""
    import mlflow  # imported here so tests and the API do not need it
    from mlflow import MlflowClient

    version = MlflowClient().get_model_version_by_alias(MODEL_NAME,
                                                        CHAMPION_ALIAS)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(mlflow.artifacts.download_artifacts(
            artifact_uri=CHAMPION_URI, dst_path=os.path.join(tmp, "model")))
        model = joblib.load(folder / "artifacts" / "model.joblib")
        meta_path = mlflow.artifacts.download_artifacts(
            run_id=version.run_id, artifact_path="metadata.json",
            dst_path=os.path.join(tmp, "run"))
        with open(meta_path) as f:
            meta = json.load(f)
    meta["registry_version"] = str(version.version)
    return model, meta, str(version.version)


def load_champion_model(folder: str = "models"):
    """(model, meta, source): the registry first, the files as fallback.

    source is "registry v<N>" or "files". Raises FileNotFoundError when
    there is no champion anywhere (first training).
    """
    if os.environ.get("MLFLOW_TRACKING_URI"):
        try:
            model, meta, version = load_from_registry()
            return model, meta, f"registry v{version}"
        except Exception as exc:  # no server, no champion yet, ...
            print(f"MLflow registry: champion not loaded "
                  f"({type(exc).__name__}: {exc}); using the files in {folder}")
    model, meta = load_model(folder)
    return model, meta, "files"


def sync_files(model: dict, meta: dict, folder: str = "models") -> bool:
    """Write the registry champion to models/ when the copy there is another
    version (e.g. after a rollback in the MLflow UI). True if written."""
    path = os.path.join(folder, "metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            current = json.load(f)
        if (current.get("model_version") == meta.get("model_version")
                and current.get("registry_version") == meta.get("registry_version")):
            return False
    save_model(model, meta, folder)
    return True

"""The champion comes from the MLflow registry, the files are the fallback."""
import json

import joblib
import pytest

from src.models import registry
from src.models.model import save_model
from src.models.registry import load_champion_model, sync_files

MODEL = {"k": 4.0, "refs": {"mstl": {"median": 0.0, "mad": 1.0}}}


def files_champion(folder, version="files-1"):
    save_model(MODEL, {"model_version": version}, str(folder))


def test_without_tracking_server_the_files_are_used(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    files_champion(tmp_path)
    _, meta, source = load_champion_model(str(tmp_path))
    assert source == "files" and meta["model_version"] == "files-1"


def test_registry_is_used_first(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    files_champion(tmp_path)
    monkeypatch.setattr(registry, "load_from_registry", lambda: (
        MODEL, {"model_version": "reg-6", "registry_version": "6"}, "6"))
    _, meta, source = load_champion_model(str(tmp_path))
    assert source == "registry v6" and meta["model_version"] == "reg-6"


def test_registry_down_falls_back_to_files(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    files_champion(tmp_path)

    def down():
        raise ConnectionError("mlflow is down")
    monkeypatch.setattr(registry, "load_from_registry", down)
    _, meta, source = load_champion_model(str(tmp_path))
    assert source == "files" and meta["model_version"] == "files-1"


def test_no_champion_anywhere(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(FileNotFoundError):
        load_champion_model(str(tmp_path))


def test_sync_writes_only_another_version(tmp_path):
    files_champion(tmp_path, "old")
    new_meta = {"model_version": "new", "registry_version": "7"}
    assert sync_files(MODEL, new_meta, str(tmp_path)) is True
    with open(tmp_path / "metadata.json") as f:
        assert json.load(f) == new_meta
    assert sync_files(MODEL, new_meta, str(tmp_path)) is False


def test_real_registry_round_trip(tmp_path, monkeypatch):
    """Register a model in a throw-away MLflow (SQLite) and load it back."""
    mlflow = pytest.importorskip("mlflow")
    from src.models.tracking import register_champion

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    mlflow.set_tracking_uri(uri)
    model_dir = tmp_path / "candidate"
    save_model(MODEL, {"model_version": "v-test"}, str(model_dir))

    mlflow.set_experiment("test")
    with mlflow.start_run():
        mlflow.log_artifact(str(model_dir / "metadata.json"))
        version = register_champion(str(model_dir))

    model, meta, got = registry.load_from_registry()
    assert got == version
    assert model == joblib.load(model_dir / "model.joblib")
    assert meta == {"model_version": "v-test", "registry_version": version}

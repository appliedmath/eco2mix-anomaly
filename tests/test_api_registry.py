"""GET /champion: the champion in the MLflow registry vs the version in use."""
import io
import json
import urllib.error

import pytest
from fastapi.testclient import TestClient

from src.api import registry_client
from src.api.main import app, get_meta, get_registry
from src.api.registry_client import RegistryError, champion_version

META = {"model_version": "20261104-121941", "registry_version": "6"}


def fake_registry(version="6", error=None):
    def call():
        if error:
            raise error
        return {"name": "grid-anomaly-detector", "alias": "champion",
                "version": version, "run_id": "abc"}
    return lambda: call


@pytest.fixture
def client():
    app.dependency_overrides[get_meta] = lambda: META
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_champion_in_sync(client):
    app.dependency_overrides[get_registry] = fake_registry("6")
    body = client.get("/champion").json()
    assert body["registry"]["version"] == "6"
    assert body["in_use"] == {"model_version": "20261104-121941",
                              "registry_version": "6"}
    assert body["in_sync"] is True


def test_alias_moved_not_yet_used(client):
    app.dependency_overrides[get_registry] = fake_registry("5")   # rollback
    assert client.get("/champion").json()["in_sync"] is False


def test_no_champion_yet_is_404(client):
    app.dependency_overrides[get_registry] = fake_registry(
        error=RegistryError(404, "no version has the alias @champion yet"))
    assert client.get("/champion").status_code == 404


def test_mlflow_down_is_503(client):
    app.dependency_overrides[get_registry] = fake_registry(
        error=RegistryError(None, "MLflow is not reachable"))
    assert client.get("/champion").status_code == 503


def test_no_tracking_server_configured_is_503(client, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    response = client.get("/champion")
    assert response.status_code == 503
    assert "MLFLOW_TRACKING_URI" in response.json()["detail"]


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_client_reads_the_alias(monkeypatch):
    calls = []

    def urlopen(url, timeout):
        calls.append(url)
        return FakeResponse(json.dumps(
            {"model_version": {"version": "6", "run_id": "abc"}}).encode())
    monkeypatch.setattr(registry_client.urllib.request, "urlopen", urlopen)
    entry = champion_version("http://mlflow:5000/")
    assert entry["version"] == "6" and entry["run_id"] == "abc"
    assert calls[0] == ("http://mlflow:5000/api/2.0/mlflow/registered-models/"
                        "alias?name=grid-anomaly-detector&alias=champion")


def test_client_unknown_alias_is_404(monkeypatch):
    def urlopen(url, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(
            b'{"error_code": "RESOURCE_DOES_NOT_EXIST"}'))
    monkeypatch.setattr(registry_client.urllib.request, "urlopen", urlopen)
    with pytest.raises(RegistryError) as err:
        champion_version("http://mlflow:5000")
    assert err.value.status == 404


def test_client_server_down(monkeypatch):
    def urlopen(url, timeout):
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(registry_client.urllib.request, "urlopen", urlopen)
    with pytest.raises(RegistryError) as err:
        champion_version("http://mlflow:5000")
    assert err.value.status is None

"""X-API-Key on POST /train and POST /predict (real lock, fake Airflow)."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_airflow

KEY = "test-secret-key"
LOCKED = ["/train", "/predict?days=1"]


class FakeAirflow:
    def is_paused(self, dag_id):
        return False

    def trigger(self, dag_id, run_id, conf):
        return {"dag_run_id": run_id, "state": "queued"}

    def get_run(self, dag_id, run_id):
        return {"state": "success", "start_date": None, "end_date": None}

    def task_states(self, dag_id, run_id):
        return []


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    app.dependency_overrides[get_airflow] = lambda: FakeAirflow()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.mark.parametrize("path", LOCKED)
def test_missing_key_returns_401(client, path):
    assert client.post(path).status_code == 401


@pytest.mark.parametrize("path", LOCKED)
def test_wrong_key_returns_403(client, path):
    response = client.post(path, headers={"X-API-Key": "wrong"})
    assert response.status_code == 403


@pytest.mark.parametrize("path", LOCKED)
def test_right_key_returns_202(client, path):
    response = client.post(path, headers={"X-API-Key": KEY})
    assert response.status_code == 202


def test_server_without_key_refuses(client, monkeypatch):
    monkeypatch.delenv("API_KEY")
    response = client.post("/train", headers={"X-API-Key": KEY})
    assert response.status_code == 500


def test_run_status_needs_no_key(client):
    assert client.get("/runs/retrain_dag/some-run").status_code == 200

"""POST /train, POST /predict and GET /runs with a fake Airflow (no server needed)."""
import pytest
from fastapi.testclient import TestClient

from src.api.airflow_client import AirflowClient, AirflowError, AirflowUnavailable
from src.api.main import app, get_airflow, new_run_id
from src.api.security import require_api_key


class FakeAirflow:
    def __init__(self, paused=False, down=False, wrong_login=False):
        self.paused = paused
        self.down = down
        self.wrong_login = wrong_login
        self.triggered = []

    def _check(self):
        if self.down:
            raise AirflowUnavailable("connection refused")
        if self.wrong_login:
            raise AirflowError(403, "FORBIDDEN")

    def is_paused(self, dag_id):
        self._check()
        return self.paused

    def trigger(self, dag_id, run_id, conf):
        self._check()
        self.triggered.append((dag_id, run_id, conf))
        return {"dag_run_id": run_id, "state": "queued"}

    def get_run(self, dag_id, run_id):
        self._check()
        if run_id == "missing":
            raise AirflowError(404, "DAGRun not found")
        return {"state": "success", "start_date": "2026-10-30T10:28:48+00:00",
                "end_date": "2026-10-30T10:29:25+00:00"}

    def task_states(self, dag_id, run_id):
        return [{"task_id": "train", "state": "success"}]


@pytest.fixture
def make_client():
    def _make(**kwargs):
        fake = FakeAirflow(**kwargs)
        app.dependency_overrides[get_airflow] = lambda: fake
        app.dependency_overrides[require_api_key] = lambda: None
        return TestClient(app), fake
    yield _make
    app.dependency_overrides.clear()


def test_train_starts_retrain_dag(make_client):
    client, fake = make_client()
    response = client.post("/train")
    assert response.status_code == 202
    body = response.json()
    assert body["dag_id"] == "retrain_dag"
    assert body["state"] == "queued"
    assert body["run_id"].startswith("api__")
    assert body["status_url"] == f"/runs/retrain_dag/{body['run_id']}"
    assert fake.triggered == [("retrain_dag", body["run_id"], {})]


def test_predict_passes_days_to_batch_dag(make_client):
    client, fake = make_client()
    response = client.post("/predict?days=7")
    assert response.status_code == 202
    assert fake.triggered[0][0] == "batch_dag"
    assert fake.triggered[0][2] == {"days": 7, "source": "replay"}


@pytest.mark.parametrize("days", [0, 32, -1])
def test_predict_rejects_bad_days(make_client, days):
    client, fake = make_client()
    assert client.post(f"/predict?days={days}").status_code == 422
    assert fake.triggered == []


def test_paused_dag_returns_409(make_client):
    client, fake = make_client(paused=True)
    response = client.post("/train")
    assert response.status_code == 409
    assert "paused" in response.json()["detail"]
    assert fake.triggered == []


def test_airflow_down_returns_503(make_client):
    client, _ = make_client(down=True)
    assert client.post("/train").status_code == 503
    assert client.get("/runs/retrain_dag/x").status_code == 503


def test_wrong_airflow_login_is_explained(make_client):
    client, _ = make_client(wrong_login=True)
    response = client.post("/train")
    assert response.status_code == 502
    assert "AIRFLOW_ADMIN_PASSWORD" in response.json()["detail"]


def test_run_status(make_client):
    client, _ = make_client()
    response = client.get("/runs/retrain_dag/api__20261030T102848_ab12")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "success"
    assert body["tasks"] == [{"task_id": "train", "state": "success"}]


def test_unknown_run_returns_404(make_client):
    client, _ = make_client()
    assert client.get("/runs/retrain_dag/missing").status_code == 404


def test_unknown_dag_returns_422(make_client):
    client, _ = make_client()
    assert client.get("/runs/other_dag/x").status_code == 422


def test_run_ids_are_unique():
    assert len({new_run_id() for _ in range(50)}) == 50


def test_client_sends_basic_auth():
    client = AirflowClient("http://airflow:8080/", "admin", "secret")
    assert client.base_url == "http://airflow:8080"
    assert client.headers["Authorization"] == "Basic YWRtaW46c2VjcmV0"


def test_client_reports_unreachable_airflow():
    client = AirflowClient("http://127.0.0.1:9", "admin", "x", timeout=1)
    with pytest.raises(AirflowUnavailable):
        client.is_paused("batch_dag")

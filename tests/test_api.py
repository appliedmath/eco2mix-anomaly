import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app, get_detections, get_meta, get_series_data

# Two days of half-hours: 14 July (first 48, a holiday) and 15 July
INDEX = pd.date_range("2026-07-14", periods=96, freq="30min", tz="UTC")
META = {"model_version": "test-1", "data_end": "2026-07-15 23:30:00+00:00",
        "k": 4.0}


def make_data():
    data = pd.DataFrame({
        "consumption_mw": [1000.0 + i for i in range(96)],
        "forecast_j1_mw": 900.0,
        "forecast_j_mw": 950.0,
        "is_holiday": [i < 48 for i in range(96)],
        "is_future": [i == 95 for i in range(96)],
    }, index=INDEX)
    data.index.name = "timestamp_utc"
    data.loc[INDEX[95], "consumption_mw"] = float("nan")  # not published yet
    return data


def make_detections():
    z = [0.0] * 96
    z[5] = -8.2    # alert on day 1
    z[60] = 5.0    # alert on day 2
    mstl = pd.DataFrame({
        "detector": "mstl",
        "timestamp_utc": INDEX,
        "expected_mw": 1000.0,
        "z_score": z,
        "is_anomaly": [abs(v) > 4 for v in z],
    })
    gap = pd.DataFrame({
        "detector": "forecast_gap",
        "timestamp_utc": INDEX,
        "expected_mw": 900.0,
        "z_score": 0.0,
        "is_anomaly": False,
    })
    return pd.concat([mstl, gap], ignore_index=True)


@pytest.fixture
def client():
    app.dependency_overrides[get_meta] = lambda: META
    app.dependency_overrides[get_detections] = make_detections
    app.dependency_overrides[get_series_data] = make_data
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "test-1",
        "last_data": "2026-07-15 23:30:00+00:00",
    }


def test_model_info(client):
    assert client.get("/model-info").json() == META


def test_anomalies_returns_flagged_points(client):
    response = client.get("/anomalies",
                          params={"start": "2026-07-14", "end": "2026-07-15"})
    assert response.status_code == 200
    first, second = response.json()
    assert first["timestamp_utc"] == INDEX[5].isoformat()
    assert first["z_score"] == -8.2
    assert first["consumption_mw"] == 1005.0
    assert first["expected_mw"] == 1000.0
    assert first["is_holiday"] is True
    assert first["detector"] == "mstl"
    assert second["timestamp_utc"] == INDEX[60].isoformat()
    assert second["is_holiday"] is False


def test_anomalies_single_day_includes_both_ends(client):
    response = client.get("/anomalies",
                          params={"start": "2026-07-14", "end": "2026-07-14"})
    assert len(response.json()) == 1


def test_anomalies_other_detector_has_none(client):
    response = client.get("/anomalies", params={
        "start": "2026-07-14", "end": "2026-07-15",
        "detector": "forecast_gap"})
    assert response.json() == []


def test_anomalies_can_include_normal_points(client):
    response = client.get("/anomalies", params={
        "start": "2026-07-14", "end": "2026-07-14",
        "only_anomalies": "false"})
    assert len(response.json()) == 48


def test_anomalies_rejects_bad_dates(client):
    bad_text = client.get("/anomalies",
                          params={"start": "not-a-date", "end": "2026-07-14"})
    assert bad_text.status_code == 422
    backwards = client.get("/anomalies",
                           params={"start": "2026-07-15", "end": "2026-07-14"})
    assert backwards.status_code == 422


def test_series_returns_one_row_per_half_hour(client):
    response = client.get("/series",
                          params={"start": "2026-07-14", "end": "2026-07-14"})
    rows = response.json()
    assert len(rows) == 48
    assert rows[0]["consumption_mw"] == 1000.0
    assert rows[0]["forecast_j1_mw"] == 900.0
    assert rows[0]["is_holiday"] is True


def test_series_unpublished_consumption_is_null(client):
    response = client.get("/series",
                          params={"start": "2026-07-15", "end": "2026-07-15"})
    last = response.json()[-1]
    assert last["consumption_mw"] is None
    assert last["is_future"] is True

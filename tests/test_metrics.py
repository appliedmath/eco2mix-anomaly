"""GET /metrics: HTTP traffic counters and model/data/batch gauges."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from src.api.main import app
from src.api.metrics import StateCollector

META = {"model_version": "20261104-121941", "k": 4.0,
        "trained_at": "2026-11-04T12:19:41+00:00",
        "metrics": {"mstl": {"recall": 0.875, "false_alarms_per_week": 0.4},
                    "last_week": {"recall": 0.5, "false_alarms_per_week": 2.1}}}
INDEX = pd.date_range("2026-01-01", periods=48 * 10, freq="30min", tz="UTC")


def data():
    df = pd.DataFrame({"consumption_mw": 50000.0, "is_future": False}, index=INDEX)
    df.loc[INDEX[-1], "is_future"] = True        # forecast only, not measured
    return df


def detections():
    return pd.DataFrame({
        "detector": "mstl",
        "timestamp_utc": INDEX[:-1],
        "is_anomaly": [i in (5, 470, 471) for i in range(len(INDEX) - 1)],
    })


def scrape(collector) -> dict:
    """{'name{labels}': value} of one scrape."""
    registry = CollectorRegistry()
    registry.register(collector)
    values = {}
    for family in registry.collect():
        for s in family.samples:
            labels = ",".join(f'{k}="{v}"' for k, v in sorted(s.labels.items()))
            values[f"{s.name}{{{labels}}}" if labels else s.name] = s.value
    return values


def test_model_and_data_gauges(tmp_path):
    status = tmp_path / "latest.json"
    status.write_text(json.dumps({
        "batch_end": "2026-01-10T22:30:00Z", "n_rows": 48, "source": "replay",
        "quality_failed": False, "quality_problems": [],
        "drift_detected": None, "drift_share": None, "anomaly_rate": None}))
    data_file = tmp_path / "data.parquet"
    data_file.write_bytes(b"x")
    m = scrape(StateCollector(lambda: META, data, detections, lambda: str(status),
                              data_path=lambda: str(data_file)))

    assert m["data_updated_timestamp_seconds"] == data_file.stat().st_mtime
    assert m['model_info{version="20261104-121941"}'] == 1
    assert m["model_k"] == 4.0
    assert m['model_recall{detector="mstl"}'] == 0.875
    assert m['model_false_alarms_per_week{detector="last_week"}'] == 2.1
    assert m["data_last_timestamp_seconds"] == INDEX[-2].timestamp()
    assert m["detections_last_timestamp_seconds"] == INDEX[-2].timestamp()
    assert m['anomalies_last_7d{detector="mstl"}'] == 2     # half-hour 5 is older
    assert m["batch_rows"] == 48
    assert m["data_quality_failed"] == 0
    assert "drift_detected" not in m and "drift_share" not in m   # null: not exported


def test_quality_failure_and_drift_are_exported(tmp_path):
    status = tmp_path / "latest.json"
    status.write_text(json.dumps({"n_rows": 48, "quality_failed": True,
                                  "drift_detected": True, "drift_share": 0.4,
                                  "anomaly_rate": 0.02}))
    m = scrape(StateCollector(lambda: META, data, detections, lambda: str(status)))
    assert m["data_quality_failed"] == 1
    assert m["drift_detected"] == 1
    assert m["drift_share"] == 0.4
    assert m["anomaly_rate_last_batch"] == 0.02


def test_missing_files_export_nothing_instead_of_crashing(tmp_path):
    def missing():
        raise FileNotFoundError("models/metadata.json")
    m = scrape(StateCollector(missing, missing, missing,
                              lambda: str(tmp_path / "nope.json")))
    assert m == {}


@pytest.fixture
def client():
    return TestClient(app)


def test_requests_are_counted_by_route(client):
    before = client.get("/metrics").text
    client.get("/runs/batch_dag/api__1")          # 404 or 503: still counted
    client.get("/runs/batch_dag/api__2")
    text = client.get("/metrics").text
    assert text.startswith("# HELP") and text != before
    lines = [line for line in text.splitlines()
             if line.startswith("api_requests_total{")
             and 'endpoint="/runs/{dag_id}/{run_id}"' in line]
    assert lines, text                                 # one series for all run ids
    assert "api__1" not in text
    assert "api_request_duration_seconds_bucket" in text


def test_metrics_endpoint_itself_is_not_counted(client):
    client.get("/metrics")
    text = client.get("/metrics").text
    assert 'endpoint="/metrics"' not in text


def test_metrics_is_prometheus_text(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")

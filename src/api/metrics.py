"""Prometheus metrics of the API, served on GET /metrics.

Two kinds of numbers:

1. HTTP traffic (counted by a middleware in main.py, for every endpoint):
     api_requests_total{endpoint, method, status_code}            Counter
     api_request_duration_seconds{endpoint, method, status_code}  Histogram
   -> request rate, P95 latency, error rate (401/403/429/5xx) in Grafana.

2. Model and data state, read from the files at the moment Prometheus asks
   (every 15 s), so they are always current without any extra job:
     model_info{version}, model_k, model_trained_timestamp_seconds
     model_recall{detector}, model_false_alarms_per_week{detector}
     data_last_timestamp_seconds          newest measured half-hour (freshness)
     detections_last_timestamp_seconds    newest scored half-hour
     anomalies_last_7d{detector}          flagged half-hours in the last 7 days
     batch_* / data_quality_failed / drift_*   from models/drift/latest.json
                                               (written by the batch_dag tasks)
   A value that is not available yet (no file, field is null) is simply not
   exported, so Grafana shows "no data" instead of a wrong 0.

Own CollectorRegistry (as in the course): only our metrics, no default
Python/process metrics.
"""
import json
import os
from collections.abc import Callable

import pandas as pd
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily

REGISTRY = CollectorRegistry()

LABELS = ["endpoint", "method", "status_code"]
REQUESTS = Counter("api_requests_total", "HTTP requests handled by the API",
                   LABELS, registry=REGISTRY)
LATENCY = Histogram("api_request_duration_seconds",
                    "Time to answer an HTTP request", LABELS,
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
                    registry=REGISTRY)


def observe(endpoint: str, method: str, status_code: int, seconds: float) -> None:
    labels = (endpoint, method, str(status_code))
    REQUESTS.labels(*labels).inc()
    LATENCY.labels(*labels).observe(seconds)


def _seconds(ts) -> float:
    return pd.Timestamp(ts).timestamp()


def _gauge(name: str, doc: str, value, labels: dict | None = None):
    """GaugeMetricFamily with one sample, or None when value is missing."""
    if value is None:
        return None
    family = GaugeMetricFamily(name, doc, labels=list(labels or {}))
    family.add_metric(list((labels or {}).values()), float(value))
    return family


class StateCollector:
    """Reads model, data and batch state when Prometheus scrapes /metrics."""

    def __init__(self, get_meta: Callable, get_data: Callable,
                 get_detections: Callable, status_path: Callable[[], str]):
        self.get_meta = get_meta
        self.get_data = get_data
        self.get_detections = get_detections
        self.status_path = status_path

    def collect(self):
        for part in (self._model, self._data, self._batch):
            try:
                yield from (m for m in part() if m is not None)
            except (FileNotFoundError, KeyError, ValueError):
                continue   # that part is not available yet: export nothing

    def _model(self):
        meta = self.get_meta()
        info = GaugeMetricFamily("model_info", "Champion model in use (value 1)",
                                 labels=["version"])
        info.add_metric([meta["model_version"]], 1)
        yield info
        yield _gauge("model_k", "Threshold K on the robust z-score", meta.get("k"))
        if meta.get("trained_at"):
            yield _gauge("model_trained_timestamp_seconds",
                         "When the champion was trained", _seconds(meta["trained_at"]))
        metrics = meta.get("metrics", {})
        for key, doc in (("recall", "Share of reference events caught"),
                         ("false_alarms_per_week", "Alert episodes outside events per week")):
            family = GaugeMetricFamily(f"model_{key}", doc, labels=["detector"])
            for detector, scores in metrics.items():
                if scores.get(key) is not None:
                    family.add_metric([detector], float(scores[key]))
            yield family

    def _data(self):
        data = self.get_data()
        measured = data.index[~data["is_future"].astype(bool)]
        if len(measured):
            yield _gauge("data_last_timestamp_seconds",
                         "Newest measured half-hour in the processed data",
                         _seconds(measured.max()))
        detections = self.get_detections()
        if len(detections):
            last = detections["timestamp_utc"].max()
            yield _gauge("detections_last_timestamp_seconds",
                         "Newest half-hour scored by the model", _seconds(last))
            recent = detections[detections["timestamp_utc"] > last - pd.Timedelta(days=7)]
            family = GaugeMetricFamily("anomalies_last_7d",
                                       "Flagged half-hours in the last 7 scored days",
                                       labels=["detector"])
            for detector, group in recent.groupby("detector"):
                family.add_metric([detector], float(group["is_anomaly"].sum()))
            yield family

    def _batch(self):
        with open(self.status_path()) as f:
            status = json.load(f)
        flag = lambda v: None if v is None else float(bool(v))  # noqa: E731
        yield _gauge("batch_rows", "Rows in the last injected batch", status.get("n_rows"))
        if status.get("batch_end"):
            yield _gauge("batch_end_timestamp_seconds", "End of the last batch",
                         _seconds(status["batch_end"]))
        yield _gauge("data_quality_failed",
                     "1 if the last batch failed the data quality checks",
                     flag(status.get("quality_failed")))
        yield _gauge("drift_detected", "1 if drift was detected on the last batch",
                     flag(status.get("drift_detected")))
        yield _gauge("drift_share", "Share of drifted columns in the last batch",
                     status.get("drift_share"))
        yield _gauge("anomaly_rate_last_batch",
                     "Share of flagged half-hours in the last batch",
                     status.get("anomaly_rate"))


def status_path_from_env(model_dir: str) -> Callable[[], str]:
    return lambda: os.environ.get("DRIFT_STATUS_PATH",
                                  os.path.join(model_dir, "drift", "latest.json"))

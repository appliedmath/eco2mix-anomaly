"""Build the Grafana dashboards (JSON) from one small Python description.

    python monitoring/grafana/build_dashboards.py

Writes monitoring/grafana/dashboards/*.json, which Grafana loads at start-up
(provisioning). The JSON files are committed; this script only keeps them
consistent (same data source, ids, layout). Change a panel here, run the
script, commit both.
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "dashboards"
DS = {"type": "prometheus", "uid": "prometheus"}
HOUR = 3600

GREEN_RED = {"mode": "absolute", "steps": [{"color": "green", "value": None},
                                           {"color": "red", "value": 1}]}
UP_MAP = [{"type": "value", "options": {
    "0": {"text": "DOWN", "color": "red", "index": 0}}},
    {"type": "range", "options": {"from": 1, "to": 1000,
                                  "result": {"text": "UP", "color": "green", "index": 1}}}]
YES_NO = [{"type": "value", "options": {
    "0": {"text": "no", "color": "green", "index": 0},
    "1": {"text": "YES", "color": "red", "index": 1}}}]
OK_FAILED = [{"type": "value", "options": {
    "0": {"text": "OK", "color": "green", "index": 0},
    "1": {"text": "FAILED", "color": "red", "index": 1}}}]


class Layout:
    """Places panels left to right in rows of 24 units."""

    def __init__(self):
        self.x = self.y = self.row_h = 0
        self.next_id = 1

    def place(self, w, h):
        if self.x + w > 24:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        pos = {"x": self.x, "y": self.y, "w": w, "h": h}
        self.x += w
        self.row_h = max(self.row_h, h)
        return pos

    def row(self, title):
        if self.x:
            self.x, self.y, self.row_h = 0, self.y + self.row_h, 0
        panel = {"id": self.new_id(), "type": "row", "title": title, "collapsed": False,
                 "gridPos": {"x": 0, "y": self.y, "w": 24, "h": 1}, "panels": []}
        self.y += 1
        return panel

    def new_id(self):
        self.next_id += 1
        return self.next_id - 1


def target(expr, legend="", instant=False):
    return {"refId": "A", "datasource": DS, "expr": expr, "legendFormat": legend,
            "instant": instant, "range": not instant}


def panel(layout, kind, title, exprs, w=6, h=6, unit=None, description="",
          mappings=None, thresholds=None, decimals=None, options=None,
          min_=None, max_=None, text_mode=None):
    """One panel. exprs: list of (expr, legend)."""
    instant = kind in ("stat", "bargauge", "gauge")
    targets = []
    for i, (expr, legend) in enumerate(exprs):
        t = target(expr, legend, instant)
        t["refId"] = chr(ord("A") + i)
        targets.append(t)
    defaults = {}
    if unit:
        defaults["unit"] = unit
    if mappings:
        defaults["mappings"] = mappings
    if thresholds:
        defaults["thresholds"] = thresholds
    if decimals is not None:
        defaults["decimals"] = decimals
    if min_ is not None:
        defaults["min"] = min_
    if max_ is not None:
        defaults["max"] = max_
    if kind == "timeseries":
        defaults["custom"] = {"lineWidth": 2, "fillOpacity": 10, "showPoints": "never",
                              "spanNulls": True}
    opts = {}
    if kind == "stat":
        opts = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "colorMode": "background", "graphMode": "none",
                "textMode": text_mode or "value_and_name", "justifyMode": "center"}
    elif kind == "bargauge":
        opts = {"reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
                "orientation": "horizontal", "displayMode": "gradient",
                "showUnfilled": True}
    elif kind == "timeseries":
        opts = {"legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
                "tooltip": {"mode": "multi", "sort": "desc"}}
    opts.update(options or {})
    return {"id": layout.new_id(), "type": kind, "title": title, "description": description,
            "datasource": DS, "gridPos": layout.place(w, h), "targets": targets,
            "fieldConfig": {"defaults": defaults, "overrides": []}, "options": opts}


def dashboard(uid, title, description, panels, time_from="now-6h"):
    return {"uid": uid, "title": title, "description": description, "tags": ["eco2mix"],
            "schemaVersion": 39, "version": 1, "editable": False, "refresh": "30s",
            "time": {"from": time_from, "to": "now"}, "timezone": "browser",
            "panels": panels, "templating": {"list": []}, "annotations": {"list": []}}


def thresholds(*steps):
    """thresholds(("green", None), ("red", 36 * HOUR))"""
    return {"mode": "absolute",
            "steps": [{"color": c, "value": v} for c, v in steps]}


# ---------------------------------------------------------------- 1. API
def api_dashboard():
    L = Layout()
    p = [
        L.row("Traffic (RED: rate, errors, duration)"),
        panel(L, "stat", "API containers up", [("sum(up{job=\"api\"})", "api")], w=4,
              mappings=UP_MAP, thresholds=thresholds(("red", None), ("green", 1)),
              description="Number of API containers Prometheus can scrape (DNS discovery: "
                          "with --scale api=3 or 3 Kubernetes replicas this shows 3)."),
        panel(L, "stat", "Requests per second", [
            ("sum(rate(api_requests_total[5m]))", "req/s")], w=5, unit="reqps",
            decimals=2, text_mode="value"),
        panel(L, "stat", "Server errors (5xx)", [
            ("(sum(rate(api_requests_total{status_code=~\"5..\"}[5m])) or vector(0))"
             " / clamp_min(sum(rate(api_requests_total[5m])), 1e-9)", "5xx")],
            w=5, unit="percentunit", decimals=1, text_mode="value",
            thresholds=thresholds(("green", None), ("orange", 0.01), ("red", 0.05)),
            description="Share of requests answered with 5xx in the last 5 minutes."),
        panel(L, "stat", "P95 latency", [
            ("histogram_quantile(0.95, sum by (le) "
             "(rate(api_request_duration_seconds_bucket[5m])))", "p95")],
            w=5, unit="s", decimals=3, text_mode="value",
            thresholds=thresholds(("green", None), ("orange", 0.2), ("red", 0.5)),
            description="SLO: 95 % of the requests answered in under 200 ms."),
        panel(L, "stat", "Rejected by API key (1 h)", [
            ("sum(increase(api_requests_total{status_code=~\"401|403\"}[1h])) or vector(0)",
             "401/403")], w=5, decimals=0, text_mode="value",
            thresholds=thresholds(("green", None), ("orange", 10)),
            description="POST /train or /predict without (401) or with a wrong (403) key. "
                        "Requests above the NGINX rate limit (429) never reach the API, "
                        "so they are counted by NGINX, not here."),
        panel(L, "timeseries", "Requests per second by endpoint", [
            ("sum by (endpoint) (rate(api_requests_total[5m]))", "{{endpoint}}")],
            w=12, h=8, unit="reqps"),
        panel(L, "timeseries", "P95 latency by endpoint", [
            ("histogram_quantile(0.95, sum by (le, endpoint) "
             "(rate(api_request_duration_seconds_bucket[5m])))", "{{endpoint}}")],
            w=12, h=8, unit="s"),
        panel(L, "timeseries", "Responses by status code", [
            ("sum by (status_code) (rate(api_requests_total[5m]))", "{{status_code}}")],
            w=24, h=8, unit="reqps",
            description="2xx = OK, 202 = job started in Airflow, 401/403 = API key, "
                        "404 = unknown run, 409 = DAG paused, 503 = Airflow down."),
    ]
    return dashboard("eco2mix-api", "1 · API performance",
                     "Is the API up, fast and answering correctly?", p, "now-1h")


# ------------------------------------------------------- 2. Model & data
def model_dashboard():
    L = Layout()
    det = "{{detector}}"
    p = [
        L.row("Model in use (champion)"),
        panel(L, "stat", "Champion version", [("model_info", "{{version}}")], w=8,
              text_mode="name", options={"colorMode": "none"},
              description="Version of the champion model the API serves "
                          "(changes after a promoted retraining)."),
        panel(L, "stat", "Threshold K", [("model_k", "K")], w=4, decimals=1,
              text_mode="value", options={"colorMode": "none"}),
        panel(L, "stat", "Trained", [("model_trained_timestamp_seconds * 1000", "trained")],
              w=4, unit="dateTimeFromNow", text_mode="value", options={"colorMode": "none"}),
        panel(L, "bargauge", "Recall on reference events (training)", [
            ("model_recall", det)], w=4, unit="percentunit", min_=0, max_=1,
            thresholds=thresholds(("red", None), ("orange", 0.5), ("green", 0.7))),
        panel(L, "bargauge", "False alarms per week (training)", [
            ("model_false_alarms_per_week", det)], w=4, decimals=2, min_=0,
            thresholds=thresholds(("green", None), ("orange", 1), ("red", 2))),

        L.row("Data (freshness and quality)"),
        panel(L, "stat", "Data updated", [
            ("time() - data_updated_timestamp_seconds", "age")], w=6, unit="s",
            text_mode="value",
            thresholds=thresholds(("green", None), ("red", 36 * HOUR)),
            description="Time since the processed data file last changed. Red after 36 h "
                        "= the same limit as the NoNewData alert."),
        panel(L, "stat", "Newest measured half-hour", [
            ("data_last_timestamp_seconds * 1000", "newest")], w=6, unit="dateTimeAsIso",
            text_mode="value", options={"colorMode": "none"},
            description="Time of the newest real measurement (in replay mode: historical)."),
        panel(L, "stat", "Last batch: rows", [("batch_rows", "rows")], w=4, decimals=0,
              text_mode="value", options={"colorMode": "none"},
              description="0 = the batch ran but there was no new data."),
        panel(L, "stat", "Last batch: data quality", [("data_quality_failed", "quality")],
              w=4, mappings=OK_FAILED, thresholds=GREEN_RED, text_mode="value",
              description="FAILED = the batch was moved to data/rejected/ and not scored."),
        panel(L, "stat", "Last batch: end", [("batch_end_timestamp_seconds * 1000", "end")],
              w=4, unit="dateTimeAsIso", text_mode="value", options={"colorMode": "none"}),

        L.row("Drift (new data vs the same weeks of the 3 earlier years) — alert only"),
        panel(L, "stat", "Input drift", [("input_drift_detected", "inputs")], w=4,
              mappings=YES_NO, thresholds=GREEN_RED, text_mode="value",
              description="Consumption and RTE forecast compared with the seasonal "
                          "reference (Evidently). Drift = alert, never automatic retraining."),
        panel(L, "stat", "Input drift share", [("input_drift_share", "share")], w=4,
              unit="percentunit", text_mode="value",
              thresholds=thresholds(("green", None), ("red", 0.5))),
        panel(L, "stat", "Prediction drift", [("prediction_drift_detected", "z-score")],
              w=4, mappings=YES_NO, thresholds=GREEN_RED, text_mode="value",
              description="Distribution of the model's z-score vs the reference."),
        panel(L, "stat", "Prediction drift score", [
            ("prediction_drift_score", "Wasserstein")], w=4, decimals=3, text_mode="value",
            thresholds=thresholds(("green", None), ("red", 0.2)),
            description="Wasserstein distance of the z-scores; threshold 0.2 "
                        "(≈ 0.1 is sampling noise for one week)."),
        panel(L, "stat", "Drift (all columns)", [("drift_detected", "all")], w=4,
              mappings=YES_NO, thresholds=GREEN_RED, text_mode="value"),
        panel(L, "stat", "Anomaly rate, last batch", [("anomaly_rate_last_batch", "rate")],
              w=4, unit="percentunit", decimals=1, text_mode="value",
              options={"colorMode": "none"}),

        L.row("Model quality on live data (no daily labels → proxies) and business KPI"),
        panel(L, "stat", "Rolling recall (weekday holidays)", [
            ("model_recall_rolling", "recall")], w=6, unit="percentunit", text_mode="value",
            thresholds=thresholds(("red", None), ("orange", 0.5), ("green", 0.7)),
            description="Share of the weekday public holidays of the last 365 days that "
                        "the model flagged. Partly in-sample: the window still includes "
                        "days the model was trained on."),
        panel(L, "stat", "Rolling false alarms per week", [
            ("false_alarms_per_week_rolling", "per week")], w=6, decimals=2,
            text_mode="value",
            thresholds=thresholds(("green", None), ("orange", 1), ("red", 2)),
            description="Alert days of the last 28 days that are not holidays, per week "
                        "(upper bound: some can be real events)."),
        panel(L, "bargauge", "Alert days in the last 7 days (operator load)", [
            ("anomaly_days_last_7d", det)], w=6, decimals=0, min_=0, max_=7,
            thresholds=thresholds(("green", None), ("orange", 2), ("red", 4)),
            description="Business KPI: on how many of the last 7 days each detector "
                        "would have alerted the operators."),
        panel(L, "bargauge", "Flagged half-hours, last 7 days", [
            ("anomalies_last_7d", det)], w=6, decimals=0, min_=0),
    ]
    return dashboard("eco2mix-model", "2 · Model & data",
                     "Champion, data freshness and quality, drift, and model quality "
                     "on live data.", p, "now-7d")


# --------------------------------------------------------- 3. Infrastructure
def infra_dashboard():
    L = Layout()
    p = [
        L.row("Targets"),
        panel(L, "stat", "Scraped targets", [("up", "{{job}} {{instance}}")], w=24, h=4,
              mappings=UP_MAP, thresholds={"mode": "absolute", "steps": [
                  {"color": "red", "value": None}, {"color": "green", "value": 1}]},
              text_mode="value_and_name"),
        L.row("Host (on Docker Desktop: the Docker VM)"),
        panel(L, "gauge", "CPU used", [
            ("100 * (1 - avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))", "cpu")],
            w=8, unit="percent", min_=0, max_=100,
            thresholds=thresholds(("green", None), ("orange", 70), ("red", 90))),
        panel(L, "gauge", "RAM used", [
            ("100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
             "ram")], w=8, unit="percent", min_=0, max_=100,
            thresholds=thresholds(("green", None), ("orange", 80), ("red", 90))),
        panel(L, "gauge", "Disk used (/)", [
            ("100 * (1 - node_filesystem_avail_bytes{mountpoint=\"/\"} "
             "/ node_filesystem_size_bytes{mountpoint=\"/\"})", "disk")],
            w=8, unit="percent", min_=0, max_=100,
            thresholds=thresholds(("green", None), ("orange", 80), ("red", 90))),
        panel(L, "timeseries", "CPU used over time", [
            ("100 * (1 - avg(rate(node_cpu_seconds_total{mode=\"idle\"}[5m])))", "cpu %")],
            w=12, h=8, unit="percent", min_=0, max_=100,
            description="Peaks = training or batch jobs started by Airflow."),
        panel(L, "timeseries", "RAM used over time", [
            ("node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes", "used"),
            ("node_memory_MemTotal_bytes", "total")], w=12, h=8, unit="bytes"),
        panel(L, "timeseries", "Network", [
            ("sum(rate(node_network_receive_bytes_total{device!=\"lo\"}[5m]))", "in"),
            ("sum(rate(node_network_transmit_bytes_total{device!=\"lo\"}[5m]))", "out")],
            w=12, h=8, unit="Bps"),
        panel(L, "timeseries", "Load average (1 min)", [("node_load1", "load1")], w=12, h=8),
    ]
    return dashboard("eco2mix-infra", "3 · Infrastructure",
                     "CPU, RAM, disk and network (node-exporter) and which targets are up.",
                     p, "now-6h")


# ----------------------------------------------------------- stack health
def stack_dashboard():
    L = Layout()
    p = [
        panel(L, "stat", "Targets up (per job)", [("sum by (job) (up)", "{{job}}")], w=8,
              mappings=UP_MAP, thresholds={"mode": "absolute", "steps": [
                  {"color": "red", "value": None}, {"color": "green", "value": 1}]}),
        panel(L, "stat", "Champion model", [("model_info", "{{version}}")], w=8,
              text_mode="name", options={"colorMode": "none"}),
        panel(L, "stat", "Data updated (age)", [
            ("time() - data_updated_timestamp_seconds", "age")], w=8, unit="s",
            text_mode="value", thresholds=thresholds(("green", None), ("red", 36 * HOUR))),
    ]
    return dashboard("eco2mix-stack", "0 · Stack health",
                     "One-glance check: targets up, champion, data age.", p, "now-1h")


DASHBOARDS = {
    "stack_health.json": stack_dashboard,
    "api_performance.json": api_dashboard,
    "model_and_data.json": model_dashboard,
    "infrastructure.json": infra_dashboard,
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, build in DASHBOARDS.items():
        (OUT / name).write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
        print("wrote", OUT / name)


if __name__ == "__main__":
    main()

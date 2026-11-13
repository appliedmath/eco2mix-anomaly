import os
import time
import uuid
from datetime import UTC, date, datetime
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.airflow_client import (
    AirflowClient,
    AirflowError,
    AirflowUnavailable,
    client_from_env,
)
from src.api.metrics import REGISTRY, StateCollector, observe, status_path_from_env
from src.api.registry_client import RegistryError, champion_version, tracking_url
from src.api.security import require_api_key
from src.models.anomaly_days import paris_day_bounds, summarize_days
from src.models.model import load_model

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DATA_PATH = os.environ.get(
    "DATA_PATH", "data/processed/consumption_30min.parquet")
MAX_DAYS = 366

app = FastAPI(title="Grid anomaly detection")

DetectorName = Literal["mstl", "forecast_gap", "last_week"]
DagName = Literal["batch_dag", "retrain_dag"]


_cache: dict = {}


def _read_if_changed(path: str, reader):
    """Read a file again only when it changed on disk (e.g. after predict)."""
    mtime = os.path.getmtime(path)
    hit = _cache.get(path)
    if hit is None or hit[0] != mtime:
        _cache[path] = (mtime, reader(path))
    return _cache[path][1]


def get_meta() -> dict:
    return _read_if_changed(os.path.join(MODEL_DIR, "metadata.json"),
                            lambda p: load_model(MODEL_DIR)[1])


def get_detections():
    return _read_if_changed(os.path.join(MODEL_DIR, "detections.parquet"),
                            pd.read_parquet)


def get_series_data():
    return _read_if_changed(DATA_PATH,
                            lambda p: pd.read_parquet(p).sort_index())


def num(value):
    """JSON cannot hold NaN: return None instead."""
    return None if pd.isna(value) else float(value)


def day_range(start: date, end: date):
    """Both days included: [start 00:00 UTC, end+1 day 00:00 UTC)."""
    if end < start:
        raise HTTPException(status_code=422,
                            detail="end must not be before start")
    if (end - start).days > MAX_DAYS:
        raise HTTPException(status_code=422,
                            detail=f"range is limited to {MAX_DAYS} days")
    low = pd.Timestamp(start, tz="UTC")
    high = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return low, high


def newest_measured(data) -> str | None:
    measured = data.index[~data["is_future"].astype(bool)]
    return str(measured.max()) if len(measured) else None


@app.get("/health")
def health(meta=Depends(get_meta), data=Depends(get_series_data)):
    return {
        "status": "ok",
        "model_version": meta["model_version"],
        "last_data": meta["data_end"],         # end of the training data
        "newest_data": newest_measured(data),  # newest measured half-hour
    }


# ---------- Monitoring: Prometheus metrics ----------

REGISTRY.register(StateCollector(get_meta, get_series_data, get_detections,
                                 status_path_from_env(MODEL_DIR),
                                 data_path=lambda: DATA_PATH))


@app.middleware("http")
async def count_requests(request: Request, call_next):
    """Count every request and measure its duration (labels: route template,
    so /runs/x/y is one series, not one per run id)."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        if endpoint != "/metrics":
            observe(endpoint, request.method, status, time.perf_counter() - start)


@app.get("/metrics", include_in_schema=False)
def metrics():
    """For Prometheus (inside Docker). NGINX blocks it from outside."""
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.get("/model-info")
def model_info(meta=Depends(get_meta)):
    return meta


# ---------- Where the model comes from: MLflow Model Registry ----------

def get_registry():
    """Function that returns the champion entry of the registry (replaced in tests)."""
    url = tracking_url()
    if url is None:
        raise HTTPException(status_code=503,
                            detail="MLFLOW_TRACKING_URI is not set")
    return lambda: champion_version(url)


@app.get("/champion")
def champion(meta=Depends(get_meta), registry=Depends(get_registry)):
    """The champion in the MLflow Model Registry, and the version this API shows.

    The batch job loads the model from the registry (alias @champion) and
    keeps a copy in models/; in_sync tells whether that copy is the same
    version as the registry's champion.
    """
    try:
        entry = registry()
    except RegistryError as err:
        code = 404 if err.status == 404 else 503
        raise HTTPException(status_code=code, detail=err.message) from err
    in_use = meta.get("registry_version")
    return {
        "registry": entry,
        "in_use": {"model_version": meta["model_version"],
                   "registry_version": in_use},
        "in_sync": in_use == entry["version"],
    }


@app.get("/anomalies")
def anomalies(
    start: date,
    end: date,
    detector: DetectorName = "mstl",
    only_anomalies: bool = True,
    detections=Depends(get_detections),
    data=Depends(get_series_data),
):
    low, high = day_range(start, end)
    d = detections[(detections["detector"] == detector)
                   & (detections["timestamp_utc"] >= low)
                   & (detections["timestamp_utc"] < high)]
    if only_anomalies:
        d = d[d["is_anomaly"]]

    stamps = d["timestamp_utc"]
    consumption = data["consumption_mw"].reindex(stamps).values
    holiday = data["is_holiday"].reindex(stamps).fillna(False).astype(bool).values

    return [
        {
            "timestamp_utc": row.timestamp_utc.isoformat(),
            "consumption_mw": num(consumption[i]),
            "expected_mw": num(row.expected_mw),
            "z_score": num(row.z_score),
            "is_anomaly": bool(row.is_anomaly),
            "detector": detector,
            "is_holiday": bool(holiday[i]),
        }
        for i, row in enumerate(d.itertuples())
    ]


@app.get("/series")
def series(start: date, end: date, data=Depends(get_series_data)):
    low, high = day_range(start, end)
    d = data[(data.index >= low) & (data.index < high)]
    return [
        {
            "timestamp_utc": row.Index.isoformat(),
            "consumption_mw": num(row.consumption_mw),
            "forecast_j1_mw": num(row.forecast_j1_mw),
            "forecast_j_mw": num(row.forecast_j_mw),
            "is_holiday": bool(row.is_holiday),
            "is_future": bool(row.is_future),
        }
        for row in d.itertuples()
    ]


@app.get("/anomaly-day")
def anomaly_day(
    day: date,
    detector: DetectorName = "mstl",
    detections=Depends(get_detections),
    data=Depends(get_series_data),
):
    """Was there an anomaly on this calendar day (Paris time)?"""
    low, high = paris_day_bounds(day)
    d = detections[(detections["detector"] == detector)
                   & (detections["timestamp_utc"] >= low)
                   & (detections["timestamp_utc"] < high)]
    if d.empty:
        raise HTTPException(status_code=404,
                            detail=f"no data for {day.isoformat()}")

    flagged = d[d["is_anomaly"]]
    summary = summarize_days(flagged)
    stamps = flagged["timestamp_utc"]
    consumption = data["consumption_mw"].reindex(stamps).values
    holiday = data["is_holiday"].reindex(d["timestamp_utc"]).fillna(False)

    result = {
        "day": day.isoformat(),
        "detector": detector,
        "was_anomaly": not flagged.empty,
        "is_holiday": bool(holiday.any()),
        "n_points": 0,
        "n_episodes": 0,
        "max_abs_z": None,
        "peak_utc": None,
        "points": [
            {
                "timestamp_utc": row.timestamp_utc.isoformat(),
                "consumption_mw": num(consumption[i]),
                "expected_mw": num(row.expected_mw),
                "z_score": num(row.z_score),
            }
            for i, row in enumerate(flagged.itertuples())
        ],
    }
    if not summary.empty:
        top = summary.iloc[0]
        result.update(n_points=int(top["n_points"]),
                      n_episodes=int(top["n_episodes"]),
                      max_abs_z=float(top["max_abs_z"]),
                      peak_utc=top["peak_utc"])
    return result


# ---------- Jobs through Airflow (Phase 3) ----------
# The API never trains or predicts itself: it asks Airflow to start a DAG run.
# The workflow (steps, retries, one-job-at-a-time pool) lives only in Airflow.

def get_airflow() -> AirflowClient:
    return client_from_env()


def new_run_id() -> str:
    """Readable, unique run id, so API-started runs are easy to spot in Airflow.

    8 random hex characters (4 billion values): even many requests in the
    same second practically never get the same id."""
    return f"api__{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"


def airflow_problem(err: AirflowError) -> str:
    if err.status in (401, 403):
        return ("Airflow refused the API's login: check AIRFLOW_ADMIN_USER and "
                "AIRFLOW_ADMIN_PASSWORD in .env")
    return err.message


def start_dag(airflow: AirflowClient, dag_id: str, conf: dict) -> dict:
    try:
        if airflow.is_paused(dag_id):
            raise HTTPException(
                status_code=409,
                detail=f"{dag_id} is paused in Airflow: switch it on first")
        run = airflow.trigger(dag_id, new_run_id(), conf)
    except AirflowUnavailable as err:
        raise HTTPException(status_code=503,
                            detail=f"Airflow is not reachable: {err}") from err
    except AirflowError as err:
        raise HTTPException(status_code=502, detail=airflow_problem(err)) from err
    return {
        "dag_id": dag_id,
        "run_id": run["dag_run_id"],
        "state": run["state"],
        "status_url": f"/runs/{dag_id}/{run['dag_run_id']}",
    }


@app.post(
    "/train", status_code=202, dependencies=[Depends(require_api_key)]
)
def train(airflow: AirflowClient = Depends(get_airflow)):
    """Retrain now (retrain_dag): promote-or-reject decides if the new model is used."""
    return start_dag(airflow, "retrain_dag", {})


@app.post(
    "/predict", status_code=202, dependencies=[Depends(require_api_key)]
)
def predict(days: int = Query(1, ge=1, le=31),
            airflow: AirflowClient = Depends(get_airflow)):
    """Inject the next `days` of data and score them (batch_dag)."""
    return start_dag(airflow, "batch_dag", {"days": days, "source": "replay"})


@app.get("/runs/{dag_id}/{run_id}")
def run_status(dag_id: DagName, run_id: str,
               airflow: AirflowClient = Depends(get_airflow)):
    """State of one run and of its tasks (queued, running, success, failed)."""
    try:
        run = airflow.get_run(dag_id, run_id)
        tasks = airflow.task_states(dag_id, run_id)
    except AirflowUnavailable as err:
        raise HTTPException(status_code=503,
                            detail=f"Airflow is not reachable: {err}") from err
    except AirflowError as err:
        code = 404 if err.status == 404 else 502
        raise HTTPException(status_code=code, detail=airflow_problem(err)) from err
    return {
        "dag_id": dag_id,
        "run_id": run_id,
        "state": run["state"],
        "start_date": run.get("start_date"),
        "end_date": run.get("end_date"),
        "tasks": tasks,
    }

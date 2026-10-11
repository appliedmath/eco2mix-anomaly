import os
from datetime import date
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from src.models.model import load_model

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
DATA_PATH = os.environ.get(
    "DATA_PATH", "data/processed/consumption_30min.parquet")
MAX_DAYS = 366

app = FastAPI(title="Grid anomaly detection")

DetectorName = Literal["mstl", "forecast_gap", "last_week"]


@lru_cache(maxsize=1)
def get_meta() -> dict:
    _, meta = load_model(MODEL_DIR)
    return meta


@lru_cache(maxsize=1)
def get_detections():
    return pd.read_parquet(os.path.join(MODEL_DIR, "detections.parquet"))


@lru_cache(maxsize=1)
def get_series_data():
    return pd.read_parquet(DATA_PATH).sort_index()


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


@app.get("/health")
def health(meta=Depends(get_meta)):
    return {
        "status": "ok",
        "model_version": meta["model_version"],
        "last_data": meta["data_end"],
    }


@app.get("/model-info")
def model_info(meta=Depends(get_meta)):
    return meta


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

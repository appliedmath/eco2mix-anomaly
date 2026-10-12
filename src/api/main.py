import os
from datetime import date
from functools import lru_cache
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from src.models.anomaly_days import paris_day_bounds, summarize_days
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

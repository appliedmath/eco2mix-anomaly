"""Batch prediction: score only the new half-hours with the saved model.

The saved model is what training learned on the history: for each
detector the median and MAD of its normal error, plus K. Nothing is
re-learned here. New results are appended to models/detections.parquet
and models/anomaly_days.csv is rebuilt, so the API shows them at once.

    python -m src.models.predict
"""
import os

import pandas as pd

from src.models.anomaly_days import ANOMALY_DAYS, summarize_days
from src.models.detect import WEEK, mstl_residual
from src.models.events import build_reference_events
from src.models.model import load_model
from src.models.train import DATA, DETECTIONS, make_detection

MODEL_DIR = "models"
CONTEXT_WEEKS = 8   # history MSTL sees before the new points


def expected_values(df: pd.DataFrame, new_index: pd.DatetimeIndex) -> dict:
    """What each detector expects at the new half-hours."""
    y = df.loc[~df["is_future"], "consumption_mw"]
    start = new_index.min() - pd.Timedelta(weeks=CONTEXT_WEEKS)
    window = y[y.index >= start]
    mstl_expected = window - mstl_residual(window)
    return {
        "mstl": mstl_expected.reindex(new_index),
        "forecast_gap": df["forecast_j1_mw"].reindex(new_index),
        "last_week": y.shift(WEEK).reindex(new_index),
    }


def score_new(df: pd.DataFrame, detections: pd.DataFrame,
              model: dict) -> pd.DataFrame:
    """Rows for the measured half-hours after the last scored one."""
    last_scored = detections["timestamp_utc"].max()
    y = df.loc[~df["is_future"], "consumption_mw"]
    new_index = y.index[y.index > last_scored]
    if new_index.empty:
        return detections.iloc[:0]

    parts = {}
    for name, expected in expected_values(df, new_index).items():
        parts[name] = make_detection(y.reindex(new_index), expected,
                                     model["refs"][name], model["k"])
    return pd.concat(parts, names=["detector", "timestamp_utc"]).reset_index()


def main():
    model, _ = load_model(MODEL_DIR)
    df = pd.read_parquet(DATA).sort_index()
    detections = pd.read_parquet(DETECTIONS)

    new_rows = score_new(df, detections, model)
    if new_rows.empty:
        print("Nothing new to score.")
        return

    table = pd.concat([detections, new_rows], ignore_index=True)
    table.to_parquet(DETECTIONS, index=False)
    days = summarize_days(table, build_reference_events(df))
    days.to_csv(ANOMALY_DAYS, index=False)

    n_points = new_rows["timestamp_utc"].nunique()
    flagged = new_rows[new_rows["is_anomaly"]]
    print(f"Scored {n_points} new half-hours: "
          f"{new_rows['timestamp_utc'].min()} -> "
          f"{new_rows['timestamp_utc'].max()}")
    for name, group in flagged.groupby("detector"):
        print(f"  {name}: {len(group)} anomalous half-hours")
    print("Updated", DETECTIONS, "and", ANOMALY_DAYS)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    main()

"""Seasonal reference, input/prediction drift and rolling model drift (no internet)."""
import numpy as np
import pandas as pd
import pytest

from src.data.drift import drift_frames, rolling_model_drift, run_drift


def make_data(shift=0.0, z_shift=0.0, seed=0):
    idx = pd.date_range("2023-01-01", "2026-01-31 23:30", freq="30min", tz="UTC")
    rng = np.random.default_rng(seed)
    winter = np.where(idx.month.isin([12, 1, 2]), 15000.0, 0.0)
    cons = 50000 + winter + rng.normal(0, 2000, len(idx))
    processed = pd.DataFrame(
        {"consumption_mw": cons, "forecast_j_mw": cons + rng.normal(0, 500, len(idx)),
         "is_future": False},
        index=idx,
    )
    detections = pd.DataFrame({
        "detector": "mstl", "timestamp_utc": idx,
        "z_score": rng.normal(0, 1, len(idx)), "is_anomaly": False,
    })
    late = processed.index >= pd.Timestamp("2026-01-19", tz="UTC")
    processed.loc[late, ["consumption_mw", "forecast_j_mw"]] += shift
    detections.loc[late, "z_score"] += z_shift
    return processed, detections


def test_window_is_at_least_7_days_and_reference_is_seasonal():
    processed, detections = make_data()
    # a one-day batch still gives a 7-day window
    current, reference = drift_frames(processed, detections, "2026-01-25", "2026-01-25 23:30")
    assert len(current) == 7 * 48
    assert reference.index.max() < current.index.min()
    assert set(reference.index.year) == {2023, 2024, 2025}
    assert set(reference.index.month) <= {1, 2}


def test_input_and_prediction_drift_are_separate():
    pytest.importorskip("evidently")
    cases = [  # (consumption shift, z-score shift) -> (input drift, prediction drift)
        (8000.0, 0.0, True, False),
        (0.0, 1.5, False, True),
        (0.0, 0.0, False, False),
    ]
    for shift, z_shift, input_drift, prediction_drift in cases:
        processed, detections = make_data(shift=shift, z_shift=z_shift)
        current, reference = drift_frames(processed, detections, "2026-01-19", "2026-01-25 23:30")
        result = run_drift(current, reference)
        assert result["input_drift_detected"] is input_drift
        assert result["prediction_drift_detected"] is prediction_drift
        assert result["anomaly_rate"] == 0.0


def test_rolling_model_drift():
    idx = pd.date_range("2026-01-01", "2026-07-31 22:30", freq="30min", tz="UTC")
    holidays = {pd.Timestamp(d).date() for d in
                ["2026-05-01", "2026-05-08", "2026-06-06", "2026-07-14"]}  # 06-06 is a Saturday
    paris_days = pd.DatetimeIndex(idx).tz_convert("Europe/Paris").date
    processed = pd.DataFrame({"is_holiday": [d in holidays for d in paris_days]}, index=idx)
    anomaly_days = pd.DataFrame({
        "detector": ["mstl", "mstl", "mstl", "last_week"],
        "day": ["2026-05-01", "2026-07-14", "2026-07-20", "2026-05-08"],
    })
    result = rolling_model_drift(processed, anomaly_days, "2026-07-31 22:30")
    # weekday holidays 05-01, 05-08, 07-14; mstl flagged 05-01 and 07-14
    assert result["weekday_holidays_in_window"] == 3
    assert result["model_recall_rolling"] == 0.667
    # last 28 days: 07-14 is a holiday, only 07-20 counts -> 1 day in 4 weeks
    assert result["false_alarms_per_week_rolling"] == 0.25

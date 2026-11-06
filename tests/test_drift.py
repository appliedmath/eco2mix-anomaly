"""Seasonal reference and Evidently drift (no Airflow, no internet)."""
import numpy as np
import pandas as pd
import pytest

from src.data.drift import drift_frames, run_drift


def make_data(start="2023-01-01", end="2026-01-31 23:30", shift=0.0, seed=0):
    idx = pd.date_range(start, end, freq="30min", tz="UTC")
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
    late = processed.index >= pd.Timestamp("2026-01-20", tz="UTC")
    processed.loc[late, ["consumption_mw", "forecast_j_mw"]] += shift
    detections.loc[late, "z_score"] += shift / 2000
    return processed, detections


def test_reference_is_the_same_weeks_of_earlier_years():
    processed, detections = make_data()
    current, reference = drift_frames(processed, detections, "2026-01-20", "2026-01-25 23:30")
    assert len(current) == 6 * 48
    assert reference.index.max() < pd.Timestamp("2026-01-20", tz="UTC")
    assert set(reference.index.year) == {2023, 2024, 2025}
    assert set(reference.index.month) <= {1, 2}


def test_shift_is_detected_and_no_shift_is_not():
    pytest.importorskip("evidently")
    for shift, expected in [(8000.0, True), (0.0, False)]:
        processed, detections = make_data(shift=shift)
        current, reference = drift_frames(processed, detections, "2026-01-20", "2026-01-25 23:30")
        result = run_drift(current, reference)
        assert result["drift_detected"] is expected
        assert 0.0 <= result["drift_share"] <= 1.0
        assert result["anomaly_rate"] == 0.0

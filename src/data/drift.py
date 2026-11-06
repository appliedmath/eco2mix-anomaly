"""Drift of a new batch against a seasonal reference (same weeks of earlier years).

Input drift: consumption and the RTE forecast. Prediction drift: the z-score of
the main detector. A winter batch is compared with earlier winters, otherwise
every season change would look like drift.
"""
from pathlib import Path

import pandas as pd

INPUT_COLUMNS = ["consumption_mw", "forecast_j_mw"]
PREDICTION_COLUMNS = ["z_score"]
DETECTOR = "mstl"
REPORT_DIR = Path("models/drift/reports")


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def join_detections(processed: pd.DataFrame, detections: pd.DataFrame,
                    detector: str = DETECTOR) -> pd.DataFrame:
    """Measured half-hours with the inputs and the detector output side by side."""
    det = detections[detections["detector"] == detector].set_index("timestamp_utc")
    data = processed.loc[~processed["is_future"], INPUT_COLUMNS]
    return data.join(det[["z_score", "is_anomaly"]], how="inner")


def drift_frames(processed, detections, start, end, years: int = 3, margin_days: int = 14):
    """Current batch [start, end] and its seasonal reference (same weeks, earlier years)."""
    start, end = _utc(start), _utc(end)
    data = join_detections(processed, detections)
    current = data[(data.index >= start) & (data.index <= end)]
    margin = pd.Timedelta(days=margin_days)
    parts = []
    for k in range(1, years + 1):
        low = start - pd.DateOffset(years=k) - margin
        high = end - pd.DateOffset(years=k) + margin
        parts.append(data[(data.index >= low) & (data.index <= high)])
    reference = pd.concat(parts)
    reference = reference[reference.index < start]  # never compare the batch with itself
    return current, reference


def run_drift(current: pd.DataFrame, reference: pd.DataFrame, report_path=None) -> dict:
    """Evidently data drift on inputs + z-score; drift = at least half of the columns drifted."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    columns = INPUT_COLUMNS + PREDICTION_COLUMNS
    snapshot = Report([DataDriftPreset()]).run(
        current_data=current[columns].reset_index(drop=True),  # Evidently 0.7.2 + pandas 3:
        reference_data=reference[columns].reset_index(drop=True),  # no datetime index
    )
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_path))
    share = next(
        m["value"]["share"] for m in snapshot.dict()["metrics"]
        if m["metric_id"].startswith("DriftedColumnsCount")
    )
    return {
        "drift_share": round(float(share), 3),
        "drift_detected": bool(share >= 0.5),
        "anomaly_rate": round(float(current["is_anomaly"].mean()), 4),
    }

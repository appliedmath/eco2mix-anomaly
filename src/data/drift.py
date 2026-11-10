"""Drift of the new data against a seasonal reference (same weeks of earlier years).

Input drift: consumption and the RTE forecast. Prediction drift: the z-score of
the main detector. The current window is the last 7 days (or the whole batch if
longer), so a full week is compared with full weeks and a normal Sunday does not
look like drift. Model drift on live data: recall on weekday public holidays and
non-holiday anomaly days per week.
"""
import datetime as dt
import re
from pathlib import Path

import pandas as pd

INPUT_COLUMNS = ["consumption_mw", "forecast_j_mw"]
PREDICTION_COLUMNS = ["z_score"]
DETECTOR = "mstl"
REPORT_DIR = Path("models/drift/reports")
# Wasserstein distance of the z-scores: a week of unchanged z-scores already
# reaches ~0.1 by sampling noise (they are autocorrelated), so 0.2, not 0.1
PREDICTION_DRIFT_THRESHOLD = 0.2
PARIS = "Europe/Paris"


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def join_detections(processed: pd.DataFrame, detections: pd.DataFrame,
                    detector: str = DETECTOR) -> pd.DataFrame:
    """Measured half-hours with the inputs and the detector output side by side."""
    det = detections[detections["detector"] == detector].set_index("timestamp_utc")
    data = processed.loc[~processed["is_future"], INPUT_COLUMNS]
    return data.join(det[["z_score", "is_anomaly"]], how="inner")


def drift_frames(processed, detections, start, end, years: int = 3,
                 margin_days: int = 14, window_days: int = 7):
    """Current window and its seasonal reference (same weeks, earlier years).

    The current window is [start, end], but at least the last `window_days` days.
    """
    start, end = _utc(start), _utc(end)
    start = min(start, end - pd.Timedelta(days=window_days) + pd.Timedelta(minutes=30))
    data = join_detections(processed, detections)
    current = data[(data.index >= start) & (data.index <= end)]
    margin = pd.Timedelta(days=margin_days)
    parts = []
    for k in range(1, years + 1):
        low = start - pd.DateOffset(years=k) - margin
        high = end - pd.DateOffset(years=k) + margin
        parts.append(data[(data.index >= low) & (data.index <= high)])
    reference = pd.concat(parts)
    reference = reference[reference.index < start]  # never compare the window with itself
    return current, reference


def _evidently_drift(current, reference, columns):
    """Evidently DataDriftPreset: (snapshot, share of drifted columns, score per column)."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    # the z-score uses our threshold (also in the HTML report), the inputs Evidently's default
    thresholds = {c: PREDICTION_DRIFT_THRESHOLD for c in columns if c in PREDICTION_COLUMNS}
    snapshot = Report([DataDriftPreset(per_column_threshold=thresholds or None)]).run(
        current_data=current[columns].reset_index(drop=True),  # Evidently 0.7.2 + pandas 3:
        reference_data=reference[columns].reset_index(drop=True),  # no datetime index
    )
    share, scores = 0.0, {}
    for metric in snapshot.dict()["metrics"]:
        metric_id = metric["metric_id"]
        if metric_id.startswith("DriftedColumnsCount"):
            share = float(metric["value"]["share"])
        elif metric_id.startswith("ValueDrift"):
            column = re.search(r"column=([^,)]+)", metric_id)
            if column:
                scores[column.group(1)] = float(metric["value"])
    return snapshot, share, scores


def run_drift(current: pd.DataFrame, reference: pd.DataFrame, report_path=None) -> dict:
    """Drift on all columns (report), on the inputs and on the z-score separately.

    drift = at least half of the columns drifted (Evidently default).
    """
    snapshot, share, _ = _evidently_drift(current, reference, INPUT_COLUMNS + PREDICTION_COLUMNS)
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_path))
    _, input_share, _ = _evidently_drift(current, reference, INPUT_COLUMNS)
    _, _, prediction_scores = _evidently_drift(current, reference, PREDICTION_COLUMNS)
    prediction_score = prediction_scores.get("z_score", 0.0)
    return {
        "drift_share": round(share, 3),
        "drift_detected": bool(share >= 0.5),
        "input_drift_share": round(input_share, 3),
        "input_drift_detected": bool(input_share >= 0.5),
        "prediction_drift_detected": bool(prediction_score >= PREDICTION_DRIFT_THRESHOLD),
        "prediction_drift_score": round(prediction_score, 4),
        "anomaly_rate": round(float(current["is_anomaly"].mean()), 4),
    }


def holiday_days(processed, start, end, weekdays_only: bool = False) -> set:
    """Paris dates of the public holidays in (start, end]."""
    window = processed[(processed.index > start) & (processed.index <= end)]
    days = pd.DatetimeIndex(window.index[window["is_holiday"]]).tz_convert(PARIS)
    return {d.date() for d in days if not weekdays_only or d.weekday() < 5}


def rolling_model_drift(processed, anomaly_days, end, recall_days: int = 365,
                        alarm_days: int = 28, detector: str = DETECTOR) -> dict:
    """Model drift on live data, without labels.

    model_recall_rolling: share of the weekday public holidays of the last
    `recall_days` that the detector flagged (a holiday on a weekend does not
    change consumption). false_alarms_per_week_rolling: flagged days of the
    last `alarm_days` that are not holidays, per week (an upper bound: some
    can be real events).
    """
    end = _utc(end)
    flagged = {
        pd.Timestamp(day).date()
        for day in anomaly_days.loc[anomaly_days["detector"] == detector, "day"]
    }
    holidays = holiday_days(processed, end - pd.Timedelta(days=recall_days), end, weekdays_only=True)
    recall = round(len(holidays & flagged) / len(holidays), 3) if holidays else None

    last_day = end.tz_convert(PARIS).date()
    first_day = last_day - dt.timedelta(days=alarm_days - 1)
    any_holiday = holiday_days(processed, end - pd.Timedelta(days=alarm_days), end)
    alarms = {d for d in flagged if first_day <= d <= last_day and d not in any_holiday}
    return {
        "model_recall_rolling": recall,
        "weekday_holidays_in_window": len(holidays),
        "false_alarms_per_week_rolling": round(len(alarms) / (alarm_days / 7), 2),
    }

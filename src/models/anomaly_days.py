"""Collect anomalies per calendar day (Paris time).

One row per (detector, day) that has at least one flagged half-hour.
train.py writes this table to models/anomaly_days.csv, and the API uses
the same function to answer "was there an anomaly on this day?".
"""
import pandas as pd

PARIS = "Europe/Paris"
ANOMALY_DAYS = "models/anomaly_days.csv"
COLUMNS = ["detector", "day", "n_points", "n_episodes", "max_abs_z",
           "peak_utc", "first_utc", "last_utc", "event"]


def paris_day_bounds(day) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of one Paris calendar day, in UTC (23, 24 or 25 h)."""
    start = pd.Timestamp(day, tz=PARIS)
    end = start + pd.DateOffset(days=1)  # calendar day, not 24 h
    return start.tz_convert("UTC"), end.tz_convert("UTC")


def _episodes(stamps: pd.Series) -> int:
    """Consecutive flagged half-hours count as one episode."""
    gaps = stamps.sort_values().diff() != pd.Timedelta(minutes=30)
    return int(gaps.sum())


def _event_label(first, last, events: pd.DataFrame | None) -> str | None:
    """Labels of the reference events that overlap [first, last]."""
    if events is None or events.empty:
        return None
    hit = events[(events["start_utc"] <= last) & (events["end_utc"] >= first)]
    return "; ".join(hit["label"]) if len(hit) else None


def summarize_days(detections: pd.DataFrame,
                   events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn the long detections table into one row per anomalous day.

    detections: columns detector, timestamp_utc, z_score, is_anomaly
    events:     optional reference events (start_utc, end_utc, label)
    """
    flagged = detections[detections["is_anomaly"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=COLUMNS)

    flagged["day"] = flagged["timestamp_utc"].dt.tz_convert(PARIS).dt.date
    flagged["abs_z"] = flagged["z_score"].abs()

    rows = []
    for (detector, day), g in flagged.groupby(["detector", "day"], sort=True):
        peak = g.loc[g["abs_z"].idxmax()]
        first, last = g["timestamp_utc"].min(), g["timestamp_utc"].max()
        rows.append({
            "detector": detector,
            "day": day.isoformat(),
            "n_points": len(g),
            "n_episodes": _episodes(g["timestamp_utc"]),
            "max_abs_z": round(float(peak["abs_z"]), 2),
            "peak_utc": peak["timestamp_utc"].isoformat(),
            "first_utc": first.isoformat(),
            "last_utc": last.isoformat(),
            "event": _event_label(first, last, events),
        })
    return pd.DataFrame(rows, columns=COLUMNS)

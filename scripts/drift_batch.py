"""Drift of the new data against the same weeks of earlier years.

    python -m scripts.drift_batch

Reads models/batch/last_batch.json, writes an HTML report to models/drift/reports/
and the drift and model-drift fields of models/drift/latest.json. Drift only
raises an alert (Grafana): the task never fails because of drift.
"""
import json
from pathlib import Path

import pandas as pd

from src.data.batches import BATCH_INFO_PATH, PROCESSED_PATH
from src.data.drift import REPORT_DIR, drift_frames, rolling_model_drift, run_drift

STATUS_PATH = Path("models/drift/latest.json")
DETECTIONS_PATH = Path("models/detections.parquet")
ANOMALY_DAYS_PATH = Path("models/anomaly_days.csv")


def main() -> None:
    info = json.loads(BATCH_INFO_PATH.read_text())
    if info["rows"] == 0:
        print("Nothing new was injected: no drift check.")
        return

    processed = pd.read_parquet(PROCESSED_PATH)
    detections = pd.read_parquet(DETECTIONS_PATH)
    current, reference = drift_frames(processed, detections, info["start"], info["end"])
    if current.empty or reference.empty:
        print(f"Not enough data for a drift check: {len(current)} current, "
              f"{len(reference)} reference rows.")
        return

    report = REPORT_DIR / f"drift_{pd.Timestamp(info['end']).date()}.html"
    result = run_drift(current, reference, report)
    result.update(rolling_model_drift(processed, pd.read_csv(ANOMALY_DAYS_PATH), info["end"]))

    status = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
    status.update(result, report=str(report), reference_rows=len(reference),
                  window_start=current.index.min().isoformat())
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"{len(current)} current vs {len(reference)} reference rows; report: {report}")


if __name__ == "__main__":
    main()

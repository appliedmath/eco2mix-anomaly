"""Drift of the last injected batch against the same weeks of earlier years.

    python -m scripts.drift_batch

Reads models/batch/last_batch.json, writes an HTML report to models/drift/reports/
and fills the drift fields of models/drift/latest.json. Drift only raises an alert
(Grafana): the task never fails because of drift.
"""
import json
from pathlib import Path

import pandas as pd

from src.data.batches import BATCH_INFO_PATH, PROCESSED_PATH
from src.data.drift import REPORT_DIR, drift_frames, run_drift

STATUS_PATH = Path("models/drift/latest.json")
DETECTIONS_PATH = Path("models/detections.parquet")


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

    status = json.loads(STATUS_PATH.read_text()) if STATUS_PATH.exists() else {}
    status.update(result, report=str(report), reference_rows=len(reference))
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    print(f"Drift share {result['drift_share']}, drift detected: {result['drift_detected']}, "
          f"anomaly rate {result['anomaly_rate']} ({len(current)} vs {len(reference)} rows)")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()

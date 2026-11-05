"""Check the data quality of the last injected batch; fail if there is a problem.

    python -m scripts.validate_batch

Reads models/batch/last_batch.json (written by inject_batch) and writes the
result to models/drift/latest.json (read by the API for Grafana).
A failed check exits with code 1, so Airflow does not run predict.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from src.data.batches import BATCH_INFO_PATH, PROCESSED_PATH, check_batch

STATUS_PATH = Path("models/drift/latest.json")


def main() -> int:
    info = json.loads(BATCH_INFO_PATH.read_text())
    if info["rows"] == 0:
        print("Nothing new was injected: nothing to check.")
        return 0

    processed = pd.read_parquet(PROCESSED_PATH)
    problems = check_batch(processed, info["start"], info["end"])

    status = {
        "batch_end": info["end"],
        "n_rows": info["rows"],
        "source": info["source"],
        "quality_failed": bool(problems),
        "quality_problems": problems,
        "drift_detected": None,
        "drift_share": None,
        "anomaly_rate": None,
        "report": None,
    }
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, indent=2))

    if problems:
        for p in problems:
            print("Data quality problem:", p)
        return 1
    print(f"Data quality OK: {info['rows']} rows, {info['start']} -> {info['end']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare detectors and thresholds (k); log every combination to MLflow.

Run:  python -m src.models.compare
Set MLFLOW_TRACKING_URI to log the runs (experiment: detector-comparison).
Without it, the table is only printed.
"""
import os

import pandas as pd

from src.models.detect import WEEK, detect_mstl
from src.models.events import build_reference_events
from src.models.model import fit_reference
from src.models.train import DATA, evaluate, inside_events, make_detection

EXPERIMENT = "detector-comparison"
KS = (2.0, 3.0, 4.0, 5.0, 6.0)


def log_to_mlflow(table: pd.DataFrame, n_points: int, n_events: int) -> None:
    """One MLflow run per (detector, k)."""
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        print("MLflow: MLFLOW_TRACKING_URI not set, skipping tracking")
        return

    import mlflow  # imported here so the script also runs without a server

    mlflow.set_experiment(EXPERIMENT)
    for row in table.to_dict("records"):
        with mlflow.start_run(run_name=f"{row['detector']}_k{row['k']:g}"):
            mlflow.log_params({
                "detector": row["detector"],
                "k": row["k"],
                "n_points": n_points,
                "n_events": n_events,
            })
            mlflow.log_metrics({
                key: float(value) for key, value in row.items()
                if key not in ("detector", "k")
            })
    print("MLflow: logged", len(table), "runs")


def main():
    df = pd.read_parquet(DATA).sort_index()
    y = df.loc[~df["is_future"], "consumption_mw"]

    events = build_reference_events(df)
    in_event = inside_events(y.index, events)
    weeks = (y.index.max() - y.index.min()).days / 7

    expected = {
        "mstl": detect_mstl(y)["expected_mw"],
        "forecast_gap": df.loc[y.index, "forecast_j1_mw"],
        "last_week": y.shift(WEEK),
    }

    rows = []
    for name, values in expected.items():
        ref = fit_reference(y - values)
        for k in KS:
            flags = make_detection(y, values, ref, k)["is_anomaly"]
            rows.append({"detector": name, "k": k,
                         **evaluate(flags, events, in_event, weeks)})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    log_to_mlflow(table, len(y), len(events))


if __name__ == "__main__":
    main()

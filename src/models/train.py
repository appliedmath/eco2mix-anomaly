from datetime import datetime, timezone

import mlflow
import pandas as pd

from src.models.detect import DAY, WEEK, detect_mstl
from src.models.events import build_reference_events, events_caught
from src.models.model import fit_reference, save_model

K = 4.0
DATA = "data/processed/consumption_30min.parquet"
DETECTIONS = "models/detections.parquet"
MLFLOW_URI = "http://127.0.0.1:5001"
EXPERIMENT = "eco2mix-anomaly"


def episodes(flags: pd.Series) -> int:
    """Count alerts as episodes: consecutive flagged points count once."""
    return int((flags & ~flags.shift(1, fill_value=False)).sum())


def inside_events(index, events) -> pd.Series:
    """True for every point that lies inside a reference event."""
    mask = pd.Series(False, index=index)
    for start, end in zip(events["start_utc"], events["end_utc"]):
        mask.loc[start:end] = True
    return mask


def make_detection(actual, expected, ref: dict, k: float) -> pd.DataFrame:
    """Expected value, robust z-score and alert flag for every point."""
    z = (actual - expected - ref["median"]) / (1.4826 * ref["mad"])
    return pd.DataFrame({
        "expected_mw": expected,
        "z_score": z,
        "is_anomaly": z.abs() > k,
    })


def evaluate(flags, events, in_event, weeks) -> dict:
    caught = events_caught(events, flags)
    out = {
        "events_caught": int(caught.sum()),
        "recall": round(float(caught.mean()), 3),
        "false_alarms_per_week": round(episodes(flags & ~in_event) / weeks, 2),
        "alert_episodes": episodes(flags),
    }
    for kind in events["kind"].unique():
        selected = (events["kind"] == kind).values
        out[f"recall_{kind}"] = round(float(caught[selected].mean()), 3)
    return out


def main():
    df = pd.read_parquet(DATA).sort_index()
    y = df.loc[~df["is_future"], "consumption_mw"]

    events = build_reference_events(df)
    in_event = inside_events(y.index, events)
    weeks = (y.index.max() - y.index.min()).days / 7

    # What each detector expects at every half-hour
    expected = {
        # A: MSTL removes trend + daily + weekly cycles
        "mstl": detect_mstl(y, k=K)["expected_mw"],
        # B: RTE day-before forecast
        "forecast_gap": df.loc[y.index, "forecast_j1_mw"],
        # C: same half-hour one week earlier
        "last_week": y.shift(WEEK),
    }

    refs, detections = {}, {}
    for name, values in expected.items():
        refs[name] = fit_reference(y - values)
        detections[name] = make_detection(y, values, refs[name], K)

    flags = {name: d["is_anomaly"] for name, d in detections.items()}
    metrics = {name: evaluate(f, events, in_event, weeks)
               for name, f in flags.items()}

    trained_at = datetime.now(timezone.utc)
    model = {"k": K, "periods": [DAY, WEEK], "refs": refs}
    meta = {
        "model_version": trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at": trained_at.isoformat(),
        "data_start": str(y.index.min()),
        "data_end": str(y.index.max()),
        "n_points": len(y),
        "n_events": len(events),
        "k": K,
        "metrics": metrics,
    }
    save_model(model, meta)

    # One long table (detector, timestamp_utc, ...) that the API reads
    table = pd.concat(
        detections, names=["detector", "timestamp_utc"]).reset_index()
    table.to_parquet(DETECTIONS, index=False)

    print("Points:", len(y), "| weeks:", round(weeks, 1),
          "| events:", len(events))
    print("Detections saved:", len(table), "rows ->", DETECTIONS)
    for name, m in metrics.items():
        print(f"\n{name}")
        for key, value in m.items():
            print(f"  {key}: {value}")


    # Log evaluation results to MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    for name, m in metrics.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("detector", name)
            mlflow.log_param("k", K)
            mlflow.log_param("n_points", len(y))
            mlflow.log_param("n_events", len(events))
            mlflow.log_metrics({
                key: float(value)
                for key, value in m.items()
            })
            mlflow.log_param("model_version", meta["model_version"])

    # Extra runs: same detector (mstl), different k
    for k in (2.0, 3.0, 5.0, 6.0):
        d = make_detection(y, expected["mstl"], refs["mstl"], k)
        m = evaluate(d["is_anomaly"], events, in_event, weeks)
        with mlflow.start_run(run_name=f"mstl_k{k:g}"):
            mlflow.log_param("detector", "mstl")
            mlflow.log_param("k", k)
            mlflow.log_metrics({key: float(v) for key, v in m.items()})


if __name__ == "__main__":
    main()

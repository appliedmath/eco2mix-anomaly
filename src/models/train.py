"""Train the detectors and keep the new model only if it is not worse.

    python -m src.models.train            # K = 4
    python -m src.models.train --k 2      # try another K (demo: rejected)

The new model is written to models/candidate/, compared with the model in
use on the same data (see promote.py), then promoted to models/ or rejected.
"""
import argparse
import os
from datetime import UTC, datetime

import pandas as pd

from src.models.anomaly_days import summarize_days
from src.models.detect import DAY, WEEK, detect_mstl
from src.models.events import build_reference_events, events_caught
from src.models.model import fit_reference, save_model
from src.models.promote import (
    CANDIDATE_DIR,
    MAIN_DETECTOR,
    MODEL_DIR,
    decide,
    load_champion,
    promote_files,
)
from src.models.tracking import log_run

K = 4.0
DATA = "data/processed/consumption_30min.parquet"
DETECTIONS = "models/detections.parquet"


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


def score_champion(y, expected_main, events, in_event, weeks,
                   folder: str = MODEL_DIR) -> dict | None:
    """The model in use, scored on the same data as the candidate."""
    champion = load_champion(folder)
    if champion is None:
        return None
    model, meta = champion
    flags = make_detection(y, expected_main, model["refs"][MAIN_DETECTOR],
                           model["k"])["is_anomaly"]
    scores = evaluate(flags, events, in_event, weeks)
    scores["k"] = model["k"]
    scores["model_version"] = meta["model_version"]
    return scores


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train, then promote or reject.")
    parser.add_argument("--k", type=float, default=K, help="alert threshold on |z|")
    k = parser.parse_args(argv).k

    df = pd.read_parquet(DATA).sort_index()
    y = df.loc[~df["is_future"], "consumption_mw"]

    events = build_reference_events(df)
    in_event = inside_events(y.index, events)
    weeks = (y.index.max() - y.index.min()).days / 7

    # What each detector expects at every half-hour
    expected = {
        # A: MSTL removes trend + daily + weekly cycles
        "mstl": detect_mstl(y, k=k)["expected_mw"],
        # B: RTE day-before forecast
        "forecast_gap": df.loc[y.index, "forecast_j1_mw"],
        # C: same half-hour one week earlier
        "last_week": y.shift(WEEK),
    }

    refs, detections = {}, {}
    for name, values in expected.items():
        refs[name] = fit_reference(y - values)
        detections[name] = make_detection(y, values, refs[name], k)

    flags = {name: d["is_anomaly"] for name, d in detections.items()}
    metrics = {name: evaluate(f, events, in_event, weeks)
               for name, f in flags.items()}

    # Promote or reject: compare with the model in use on the same data
    champion = score_champion(y, expected[MAIN_DETECTOR], events, in_event, weeks)
    promote, reason = decide(metrics[MAIN_DETECTOR], champion)

    trained_at = datetime.now(UTC)
    model = {"k": k, "periods": [DAY, WEEK], "refs": refs}
    meta = {
        "model_version": trained_at.strftime("%Y%m%d-%H%M%S"),
        "trained_at": trained_at.isoformat(),
        "data_start": str(y.index.min()),
        "data_end": str(y.index.max()),
        "n_points": len(y),
        "n_events": len(events),
        "k": k,
        "metrics": metrics,
        "promotion": {
            "decision": "promoted" if promote else "rejected",
            "reason": reason,
            "champion_version": champion["model_version"] if champion else None,
            "champion_scores": champion,
        },
    }
    save_model(model, meta, CANDIDATE_DIR)

    # One long table (detector, timestamp_utc, ...) that the API reads
    table = pd.concat(
        detections, names=["detector", "timestamp_utc"]).reset_index()
    table.to_parquet(os.path.join(CANDIDATE_DIR, "detections.parquet"), index=False)

    # One row per (detector, Paris day) with at least one anomaly
    days = summarize_days(table, events)
    days.to_csv(os.path.join(CANDIDATE_DIR, "anomaly_days.csv"), index=False)

    log_run(meta, model_dir=CANDIDATE_DIR, register=promote)
    if promote:
        promote_files(CANDIDATE_DIR, MODEL_DIR)

    print("Points:", len(y), "| weeks:", round(weeks, 1),
          "| events:", len(events))
    print("Detections:", len(table), "rows | anomaly days:", len(days), "rows")
    for name, m in metrics.items():
        print(f"\n{name}")
        for key, value in m.items():
            print(f"  {key}: {value}")

    print(f"\nDecision: {meta['promotion']['decision'].upper()} - {reason}")
    if promote:
        print("New model in use ->", MODEL_DIR)
    else:
        print("Model in use unchanged; candidate kept in", CANDIDATE_DIR)


if __name__ == "__main__":
    main()

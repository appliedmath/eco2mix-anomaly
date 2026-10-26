"""Promote or reject: a new training run replaces the model in use only if
it is not worse.

The model in use (the "champion") lives in models/. A new run (the
"candidate") is first written to models/candidate/. Both are scored on the
SAME data with the main detector (mstl). The candidate is promoted when

  - its recall is not lower than the champion's, and
  - its false alarms per week are at most MAX_FA_INCREASE (10 %) higher.

Otherwise the champion stays in models/ and the run is tagged "rejected"
in MLflow. A promoted model is also registered in the MLflow Model
Registry with the alias "champion".
"""
import os

from src.models.model import load_model

MAIN_DETECTOR = "mstl"
MAX_FA_INCREASE = 0.10
MODEL_DIR = "models"
CANDIDATE_DIR = os.path.join(MODEL_DIR, "candidate")
FILES = ("model.joblib", "metadata.json", "detections.parquet",
         "anomaly_days.csv")


def decide(candidate: dict, champion: dict | None,
           max_fa_increase: float = MAX_FA_INCREASE) -> tuple[bool, str]:
    """(promote?, reason) from the main detector's scores of both models."""
    if champion is None:
        return True, "no champion yet"

    rec_new, rec_old = candidate["recall"], champion["recall"]
    fa_new, fa_old = (candidate["false_alarms_per_week"],
                      champion["false_alarms_per_week"])
    if rec_new < rec_old:
        return False, f"recall dropped ({rec_old} -> {rec_new})"
    limit = fa_old * (1 + max_fa_increase)
    if fa_new > limit + 1e-9:
        return False, (f"false alarms/week rose too much ({fa_old} -> "
                       f"{fa_new}, limit {limit:.2f})")
    return True, (f"recall {rec_old} -> {rec_new}, "
                  f"false alarms/week {fa_old} -> {fa_new}")


def load_champion(folder: str = MODEL_DIR):
    """(model, meta) of the model in use, or None if there is none yet."""
    needed = [os.path.join(folder, n) for n in ("model.joblib", "metadata.json")]
    if not all(os.path.exists(p) for p in needed):
        return None
    return load_model(folder)


def promote_files(candidate_dir: str = CANDIDATE_DIR,
                  target_dir: str = MODEL_DIR) -> list[str]:
    """Move the candidate's files over the champion's (one file at a time,
    os.replace is atomic, so the API never reads a half-written file)."""
    moved = []
    for name in FILES:
        src = os.path.join(candidate_dir, name)
        if os.path.exists(src):
            os.replace(src, os.path.join(target_dir, name))
            moved.append(name)
    return moved

import json
import os

import joblib
import pandas as pd


def fit_reference(residuals: pd.Series) -> dict:
    """Learn the 'normal' model error: its median and MAD."""
    residuals = residuals.dropna()
    median = float(residuals.median())
    mad = float((residuals - median).abs().median())
    return {"median": median, "mad": mad}


def flag_with_reference(actual, expected, ref: dict, k: float = 4.0):
    """True where the error is far from normal, using learned stats."""
    gap = actual - expected
    score = (gap - ref["median"]) / (1.4826 * ref["mad"])
    return score.abs() > k


def save_model(model, meta: dict, folder: str = "models") -> None:
    os.makedirs(folder, exist_ok=True)
    joblib.dump(model, os.path.join(folder, "model.joblib"))
    with open(os.path.join(folder, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def load_model(folder: str = "models"):
    model = joblib.load(os.path.join(folder, "model.joblib"))
    with open(os.path.join(folder, "metadata.json")) as f:
        meta = json.load(f)
    return model, meta

def fit_bias(residuals: pd.Series) -> float:
    """Learn RTE's usual error: the median of (actual - forecast)."""
    return float(residuals.dropna().median())


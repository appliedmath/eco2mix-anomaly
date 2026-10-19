import numpy as np
import pandas as pd

from src.models.model import fit_reference
from src.models.predict import score_new
from src.models.train import make_detection

# 6 weeks of half-hours with a daily + weekly pattern
INDEX = pd.date_range("2024-03-04", periods=6 * 336, freq="30min", tz="UTC")


def make_data():
    t = np.arange(len(INDEX))
    rng = np.random.default_rng(1)
    y = (50000 + 8000 * np.sin(2 * np.pi * t / 48)
         + 3000 * np.sin(2 * np.pi * t / 336) + rng.normal(0, 300, len(t)))
    return pd.DataFrame({"consumption_mw": y, "forecast_j1_mw": y,
                         "is_future": False}, index=INDEX)


def make_model_and_detections(df, scored_until):
    """Pretend training scored everything up to `scored_until`."""
    y = df.loc[df.index <= scored_until, "consumption_mw"]
    refs = {name: fit_reference(pd.Series(np.random.default_rng(2)
                                          .normal(0, 300, 500)))
            for name in ("mstl", "forecast_gap", "last_week")}
    det = make_detection(y, y, refs["mstl"], 4.0)
    table = pd.concat({"mstl": det}, names=["detector", "timestamp_utc"])
    return {"k": 4.0, "refs": refs}, table.reset_index()


def test_scores_only_new_half_hours():
    df = make_data()
    cut = INDEX[5 * 336 - 1]                 # 5 weeks already scored
    model, detections = make_model_and_detections(df, cut)
    new = score_new(df, detections, model)
    assert new["timestamp_utc"].min() > cut
    assert new["timestamp_utc"].nunique() == 336   # the last week
    assert set(new["detector"]) == {"mstl", "forecast_gap", "last_week"}


def test_spike_in_new_week_is_flagged_by_mstl():
    df = make_data()
    cut = INDEX[5 * 336 - 1]
    spike = INDEX[5 * 336 + 100]
    df.loc[spike, "consumption_mw"] += 15000
    model, detections = make_model_and_detections(df, cut)
    new = score_new(df, detections, model)
    mstl = new[new["detector"] == "mstl"].set_index("timestamp_utc")
    assert mstl.loc[spike, "is_anomaly"]


def test_nothing_new_returns_empty():
    df = make_data()
    model, detections = make_model_and_detections(df, INDEX[-1])
    assert score_new(df, detections, model).empty

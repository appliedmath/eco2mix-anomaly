import numpy as np
import pandas as pd

from src.models.detect import detect_mstl

N = 6 * 336  # six weeks of 30-minute points


def make_series(spike_at=None):
    t = np.arange(N)
    daily = 4000 * np.sin(2 * np.pi * t / 48)
    weekly = 2000 * np.sin(2 * np.pi * t / 336)
    noise = np.random.RandomState(0).normal(0, 50, N)
    values = 40000 + daily + weekly + noise
    if spike_at is not None:
        values[spike_at] += 3000
    index = pd.date_range("2026-01-01", periods=N, freq="30min", tz="UTC")
    return pd.Series(values, index=index)


def test_detect_flags_the_spike():
    out = detect_mstl(make_series(spike_at=1000))
    assert out["is_anomaly"].iloc[1000]
    # the spike is the strongest point, clearly above any echo
    top = out["z_score"].abs().sort_values(ascending=False)
    assert top.index[0] == out.index[1000]
    assert top.iloc[0] > 2 * top.iloc[1]



def test_detect_quiet_series_has_no_alerts():
    out = detect_mstl(make_series(), k=6.0)
    assert out["is_anomaly"].sum() == 0


def test_detect_uses_the_given_reference():
    # A huge saved MAD means almost nothing counts as unusual
    ref = {"median": 0.0, "mad": 1_000_000.0}
    out = detect_mstl(make_series(spike_at=1000), ref=ref)
    assert out["is_anomaly"].sum() == 0

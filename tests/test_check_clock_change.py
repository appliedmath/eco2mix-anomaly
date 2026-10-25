"""Tests for scripts/check_clock_change.py on a hand-made October clock-change day."""
import numpy as np
import pandas as pd

from scripts.check_clock_change import check_day, is_ok
from src.data import process


def raw_october(lose_hour: bool) -> pd.DataFrame:
    """15-min raw data around 25 Oct 2026 (Paris day = 24 Oct 22:00 to 25 Oct 23:00 UTC)."""
    ts = pd.date_range("2026-10-24 20:00", "2026-10-26 02:00", freq="15min", tz="UTC")
    df = pd.DataFrame({
        "date_heure": ts,
        "consommation": np.where(ts.minute.isin([0, 30]), 45000.0, np.nan),
        "prevision_j1": 44000.0,
        "prevision_j": 44500.0,
    })
    if lose_hour:  # the repeated hour (01:00-02:00 UTC) missing, as in the RTE export
        df = df[~df.date_heure.between("2026-10-25 01:00+00:00", "2026-10-25 01:45+00:00")]
    return df


def test_change_day_has_50_half_hours():
    df = process.add_flags(process.to_30min_grid(raw_october(lose_hour=False)))
    result = check_day(df, "2026-10-25")
    assert result["length_hours"] == 25
    assert result["half_hours"] == result["expected_half_hours"] == 50
    assert result["filled"] == 0
    assert is_ok(result)


def test_lost_hour_is_filled_and_reported():
    df = process.add_flags(process.to_30min_grid(raw_october(lose_hour=True)))
    result = check_day(df, "2026-10-25")
    assert result["half_hours"] == 50
    assert result["missing"] == 0
    assert result["filled_at_utc"] == ["01:00", "01:30"]
    assert is_ok(result)


def test_normal_day_has_48():
    df = process.add_flags(process.to_30min_grid(raw_october(lose_hour=False)))
    assert check_day(df, "2026-10-24")["expected_half_hours"] == 48

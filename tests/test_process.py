"""Tests for src/data/process.py on a small hand-made raw table (14-15 July 2024, Paris)."""

import numpy as np
import pandas as pd
import pytest

from src.data import process


@pytest.fixture
def raw() -> pd.DataFrame:
    """15-min raw data like the API: consumption at :00/:30 only, one doubled hour, one lost hour, future rows."""
    ts = pd.date_range("2024-07-13 22:00", "2024-07-15 21:45", freq="15min", tz="UTC")
    df = pd.DataFrame({
        "date_heure": ts,
        "consommation": np.where(ts.minute.isin([0, 30]), 40000.0, np.nan),
        "prevision_j1": 39000.0,
        "prevision_j": 39500.0,
    })
    doubled = df[(df.date_heure >= "2024-07-14 01:00+00:00") & (df.date_heure <= "2024-07-14 01:45+00:00")]
    df = pd.concat([df, doubled.assign(consommation=99999.0)])               # DST-like duplicate (must be dropped)
    df = df[~df.date_heure.between("2024-07-14 03:00+00:00", "2024-07-14 03:45+00:00")]  # lost hour
    df.loc[df.date_heure > "2024-07-15 20:00+00:00", "consommation"] = np.nan  # not published yet
    return df


def test_grid_is_regular_30min_without_duplicates(raw):
    df = process.to_30min_grid(raw)
    assert not df.index.has_duplicates
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("30min")).all()
    assert (df["consumption_mw"] != 99999.0).all()       # first (original) row kept, duplicate dropped


def test_flags_fill_short_gap_and_keep_future_empty(raw):
    df = process.add_flags(process.to_30min_grid(raw))
    assert df["was_filled"].sum() == 2                     # lost hour = 2 half-hours, interpolated
    future = df[df["is_future"]]
    assert len(future) > 0 and future["consumption_mw"].isna().all()
    assert process.validate(df) == []


def test_bastille_day_is_a_holiday(raw):
    df = process.add_flags(process.to_30min_grid(raw))
    paris_day = df.index.tz_convert("Europe/Paris").day
    assert df.loc[paris_day == 14, "is_holiday"].all()
    assert not df.loc[paris_day == 15, "is_holiday"].any()


def test_validate_reports_out_of_range_values(raw):
    df = process.add_flags(process.to_30min_grid(raw))
    df.iloc[0, df.columns.get_loc("consumption_mw")] = 5_000.0
    assert any("outside" in problem for problem in process.validate(df))
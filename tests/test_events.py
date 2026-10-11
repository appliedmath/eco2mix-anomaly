import pandas as pd

from src.models.events import (
    clock_change_events,
    events_caught,
    holiday_events,
)


def test_clock_change_dates_in_2026():
    events = clock_change_events([2026])
    assert events["label"].tolist() == [
        "Clock change 2026-03-29",
        "Clock change 2026-10-25",
    ]


def test_paris_day_is_23_hours_when_clocks_go_forward():
    event = clock_change_events([2026]).iloc[0]  # 29 March 2026
    # a 23-hour day: first to last half-hour is 22 h 30 min apart
    assert event["end_utc"] - event["start_utc"] == pd.Timedelta(
        hours=22, minutes=30)


def test_holiday_events_one_event_per_paris_day():
    # 4 half-hours that all fall on 14 July in Paris (UTC+2 in summer)
    index = pd.date_range("2026-07-13 22:00", periods=4, freq="30min", tz="UTC")
    df = pd.DataFrame({"is_holiday": [True] * 4}, index=index)
    events = holiday_events(df)
    assert len(events) == 1
    assert events["start_utc"].iloc[0] == pd.Timestamp("2026-07-13 22:00", tz="UTC")


def test_events_caught():
    events = pd.DataFrame({
        "start_utc": pd.to_datetime(["2026-01-01 00:00", "2026-01-02 00:00"], utc=True),
        "end_utc": pd.to_datetime(["2026-01-01 23:30", "2026-01-02 23:30"], utc=True),
    })
    index = pd.date_range("2026-01-01", periods=96, freq="30min", tz="UTC")
    flags = pd.Series(False, index=index)
    flags.loc["2026-01-01 10:00"] = True  # only inside the first event
    assert events_caught(events, flags).tolist() == [True, False]

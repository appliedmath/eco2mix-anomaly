import os

import pandas as pd

MANUAL_FILE = "config/reference_events_manual.csv"
COLUMNS = ["start_utc", "end_utc", "kind", "label"]


def _paris_day(day):
    """First and last half-hour (in UTC) of one calendar day in Paris."""
    start = pd.Timestamp(day, tz="Europe/Paris")
    next_midnight = start + pd.DateOffset(days=1)  # calendar day, not 24 h
    last = next_midnight - pd.Timedelta(minutes=30)
    return start.tz_convert("UTC"), last.tz_convert("UTC")


def holiday_events(df: pd.DataFrame) -> pd.DataFrame:
    """One event per public holiday (a whole Paris day)."""
    paris_days = pd.Series(df.index.tz_convert("Europe/Paris").date,
                           index=df.index)
    rows = []
    for day in paris_days[df["is_holiday"]].unique():
        start, end = _paris_day(day)
        rows.append((start, end, "holiday", f"Public holiday {day}"))
    return pd.DataFrame(rows, columns=COLUMNS)


def clock_change_events(years) -> pd.DataFrame:
    """Last Sunday of March and of October, for each year."""
    rows = []
    for year in years:
        for month in (3, 10):
            last_day = pd.Timestamp(year, month, 31)
            sunday = last_day - pd.Timedelta(days=(last_day.weekday() + 1) % 7)
            start, end = _paris_day(sunday.date())
            rows.append((start, end, "clock_change",
                         f"Clock change {sunday.date()}"))
    return pd.DataFrame(rows, columns=COLUMNS)


def build_reference_events(df: pd.DataFrame,
                           manual_file: str = MANUAL_FILE) -> pd.DataFrame:
    """Automatic events + the ones added by hand in the CSV file."""
    years = range(df.index.min().year, df.index.max().year + 1)
    parts = [holiday_events(df), clock_change_events(years)]

    if os.path.exists(manual_file):
        manual = pd.read_csv(manual_file)
        if len(manual):
            for col in ("start_utc", "end_utc"):
                manual[col] = pd.to_datetime(manual[col], utc=True)
            parts.append(manual[COLUMNS])

    events = pd.concat(parts, ignore_index=True)
    inside = ((events["start_utc"] >= df.index.min())
              & (events["end_utc"] <= df.index.max()))
    return events[inside].sort_values("start_utc").reset_index(drop=True)


def events_caught(events: pd.DataFrame, flags: pd.Series) -> pd.Series:
    """For each event: was at least one point flagged inside it?"""
    flagged = flags[flags].index
    caught = [bool(((flagged >= s) & (flagged <= e)).any())
              for s, e in zip(events["start_utc"], events["end_utc"])]
    return pd.Series(caught, index=events.index)

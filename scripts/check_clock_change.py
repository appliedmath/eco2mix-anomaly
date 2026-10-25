"""Check that a clock-change day comes out right after processing.

Downloads only the RTE real-time data (in memory, nothing is written to
data/), runs the same cleaning as process.py and reports one Paris day.

    python -m scripts.check_clock_change               # 25 Oct 2026 (CEST -> CET)
    python -m scripts.check_clock_change --day 2026-03-29

Expected: 50 half-hours on the October change day (25 h), 46 in March (23 h),
no duplicates, no missing values.
"""
import argparse

import pandas as pd

from src.data import collect, process
from src.models.anomaly_days import paris_day_bounds


def check_day(df: pd.DataFrame, day: str) -> dict:
    """Summary of one Paris calendar day of the processed 30-min table."""
    start, end = paris_day_bounds(day)
    hours = (end - start) / pd.Timedelta("1h")
    part = df[(df.index >= start) & (df.index < end)]
    measured = part[~part["is_future"]]
    return {
        "day": day,
        "length_hours": hours,
        "expected_half_hours": int(hours * 2),
        "half_hours": len(part),
        "duplicates": int(part.index.duplicated().sum()),
        "missing": int(measured["consumption_mw"].isna().sum()),
        "filled": int(measured["was_filled"].sum()),
        "future": int(part["is_future"].sum()),
        "filled_at_utc": [t.strftime("%H:%M") for t in part.index[part["was_filled"]]],
    }


def is_ok(result: dict) -> bool:
    return (result["half_hours"] == result["expected_half_hours"]
            and result["duplicates"] == 0 and result["missing"] == 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--day", default="2026-10-25", help="Paris calendar day, YYYY-MM-DD")
    args = parser.parse_args()

    raw = collect.download("tr")
    df = process.add_flags(process.to_30min_grid(raw))
    result = check_day(df, args.day)

    for key, value in result.items():
        print(f"{key:>20}: {value}")
    print("\nOK" if is_ok(result) else "\nPROBLEM - see the numbers above")
    for problem in process.validate(df):
        print("validate():", problem)


if __name__ == "__main__":
    main()

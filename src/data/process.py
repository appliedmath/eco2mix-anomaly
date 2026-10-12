"""Turn raw éCO2mix downloads into one clean 30-min series: data/processed/consumption_30min.parquet."""

import logging
from pathlib import Path

import holidays
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_PATH = Path("data/processed/consumption_30min.parquet")
RENAME = {
    "consommation": "consumption_mw",
    "prevision_j1": "forecast_j1_mw",
    "prevision_j": "forecast_j_mw",
}
MAX_FILL_STEPS = 4                  # fill gaps of at most 4 half-hours (= 2 h)
VALID_RANGE_MW = (20_000, 100_000)  # plausible national consumption

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("process")


def load_latest(source: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the newest raw file of one source ('cons-def' or 'tr')."""
    files = sorted(raw_dir.glob(f"{source}_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No raw file for '{source}' in {raw_dir} - run collect.py first")
    log.info("Loading %s", files[-1])
    return pd.read_parquet(files[-1])


def to_30min_grid(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per half-hour (UTC): drop DST duplicates, keep minutes 0/30, reindex on a regular grid."""
    df = raw.sort_values("date_heure", kind="stable")               # stable: consolidated rows stay first
    df = df.drop_duplicates(subset="date_heure", keep="first")      # DST bug: doubled hour in March
    df = df.set_index("date_heure")[list(RENAME)].rename(columns=RENAME)
    df = df[pd.DatetimeIndex(df.index).minute.isin([0, 30])]                          # history has consumption only at :00/:30

    grid = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    missing = len(grid.difference(pd.DatetimeIndex(df.index)))                        # DST bug: lost hour in October
    df = df.reindex(grid)
    df.index.name = "timestamp_utc"
    log.info("30-min grid: %d rows, %d half-hours were missing", len(df), missing)
    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Mark future rows, fill short gaps, mark filled values and French public holidays."""
    df = df.copy()
    observed = df["consumption_mw"].notna()
    last_actual = df.index[observed].max()
    df["is_future"] = df.index > last_actual                        # not published yet: never an anomaly

    was_missing = ~observed & ~df["is_future"]
    cols = list(RENAME.values())
    df[cols] = df[cols].interpolate(limit=MAX_FILL_STEPS, limit_area="inside")
    df["was_filled"] = was_missing & df["consumption_mw"].notna()

    local_dates = pd.DatetimeIndex(df.index).tz_convert("Europe/Paris").date          # holidays follow the French calendar
    fr_holidays = holidays.France(years=range(df.index.min().year, df.index.max().year + 1))
    df["is_holiday"] = [d in fr_holidays for d in local_dates]
    return df


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of data-quality problems (empty list = all good)."""
    problems = []
    if df.index.has_duplicates:
        problems.append("duplicated timestamps")
    steps = df.index.to_series().diff().dropna().unique()
    if len(steps) != 1 or steps[0] != pd.Timedelta("30min"):
        problems.append(f"irregular time steps: {steps}")

    past = df.loc[~df["is_future"], "consumption_mw"]
    if past.isna().any():
        problems.append(f"{past.isna().sum()} past half-hours still missing (gap longer than 2 h)")
    low, high = VALID_RANGE_MW
    outside = past[(past < low) | (past > high)]
    if len(outside):
        problems.append(f"{len(outside)} values outside {low}-{high} MW")
    filled_per_day = df["was_filled"].groupby(pd.DatetimeIndex(df.index).date).sum()
    if (filled_per_day > MAX_FILL_STEPS).any():
        problems.append(f"more than {MAX_FILL_STEPS} filled half-hours on some days")
    return problems


def main() -> None:
    raw = pd.concat([load_latest("cons-def"), load_latest("tr")], ignore_index=True)
    df = add_flags(to_30min_grid(raw))

    for problem in validate(df):
        log.warning("Data quality: %s", problem)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED_PATH)
    log.info(
        "Saved %s: %s -> %s | %d filled, %d future, %d holiday half-hours",
        PROCESSED_PATH, df.index.min(), df.index.max(),
        df["was_filled"].sum(), df["is_future"].sum(), df["is_holiday"].sum(),
    )


if __name__ == "__main__":
    main()
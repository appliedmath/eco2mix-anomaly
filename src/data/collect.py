"""Download RTE éCO2mix national data (real-time + consolidated history) into data/raw/."""

import argparse
import io
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/exports/csv"
DATASETS = {"tr": "eco2mix-national-tr", "cons-def": "eco2mix-national-cons-def"}
COLUMNS = ["date_heure", "consommation", "prevision_j1", "prevision_j"]
RAW_DIR = Path("data/raw")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collect")


def download(source: str, start: str | None = None, timeout: int = 300) -> pd.DataFrame:
    """Download one dataset as a DataFrame (one API call = one CSV export)."""
    params = {"delimiter": ";", "select": ",".join(COLUMNS)}
    if start:
        params["where"] = f"date_heure >= date'{start}'"

    url = BASE_URL.format(dataset=DATASETS[source])
    log.info("Downloading %s (start=%s)", DATASETS[source], start)
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text), sep=";")
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Columns missing in API answer: {missing}")

    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df["source"] = source
    log.info("Got %d rows, %s -> %s", len(df), df["date_heure"].min(), df["date_heure"].max())
    return df


def save(df: pd.DataFrame, source: str, out_dir: Path = RAW_DIR) -> Path:
    """Save as data/raw/<source>_<UTC timestamp>.parquet and return the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{source}_{stamp}.parquet"
    df.to_parquet(path, index=False)
    log.info("Saved %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download RTE éCO2mix national data.")
    parser.add_argument("--source", choices=["tr", "cons-def", "all"], default="all")
    parser.add_argument("--start", default="2023-01-01",
                        help="First date of the history download (cons-def only).")
    args = parser.parse_args()

    sources = ["cons-def", "tr"] if args.source == "all" else [args.source]
    for source in sources:
        df = download(source, start=args.start if source == "cons-def" else None)
        save(df, source)


if __name__ == "__main__":
    main()
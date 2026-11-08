"""Simulate new data arriving over time (Kilyan, 12 Oct).

The last part of the cleaned series is held back in data/incoming/.
inject_batch moves the next few days from incoming back into the
processed file, as if RTE had just published them.
"""
from pathlib import Path

import pandas as pd

PROCESSED_PATH = Path("data/processed/consumption_30min.parquet")
INCOMING_PATH = Path("data/incoming/consumption_30min_incoming.parquet")
PARIS = "Europe/Paris"


def paris_midnight(ts: pd.Timestamp) -> pd.Timestamp:
    """Start (in UTC) of the Paris calendar day that contains ts."""
    day = ts.tz_convert(PARIS).normalize()
    return day.tz_convert("UTC")


def split_holdout(df: pd.DataFrame, frac: float = 0.2):
    """Split at a Paris midnight so ~frac of the measured rows are held back.

    Returns (training part, incoming part). Rows from the cut onwards,
    including future forecast-only rows, go to incoming.
    """
    if not 0 < frac < 1:
        raise ValueError("frac must be between 0 and 1")
    df = df.sort_index()
    actual = df.index[~df["is_future"]]
    cut = paris_midnight(actual[int(len(actual) * (1 - frac))])
    return df[df.index < cut], df[df.index >= cut]


def take_batch(processed: pd.DataFrame, incoming: pd.DataFrame,
               days: int = 7):
    """Move the next `days` Paris calendar days from incoming to processed.

    Returns (new processed, new incoming, the batch that was moved).
    """
    if incoming.empty:
        return processed, incoming, incoming
    start = paris_midnight(incoming.index.min())
    end = (start.tz_convert(PARIS) + pd.DateOffset(days=days)).tz_convert("UTC")
    batch = incoming[incoming.index < end]
    rest = incoming[incoming.index >= end]
    return pd.concat([processed, batch]).sort_index(), rest, batch


def fetch_live_batch(processed: pd.DataFrame, download=None):
    """Download the RTE days after the last processed one (complete Paris days only).

    Uses the same steps as Phase 1 (collect -> 30-min grid -> flags).
    Returns (new processed, the batch that was added).
    """
    from src.data.collect import download as rte_download
    from src.data.process import add_flags, to_30min_grid

    download = download or rte_download
    last = processed.index[~processed["is_future"]].max()
    start = (paris_midnight(last) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # consolidated first, like Phase 1; an empty part (cons-def ends on 2026-08-31)
    # would turn every column into object dtype, so it is left out
    parts = [download(source, start=start) for source in ("cons-def", "tr")]
    parts = [part for part in parts if len(part)]
    if not parts:
        return processed, processed.iloc[0:0]
    raw = pd.concat(parts, ignore_index=True)
    fresh = add_flags(to_30min_grid(raw))
    actual = fresh[~fresh["is_future"]]
    if actual.empty:
        return processed, actual
    end = paris_midnight(actual.index.max() + pd.Timedelta(minutes=30))
    batch = actual[(actual.index > last) & (actual.index < end)]
    return pd.concat([processed, batch]).sort_index(), batch


BATCH_INFO_PATH = Path("models/batch/last_batch.json")


def _utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def check_batch(processed: pd.DataFrame, start, end) -> list[str]:
    """Data-quality problems of the injected batch [start, end] (UTC).

    Uses the Phase 1 validate(); the half-hour before the batch is included,
    so a gap between the old and the new data is found too.
    """
    from src.data.process import validate

    start, end = _utc(start), _utc(end)
    batch = processed[(processed.index >= start) & (processed.index <= end)]
    if batch.empty:
        return ["the injected batch is empty"]
    before = processed[processed.index < start].tail(1)
    return validate(pd.concat([before, batch]))


REJECTED_DIR = Path("data/rejected")


def reject_batch(processed: pd.DataFrame, start, end):
    """Split the batch [start, end] (UTC) off the processed data: (kept, rejected)."""
    start, end = _utc(start), _utc(end)
    in_batch = (processed.index >= start) & (processed.index <= end)
    return processed[~in_batch], processed[in_batch]

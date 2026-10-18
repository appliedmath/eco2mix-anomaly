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

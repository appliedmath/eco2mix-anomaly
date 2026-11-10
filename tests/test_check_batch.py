"""check_batch: data-quality checks on the newly injected days only."""
import numpy as np
import pandas as pd

from src.data.batches import check_batch, reject_batch
from src.data.process import RENAME, add_flags, to_30min_grid

START = "2026-01-08 23:00"
END = "2026-01-10 22:30"


def make_processed():
    times = pd.date_range("2026-01-01 23:00", END, freq="30min", tz="UTC")
    raw = pd.DataFrame({"date_heure": times})
    for col in RENAME:
        raw[col] = 50000.0
    return add_flags(to_30min_grid(raw))


def test_clean_batch_has_no_problem():
    assert check_batch(make_processed(), START, END) == []


def test_bad_values_in_batch_are_found():
    df = make_processed()
    df.loc[pd.Timestamp("2026-01-09 12:00", tz="UTC"), "consumption_mw"] = -5.0
    df.loc[pd.Timestamp("2026-01-10 03:00", tz="UTC"), "consumption_mw"] = np.nan
    problems = check_batch(df, START, END)
    assert any("outside" in p for p in problems)
    assert any("missing" in p for p in problems)


def test_problems_before_the_batch_are_ignored():
    df = make_processed()
    df.loc[pd.Timestamp("2026-01-03 12:00", tz="UTC"), "consumption_mw"] = -5.0
    assert check_batch(df, START, END) == []


def test_reject_batch_removes_only_the_batch():
    df = make_processed()
    kept, rejected = reject_batch(df, START, END)
    assert len(rejected) == 2 * 48
    assert kept.index.max() < pd.Timestamp(START, tz="UTC")
    assert len(kept) + len(rejected) == len(df)


def test_after_a_rejected_batch_the_next_clean_batch_passes():
    df = make_processed()
    batch1 = ("2026-01-08 23:00", "2026-01-09 22:30")
    batch2 = ("2026-01-09 23:00", "2026-01-10 22:30")
    kept, _ = reject_batch(df, *batch1)
    # without the record the gap is still found
    assert any("irregular" in p for p in check_batch(kept, *batch2))
    # with the rejected range the next clean batch passes
    assert check_batch(kept, *batch2, rejected_ranges=[batch1]) == []

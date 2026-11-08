"""fetch_live_batch with a fake RTE download (no internet needed)."""
import pandas as pd

from src.data.batches import fetch_live_batch
from src.data.process import RENAME, add_flags, to_30min_grid


def make_raw(start, end):
    times = pd.date_range(start, end, freq="30min", tz="UTC")
    raw = pd.DataFrame({"date_heure": times})
    for col in RENAME:
        raw[col] = 50000.0
    return raw


def fake_download(source, start=None, timeout=300):
    if source == "cons-def":
        return make_raw("2026-01-06 00:00", "2026-01-10 22:30")
    return make_raw("2026-01-10 23:00", "2026-01-12 10:00")  # Jan 12 is not complete


def test_live_batch_adds_complete_days_without_gaps():
    processed = add_flags(to_30min_grid(make_raw("2026-01-01 23:00", "2026-01-08 22:30")))
    new, batch = fetch_live_batch(processed, download=fake_download)

    assert batch.index.min() == pd.Timestamp("2026-01-08 23:00", tz="UTC")
    assert batch.index.max() == pd.Timestamp("2026-01-11 22:30", tz="UTC")
    assert list(batch.columns) == list(processed.columns)
    assert not new.index.duplicated().any()
    assert (new.index.to_series().diff().dropna() == pd.Timedelta("30min")).all()


def test_live_batch_nothing_new():
    processed = add_flags(to_30min_grid(make_raw("2026-01-01 23:00", "2026-01-12 10:00")))
    new, batch = fetch_live_batch(processed, download=fake_download)

    assert batch.empty
    assert len(new) == len(processed)


def test_live_batch_when_cons_def_is_empty():
    """After 2026-08-31 the consolidated dataset is empty: only tr is used."""
    def download(source, start=None, timeout=300):
        if source == "cons-def":
            return pd.DataFrame(columns=["date_heure", *RENAME])
        return make_raw("2026-01-08 23:00", "2026-01-10 10:00")

    processed = add_flags(to_30min_grid(make_raw("2026-01-01 23:00", "2026-01-08 22:30")))
    new, batch = fetch_live_batch(processed, download=download)
    assert batch.index.min() == pd.Timestamp("2026-01-08 23:00", tz="UTC")
    assert batch.index.max() == pd.Timestamp("2026-01-09 22:30", tz="UTC")
    assert not new.index.duplicated().any()

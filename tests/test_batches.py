import pandas as pd
import pytest

from src.data.batches import split_holdout, take_batch

# 10 Paris days of half-hours (UTC+2 in July), last 2 rows = future
INDEX = pd.date_range("2024-07-01 22:00", periods=480, freq="30min", tz="UTC")


def make_df():
    df = pd.DataFrame({"consumption_mw": 1.0, "is_future": False}, index=INDEX)
    df.iloc[-2:, df.columns.get_loc("is_future")] = True
    return df


def test_split_keeps_every_row_once_and_in_order():
    df = make_df()
    train, incoming = split_holdout(df, 0.2)
    assert len(train) + len(incoming) == len(df)
    assert train.index.max() < incoming.index.min()


def test_split_cuts_at_paris_midnight():
    _, incoming = split_holdout(make_df(), 0.2)
    first = incoming.index.min().tz_convert("Europe/Paris")
    assert (first.hour, first.minute) == (0, 0)
    # cut rounds down to midnight: at least 20%, in whole days (3 of 10)
    assert len(incoming) % 48 == 0
    assert len(incoming) >= 0.2 * len(make_df())


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        split_holdout(make_df(), 1.5)


def test_take_batch_moves_whole_days():
    train, incoming = split_holdout(make_df(), 0.2)
    new_train, rest, batch = take_batch(train, incoming, days=1)
    assert len(batch) == 48
    assert len(rest) == len(incoming) - 48
    assert len(new_train) == len(train) + 48
    assert new_train.index.is_monotonic_increasing


def test_take_batch_when_nothing_left():
    train, _ = split_holdout(make_df(), 0.2)
    empty = train.iloc[:0]
    same, _rest, batch = take_batch(train, empty)
    assert batch.empty and len(same) == len(train)

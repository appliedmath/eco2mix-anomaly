"""Tests for src/data/collect.py - the API is replaced by a fake answer, so no internet is needed."""

import pandas as pd
import pytest

from src.data import collect

FAKE_CSV = (
    "date_heure;consommation;prevision_j1;prevision_j\n"
    "2026-10-01T00:00:00+02:00;40000;39500;39800\n"
    "2026-10-01T00:15:00+02:00;;39400;39700\n"
)


class FakeResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_download_parses_csv_to_utc(monkeypatch):
    monkeypatch.setattr(collect.requests, "get", lambda *args, **kwargs: FakeResponse(FAKE_CSV))
    df = collect.download("tr")
    assert len(df) == 2
    assert str(df["date_heure"].dt.tz) == "UTC"
    assert df["date_heure"].iloc[0] == pd.Timestamp("2026-09-30 22:00", tz="UTC")  # Paris 00:00 = UTC 22:00
    assert (df["source"] == "tr").all()


def test_download_fails_when_a_column_is_missing(monkeypatch):
    bad_csv = "date_heure;consommation\n2026-10-01T00:00:00+02:00;40000\n"
    monkeypatch.setattr(collect.requests, "get", lambda *args, **kwargs: FakeResponse(bad_csv))
    with pytest.raises(ValueError, match="Columns missing"):
        collect.download("tr")


def test_save_writes_timestamped_parquet(tmp_path):
    df = pd.DataFrame({"date_heure": pd.to_datetime(["2026-10-01"], utc=True), "consommation": [40000.0]})
    path = collect.save(df, "tr", out_dir=tmp_path)
    assert path.exists() and path.name.startswith("tr_") and path.suffix == ".parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)
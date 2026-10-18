import pandas as pd

from src.models.anomaly_days import paris_day_bounds, summarize_days

INDEX = pd.date_range("2026-07-14", periods=96, freq="30min", tz="UTC")


def make_detections(flag_at, z_at):
    z = [0.0] * 96
    for i, value in zip(flag_at, z_at):
        z[i] = value
    return pd.DataFrame({
        "detector": "mstl",
        "timestamp_utc": INDEX,
        "z_score": z,
        "is_anomaly": [abs(v) > 4 for v in z],
    })


def test_one_row_per_anomalous_day():
    # 10-11 = one episode on 14 July, 60 = 15 July (Paris time)
    days = summarize_days(make_detections([10, 11, 60], [5.0, -9.0, 6.0]))
    assert list(days["day"]) == ["2026-07-14", "2026-07-15"]
    first = days.iloc[0]
    assert first["n_points"] == 2
    assert first["n_episodes"] == 1
    assert first["max_abs_z"] == 9.0
    assert first["peak_utc"] == INDEX[11].isoformat()


def test_separate_alerts_count_as_two_episodes():
    days = summarize_days(make_detections([2, 20], [5.0, 5.0]))
    assert days.iloc[0]["n_episodes"] == 2


def test_day_is_paris_day_not_utc_day():
    # 22:30 UTC on 14 July is already 00:30 on 15 July in Paris (UTC+2)
    days = summarize_days(make_detections([45], [7.0]))
    assert list(days["day"]) == ["2026-07-15"]


def test_no_anomalies_gives_empty_table():
    assert summarize_days(make_detections([], [])).empty


def test_event_label_is_attached():
    events = pd.DataFrame({
        "start_utc": [INDEX[0]], "end_utc": [INDEX[30]],
        "kind": ["holiday"], "label": ["Public holiday 2026-07-14"]})
    days = summarize_days(make_detections([10], [5.0]), events)
    assert days.iloc[0]["event"] == "Public holiday 2026-07-14"


def test_clock_change_day_has_23_hours():
    low, high = paris_day_bounds("2026-03-29")
    assert high - low == pd.Timedelta(hours=23)

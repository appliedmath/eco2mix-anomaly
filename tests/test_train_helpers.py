import pandas as pd

from src.models.train import episodes, evaluate, inside_events


def make_index():
    return pd.date_range("2026-01-01", periods=10, freq="30min", tz="UTC")


def test_episodes_count_consecutive_flags_once():
    flags = pd.Series([False, True, True, False, True])
    assert episodes(flags) == 2


def test_evaluate_recall_and_false_alarms():
    index = make_index()
    events = pd.DataFrame({
        "start_utc": [index[0], index[5]],
        "end_utc": [index[1], index[6]],
        "kind": ["holiday", "clock_change"],
    })
    in_event = inside_events(index, events)
    flags = pd.Series(False, index=index)
    flags.iloc[1] = True   # inside the first event
    flags.iloc[8] = True   # outside every event

    result = evaluate(flags, events, in_event, weeks=1.0)

    assert result["recall"] == 0.5
    assert result["recall_holiday"] == 1.0
    assert result["recall_clock_change"] == 0.0
    assert result["false_alarms_per_week"] == 1.0
    assert result["alert_episodes"] == 2


from src.models.train import make_detection


def test_make_detection_scores_the_gap():
    ref = {"median": 0.0, "mad": 10.0}
    out = make_detection(
        pd.Series([100.0, 110.0, 200.0]),
        pd.Series([100.0, 100.0, 100.0]),
        ref, k=4.0)
    assert out["is_anomaly"].tolist() == [False, False, True]
    assert abs(out["z_score"].iloc[2] - 100 / 14.826) < 1e-6

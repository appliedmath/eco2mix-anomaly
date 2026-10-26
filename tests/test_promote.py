"""Promote-or-reject: the decision rule, moving files, scoring the champion."""
import pandas as pd
import pytest

from src.models.model import save_model
from src.models.promote import FILES, decide, load_champion, promote_files
from src.models.tracking import champion_metrics, run_tags
from src.models.train import score_champion

CHAMPION = {"recall": 0.70, "false_alarms_per_week": 0.50}


def test_first_model_is_always_promoted():
    assert decide({"recall": 0.1, "false_alarms_per_week": 9.0}, None) == \
        (True, "no champion yet")


def test_same_or_better_model_is_promoted():
    ok, _ = decide({"recall": 0.70, "false_alarms_per_week": 0.50}, CHAMPION)
    assert ok
    ok, _ = decide({"recall": 0.75, "false_alarms_per_week": 0.40}, CHAMPION)
    assert ok


def test_lower_recall_is_rejected():
    ok, reason = decide({"recall": 0.65, "false_alarms_per_week": 0.30}, CHAMPION)
    assert not ok
    assert "recall dropped" in reason


def test_small_rise_in_false_alarms_is_accepted():
    ok, _ = decide({"recall": 0.70, "false_alarms_per_week": 0.55}, CHAMPION)
    assert ok  # 0.55 = limit (+10 %)


def test_large_rise_in_false_alarms_is_rejected():
    ok, reason = decide({"recall": 0.80, "false_alarms_per_week": 1.20}, CHAMPION)
    assert not ok
    assert "false alarms" in reason


def test_promote_files_moves_candidate_over_champion(tmp_path):
    champion, candidate = tmp_path / "models", tmp_path / "models" / "candidate"
    candidate.mkdir(parents=True)
    for name in FILES:
        (champion / name).write_text("old")
        (candidate / name).write_text("new")
    (champion / "other.txt").write_text("keep")

    moved = promote_files(str(candidate), str(champion))

    assert moved == list(FILES)
    assert all((champion / n).read_text() == "new" for n in FILES)
    assert (champion / "other.txt").read_text() == "keep"
    assert not any((candidate / n).exists() for n in FILES)


def test_no_champion_when_folder_is_empty(tmp_path):
    assert load_champion(str(tmp_path)) is None


def test_champion_is_scored_on_the_same_data(tmp_path):
    idx = pd.date_range("2025-01-01", periods=96, freq="30min", tz="UTC")
    y = pd.Series(50_000.0, index=idx)
    y.iloc[40] = 60_000.0                      # one clear spike
    expected = pd.Series(50_000.0, index=idx)
    events = pd.DataFrame({"start_utc": [idx[38]], "end_utc": [idx[42]],
                           "kind": ["holiday"], "label": ["test"]})
    in_event = pd.Series(False, index=idx)
    in_event.iloc[38:43] = True
    model = {"k": 4.0, "refs": {"mstl": {"median": 0.0, "mad": 100.0}}}
    save_model(model, {"model_version": "v1"}, str(tmp_path))

    scores = score_champion(y, expected, events, in_event, 2.0, str(tmp_path))

    assert scores["model_version"] == "v1"
    assert scores["recall"] == 1.0
    assert scores["false_alarms_per_week"] == 0.0


def test_run_tags_and_champion_metrics():
    meta = {"promotion": {"decision": "rejected", "reason": "recall dropped",
                          "champion_version": "v1",
                          "champion_scores": {"recall": 0.7, "model_version": "v1"}}}
    assert run_tags(meta) == {"decision": "rejected", "reason": "recall dropped",
                              "compared_with": "v1"}
    assert champion_metrics(meta) == {"champion_recall": 0.7}


def test_pyfunc_predict_matches_the_rule():
    pytest.importorskip("mlflow")
    from src.models.pyfunc_model import AnomalyModel

    m = AnomalyModel()
    m.model = {"k": 4.0, "refs": {"mstl": {"median": 0.0, "mad": 100.0}}}
    out = m.predict(None, pd.DataFrame({"actual": [50_000.0, 51_000.0],
                                        "expected": [50_000.0, 50_000.0]}))
    assert out["is_anomaly"].tolist() == [False, True]

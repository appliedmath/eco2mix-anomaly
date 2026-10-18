from src.models.tracking import flat_metrics, log_run


def test_flat_metrics_names_each_detector():
    metrics = {"mstl": {"recall": 0.7, "events_caught": 33},
               "last_week": {"recall": 0.2}}
    assert flat_metrics(metrics) == {"mstl_recall": 0.7,
                                     "mstl_events_caught": 33.0,
                                     "last_week_recall": 0.2}


def test_no_tracking_without_server(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert log_run({"model_version": "x"}) is None

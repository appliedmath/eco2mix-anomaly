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


def test_model_signature_describes_inputs_outputs_and_params():
    from src.models.tracking import model_signature
    example, signature = model_signature()
    assert list(example.columns) == ["actual", "expected"]
    assert signature.inputs.input_names() == ["actual", "expected"]
    assert signature.outputs.input_names() == ["z_score", "is_anomaly"]
    assert [p.name for p in signature.params.params] == ["detector"]

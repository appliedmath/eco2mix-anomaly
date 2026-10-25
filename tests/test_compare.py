import pandas as pd

from src.models.compare import KS, log_to_mlflow
from src.models.train import K


def test_comparison_covers_the_chosen_threshold():
    assert K in KS


def test_no_tracking_without_server(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    table = pd.DataFrame([{"detector": "mstl", "k": 4.0, "recall": 0.7}])
    assert log_to_mlflow(table, n_points=10, n_events=2) is None

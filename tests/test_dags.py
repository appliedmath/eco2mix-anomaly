"""The Airflow DAGs load without errors (skipped when Airflow is not installed,
e.g. in the normal dev venv; Airflow runs in its own container)."""
import os
from pathlib import Path

import pytest

pytest.importorskip("airflow")

DAGS = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dagbag():
    os.environ.setdefault("HOST_PROJECT_DIR", "/host/eco2mix-anomaly")
    import sys
    sys.path.insert(0, str(DAGS))
    from airflow.models import DagBag
    return DagBag(str(DAGS), include_examples=False)


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}


def test_batch_dag_injects_then_predicts(dagbag):
    dag = dagbag.dags["batch_dag"]
    assert not dag.catchup
    assert dag.get_task("inject_batch").downstream_task_ids == {"predict"}


def test_tasks_use_trainer_image_host_mounts_and_one_slot_pool(dagbag):
    for dag_id in ("batch_dag", "retrain_dag"):
        for task in dagbag.dags[dag_id].tasks:
            assert task.image.startswith("eco2mix-anomaly-trainer")
            assert task.pool == "models"
            assert {m["Target"] for m in task.mounts} == {"/app/data", "/app/models"}
            assert all(m["Source"].startswith("/host/eco2mix-anomaly")
                       for m in task.mounts)

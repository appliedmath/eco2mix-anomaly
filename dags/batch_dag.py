"""batch_dag: new data arrives, the saved model scores it."""
import os

import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from eco2mix_common import DEFAULT_ARGS, trainer_task_kwargs

with DAG(
    dag_id="batch_dag",
    description="Fetch the new days from the RTE API (or replay held-out days), then score them",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    params={"source": os.getenv("BATCH_SOURCE", "replay"), "days": 1},
    tags=["eco2mix", "batch"],
    doc_md="""
### Batch prediction
1. **inject_batch** adds new data to the processed data. `source=live`
   (BATCH_SOURCE in .env): all new complete days from the RTE API (consolidated + real-time).
   `source=replay`: the next `days` held-out days from `data/incoming/`.
2. **validate** checks the data quality of the new days (Phase 1 validate():
   duplicates, gaps, missing or out-of-range values, too many filled values).
   A problem fails the task, so predict does not run; the result is written
   to `models/drift/latest.json`.
3. **predict** scores only the new half-hours with the saved champion model
   and updates `models/detections.parquet` and `models/anomaly_days.csv`.
   The API shows the result without a restart.

Replay for a demo: `{"source": "replay", "days": 7}` as the run conf.
""",
) as dag:
    inject = DockerOperator(
        task_id="inject_batch",
        command="python -m scripts.inject_batch --source {{ params.source }} --days {{ params.days }}",
        **trainer_task_kwargs(),
    )
    validate = DockerOperator(
        task_id="validate",
        command="python -m scripts.validate_batch",
        **{**trainer_task_kwargs(), "retries": 0},  # same data fails again
    )
    predict = DockerOperator(
        task_id="predict",
        command="python -m src.models.predict",
        **trainer_task_kwargs(),
    )

    inject >> validate >> predict

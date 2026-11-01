"""batch_dag: new data arrives, the saved model scores it."""
import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from eco2mix_common import DEFAULT_ARGS, trainer_task_kwargs

with DAG(
    dag_id="batch_dag",
    description="Inject the next day(s) of data, then score them with the champion model",
    schedule="@daily",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    params={"days": 1},
    tags=["eco2mix", "batch"],
    doc_md="""
### Batch prediction
1. **inject_batch** moves the next `days` of data from `data/incoming/` into
   the processed data (simulates new data arriving).
2. **predict** scores only the new half-hours with the saved champion model
   and updates `models/detections.parquet` and `models/anomaly_days.csv`.
   The API shows the result without a restart.

Trigger with another number of days: `{"days": 7}` as the run conf.
""",
) as dag:
    inject = DockerOperator(
        task_id="inject_batch",
        command="python -m scripts.inject_batch --days {{ params.days }}",
        **trainer_task_kwargs(),
    )
    predict = DockerOperator(
        task_id="predict",
        command="python -m src.models.predict",
        **trainer_task_kwargs(),
    )

    inject >> predict

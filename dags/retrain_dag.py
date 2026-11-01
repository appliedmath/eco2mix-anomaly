"""retrain_dag: train a candidate, promote it only if it is not worse."""
import pendulum
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from eco2mix_common import DEFAULT_ARGS, trainer_task_kwargs

with DAG(
    dag_id="retrain_dag",
    description="Retrain on the current data; promote-or-reject against the champion",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["eco2mix", "training"],
    doc_md="""
### Retraining with promote-or-reject
**train** runs `src.models.train`: the new model is scored on the same data
as the champion and replaces it only if recall does not drop and false
alarms per week rise by at most 10 %. Every run is logged in MLflow
(experiment `grid-anomaly-detection`, tag `decision`); a promoted model gets
the alias `@champion` in the registry.
""",
) as dag:
    DockerOperator(
        task_id="train",
        command="python -m src.models.train",
        environment={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "MLFLOW_DISABLE_TELEMETRY": "true",
            "GIT_PYTHON_REFRESH": "quiet",
        },
        **trainer_task_kwargs(),
    )

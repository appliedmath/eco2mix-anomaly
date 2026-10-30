"""Settings shared by our DAGs: how a task starts the trainer image.

Every task is a DockerOperator: Airflow asks the host's Docker (through the
mounted socket) to start a container from our trainer image, with data/ and
models/ mounted. The mount sources must be paths ON THE HOST, so the project
folder comes from HOST_PROJECT_DIR in .env.
"""
import os
from datetime import timedelta

from docker.types import Mount

HOST_PROJECT_DIR = os.environ.get("HOST_PROJECT_DIR", "/opt/eco2mix-anomaly")
TRAINER_IMAGE = os.environ.get("TRAINER_IMAGE", "eco2mix-anomaly-trainer:latest")
NETWORK = os.environ.get("COMPOSE_NETWORK", "eco2mix-anomaly_default")
MODELS_POOL = "models"   # 1 slot: only one task at a time writes models/

DEFAULT_ARGS = {
    "owner": "eco2mix",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def mounts() -> list[Mount]:
    """data/ and models/ of the project, as in docker-compose."""
    return [
        Mount(source=f"{HOST_PROJECT_DIR}/data", target="/app/data", type="bind"),
        Mount(source=f"{HOST_PROJECT_DIR}/models", target="/app/models", type="bind"),
    ]


def trainer_task_kwargs(**extra) -> dict:
    """Common DockerOperator arguments for a job in the trainer image."""
    kwargs = {
        "image": TRAINER_IMAGE,
        "docker_url": "unix://var/run/docker.sock",
        "network_mode": NETWORK,      # reach mlflow:5000 like the other services
        "mounts": mounts(),
        "mount_tmp_dir": False,       # no extra temp mount (fails on Docker Desktop)
        "auto_remove": "success",     # keep a failed container for debugging
        "pool": MODELS_POOL,
    }
    kwargs.update(extra)
    return kwargs

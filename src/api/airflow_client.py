"""Small client for Airflow's REST API (v1), used by the API to start jobs.

The API does not run training or prediction itself: it asks Airflow to start
a DAG run, so the workflow (steps, retries, pool) is defined in one place.
Only the standard library is used, so the API image needs no new package.
"""
import base64
import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote


class AirflowUnavailable(Exception):
    """Airflow could not be reached (not started, wrong URL, timeout)."""


class AirflowError(Exception):
    """Airflow answered with an error (e.g. 401 wrong login, 404 unknown run)."""

    def __init__(self, status: int, message: str):
        super().__init__(f"Airflow answered {status}: {message}")
        self.status = status
        self.message = message


class AirflowClient:
    def __init__(self, base_url: str, user: str, password: str, timeout: float = 10):
        self.base_url = base_url.rstrip("/")
        token = base64.b64encode(f"{user}:{password}".encode()).decode()
        self.headers = {
            "Authorization": f"Basic {token}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.base_url}/api/v1{path}", data=data,
                                     headers=self.headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as err:
            try:
                message = json.loads(err.read()).get("detail") or err.reason
            except (ValueError, AttributeError):
                message = err.reason
            raise AirflowError(err.code, str(message)) from err
        except (urllib.error.URLError, TimeoutError, ConnectionError) as err:
            raise AirflowUnavailable(str(err)) from err

    def is_paused(self, dag_id: str) -> bool:
        return bool(self._request("GET", f"/dags/{quote(dag_id)}")["is_paused"])

    def trigger(self, dag_id: str, run_id: str, conf: dict) -> dict:
        return self._request("POST", f"/dags/{quote(dag_id)}/dagRuns",
                             {"dag_run_id": run_id, "conf": conf})

    def get_run(self, dag_id: str, run_id: str) -> dict:
        return self._request(
            "GET", f"/dags/{quote(dag_id)}/dagRuns/{quote(run_id, safe='')}")

    def task_states(self, dag_id: str, run_id: str) -> list[dict]:
        found = self._request(
            "GET", f"/dags/{quote(dag_id)}/dagRuns/{quote(run_id, safe='')}"
                   "/taskInstances")
        return [{"task_id": t["task_id"], "state": t["state"]}
                for t in found.get("task_instances", [])]


def client_from_env() -> AirflowClient:
    return AirflowClient(
        base_url=os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080"),
        user=os.environ.get("AIRFLOW_USER", "admin"),
        password=os.environ.get("AIRFLOW_PASSWORD", ""),
    )

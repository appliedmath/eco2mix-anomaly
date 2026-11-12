"""Ask the MLflow Model Registry which version has the alias @champion.

Plain HTTP with the standard library (MLflow REST API), so the API image
does not need the mlflow package.
"""
import json
import os
import urllib.error
import urllib.parse
import urllib.request

MODEL_NAME = "grid-anomaly-detector"
ALIAS = "champion"


class RegistryError(Exception):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def champion_version(base_url: str, timeout: float = 3.0) -> dict:
    """{'name', 'alias', 'version', 'run_id'} of the champion in the registry."""
    query = urllib.parse.urlencode({"name": MODEL_NAME, "alias": ALIAS})
    url = f"{base_url.rstrip('/')}/api/2.0/mlflow/registered-models/alias?{query}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = json.load(response)
    except urllib.error.HTTPError as err:
        text = err.read().decode(errors="replace")
        if "RESOURCE_DOES_NOT_EXIST" in text or err.code == 404:
            raise RegistryError(404, f"no version of {MODEL_NAME} has the "
                                     f"alias @{ALIAS} yet") from err
        raise RegistryError(err.code, f"MLflow answered {err.code}") from err
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise RegistryError(None, f"MLflow is not reachable: {err}") from err

    version = body["model_version"]
    return {"name": MODEL_NAME, "alias": ALIAS,
            "version": str(version["version"]),
            "run_id": version.get("run_id")}


def tracking_url() -> str | None:
    return os.environ.get("MLFLOW_TRACKING_URI") or None

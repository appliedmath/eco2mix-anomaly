"""Kubernetes manifests (k8s/): the API as 3 stateless replicas."""
from pathlib import Path

import yaml

K8S = Path(__file__).resolve().parents[1] / "k8s"


def load(name):
    return yaml.safe_load((K8S / name).read_text())


def test_deployment_three_replicas_with_probes():
    dep = load("deployment.yml")
    assert dep["kind"] == "Deployment" and dep["spec"]["replicas"] == 3
    assert dep["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    container = dep["spec"]["template"]["spec"]["containers"][0]
    assert container["imagePullPolicy"] == "Never"          # local image
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["resources"]["limits"]["memory"]


def test_secrets_come_from_a_secret_not_from_the_files():
    container = load("deployment.yml")["spec"]["template"]["spec"]["containers"][0]
    secret_env = {e["name"]: e["valueFrom"]["secretKeyRef"]["name"]
                  for e in container["env"]}
    assert secret_env == {"API_KEY": "eco2mix-api-secrets",
                          "AIRFLOW_PASSWORD": "eco2mix-api-secrets"}
    assert not (K8S / "secret.yml").exists()                # only the template
    example = load("secret.example.yml")
    assert set(example["stringData"].values()) == {"change-me"}


def test_service_selects_the_api_pods():
    svc, dep = load("service.yml"), load("deployment.yml")
    assert svc["spec"]["selector"] == dep["spec"]["selector"]["matchLabels"]
    assert svc["spec"]["ports"][0]["targetPort"] == 8000
    assert svc["spec"]["ports"][0]["port"] == 8001          # 8000 = compose NGINX


def test_configmap_has_what_the_api_reads():
    data = load("configmap.yml")["data"]
    assert {"MODEL_DIR", "DATA_PATH", "AIRFLOW_URL", "AIRFLOW_USER"} <= set(data)


def test_all_objects_in_the_eco2mix_namespace():
    for path in K8S.glob("*.yml"):
        doc = yaml.safe_load(path.read_text())
        if doc["kind"] != "Namespace":
            assert doc["metadata"]["namespace"] == "eco2mix", path.name

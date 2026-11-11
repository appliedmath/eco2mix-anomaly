"""docker-compose.yml and the NGINX config fit together (no Docker needed)."""
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]
NGINX_CONF = (ROOT / "docker" / "nginx" / "default.conf").read_text()


def test_only_nginx_has_ports_on_the_machine():
    with_ports = sorted(name for name, svc in SERVICES.items() if svc.get("ports"))
    assert with_ports == ["nginx"]


def test_nginx_keeps_the_usual_ports():
    ports = [p.split(":")[-1] for p in SERVICES["nginx"]["ports"]]
    assert ports == ["8000", "5000", "8080", "3000", "8501", "9090"]


def test_nginx_forwards_to_existing_services():
    targets = re.findall(r"http://([a-z-]+):(\d+)", NGINX_CONF)
    assert targets, "no proxy targets found"
    for host, _port in targets:
        assert host in SERVICES, f"nginx points to unknown service {host}"


def test_job_endpoints_are_rate_limited():
    block = NGINX_CONF.split("location ~ ^/(train|predict)$", 1)[1].split("}", 1)[0]
    assert "limit_req zone=jobs" in block


def test_metrics_is_closed_from_outside():
    block = NGINX_CONF.split("location = /metrics", 1)[1].split("}", 1)[0]
    assert "return 403" in block


# ---------- Monitoring (Phase 4) ----------
MONITORING = ROOT / "monitoring"


def test_monitoring_images_are_pinned():
    for name in ("prometheus", "node-exporter", "grafana"):
        image = SERVICES[name]["image"]
        assert ":" in image and not image.endswith(":latest"), image


def test_prometheus_is_published_on_localhost_only():
    ports = SERVICES["nginx"]["ports"]
    prom = [p for p in ports if p.endswith(":9090")]
    assert prom == ["127.0.0.1:${PROMETHEUS_PORT:-9090}:9090"]
    assert any(p.endswith(":3000") and not p.startswith("127.") for p in ports)


def test_grafana_password_comes_from_env():
    env = SERVICES["grafana"]["environment"]
    assert env["GF_SECURITY_ADMIN_PASSWORD"].startswith("${GRAFANA_ADMIN_PASSWORD:?")
    assert env["GF_USERS_ALLOW_SIGN_UP"] == "false"
    assert env["GF_AUTH_ANONYMOUS_ENABLED"] == "false"


def test_prometheus_scrapes_api_and_node_exporter():
    config = yaml.safe_load((MONITORING / "prometheus" / "prometheus.yml").read_text())
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert config["global"]["scrape_interval"] == "15s"
    api = jobs["api"]["dns_sd_configs"][0]       # every API replica is scraped
    assert api["names"] == ["api"] and api["port"] == 8000
    assert jobs["node"]["static_configs"][0]["targets"] == ["node-exporter:9100"]
    for target in ("api", "node-exporter", "prometheus", "grafana"):
        assert target in SERVICES


def test_alert_rules_exist():
    rules = yaml.safe_load((MONITORING / "prometheus" / "alert_rules.yml").read_text())
    alerts = {r["alert"]: r for g in rules["groups"] for r in g["rules"]}
    assert set(alerts) == {"ApiDown", "NoNewData"}
    assert "data_updated_timestamp_seconds" in alerts["NoNewData"]["expr"]


def test_grafana_provisioning_and_dashboards():
    ds = yaml.safe_load((MONITORING / "grafana" / "provisioning" / "datasources"
                         / "prometheus.yml").read_text())["datasources"][0]
    assert ds["uid"] == "prometheus" and ds["url"] == "http://prometheus:9090"
    for path in (MONITORING / "grafana" / "dashboards").glob("*.json"):
        dashboard = json.loads(path.read_text())
        for panel in dashboard["panels"]:
            assert panel["datasource"]["uid"] == "prometheus", path.name


def test_grafana_location_keeps_the_host_header():
    block = NGINX_CONF.split("listen 3000;", 1)[1].split("\n}\n", 1)[0]
    location = block.split("location / {", 1)[1]
    assert "proxy_set_header Host" in location      # not inherited from server{}
    assert "Upgrade" in location

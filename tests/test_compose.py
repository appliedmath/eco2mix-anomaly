"""docker-compose.yml and the NGINX config fit together (no Docker needed)."""
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
    assert ports == ["8000", "5000", "8080"]


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

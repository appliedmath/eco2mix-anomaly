# Short commands for the most common tasks.  `make` or `make help` lists them.
# Every target is a normal command you can also type yourself (see README).

.DEFAULT_GOAL := help
.PHONY: help init up stop down ps logs rebuild retrain test lint check dashboards \
        presentation k8s k8s-delete

help:  ## show this list
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*## "}; {printf "  make %-13s %s\n", $$1, $$2}'

# ---------- run the stack ----------
init:  ## first time only: download, clean and train the first model
	docker compose --profile init up trainer

up:  ## start all services (does not retrain)
	docker compose up -d

stop:  ## stop the services, keep containers and data
	docker compose stop

down:  ## remove the containers (data, models and volumes stay)
	docker compose down

ps:  ## state of every service
	docker compose ps -a

logs:  ## follow the logs of one service: make logs s=api
	docker compose logs -f --tail=100 $(s)

rebuild:  ## rebuild the images after a code change, then recreate
	docker compose build trainer api streamlit
	docker compose up -d --force-recreate api streamlit nginx airflow-scheduler airflow-webserver

retrain:  ## retrain once without Airflow (promote or reject)
	docker compose run --rm trainer python -m src.models.train

# ---------- quality ----------
test:  ## run the tests
	python -m pytest -q

lint:  ## ruff, like CI
	python -m ruff check src tests scripts

check: lint test  ## lint + tests before a push

dashboards:  ## regenerate the Grafana dashboard JSON files
	python monitoring/grafana/build_dashboards.py

# ---------- defense ----------
presentation:  ## presentation app on http://localhost:8502
	streamlit run streamlit_app/presentation.py --server.port 8502

k8s:  ## deploy the API to local Kubernetes (Docker Desktop), :8001
	bash k8s/deploy.sh

k8s-delete:  ## remove the Kubernetes namespace
	bash k8s/deploy.sh delete

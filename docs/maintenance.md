# Maintenance: how to run, change and repair the system

Who this is for: whoever runs the stack after us. It complements the
[README](../README.md) (first start), the [architecture](architecture.md),
the [monitoring plan](monitoring_plan.md) (what each alert means and what to
do) and the [ethics and security note](ethics_security.md).

## 1. What runs by itself

| What | When | Where to look |
|---|---|---|
| `batch_dag`: inject_batch → validate → predict → drift | daily (`@daily`, 00:00 UTC) | Airflow → batch_dag → Grid; Grafana "2 · Model & data" |
| `retrain_dag`: train → promote or reject | weekly (`@weekly`) | Airflow → retrain_dag; MLflow runs (tag `decision`) |
| Prometheus scrape (API, node-exporter) | every 15 s | http://localhost:9090/targets |
| Alerts: 2 system (Prometheus), 7 ML (Grafana) | continuously | Grafana → Alerting → Alert rules |

Daily check (1 minute): Grafana **"0 · Stack health"** (all targets UP, data
age green) and the alert list. Nothing red → nothing to do.

## 2. Start, stop, update

```bash
docker compose up -d                 # start everything (the one-shot trainer also trains once)
docker compose ps                    # all "Up"/"healthy"; trainer and airflow-init "Exited (0)"
docker compose down                  # stop (data, runs, metrics are kept in volumes and folders)
```

**After a change, what to rebuild or recreate:**

| You changed | Do | Why |
|---|---|---|
| `src/`, `scripts/` | `docker compose build trainer api && docker compose up -d` | the DAG tasks run in the trainer image, the API in the api image |
| `dags/` | nothing for Airflow; **but** rebuild the trainer image if the DAG calls a new script | Airflow reads `dags/` at once; a DAG that calls a script missing from the image fails (see lessons learned) |
| `docker/nginx/default.conf` | `docker compose up -d --force-recreate nginx` | a `restart` keeps the old mounted file on Docker Desktop/WSL |
| `monitoring/prometheus/*` | `docker compose up -d --force-recreate prometheus` | same reason |
| dashboards | edit `monitoring/grafana/build_dashboards.py` → `python monitoring/grafana/build_dashboards.py` → `docker compose up -d --force-recreate grafana` | the JSON files are generated |
| `.env` | `unset <VAR>` (or a new terminal), then `docker compose up -d --force-recreate <services>` | Compose prefers a shell variable over `.env` |
| a new service in `docker-compose.yml` | `docker compose up -d --build <service>` and `--force-recreate nginx` | NGINX must publish the new port |

## 3. Model: retrain, check, roll back

- **Retrain now:** `POST /train` with the API key (Swagger at `/docs`), the
  admin tab of the Streamlit app, or `docker compose run --rm trainer`.
- **Check the decision:** MLflow → run → tag `decision` (`promoted` /
  `rejected`) and the reason; a promotion fires the **NewChampion** alert.
- **Roll back:** today the API serves the champion files in `models/`.
  Every run is kept in MLflow (parameters, metrics, artifacts) and every
  promoted model is a registry version, so the earlier champion can be
  restored from there, or retrained on the earlier data (`dvc checkout`).
  Planned: the API loads `models:/grid-anomaly-detector@champion`, then a
  rollback is only moving the alias.
- **Never retrain because of a drift alert alone:** first read the Evidently
  report (`models/drift/reports/`) and check the data; drift can come from a
  data bug. The promotion gate still protects the champion.

## 4. Data

| Situation | What to do |
|---|---|
| A batch failed validation (`DataQualityFailed`) | read `quality_problems` in `models/drift/latest.json`; the batch is in `data/rejected/` (listed in `data/rejected/ranges.json`) and never reached predict or retraining; with `BATCH_SOURCE=live` the days are fetched again on the next run |
| No new data (`NoNewData`) | is `batch_dag` paused or failing in Airflow? Is Docker running? Is RTE reachable? |
| Back to the clean 80/20 split | `dvc checkout --force`, then `docker compose run --rm trainer` |
| Live mode | `BATCH_SOURCE=live` in `.env`; it appends to `data/processed/` (tracked by DVC): do not `dvc push` it by accident |

## 5. Secrets

- Secrets live only in `.env` (git-ignored); `.env.example` has placeholders.
- **A key that was shown (screen, chat, log) is a used key: rotate it.**
  API key: new value in `.env` → `unset API_KEY` → `docker compose up -d
  --force-recreate api` (and `bash k8s/deploy.sh` if Kubernetes runs; the
  Streamlit service reads it from `.env` too).
- Check a value without printing it: `echo ${#API_KEY}` (length only).
- Airflow admin password: change it for the user in Airflow and in `.env`
  (the API logs in with it). After recreating the user, old browser sessions
  give a 500 → delete the rows of the `session` table in the Airflow database
  and log in again.

## 6. Scaling

- Compose: `docker compose up -d --scale api=3` (NGINX and Prometheus find
  every API container by DNS).
- Kubernetes (Docker Desktop): `bash k8s/deploy.sh` (snapshot image, Secret
  from `.env`, 3 replicas, probes on `/health`, Service on `localhost:8001`);
  `bash k8s/deploy.sh delete` removes it. Training, Airflow and MLflow stay
  in Compose.

## 7. Lessons learned (what went wrong and what we do now)

| Problem | Cause | What we do now |
|---|---|---|
| NGINX / Prometheus kept the old config; `restart` failed ("no such file or directory") | Docker Desktop/WSL keeps the old mounted file after git replaces it | `docker compose up -d --force-recreate <service>` |
| New API key → 403 | Compose takes a variable from the shell before `.env` | `unset API_KEY` before `docker compose up` |
| Airflow 500 after recreating the admin user | old sessions point to a deleted user | delete the sessions, log in again |
| validate failed: `No module named scripts.validate_batch` | Airflow read the new DAG at once, the trainer image was still old | rebuild the trainer image after a DAG change; next step: versioned images from CI, the DAG pins the tag |
| inject_batch "heartbeat timeout" after 17 h | the laptop/WSL was asleep → zombie task | Airflow retried it (try 2 = success); production needs a server |
| ruff different locally and in CI | local `mlflow/`, `airflow/` folders looked like packages | `known-first-party` / `known-third-party` in `pyproject.toml` |
| CI failed on duplicate run ids | 4 random hex characters → collisions in the same second | 8 characters, tested with 1000 ids |
| One rejected batch blocked every later batch (replay) | the gap left by the rejected days looked like missing data | rejected ranges are recorded and allowed (`data/rejected/ranges.json`) |
| RTE consolidated data empty after 31 Aug 2026 → object dtypes | change at the source | explicit types when reading; a test for it |
| Secrets in shared terminal output | `grep … .env` prints them | print only the length; rotate a shown key |
| "Orphan container" warning (streamlit) on a branch without that service | the container runs, the compose file of this branch does not know it | harmless; do not use `--remove-orphans` if you still need it |

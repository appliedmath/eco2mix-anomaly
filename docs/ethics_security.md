# Ethics, data protection and security

## Data and GDPR

- Source: the RTE éCO2mix national data, public open data published by RTE on
  the ODRÉ portal (consolidated and real-time datasets). We cite the source.
- The data is aggregated for the whole of France (consumption and RTE forecasts
  every 30 minutes). It contains no personal data and nothing at household or
  customer level, so GDPR obligations (consent, data subject rights) do not
  apply in practice.
- Data versions are kept with DVC on a private DagsHub repository; access needs
  a personal token, stored in `.dvc/config.local` (not in git).

## Security

| What | Now (project) | In production |
|---|---|---|
| Job endpoints (`POST /train`, `POST /predict`) | `X-API-Key` from `.env`: missing 401, wrong 403, not configured 500 (closed by default); compared with `secrets.compare_digest` | JWT with roles (user: predict, admin: train) |
| Entry point | NGINX is the only service with open ports; rate limit on job endpoints (429), 1 MB body limit, security headers | same, plus TLS |
| Grafana | admin password from `.env`, no sign-up, no anonymous access | SSO |
| Prometheus | UI only on 127.0.0.1 (it has no login) | behind authentication |
| MLflow | UI without login, local only | login and roles: only the trainer service account may register models and move `@champion`; the API only reads |
| API to Airflow | Airflow admin user from `.env` | service account limited to starting DAG runs (Kilyan) |
| Secrets | `.env` and `.env.bak` are git-ignored; `.env.example` holds placeholders only | a secret store (GitHub Actions secrets, Kubernetes secrets, a vault) |

## Limits of the model

- National level only: it cannot locate a regional problem.
- It flags deviations from the expected pattern, not their causes. Weekday
  public holidays are flagged on purpose; we use them to check the model.
- There is no weather input: a cold week raises the level slowly, the trend
  follows it, so it shows up as input drift rather than as an anomaly.
- Recent days come from the real-time RTE dataset, which is provisional and
  consolidated later; the reference years are consolidated data.
- Thresholds (z-score for anomalies, 0.2 for prediction drift) were chosen on
  our data and should be reviewed with operators.

## Responsible alerting

- Alerts inform people; the system takes no action on the grid.
- Drift only raises an alert and never retrains on its own, because drift can
  come from corrupted data. A new model goes live only if it passes the
  promote-or-reject gate, and every model version is traceable in MLflow.
- A batch that fails the data-quality checks is moved to `data/rejected/`, so
  bad data never reaches prediction or retraining.
- Alert fatigue is monitored: anomaly days per week (operator alert load) and
  non-holiday alarms per week.

## Environmental footprint

The detector is a small statistical model on CPU: a retraining takes about
10 to 40 seconds and a daily batch a few seconds. No GPU is needed.

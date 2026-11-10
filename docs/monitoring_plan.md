# Monitoring plan

What can go wrong once the system runs on its own, which metric shows it, which
alert fires, and what to do. Metrics come from the API's `/metrics` (scraped by
Prometheus every 15 s), from node-exporter, and from `models/drift/latest.json`,
which the `validate` and `drift` tasks of `batch_dag` write after each batch.
System alerts are Prometheus rules (`monitoring/prometheus/alert_rules.yml`);
ML alerts are Grafana alerts.

We have no daily labels, so we cannot measure accuracy every day. Input drift,
prediction drift and the rolling holiday recall are our proxies for model
performance between retrainings.

| Category | Question | Owner of the check |
|---|---|---|
| 1. System health | Is the service up and fast? | API `/metrics`, node-exporter |
| 2. Data quality | Is the new data complete, valid and on time? | `validate` task, API |
| 3. Input drift | Does the new data look like the same weeks of earlier years? | `drift` task |
| 4. Prediction drift | Does the model output look like before? | `drift` task |
| 5. Model drift | Is the model still good? | `drift` task, promote-or-reject |
| 6. Business KPI | Is it useful for the grid operator? | API |

## 1. System health

| Metric | Alert | Threshold | Severity | Action |
|---|---|---|---|---|
| `up{job="api"}` | `ApiDown` (Prometheus) | no API container answers for 1 min | critical | `docker compose ps`, API logs, restart the api service |
| request latency (p95) | dashboard | > 1 s for 5 min | warning | check CPU/memory and the NGINX rate limit; scale (`--scale api=N`) |
| 5xx error rate | dashboard | > 5 % for 5 min | warning | API logs; Airflow down gives 503 on the job endpoints |
| CPU, memory, disk (node-exporter) | dashboard | > 90 % for 10 min | warning | check running DAGs (pool `models` allows one heavy job at a time) |

No GPU: the MSTL detector is statistical and runs on CPU.

## 2. Data quality

| Metric | Alert | Threshold | Severity | Action |
|---|---|---|---|---|
| `data_quality_failed` | `DataQualityFailed` (Grafana) | = 1 | warning | read `quality_problems` in latest.json; the batch is in `data/rejected/` and predict did not run; with `BATCH_SOURCE=live` the days are fetched again on the next run |
| `data_updated_timestamp_seconds` | `NoNewData` (Prometheus) | file not updated for 36 h | warning | is `batch_dag` paused or failing? Docker running? RTE reachable? |
| `data_last_timestamp_seconds` (freshness) | dashboard | newest measured half-hour older than 2 days | warning | as above; also catches several rejected days in a row |
| `batch_rows` | dashboard | 0 on 2 runs in a row (live) | info | RTE has not published a complete new day yet |

Checks in `validate` (Phase 1 `validate()` on the new batch plus the half-hour
before it): duplicated timestamps, irregular 30-min steps, missing past values,
values outside 20 000 to 100 000 MW, too many filled half-hours per day. A
rejected batch is moved out of the processed data, so predict and retraining
never see it; a gap that is exactly an earlier rejected batch is allowed.

## 3. Input drift

| Metric | Alert | Threshold | Severity | Action |
|---|---|---|---|---|
| `input_drift_detected` | `InputDrift` (Grafana) | = 1 | warning | open the Evidently report (`report` in latest.json), compare with the weather; no automatic retraining |
| `input_drift_share` | dashboard | | | share of drifted input columns (consumption, RTE forecast) |

Window: the last 7 days (at least), against the same weeks plus/minus 14 days
of the three earlier years, so a winter week is compared with earlier winters
and a single Sunday does not look like drift. Evidently DataDriftPreset (normed
Wasserstein distance for these sample sizes); drift if at least half of the
columns drifted.

## 4. Prediction drift

| Metric | Alert | Threshold | Severity | Action |
|---|---|---|---|---|
| `prediction_drift_score` (z-score) | `PredictionDrift` (Grafana) | > 0.2 | warning | report; does input drift explain it? |
| `anomaly_rate_last_batch` | dashboard | > 5 % | warning | normal is about 1 to 2 %; a jump means unusual data or a broken model |

The z-score threshold is 0.2, not Evidently's default 0.1: a week of unchanged
z-scores already reaches about 0.1 by sampling noise.

## 5. Model drift

| Metric | Alert | Threshold | Severity | Action |
|---|---|---|---|---|
| `model_recall_rolling` | `ModelRecallLow` (Grafana) | < 0.7 | warning | the model misses known events: `POST /train`, compare in MLflow |
| `false_alarms_per_week_rolling` | `FalseAlarmsHigh` (Grafana) | > 1 | warning | check the days in `models/anomaly_days.csv` |
| `model_recall`, `model_false_alarms_per_week` | dashboard | | | measured on the reference events at each retraining |
| `model_info{version}` | dashboard annotation | change | info | a new champion is live |

Rolling recall: share of the weekday public holidays of the last 365 days that
the mstl detector flagged (a holiday on a weekend does not change consumption).
Rolling false alarms: flagged days of the last 28 days that are not holidays,
per week (an upper bound: some can be real events). Every retraining also passes
the promote-or-reject gate: recall must not drop and false alarms per week may
rise by at most 10 %.

## 6. Business KPI

| Metric | Alert | Threshold | Severity | Action |
|---|---|---|---|---|
| `anomaly_days_last_7d{detector="mstl"}` | `AlertLoadHigh` (Grafana) | > 3 | warning | operator alert load too high (alert fatigue): review the threshold |
| `anomalies_last_7d{detector}` | dashboard | | | flagged half-hours in the last 7 days |

In production we would also track how many alerts the operator confirmed
(needs human feedback).

## Drift raises alerts, it does not retrain

Drift can come from the weather or from corrupted data, and retraining on
corrupted data makes the model worse, so drift only raises an alert. Example:
in the first week of November 2026 consumption was about 10 % higher than in the
same weeks of 2023 to 2025 (input drift), while the model found all 9 weekday
holidays of the last year and raised 0.25 non-holiday alarms per week. The world
changed, the model did not break, and no retraining was needed.

Retraining is weekly (`retrain_dag`) or on demand (`POST /train`); a new model
goes live only if it passes the promote-or-reject gate.

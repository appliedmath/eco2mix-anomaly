"""Defense presentation (20 min): one page per step of the data flow, with links to the live tools.

Run: streamlit run streamlit_app/presentation.py --server.port 8502
"""
import base64
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="eco2mix-anomaly: defense", page_icon="⚡", layout="wide")

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

TOOLS = {
    "GitHub": "https://github.com/appliedmath/eco2mix-anomaly",
    "DagsHub": "https://dagshub.com/appliedmath/eco2mix-anomaly",
    # deep links: each button opens the exact page we talk about
    "MLflow": "http://localhost:5000/#/models/grid-anomaly-detector",
    "API docs": "http://localhost:8000/docs",
    "Airflow": "http://localhost:8080/dags/batch_dag/grid",
    "Grafana": "http://localhost:3000/d/eco2mix-stack",
    "Prometheus": "http://localhost:9090/alerts",
    "Operator app": "http://localhost:8501",
}


ASSETS = Path(__file__).parent / "assets"

COLORS = {
    "blue": "#1f6feb", "green": "#1a7f37", "orange": "#d4760a",
    "red": "#cf222e", "purple": "#8250df", "teal": "#0e8a8a",
}

# Styles for the cards, flows and closing sentences (only used by the pages that call the helpers below).
st.markdown(
    """<style>
.e2m-card {border-radius: 14px; padding: 18px 20px; height: 100%;
  border-left: 6px solid var(--c); background: color-mix(in srgb, var(--c) 10%, transparent);}
.e2m-value {font-size: 2.4rem; font-weight: 800; color: var(--c); line-height: 1.1;}
.e2m-title {font-size: 1.3rem; font-weight: 800; color: var(--c);}
.e2m-label {font-size: 1.05rem; font-weight: 600; margin-top: 6px;}
.e2m-sub {font-size: 0.92rem; opacity: 0.8; margin-top: 4px;}
.e2m-flow {display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin: 6px 0 18px;}
.e2m-step {padding: 9px 16px; border-radius: 999px; font-weight: 650;
  border: 2px solid var(--c); background: color-mix(in srgb, var(--c) 12%, transparent);}
.e2m-arrow {font-size: 1.3rem; opacity: 0.6;}
.e2m-lead {font-size: 1.5rem; font-weight: 650; margin: 4px 0 14px;}
.e2m-take {margin-top: 26px; padding: 16px 22px; border-radius: 14px; font-size: 1.3rem;
  font-weight: 700; color: #ffffff; background: linear-gradient(90deg, #1f6feb, #8250df);}
</style>""",
    unsafe_allow_html=True,
)


def screenshot(name: str, caption: str) -> None:
    """Backup screenshot (streamlit_app/assets/<name>), shown only if the file exists."""
    path = ASSETS / name
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def header(title: str, owner: str, minutes: int, tools=()) -> None:
    st.title(title)
    st.caption(f"{owner} · {minutes} min")
    if tools:
        cols = st.columns(len(tools))
        for col, name in zip(cols, tools):
            col.link_button(f"Open {name}", TOOLS[name], width="stretch")


def html(text: str, where=st) -> None:
    where.markdown(text, unsafe_allow_html=True)


def cards(items) -> None:
    """A row of big-number cards: (value, label, small text, color)."""
    for col, (value, label, sub, color) in zip(st.columns(len(items)), items):
        html(f'<div class="e2m-card" style="--c:{COLORS[color]}"><div class="e2m-value">{value}</div>'
             f'<div class="e2m-label">{label}</div><div class="e2m-sub">{sub}</div></div>', col)


def boxes(items) -> None:
    """A row of colored boxes: (title, lines, color); lines are joined with line breaks."""
    for col, (title, lines, color) in zip(st.columns(len(items)), items):
        body = "<br>".join(lines)
        html(f'<div class="e2m-card" style="--c:{COLORS[color]}"><div class="e2m-title">{title}</div>'
             f'<div class="e2m-sub" style="font-size:1rem">{body}</div></div>', col)


def flow(steps, color="blue", where=st) -> None:
    parts = []
    for i, step in enumerate(steps):
        if i:
            parts.append('<span class="e2m-arrow">→</span>')
        parts.append(f'<span class="e2m-step" style="--c:{COLORS[color]}">{step}</span>')
    html(f'<div class="e2m-flow">{"".join(parts)}</div>', where)


def lead(text: str) -> None:
    html(f'<div class="e2m-lead">{text}</div>')


def takeaway(text: str) -> None:
    html(f'<div class="e2m-take">{text}</div>')


def intro():
    st.title("Anomaly detection on the French electricity grid")
    st.markdown("#### Fatemeh (Shari) Tayebi & Mahboubeh (Mina) Nadaf")
    st.markdown("Mentor: Kilyan Poilly · MLOps, November 2026")
    st.caption("Mina & Shari · 2 min")
    lead("Every night, our system reads France's electricity consumption "
         "and flags the half-hours that do not look normal.")
    try:
        rows = requests.get(f"{API_URL}/anomalies", timeout=15, params={
            "start": "2026-11-05", "end": "2026-11-11", "detector": "mstl", "only_anomalies": "false"}).json()
        week = pd.DataFrame(rows)
        week["time"] = pd.to_datetime(week["timestamp_utc"], utc=True).dt.tz_convert("Europe/Paris")
        flagged = week[week["is_anomaly"]]
        fig = go.Figure()
        fig.add_scatter(x=week["time"], y=week["consumption_mw"], name="Consumption")
        fig.add_scatter(x=week["time"], y=week["expected_mw"], name="Expected by the model",
                        line={"dash": "dot", "color": "gray"})
        fig.add_scatter(x=flagged["time"], y=flagged["consumption_mw"], mode="markers", name="Anomaly",
                        marker={"color": "red", "size": 8})
        peak = flagged.loc[flagged["consumption_mw"].idxmax()]
        fig.add_annotation(x=peak["time"], y=peak["consumption_mw"],
                           text="11 November: Armistice Day,<br>flagged the same night",
                           showarrow=True, arrowhead=2, ax=-150, ay=-45,
                           bgcolor="white", bordercolor="red", borderpad=4)
        fig.update_layout(height=360, yaxis_title="MW", margin={"t": 50, "b": 10},
                          title="5-11 November 2026, read live from our API",
                          legend={"orientation": "h", "y": -0.15})
        st.plotly_chart(fig, width="stretch")
    except (requests.RequestException, KeyError, ValueError):
        st.info("Start the stack to see the live chart (docker compose up -d).")
    flow(["RTE data", "quality check", "train", "version", "serve", "monitor"])
    cards([
        ("1", "Data & model", "scoping, MSTL", "blue"),
        ("2", "Docker & MLflow", "tracked, versioned", "purple"),
        ("3", "Airflow & API", "automated, secured", "teal"),
        ("4", "Monitoring", "drift, live data, alerts", "orange"),
    ])
    takeaway("Not just a model: a system that runs every night and checks itself.")


def data():
    header("Data", "Shari", 1, ["DagsHub"])
    cards([
        ("30 min", "national consumption", "RTE éCO2mix, in UTC", "blue"),
        ("≈ 67,600", "half-hours", "Jan 2023 → Nov 2026", "purple"),
        ("49", "reference events", "holidays, clock changes", "orange"),
        ("80 / 20", "train / replay", "20 % arrive as new days", "green"),
    ])
    st.write("")
    flow(["RTE éCO2mix", "raw files", "clean 30-min grid", "DVC → DagsHub"], "teal")
    boxes([
        ("Two sources", ["consolidated history until 31 Aug", "real-time data from 1 Sep"], "blue"),
        ("Cleaning", ["clock changes → UTC", "duplicates dropped", "gaps ≤ 2 h filled and flagged"], "purple"),
        ("Two modes", ["replay: repeatable demo", "live: the RTE API every night"], "green"),
    ])
    takeaway("Clean, versioned data first: every later step can trust it.")
    with st.expander("Speaker notes"):
        st.markdown("The consolidated dataset ends on 31 August; after that only real-time data exists. "
                    "That detail caused the bug we found with the first nightly live run (page 7).")


def architecture():
    # this page only: less empty space at the top, so diagram, boxes and the closing sentence fit on one screen
    html("<style>.block-container {padding-top: 1.2rem !important; padding-bottom: 1rem !important;}"
         " h1 {padding-bottom: 0.2rem !important;}"
         " .e2m-card {padding: 12px 16px !important;} .e2m-title {font-size: 1.15rem !important;}"
         " .e2m-take {margin-top: 14px !important; padding: 12px 22px !important;}</style>")
    header("Architecture", "Mina", 2)
    st.caption("One night, from left to right: RTE data in, scored half-hours and alerts out.")
    # left to right, one colour per column; rendered from streamlit_app/assets/architecture.dot
    st.image(str(ASSETS / "architecture.png"), width="stretch")
    boxes([
        ("One trainer image", ["every job, same pinned code", "inject · validate · predict · drift · train"], "purple"),
        ("The API never trains", ["serves the champion's results", "starts jobs in Airflow"], "teal"),
        ("One door: NGINX", ["only service with ports", "API key, rate limits"], "blue"),
        ("Everything as code", ["DAGs, alerts, dashboards", "reviewed in pull requests"], "green"),
    ])
    takeaway("Simple on purpose: one machine, one door, one image for every job.")


def github_ci():
    header("Code, CI and data versioning", "Mina & Shari", 1, ["GitHub", "DagsHub"])
    flow(["branch", "pull request", "CI green", "review", "main"], "green")
    boxes([
        ("Code: GitHub", ["every change is a pull request", "main is protected: green CI + 1 approval",
                          "we review each other's work"], "blue"),
        ("CI: GitHub Actions", ["test: ruff + unit tests on pinned versions",
                                "docker-build: api, trainer and airflow images"], "purple"),
        ("Data: DVC + DagsHub", ["git keeps small receipts (data/*.dvc)", "the data itself is on DagsHub",
                                 "dvc pull: the same data for everyone"], "teal"),
    ])
    takeaway("Nothing reaches main without green CI and a second pair of eyes.")


def model_mlflow():
    header("Model and MLflow", "Shari", 2, ["MLflow"])
    flow(["consumption", "MSTL: trend + daily + weekly cycles", "residual", "robust z-score",
          "|z| > 4 → anomaly"], "purple")
    cards([
        ("K = 4", "threshold", "chosen from K = 2 to 6 in MLflow", "purple"),
        ("0.72", "recall", "on the 49 reference events", "green"),
        ("0.44", "false alarms / week", "less than one every two weeks", "orange"),
    ])
    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "**Three detectors, same events, K = 4** (MLflow run *detector-comparison*)\n\n"
            "| Detector | Recall | False alarms / week |\n|---|---|---|\n"
            "| **MSTL (ours)** | **0.72** | **0.44** |\n"
            "| Same half-hour last week | 0.20 | 0.45 |\n"
            "| RTE day-ahead forecast | 0.08 | 0.11 |"
        )
    with right:
        boxes([("Promote-or-reject gate", [
            "recall not lower", "false alarms at most +10 %",
            "live model = alias <b>@champion</b>", "rollback = move the alias"], "blue")])
    screenshot("mlflow_compare.png", "MLflow: the detector comparison")
    takeaway("A new model goes live only if it is not worse, and going back is one alias move.")
    with st.expander("Speaker notes"):
        st.markdown("Show in MLflow: the champion version, and a run with decision = rejected (k = 2): "
                    "logged with the reason, but no registry version, so a worse model cannot go live.")


def api():
    header("API, security and ethics", "Mina", 2, ["API docs"])
    flow(["client · operator app", "NGINX: rate limit", "FastAPI: API key", "Airflow: 202 + run id"], "teal")
    boxes([
        ("Read (open)", ["/health · /model-info", "/anomalies · /series", "/anomaly-day",
                         "/champion: registry vs in use"], "blue"),
        ("Jobs (API key)", ["POST /train → retrain_dag", "POST /predict → batch_dag",
                            "GET /runs/{dag}/{run}: status"], "purple"),
        ("Clear answers", ["401 no key · 403 wrong key", "429 too many requests", "409 DAG paused · 503 Airflow down",
                           "/metrics closed from outside"], "orange"),
    ])
    st.write("")
    boxes([
        ("Security", ["one door: NGINX", "admin tools only on this machine", "secrets only in .env"], "red"),
        ("Data", ["public RTE open data", "national total, no personal data", "GDPR: nothing personal to protect"],
         "teal"),
        ("Ethics", ["an anomaly is an alert for a person", "the system never acts on the grid",
                    "limits documented (no weather, national only)"], "green"),
    ])
    takeaway("Detecting is not deciding: the key protects every change, and a person decides.")


def airflow():
    header("Orchestration with Airflow", "Shari", 2, ["Airflow"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### batch_dag · every night, 00:00 UTC")
        flow(["inject_batch", "validate", "predict", "drift"], "blue")
        st.markdown("New days in, bad data out, the champion scores, drift is measured. **Never trains.**")
    with right:
        st.markdown("#### retrain_dag · weekly, or on demand")
        flow(["train", "gate", "champion or rejected"], "purple")
        st.markdown("A new model on all processed data. **On demand:** `POST /train`, from the app or the docs.")
    st.write("")
    cards([
        ("5 Nov", "first live runs", "real RTE data every night", "green"),
        ("8 Nov", "one red run", "bug found, test added, fixed", "red"),
        ("11 Nov", "Armistice Day", "flagged the same night", "orange"),
    ])
    screenshot("airflow_grid.png", "Airflow: batch_dag runs, the red run of 8 November")
    takeaway("It runs by itself every night, and bad input stops before it reaches the model.")
    with st.expander("Speaker notes"):
        st.markdown("Each task runs in a fresh container from the trainer image (DockerOperator). "
                    "In Airflow: Grid view, green nightly runs; the red run of 8 November with the orange "
                    "upstream_failed tasks: bad input stops the pipeline, nothing wrong goes further.")


def question(col, ask, answer, value, sub, color) -> None:
    html(f'<div class="e2m-card" style="--c:{COLORS[color]}"><div class="e2m-title">{ask}</div>'
         f'<div class="e2m-sub" style="font-size:1rem">{answer}</div>'
         f'<div class="e2m-value" style="margin-top:10px">{value}</div><div class="e2m-sub">{sub}</div></div>', col)


def drift():
    header("Data quality and drift", "Shari", 2)
    st.caption("Window: the last 7 days vs. the same weeks (± 2 weeks) of 2023-2025, "
               "so a winter week is compared with earlier winters.")
    c1, c2 = st.columns(2)
    question(c1, "1 · Is the data broken?", "gaps, duplicated hours, impossible values → batch rejected",
             "rejected", "the model never sees it", "red")
    question(c2, "2 · Has the world changed?", "input drift: consumption and RTE forecast (Evidently)",
             "+10 %", "first week of November vs. 2023-25", "orange")
    st.write("")
    c3, c4 = st.columns(2)
    question(c3, "3 · Does the model react differently?", "prediction drift: the z-scores, threshold 0.2",
             "0.231", "10 Nov: early warning", "purple")
    question(c4, "4 · Is the model still right?", "no labels: rolling recall on weekday holidays, "
             "false alarms per week", "1.0 · 0.25", "9 / 9 holidays · no real problem", "green")
    takeaway("The world changed, the model did not break: drift alerts a person, it never retrains.")
    reports = sorted(Path("models/drift/reports").glob("drift_*.html"))
    if reports:
        with st.expander(f"Evidently report: {reports[-1].stem.removeprefix('drift_')} (latest)"):
            components.html(reports[-1].read_text(), height=900, scrolling=True)
    with st.expander("Speaker notes"):
        st.markdown("Open the Evidently report of 10 November: reference left, current right, consumption moved right. "
                    "The z-score threshold is also passed to Evidently, so the report agrees with the status file.")


def monitoring():
    header("Monitoring: Prometheus and Grafana", "Mina", 3, ["Grafana", "Prometheus"])
    flow(["API /metrics", "Prometheus every 15 s", "4 dashboards", "9 alerts"], "orange")
    cards([
        ("0", "Stack health", "targets, champion, data age", "green"),
        ("1", "API performance", "requests, errors, P95", "blue"),
        ("2", "Model & data", "quality, drift, rolling recall", "purple"),
        ("3", "Infrastructure", "CPU, RAM, disk", "teal"),
    ])
    st.write("")
    boxes([
        ("System alerts (Prometheus)", ["ApiDown: no API for 1 min", "NoNewData: newest half-hour",
                                        "unchanged for 36 h"], "red"),
        ("ML alerts (Grafana)", ["DataQualityFailed · InputDrift · PredictionDrift",
                                 "ModelRecallLow · FalseAlarmsHigh",
                                 "AlertLoadHigh · NewChampion"], "orange"),
        ("Real story, 10 Nov", ["NoNewData fired, data age red", "cause: batch_dag paused",
                                "switched on → alert cleared by itself"], "green"),
    ])
    takeaway("If something breaks or changes, a person knows; drift alone never retrains the model.")
    with st.expander("Backup screenshots"):
        screenshot("grafana_alerts.png", "Grafana: the 9 alert rules, provisioned from files")
        screenshot("grafana_model_data.png", "Grafana: model & data dashboard")


def live_demo():
    header("Live demo: the operator app", "Shari", 2, ["Operator app"])
    flow(["operator app", "NGINX", "API", "Airflow · MLflow"], "green")
    boxes([
        ("1 · User tab", ["consumption, expected line, anomalies", "was this day an anomaly? (14 July 2026)",
                          "the model in use"], "blue"),
        ("2 · Admin tab", ["Retrain the model", "API key → Airflow → trainer → gate",
                           "the run status comes back"], "purple"),
    ])
    takeaway("The operator only talks to the API: one door, one key.")
    with st.expander("Backup screenshots"):
        screenshot("operator_app_chart.png", "Operator app: the chart")
        screenshot("operator_app_day.png", "Operator app: was this day an anomaly?")
    with st.expander("Speaker notes"):
        st.markdown("On the 14-day chart: around 2-3 November the whole curve jumps up and the expected line "
                    "follows it, so no anomaly, but input drift against earlier years.")


def scaling():
    header("Scaling and Kubernetes", "Mina", 1)
    flow(["more users", "more API replicas", "readiness on /health", "no downtime"], "blue")
    boxes([
        ("Docker Compose", ["--scale api=3", "NGINX spreads the requests", "Prometheus scrapes each one"], "teal"),
        ("Kubernetes (API only)", ["3 replicas, rolling update", "readiness + liveness on /health",
                                   "a deleted pod comes back at once"], "purple"),
    ])
    takeaway("The API can grow with its users; the rest moves to Kubernetes only when needed (future work).")
    with st.expander("Backup screenshot"):
        screenshot("k8s_pods.png", "kubectl: 3 pods, then self-healing after a delete")


def future():
    header("Future work", "Mina & Shari", 2)
    ds, ops = st.tabs(["With more time: data science", "With more time: MLOps"])
    with ds:
        cards([
            ("🌡️", "Weather data", "+ closer expected line · ? test it through the gate", "orange"),
            ("🗺️", "Regional data", "+ where and why · ? 12 series, GDPR closer", "teal"),
            ("🏷️", "Operator feedback", "+ real labels · ? operators' time", "green"),
            ("🧠", "Other models", "+ maybe better · ? less explainable", "purple"),
        ])
    with ops:
        st.markdown(
            "| Idea | Benefit | Challenge |\n|---|---|---|\n"
            "| **Versioned images** pushed to a registry in CI | deploy by tag, rollback in one step; the DAG pins the "
            "image tag (no DAG/image mismatch) | registry credentials in CI; tag discipline; storage |\n"
            "| **Alerts that notify** (Slack / e-mail) | the maintainer knows without opening a dashboard | contact-point "
            "secrets outside the repo; tuning against alert fatigue |\n"
            "| **Automatic retraining** when rolling recall drops | faster reaction to a real change | never retrain on bad "
            "data or a data bug: keep the gate and a human approval |\n"
            "| **Dev vs. prod** (bind mounts, Compose profiles, multi-stage images) | fast development without rebuilds; "
            "immutable images in production | two configurations to keep in sync |\n"
            "| **Security for real users** (login, one key per user, roles, HTTPS, secrets vault) | safe access for many "
            "users | a user database, certificates, more components to run |\n"
            "| **Storage**: S3 / a database instead of shared folders | several machines, more data | migration, "
            "credentials, cost |\n"
            "| **Kubernetes for the whole stack** (when there are many users) | autoscaling, self-healing everywhere | "
            "complexity and cost; Airflow and MLflow on Kubernetes |"
        )
    takeaway("From a reliable prototype to a useful service.")
    with st.expander("Speaker notes"):
        st.markdown(
            "Kilyan: the defense is about choices: what we built and why, what we would do with more time, "
            "and what the project looks like in 5 years. Shari: data-science tab (weather: an idea to test, "
            "not a sure win). Mina: MLOps tab. Pick 2 rows per tab, not all.\n\n"
            "Regional data, if asked why not now: \"We first wanted the whole MLOps loop working on one series. "
            "Going regional multiplies models, runs and dashboards by 12, so it's a scaling step we'd do next, "
            "with dynamic task mapping in Airflow.\"\n\n"
            "End with: a project is never finished, it is maintained."
        )


def thanks():
    photo = ASSETS / "thanks.jpg"
    background = "linear-gradient(135deg, #1f6feb, #8250df)"
    if photo.exists():
        data = base64.b64encode(photo.read_bytes()).decode()
        background = (f"linear-gradient(rgba(10, 20, 40, 0.35), rgba(10, 20, 40, 0.75)), "
                      f"url(data:image/jpeg;base64,{data}) center / cover")
    html(f'<div style="background:{background}; border-radius: 22px; min-height: 78vh; display: flex; '
         'flex-direction: column; justify-content: flex-end; padding: 48px 56px; color: #ffffff;">'
         '<div style="font-size: 4.2rem; font-weight: 800; line-height: 1.05;">Thank you<br>for your attention</div>'
         '<div style="font-size: 1.3rem; margin-top: 18px; opacity: 0.9;">'
         "Fatemeh (Shari) Tayebi & Mahboubeh (Mina) Nadaf · eco2mix-anomaly</div></div>")
    if photo.exists():
        st.caption("Photo: high-voltage line between Avelin and Tourmignies (Nord, France), "
                   "Pierre André Leclercq, CC BY-SA 4.0, Wikimedia Commons")


pages = [st.Page(f, title=t, url_path=f.__name__) for f, t in [
    (intro, "1. Introduction"), (data, "2. Data"), (architecture, "3. Architecture"),
    (github_ci, "4. Code, CI, DVC"), (model_mlflow, "5. Model & MLflow"), (api, "6. API, security, ethics"),
    (airflow, "7. Airflow"), (drift, "8. Quality & drift"), (monitoring, "9. Monitoring"),
    (live_demo, "10. Live demo"), (scaling, "11. Scaling"), (future, "12. Future work"),
    (thanks, "13. Thank you"),
]]
st.navigation(pages).run()
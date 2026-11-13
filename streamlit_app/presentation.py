"""Defense presentation (20 min): one page per step of the data flow, with links to the live tools.

Run: streamlit run streamlit_app/presentation.py --server.port 8502
"""
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
    "MLflow": "http://localhost:5000",
    "API docs": "http://localhost:8000/docs",
    "Airflow": "http://localhost:8080",
    "Grafana": "http://localhost:3000",
    "Prometheus": "http://localhost:9090",
    "Operator app": "http://localhost:8501",
}


def header(title: str, owner: str, minutes: int, tools=()) -> None:
    st.title(title)
    st.caption(f"{owner} · {minutes} min")
    if tools:
        cols = st.columns(len(tools))
        for col, name in zip(cols, tools):
            col.link_button(f"Open {name}", TOOLS[name], use_container_width=True)


def intro():
    st.title("Anomaly detection on the French electricity grid")
    st.markdown("#### ⚡️ Fatemeh Tayebi & Mina Nadaf")
    st.markdown("Mentor: Kilyan Poilly · MLOps, November 2026")
    st.caption("Mina & Shari · 2 min")
    st.markdown(
        "**Problem.** RTE publishes French national electricity consumption every 30 minutes. "
        "An unusual half-hour can mean a real event, a forecast error or bad data, and nobody can watch it all day.\n\n"
        "**Goal.** Flag abnormal half-hours automatically, and run the whole lifecycle as an MLOps system: "
        "new data every night, safe retraining, and monitoring of the data and the model."
    )
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
                           text="11 November: Armistice Day, a holiday on a Wednesday,<br>flagged by the nightly run",
                           showarrow=True, arrowhead=2, ax=-150, ay=-45,
                           bgcolor="white", bordercolor="red", borderpad=4)
        fig.update_layout(height=380, yaxis_title="MW", margin={"t": 50, "b": 10},
                          title="French national consumption, 5-11 November 2026 (scored by our model, read from our API)",
                          legend={"orientation": "h", "y": -0.15})
        st.plotly_chart(fig, use_container_width=True)
    except (requests.RequestException, KeyError, ValueError):
        st.info("Start the stack to see the live chart (docker compose up -d).")
    st.graphviz_chart(
        "digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica]; "
        '"RTE data" -> "quality check" -> "MSTL model" -> "anomalies" -> "API" -> "operator app & Grafana" }'
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Phase 1", "Data, model", "scoping, MSTL")
    c2.metric("Phase 2", "Docker, MLflow", "tracked, versioned (DVC)")
    c3.metric("Phase 3", "Airflow, API", "automated, secured, CI")
    c4.metric("Phase 4", "Monitoring", "drift, live data, alerts")


def data():
    header("Data", "Shari", 1, ["DagsHub"])
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "- **RTE éCO2mix national**: consolidated history + real-time data, no API key\n"
            "- **30-minute grid in UTC** from 1 Jan 2023: clock changes fixed, short gaps filled and marked\n"
            "- **80/20 split**: 80 % to train, 20 % held back and *replayed* as new days (repeatable demo)\n"
            "- **Live mode** (team extension): every night the new complete days come from the RTE API\n"
            "- All data versioned with **DVC**, stored on **DagsHub**"
        )
    with right:
        st.metric("Measured half-hours", "≈ 67,600", "Jan 2023 → Nov 2026")
        st.metric("Reference events", "49", "public holidays, clock changes")
    with st.expander("Speaker notes"):
        st.markdown("The consolidated dataset ends on 31 August; after that only real-time data exists. "
                    "That detail caused the bug we found with the first nightly live run (page 7).")


def architecture():
    header("Architecture", "Mina", 2)
    st.markdown("- Diagram (Mina)")


def github_ci():
    header("Code, CI and data versioning", "Mina & Shari", 1, ["GitHub", "DagsHub"])
    left, right = st.columns(2)
    with left:
        st.markdown(
            "**Code: GitHub**\n"
            "- Every change is a **pull request**; `main` is protected: green CI + 1 approval\n"
            "- We review each other's PRs (comments, fixes, reproductions)\n\n"
            "**CI: GitHub Actions**, on every PR and push to `main`\n"
            "- `test`: ruff + all unit tests on pinned versions\n"
            "- `docker-build`: builds the api, trainer and airflow images"
        )
    with right:
        st.markdown(
            "**Data: DVC + DagsHub**\n"
            "- Git keeps small *receipts* (`data/*.dvc`), the data itself is on DagsHub\n"
            "- `dvc pull` gives everyone the same data; `dvc checkout` resets it\n"
            "- Live data changes the processed file: never `dvc push` it by accident"
        )


def model_mlflow():
    header("Model and MLflow", "Shari", 2, ["MLflow"])
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            "- **MSTL** splits consumption into trend + daily + weekly cycles + **residual**\n"
            "- **Anomaly** = a residual far from usual: robust z-score, flag when **|z| > K = 4**\n"
            "- Two baselines: RTE's own forecast, same half-hour last week\n"
            "- Every training is an **MLflow run**; a new model goes live only through the "
            "**promote-or-reject gate**: recall not lower, false alarms at most +10 %\n"
            "- Live model = registry alias **champion**, loaded by the API and the batch job"
        )
    with right:
        st.metric("Recall on reference events (mstl)", "≈ 0.71-0.75")
        st.metric("Recall on public holidays", "≈ 0.85")
        st.metric("False alarms per week", "≈ 0.45")
    with st.expander("Speaker notes"):
        st.markdown("Show in MLflow: the champion version, and a run with decision = rejected (k = 2): "
                    "logged with the reason, but no registry version, so a worse model cannot go live.")


def api():
    header("API and NGINX", "Mina", 2, ["API docs"])
    st.markdown("- Endpoints\n- API key, rate limit\n- Single entry point")


def airflow():
    header("Orchestration with Airflow", "Shari", 2, ["Airflow"])
    left, right = st.columns(2)
    with left:
        st.markdown("**batch_dag** (every night, 00:00 UTC)")
        st.code("inject_batch  →  validate  →  predict  →  drift", language=None)
        st.markdown(
            "- **inject_batch**: new complete days (live RTE API, or replay)\n"
            "- **validate**: data-quality gate; a failed batch is moved to `data/rejected/`, predict does not run\n"
            "- **predict**: the champion scores the new half-hours\n"
            "- **drift**: Evidently report + status file for Prometheus/Grafana\n"
            "- never trains; the new days stay in the processed data"
        )
    with right:
        st.markdown("**retrain_dag** (every week, or on demand: `POST /train`)")
        st.code("train  (promote-or-reject)", language=None)
        st.markdown(
            "- **train**: a new model on all processed data collected so far\n"
            "- **gate**: replaces the champion only if it is not worse (recall, false alarms)\n"
            "- every training is an MLflow run, promoted or rejected\n"
            "- on demand from the operator app, the API docs or curl, through the API"
        )
    st.markdown("Each task runs in a fresh container from the trainer image (DockerOperator).")
    st.info("Real nightly runs with live data since 5 November. The first one-day run failed in inject_batch "
            "(the consolidated RTE data is empty after 31 August); found, reproduced, fixed with a test.")
    with st.expander("Speaker notes"):
        st.markdown("In Airflow: Grid view, green nightly runs; the red run of 8 November with the orange "
                    "upstream_failed tasks: bad input stops the pipeline, nothing wrong goes further.")


def drift():
    header("Data quality and drift", "Shari", 2)
    st.markdown(
        "- **Window**: the last 7 days vs. the **same weeks ± 2 weeks of 2023-2025** "
        "(a winter week is compared with earlier winters; a Sunday is not drift)\n"
        "- **Input drift**: consumption and RTE forecast (Evidently, Wasserstein)\n"
        "- **Prediction drift**: the model's z-scores, threshold 0.2\n"
        "- **Model drift without labels**: rolling recall on weekday public holidays (365 days), "
        "non-holiday alarm days per week (28 days)\n"
        "- **Drift only raises an alert**, it never retrains: drift can come from bad data"
    )
    st.markdown("**Real example, 2-10 November 2026**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Consumption vs. 2023-25", "+10 %", "input drift")
    c2.metric("Prediction drift (10 Nov)", "0.231", "> 0.2: early warning")
    c3.metric("Rolling recall", "1.0", "9 / 9 holidays")
    c4.metric("False alarms / week", "0.25", "no anomaly")
    st.success("The world changed (a colder period), the model did not break: alert a person, no retraining.")
    reports = sorted(Path("models/drift/reports").glob("drift_*.html"))
    if reports:
        with st.expander(f"Evidently report: {reports[-1].stem.removeprefix('drift_')} (latest)"):
            components.html(reports[-1].read_text(), height=900, scrolling=True)
    with st.expander("Speaker notes"):
        st.markdown("Open the Evidently report of 10 November: reference left, current right, consumption moved right. "
                    "The z-score threshold is also passed to Evidently, so the report agrees with the status file.")


def monitoring():
    header("Monitoring: Prometheus and Grafana", "Mina", 3, ["Grafana", "Prometheus"])
    st.markdown("- Dashboards\n- 9 alerts\n- NoNewData story")


def live_demo():
    header("Live demo: the operator app", "Shari", 2, ["Operator app"])
    st.markdown(
        "The operator app **talks only to the API** (through NGINX), never to files, Airflow or MLflow.\n\n"
        "1. **User tab**: consumption of the last days with the model's expected line and anomalies; "
        "*was this day an anomaly?* (try 14 July 2026); the model in use\n"
        "2. **Admin tab**: *Retrain the model* → API with the key → Airflow → trainer → MLflow gate; "
        "the run status comes back to the app"
    )
    with st.expander("Speaker notes"):
        st.markdown("On the 14-day chart: around 2-3 November the whole curve jumps up and the expected line "
                    "follows it, so no anomaly, but input drift against earlier years.")


def scaling():
    header("Scaling and Kubernetes", "Mina", 1)
    st.markdown("- --scale api=3\n- Kubernetes (Mina)")


def future():
    header("Future work", "Mina & Shari", 1)
    st.markdown(
        "- **Temperature** as an input: explains input drift, and a more precise expected consumption\n"
        "- **More reference events** (cold snaps, strikes, outages) and operator feedback on alerts\n"
        "- **MLOps** (Mina)"
    )


pages = [st.Page(f, title=t, url_path=f.__name__) for f, t in [
    (intro, "1. Introduction"), (data, "2. Data"), (architecture, "3. Architecture"),
    (github_ci, "4. Code, CI, DVC"), (model_mlflow, "5. Model & MLflow"), (api, "6. API & NGINX"),
    (airflow, "7. Airflow"), (drift, "8. Quality & drift"), (monitoring, "9. Monitoring"),
    (live_demo, "10. Live demo"), (scaling, "11. Scaling"), (future, "12. Future work"),
]]
st.navigation(pages).run()

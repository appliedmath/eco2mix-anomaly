"""Operator app (M4): consumption, anomalies and the model in use.

It talks only to the API (through NGINX), never to the files.
Run: streamlit run streamlit_app/app.py   (API_URL defaults to http://localhost:8000)
"""
import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
DETECTOR = "mstl"
PARIS = "Europe/Paris"

st.set_page_config(page_title="Grid anomalies", page_icon="⚡", layout="wide")


@st.cache_data(ttl=60)
def get(path: str, **params):
    response = requests.get(f"{API_URL}{path}", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


def api_or_stop(path: str, **params):
    try:
        return get(path, **params)
    except requests.RequestException as exc:
        st.error(f"The API is not reachable ({API_URL}{path}). Is the stack running? ({exc})")
        st.stop()


def paris(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert(PARIS)


st.title("⚡ French electricity grid: consumption anomalies")
health = api_or_stop("/health")
newest = paris(health["newest_data"])
st.caption(f"Model {health['model_version']} · newest data {newest:%d %b %Y, %H:%M} (Paris time)")

user_tab, admin_tab = st.tabs(["User", "Admin"])

with user_tab:
    # 1. Consumption with anomalies
    st.subheader("Consumption and anomalies")
    n_days = st.selectbox("Period", [7, 14, 30, 90], index=1, format_func=lambda d: f"last {d} days")
    end = newest.date()
    start = end - timedelta(days=n_days - 1)
    df = pd.DataFrame(api_or_stop("/anomalies", start=start.isoformat(), end=end.isoformat(), detector=DETECTOR,
                                  only_anomalies="false"))

    if df.empty:
        st.info("No scored data in this period.")
    else:
        df["time"] = pd.to_datetime(df["timestamp_utc"], utc=True).dt.tz_convert(PARIS)
        flagged = df[df["is_anomaly"]]
        fig = go.Figure()
        fig.add_scatter(x=df["time"], y=df["consumption_mw"], name="Consumption")
        fig.add_scatter(x=df["time"], y=df["expected_mw"], name="Expected by the model",
                        line={"dash": "dot", "color": "gray"})
        fig.add_scatter(x=flagged["time"], y=flagged["consumption_mw"], mode="markers", name="Anomaly",
                        marker={"color": "red", "size": 8})
        fig.update_layout(height=420, yaxis_title="MW", margin={"t": 10, "b": 10},
                          legend={"orientation": "h", "y": 1.1})
        st.plotly_chart(fig, width="stretch")

        left, right = st.columns(2)
        left.metric("Flagged half-hours", len(flagged))
        right.metric("Days with an alert", flagged["time"].dt.date.nunique())
        if not flagged.empty:
            by_day = (flagged.assign(day=flagged["time"].dt.date, abs_z=flagged["z_score"].abs())
                      .groupby("day")
                      .agg(half_hours=("abs_z", "size"), max_abs_z=("abs_z", "max"), holiday=("is_holiday", "any"))
                      .round(2).sort_index(ascending=False))
            st.dataframe(by_day, width="stretch")

    # 2. Was this day an anomaly?
    st.subheader("Was this day an anomaly?")
    day = st.date_input("Day (Paris)", value=end, min_value=date(2023, 1, 1), max_value=end)
    try:
        answer = get("/anomaly-day", day=day.isoformat(), detector=DETECTOR)
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            st.info("No data for this day.")
        else:
            st.error(f"API error: {exc}")
        answer = None
    except requests.RequestException as exc:
        st.error(f"The API is not reachable: {exc}")
        answer = None

    if answer is not None:
        holiday = " It is a public holiday." if answer["is_holiday"] else ""
        if answer["was_anomaly"]:
            st.error(f"Yes: {answer['n_points']} half-hours flagged in {answer['n_episodes']} episode(s), "
                     f"strongest |z| = {answer['max_abs_z']} at {paris(answer['peak_utc']):%H:%M}.{holiday}")
            points = pd.DataFrame(answer["points"])
            points["time (Paris)"] = pd.to_datetime(points["timestamp_utc"], utc=True).dt.tz_convert(PARIS).dt.strftime("%H:%M")
            st.dataframe(points[["time (Paris)", "consumption_mw", "expected_mw", "z_score"]].round(1),
                         hide_index=True, width="stretch")
        else:
            st.success(f"No anomaly on this day.{holiday}")

    # 3. Model in use
    st.subheader("Model in use")
    info = api_or_stop("/model-info")
    m = info["metrics"][DETECTOR]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold K (|z|)", info["k"])
    c2.metric("Recall on reference events", f"{m['recall']:.0%}", help=f"{m['events_caught']} of {info['n_events']} events")
    c3.metric("Recall on public holidays", f"{m['recall_holiday']:.0%}")
    c4.metric("False alarms per week", m["false_alarms_per_week"])
    st.caption(f"Version {info['model_version']}, trained {paris(info['trained_at']):%d %b %Y, %H:%M} "
               f"on {info['n_points']:,} half-hours ({info['data_start'][:10]} to {info['data_end'][:10]}).")
    with st.expander("How the detection works"):
        st.markdown(
            "The MSTL model splits consumption into a trend, daily and weekly cycles and a residual. "
            "A half-hour is an anomaly when its residual is unusually large: |z| above K. "
            "A weekday public holiday looks like a Sunday, so the model flags it; we use holidays to check the model."
        )

# Admin tab: start jobs through the API (with the key) and follow them
API_KEY = os.environ.get("API_KEY", "")
ERRORS = {
    401: "the API key is missing",
    403: "the API key is wrong",
    409: "the DAG is paused in Airflow: switch it on first",
    429: "too many requests, wait a minute (NGINX rate limit)",
    500: "the API has no API key configured",
    503: "Airflow is not reachable",
}


def start_job(path: str, **params) -> None:
    try:
        response = requests.post(f"{API_URL}{path}", params=params,
                                 headers={"X-API-Key": API_KEY}, timeout=30)
    except requests.RequestException as exc:
        st.error(f"The API is not reachable: {exc}")
        return
    if response.status_code == 202:
        st.session_state["job"] = response.json()
        st.success(f"Started {st.session_state['job']['dag_id']}.")
    else:
        st.error(f"Not started ({response.status_code}): "
                 f"{ERRORS.get(response.status_code, response.text[:200])}")


with admin_tab:
    st.subheader("Start a job")
    st.caption("Jobs run in Airflow. The API starts them with the API key from the environment; "
               "the key is never shown here.")
    if not API_KEY:
        st.warning("API_KEY is not set. Start the app with:\n\n"
                   "`export API_KEY=$(grep ^API_KEY= .env | cut -d= -f2-)`")
    else:
        left, right = st.columns(2)
        with left:
            st.markdown("**Retrain**: a new model goes live only if it passes the promote-or-reject gate (MLflow).")
            if st.button("Retrain the model", type="primary"):
                start_job("/train")
        with right:
            st.markdown("**New data**: inject and score the next days (replay of held-out days).")
            n_new = st.number_input("Days", min_value=1, max_value=31, value=1)
            if st.button("Score new days"):
                start_job("/predict", days=int(n_new))

    job = st.session_state.get("job")
    if job:
        st.subheader("Last job")
        st.button("Refresh status")
        try:
            response = requests.get(f"{API_URL}/runs/{job['dag_id']}/{job['run_id']}", timeout=15)
            response.raise_for_status()
            status = response.json()
        except requests.RequestException as exc:
            st.error(f"Could not read the run status: {exc}")
            status = None
        if status:
            state = status.get("state", "unknown")
            show = {"success": st.success, "failed": st.error}.get(state, st.info)
            show(f"{job['dag_id']}: {state}")
            tasks = status.get("tasks")
            if tasks:
                st.dataframe(pd.DataFrame(tasks), hide_index=True, width="stretch")
            with st.expander("Answer of the API"):
                st.json(status)

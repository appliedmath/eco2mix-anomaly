# Grid anomaly detection (France, RTE eCO2mix)

Flags abnormal behaviour in French national electricity consumption at
30-minute resolution. Phase 1: one series, one complete chain
(collect, process, train, API).

## How it works

Three detectors run side by side. Each one builds an expected value for
every half-hour and flags the point when the gap is far from the usual
gap (robust z-score with median and MAD, threshold `k = 4`).

| Detector | Expected value | Idea |
|---|---|---|
| `mstl` (main) | trend + daily (48) + weekly (336) cycles | removes the cycles, flags the residual |
| `forecast_gap` | RTE day-before forecast (`prevision_j1`) | baseline |
| `last_week` | same half-hour 7 days earlier | baseline |

## Data

- RTE eCO2mix national, consolidated history from 2023-01-01 to 2026-08-31.
- 30-minute grid, UTC, 64,268 points. Clock-change duplicates are dropped
  and short gaps (up to 2 hours) are interpolated (`was_filled` marks them).
- Contract: `data/processed/consumption_30min.parquet` with the columns
  `timestamp_utc` (index), `consumption_mw`, `forecast_j1_mw`,
  `forecast_j_mw`, `is_holiday`, `is_future`, `was_filled`.
- Holidays use the French calendar (`holidays` package), by Paris date.

## Results

Evaluated on 47 reference events: 40 public holidays and 7 clock changes.
Recall = share of events with at least one alert inside. False alarms =
alert episodes outside every event, per week (consecutive alerts count once).

| Detector | Recall | Holiday recall | Clock-change recall | False alarms / week |
|---|---|---|---|---|
| mstl | 0.745 (35/47) | 0.875 | 0.0 | 0.40 |
| forecast_gap | 0.085 (4/47) | 0.10 | 0.0 | 0.13 |
| last_week | 0.191 (9/47) | 0.225 | 0.0 | 0.41 |

Reading these numbers:

- The detectors do not know about holidays, so a holiday looks like a
  Sunday on a weekday. A high holiday recall for `mstl` shows it reacts to
  a known calendar effect. It does not prove it finds real incidents.
- Clock-change recall is 0 for all three. The highest |z| on those days is
  between 1.5 and 3.4, below the threshold of 4. The cause is not verified.
- 74 of the 122 `mstl` alert episodes fall outside the reference events.
  They are long (5 to 35 hours) and concentrated between November and
  January. We have not checked what they are (cold snaps, school holidays
  or something else), so "false alarm" is an upper bound here.
- One threshold is used for the whole 2023-2026 history. The residual MAD
  is 651 MW on the full history and 462 MW on June-August 2026, so the same
  event gets a smaller z on the full history (14 July 2026, 07:00 UTC:
  -8.2 with the June-August window, -6.0 with the full history).

## Run it

Python 3.12 with `uv`:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.lock.txt

./run_pipeline.sh                      # collect -> process -> train
python -m pytest -v                    # tests
uvicorn src.api.main:app --reload      # API on http://127.0.0.1:8000/docs
```

`run_pipeline.sh` uses `src/data/collect.py` and `process.py` when they
exist. Otherwise it uses `scripts/make_sample_parquet.py`, a stand-in that
downloads the history and builds the contract file.

## API

The API only reads files written by the training step
(`models/detections.parquet`, `models/metadata.json`) and the processed
parquet. It never runs a model itself.

| Endpoint | Returns |
|---|---|
| `GET /health` | status, `model_version`, `last_data` |
| `GET /model-info` | training metadata and metrics |
| `GET /anomalies?start=&end=` | flagged points; optional `detector`, `only_anomalies` |
| `GET /series?start=&end=` | consumption and RTE forecasts |

Dates are `YYYY-MM-DD`, UTC, both days included, at most 366 days.
Invalid dates return 422.

```bash
curl "http://127.0.0.1:8000/anomalies?start=2026-07-14&end=2026-07-14"
```

## Known limitations

- Evaluation is in-sample: thresholds are computed on the same data that
  is scored, and MSTL sees the whole series.
- Reference events are only holidays and clock changes. Heatwaves, cold
  snaps, strikes and energy-saving periods are not in the list yet.
- History only: no real-time collection, so `last_data` is 2026-08-31.
- The 25 October 2026 clock change is not tested on live data.
- No authentication on the API.
- `models/` and `data/` are not in git: run `./run_pipeline.sh` first.

## Layout

- `src/data/`: collect and process (data half)
- `src/models/`: MSTL detector, baselines, reference events, training
- `src/api/`: FastAPI service
- `flows/`, `streamlit_app/`, `notebooks/`: orchestration, dashboard, exploration
- `scripts/`: stand-in data script, alarm inspection
- `tests/`: unit tests
- `config/`: reference events added by hand
- `run_pipeline.sh`

## Infrastructure notes (Prefect Docker image, MLflow, deployment)

## Prefect docker official image

You can see : https://hub.docker.com/r/prefecthq/prefect/tags

Moving from `prefecthq/prefect:3-latest` to `docker pull prefecthq/prefect:3.1.16.dev2-python3.12`. See on https://hub.docker.com/r/prefecthq/prefect/tags.

It is recommended to “pin” a Docker image so that you’re locked into a specific version even if the tag (like “3‑latest”) gets updated later. However, there are a couple of points to consider:

1. Why “Pinning” Is Important

Using a mutable tag like 3-latest means that the underlying image can change over time. This could lead to unexpected behavior if a new version is published that isn’t backwards compatible. Pinning ensures that your deployment always uses the exact image you expect.

2. How to Pin an Image

A. Use a Version Tag:
Instead of using 3-latest, you can specify a specific version (for example, 3.0.0rc14-python3.11). This way, you know exactly what you’re running, and it won’t change unexpectedly.

B. Use an Immutable Digest:
Docker images have an immutable identifier called a digest. When you pull an image by digest (e.g., prefecthq/prefect@sha256:<digest>), you’re referencing a specific snapshot of that image. Even if the tag moves, the digest stays the same.

Steps to Pin Using a Digest:
1. Pull the Image by Tag: `docker pull prefecthq/prefect:3-latest`

(Make sure the tag exists. Note that in some cases, like in a GitHub issue , the 3-latest tag might not be available or could have been changed.)

2. Find the Digest:
After pulling, run: `docker inspect --format='{{index .RepoDigests 0}}' prefecthq/prefect:3-latest`

This command returns something like `prefecthq/prefect@sha256:abcdef123456....`

3. Deploy Using the Digest:
Use this digest reference in your deployment commands. For example: `docker pull prefecthq/prefect@sha256:abcdef123456...`

This guarantees that the same image is used every time, regardless of changes to the tag.

3. Special Consideration for Prefect Images

There’s been some discussion on GitHub regarding the 3-latest tag for Prefect images. One issue even reported that deploying with prefecthq/prefect:3-latest resulted in errors because the tag wasn’t found ([ ￼](https://github.com/PrefectHQ/prefect/issues/14384)). In this case, it might be a good idea to either:
	•	Use a version-specific tag that is known to exist.
	•	Or, if you prefer using the “latest” style, first check the digest on Docker Hub and then pin it using the digest.

Example in Simple Steps :

1. Decide on the Image:
Instead of prefecthq/prefect:3-latest, check Docker Hub for Prefect to see available tags.

2. Pin Using a Specific Tag or Digest:
For a specific tag: `docker pull prefecthq/prefect:3.0.0rc14-python3.11` (tested but did worked in my case). That's why I decided to go for the digest.

Or, for a digest: `docker pull prefecthq/prefect@sha256:abcdef123456...`


3. Use in Your Deployment:
Replace the image reference in your deployment configuration with the one you’ve pinned.


mlflow:
    build:
      context: .
      target: development
    command: mlflow server --host 0.0.0.0 --port 5001

# Create deployment

```sh
make deploy
make run
```
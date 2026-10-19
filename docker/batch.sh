#!/bin/sh
# One batch: new data arrives (next 7 days by default), then the saved
# model scores it. Usage: sh batch.sh [days]
set -e
python -m scripts.inject_batch --days "${1:-7}"
python -m src.models.predict

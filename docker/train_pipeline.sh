#!/bin/sh
# Download and clean the data only if it is not there yet, then train.
set -e

if [ ! -f data/processed/consumption_30min.parquet ]; then
  echo "== no processed data: collect + process =="
  python -m src.data.collect
  python -m src.data.process
else
  echo "== processed data found: skip download =="
fi

echo "== train =="
python -m src.models.train

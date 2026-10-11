#!/usr/bin/env bash
# collect -> process -> train, in one command
set -euo pipefail
cd "$(dirname "$0")"

if [ -f src/data/collect.py ] && [ -f src/data/process.py ]; then
  echo "== 1/3 collect =="
  python -m src.data.collect
  echo "== 2/3 process =="
  python -m src.data.process
else
  echo "== 1-2/3 collect + process (stand-in script) =="
  python scripts/make_sample_parquet.py
fi

echo "== 3/3 train =="
python -m src.models.train
echo "Pipeline finished."

"""Hold back the last ~20% of the cleaned data as "new data" (run once).

    python -m scripts.split_holdout            # 20%
    python -m scripts.split_holdout --frac 0.3
"""
import argparse

import pandas as pd

from src.data.batches import INCOMING_PATH, PROCESSED_PATH, split_holdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frac", type=float, default=0.2)
    args = parser.parse_args()

    if INCOMING_PATH.exists():
        raise SystemExit(f"{INCOMING_PATH} already exists: split was done "
                         "already. Use inject_batch to add data.")

    df = pd.read_parquet(PROCESSED_PATH)
    train, incoming = split_holdout(df, args.frac)

    INCOMING_PATH.parent.mkdir(parents=True, exist_ok=True)
    incoming.to_parquet(INCOMING_PATH)
    train.to_parquet(PROCESSED_PATH)

    print(f"Training part: {train.index.min()} -> {train.index.max()} "
          f"({len(train)} rows) -> {PROCESSED_PATH}")
    print(f"Held back:     {incoming.index.min()} -> {incoming.index.max()} "
          f"({len(incoming)} rows) -> {INCOMING_PATH}")


if __name__ == "__main__":
    main()

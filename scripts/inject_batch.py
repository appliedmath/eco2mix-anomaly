"""Move the next days from data/incoming/ into the processed data.

    python -m scripts.inject_batch           # next 7 days
    python -m scripts.inject_batch --days 1
"""
import argparse

import pandas as pd

from src.data.batches import INCOMING_PATH, PROCESSED_PATH, take_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    processed = pd.read_parquet(PROCESSED_PATH)
    incoming = pd.read_parquet(INCOMING_PATH)
    processed, incoming, batch = take_batch(processed, incoming, args.days)

    if batch.empty:
        print("No more incoming data: everything has been injected.")
        return
    processed.to_parquet(PROCESSED_PATH)
    incoming.to_parquet(INCOMING_PATH)
    print(f"Injected {len(batch)} rows: {batch.index.min()} -> "
          f"{batch.index.max()}")
    print(f"Still incoming: {len(incoming)} rows")


if __name__ == "__main__":
    main()

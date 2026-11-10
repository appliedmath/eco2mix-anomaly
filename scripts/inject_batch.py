"""Add new days to the processed data.

    python -m scripts.inject_batch                  # replay: next 7 held-out days
    python -m scripts.inject_batch --days 1
    python -m scripts.inject_batch --source live    # all new complete days from the RTE API

Writes models/batch/last_batch.json (source, start, end, rows) for validate_batch.
"""
import argparse
import json

import pandas as pd

from src.data.batches import (
    BATCH_INFO_PATH,
    INCOMING_PATH,
    PROCESSED_PATH,
    fetch_live_batch,
    take_batch,
)


def write_info(source: str, batch: pd.DataFrame) -> None:
    info = {"source": source, "rows": len(batch), "start": None, "end": None}
    if len(batch):
        info["start"] = batch.index.min().isoformat()
        info["end"] = batch.index.max().isoformat()
    BATCH_INFO_PATH.parent.mkdir(parents=True, exist_ok=True)
    BATCH_INFO_PATH.write_text(json.dumps(info, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="replay only")
    parser.add_argument("--source", choices=["replay", "live"], default="replay")
    args = parser.parse_args()

    processed = pd.read_parquet(PROCESSED_PATH)

    if args.source == "live":
        processed, batch = fetch_live_batch(processed)
        write_info("live", batch)
        if batch.empty:
            print("No new complete day from the RTE API yet.")
            return
        processed.to_parquet(PROCESSED_PATH)
        print(f"Injected {len(batch)} live rows: {batch.index.min()} -> "
              f"{batch.index.max()}")
        return

    incoming = pd.read_parquet(INCOMING_PATH)
    incoming = incoming[incoming.index > processed.index.max()]  # skip days already added (e.g. live)
    processed, incoming, batch = take_batch(processed, incoming, args.days)
    write_info("replay", batch)

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

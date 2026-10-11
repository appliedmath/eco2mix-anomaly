import pandas as pd

from src.models.detect import detect_mstl
from src.models.events import build_reference_events
from src.models.train import inside_events

df = pd.read_parquet("data/processed/consumption_30min.parquet").sort_index()
y = df.loc[~df["is_future"], "consumption_mw"]
events = build_reference_events(df)
in_event = inside_events(y.index, events)

out = detect_mstl(y, k=4.0)
flags = out["is_anomaly"]

# group consecutive flagged points into episodes
episode_id = (flags & ~flags.shift(1, fill_value=False)).cumsum()
rows = []
for _, g in out[flags].groupby(episode_id[flags]):
    rows.append({
        "start": g.index[0],
        "half_hours": len(g),
        "max_abs_z": round(g["z_score"].abs().max(), 1),
        "inside_event": bool(in_event.loc[g.index].any()),
    })
eps = pd.DataFrame(rows)
outside = eps[~eps["inside_event"]]
print("episodes:", len(eps), "| outside events:", len(outside))
print("\nTop 10 outside the events:")
print(outside.sort_values("max_abs_z", ascending=False).head(10).to_string(index=False))

cc = events[events["kind"] == "clock_change"]
print("\nMax |z| in each clock-change day (alert threshold is 4):")
for s, e, label in zip(cc["start_utc"], cc["end_utc"], cc["label"]):
    print(" ", label, round(out["z_score"].loc[s:e].abs().max(), 1))

import io
import os

import holidays
import pandas as pd
import requests

URL = ("https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
       "eco2mix-national-cons-def/exports/csv")
params = {
    "delimiter": ";",
    "select": "date_heure,consommation,prevision_j1,prevision_j",
    "where": "date_heure >= date'2023-01-01'",
}

# 1. One API call for the whole history
r = requests.get(URL, params=params, timeout=300)
r.raise_for_status()
hist = pd.read_csv(io.StringIO(r.text), sep=";")
hist["date_heure"] = pd.to_datetime(hist["date_heure"], utc=True)
hist = hist.sort_values("date_heure")

# 2. Daylight-saving fix: drop duplicates, keep the 30-min points
clean = hist.drop_duplicates(subset="date_heure", keep="first")
clean = clean.set_index("date_heure")
clean = clean[clean.index.minute.isin([0, 30])]

# 3. Regular 30-min grid; remember which rows were missing
grid = pd.date_range(clean.index.min(), clean.index.max(),
                     freq="30min", tz="UTC")
clean = clean.reindex(grid)
was_filled = clean.isna().any(axis=1)
clean = clean.interpolate(limit=4)  # gaps of up to 2 hours only

# 4. Build the contract columns
fr_holidays = holidays.France(years=range(2023, 2028))
paris_dates = clean.index.tz_convert("Europe/Paris").date

out = pd.DataFrame({
    "consumption_mw": clean["consommation"],
    "forecast_j1_mw": clean["prevision_j1"],
    "forecast_j_mw": clean["prevision_j"],
    "is_holiday": [d in fr_holidays for d in paris_dates],
    "is_future": False,
    "was_filled": was_filled.values,
}, index=clean.index)
out.index.name = "timestamp_utc"

os.makedirs("data/processed", exist_ok=True)
out.to_parquet("data/processed/consumption_30min.parquet")

print(out.shape)
print("From:", out.index.min(), " To:", out.index.max())
print("Missing values:", int(out[["consumption_mw", "forecast_j1_mw",
                                  "forecast_j_mw"]].isna().sum().sum()))
print("Filled points:", int(out["was_filled"].sum()))
print("Holiday rows:", int(out["is_holiday"].sum()))
print(out.dtypes)


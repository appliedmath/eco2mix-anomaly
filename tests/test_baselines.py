import pandas as pd

from src.models.baselines import (
    baseline_last_week,
    baseline_rte,
    flag_anomalies,
)

# 7 days * 24 hours * 4 records per hour.
# Hardcoded on purpose: the test must not copy the number from the code.
ONE_WEEK = 672


def make_df(n=ONE_WEEK + 10):
    return pd.DataFrame({
        "consommation": range(n),
        "prevision_j1": [x + 5 for x in range(n)],
    })


def test_last_week_baseline_shifts_by_one_week():
    df = make_df()
    expected = baseline_last_week(df)
    # first week has no history
    assert expected.iloc[:ONE_WEEK].isna().all()
    # afterwards: value = same record one week earlier
    assert expected.iloc[ONE_WEEK] == df["consommation"].iloc[0]


def test_rte_baseline_returns_forecast_column():
    df = make_df()
    assert baseline_rte(df).equals(df["prevision_j1"])


def test_flag_anomalies_finds_only_the_spike():
    expected = pd.Series([100.0] * 50)
    # small regular noise: -0, +1, -2, +3, -4, ...
    noise = pd.Series([(-1) ** i * (i % 5) for i in range(50)])
    actual = expected + noise
    actual.iloc[25] = 1000.0  # one huge spike
    flags = flag_anomalies(actual, expected)
    assert flags.sum() == 1
    assert flags.iloc[25]


def test_flag_anomalies_quiet_data_has_no_alerts():
    expected = pd.Series([100.0] * 50)
    noise = pd.Series([(-1) ** i * (i % 5) for i in range(50)])
    flags = flag_anomalies(expected + noise, expected)
    assert flags.sum() == 0

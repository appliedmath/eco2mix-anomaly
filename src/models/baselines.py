import pandas as pd

STEPS_PER_WEEK = 7 * 24 * 4  # 672 records of 15 minutes


def baseline_last_week(df: pd.DataFrame) -> pd.Series:
    """B1: expected = consumption at the same time last week."""
    return df["consommation"].shift(STEPS_PER_WEEK)


def baseline_rte(df: pd.DataFrame) -> pd.Series:
    """B2: expected = RTE day-ahead forecast."""
    return df["prevision_j1"]


def flag_anomalies(actual, expected, k: float = 4.0) -> pd.Series:
    """True where the gap is unusually large (robust z-score > k)."""
    gap = (actual - expected).dropna()
    median = gap.median()
    mad = (gap - median).abs().median()
    score = (gap - median) / (1.4826 * mad)
    return score.abs() > k


if __name__ == "__main__":
    df = pd.read_csv("data/clean.csv", index_col=0, parse_dates=True)
    actual = df["consommation"]

    for name, expected in [
        ("B1 last week", baseline_last_week(df)),
        ("B2 RTE forecast", baseline_rte(df)),
    ]:
        flags = flag_anomalies(actual, expected)
        print(name, "-> records checked:", len(flags),
              "| alerts:", int(flags.sum()))
        print(flags[flags].index[:5].tolist())

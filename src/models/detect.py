import pandas as pd
from statsmodels.tsa.seasonal import MSTL

from src.models.model import fit_reference

DAY = 48     # half-hours in a day
WEEK = 336   # half-hours in a week


def mstl_residual(series: pd.Series, periods=(DAY, WEEK)) -> pd.Series:
    """Remove trend + daily + weekly cycles; return what is left."""
    return MSTL(series, periods=periods).fit().resid


def detect_mstl(series: pd.Series, k: float = 4.0, ref: dict | None = None,
                periods=(DAY, WEEK)) -> pd.DataFrame:
    """Flag points whose residual is far from the usual residual."""
    resid = mstl_residual(series, periods)
    if ref is None:
        ref = fit_reference(resid)  # median and MAD of the residual
    z = (resid - ref["median"]) / (1.4826 * ref["mad"])
    return pd.DataFrame({
        "consumption_mw": series,
        "expected_mw": series - resid,
        "z_score": z,
        "is_anomaly": z.abs() > k,
    })

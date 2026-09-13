"""
forecaster.py
-------------
Made BY DHRUV BATHLA
Wraps Facebook Prophet (primary) with a SARIMAX fallback.

Public API
----------
run_forecast(df_daily, horizon_days) -> list[dict]
    df_daily  : DataFrame with columns ["ds", "y"] (date, revenue) — daily granularity
    horizon_days : int, forecast horizon (e.g. 30 or 90)

Returns a list of dicts:
    [{"date": "YYYY-MM-DD", "yhat": float, "yhat_lower": float, "yhat_upper": float}, ...]

Why Prophet?
------------
Prophet is additive by design: it decomposes a time series into
  trend + seasonality + holiday effects + error.
For e-commerce revenue it is particularly well-suited because:
  1. It handles dual seasonality (weekly + yearly) without manual tuning.
  2. Missing data / outliers are handled gracefully.
  3. Confidence intervals ("yhat_lower", "yhat_upper") are computed via a
     Monte Carlo posterior predictive simulation — they represent the 80 %
     credible interval of the model's forecast distribution.

Why SARIMA as fallback?
-----------------------
SARIMA(1,1,1)(1,1,0)[7] is a classical frequentist model.
  * (1,1,1) : AR(1) + one differencing for trend + MA(1) smoothing.
  * (1,1,0)[7] : seasonal AR(1) + seasonal differencing with period 7 (weekly).
Confidence intervals from `get_forecast()` are 80 % prediction intervals
derived from the asymptotic distribution of the one-step-ahead residuals.
"""

from __future__ import annotations
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Prophet implementation
# ---------------------------------------------------------------------------

def _forecast_prophet(df: pd.DataFrame, horizon_days: int) -> List[Dict]:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=horizon_days, freq="D")
    forecast = model.predict(future)

    # Keep only the forecasted portion (beyond training data)
    last_train_date = df["ds"].max()
    fc = forecast[forecast["ds"] > last_train_date].copy()

    return [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "yhat": round(float(row["yhat"]), 2),
            "yhat_lower": round(float(row["yhat_lower"]), 2),
            "yhat_upper": round(float(row["yhat_upper"]), 2),
        }
        for _, row in fc.iterrows()
    ]


# ---------------------------------------------------------------------------
# SARIMAX fallback
# ---------------------------------------------------------------------------

def _forecast_sarima(df: pd.DataFrame, horizon_days: int) -> List[Dict]:
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

    series = df.set_index("ds")["y"].asfreq("D").fillna(method="ffill")

    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False, maxiter=200)

    forecast_obj = fit.get_forecast(steps=horizon_days, alpha=0.20)  # 80 % CI
    pred_mean = forecast_obj.predicted_mean
    ci = forecast_obj.conf_int()

    results = []
    for date, yhat in pred_mean.items():
        lo = float(ci.loc[date].iloc[0])
        hi = float(ci.loc[date].iloc[1])
        results.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "yhat": round(float(yhat), 2),
                "yhat_lower": round(lo, 2),
                "yhat_upper": round(hi, 2),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_forecast(df_daily: pd.DataFrame, horizon_days: int = 30) -> List[Dict]:
    """
    Train a forecasting model on df_daily and return horizon_days predictions.

    df_daily must contain columns: date (or ds), revenue (or y).
    """
    # Normalise column names to Prophet convention (ds, y)
    col_map = {}
    for col in df_daily.columns:
        if col.lower() in ("date", "ds"):
            col_map[col] = "ds"
        elif col.lower() in ("revenue", "y"):
            col_map[col] = "y"

    df = df_daily.rename(columns=col_map)[["ds", "y"]].copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna().sort_values("ds")

    if len(df) < 14:
        raise ValueError("Need at least 14 days of data to train a forecast model.")

    if PROPHET_AVAILABLE:
        try:
            return _forecast_prophet(df, horizon_days)
        except Exception as exc:
            print(f"[forecaster] Prophet failed ({exc}), falling back to SARIMA …")

    return _forecast_sarima(df, horizon_days)

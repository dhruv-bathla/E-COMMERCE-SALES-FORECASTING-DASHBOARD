"""
main.py  —  FastAPI backend for the E-Commerce Sales Forecasting Dashboard
---------------------------------------------------------------------------
Made BY DHRUV BATHLA

Start with:
    uvicorn backend.main:app --reload --port 8001

Developer UI (Swagger):  http://localhost:8001/docs
Alternative UI (ReDoc):  http://localhost:8001/redoc

Endpoints
---------
GET /api/health
GET /api/sales/historical
GET /api/sales/forecast
GET /api/kpis/summary
GET /api/products/top
"""

from __future__ import annotations
import os
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.data_loader import load_sales, aggregate_sales
from backend.forecaster import run_forecast

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="E-Commerce Forecasting API",
    description=(
        "Real time-series sales forecasting powered by Prophet / SARIMA.\n\n"
        "**Author:** Dhruv Bathla  \n"
        "Use the endpoints below to explore historical sales, run forecasts, "
        "and retrieve KPIs. All values are computed live from the dataset."
    ),
    version="1.0.0",
    contact={"name": "Dhruv Bathla"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str


class SalesPoint(BaseModel):
    date: str
    revenue: float
    quantity: float
    orders: int


class ForecastPoint(BaseModel):
    date: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class KPISummary(BaseModel):
    total_revenue: float
    mom_growth_pct: float
    best_category: str
    forecast_next_30_revenue: float


class CategoryRevenue(BaseModel):
    category: str
    revenue: float


# ---------------------------------------------------------------------------
# Helper: load data once per request (simple; no heavy caching needed here)
# ---------------------------------------------------------------------------


def _get_df() -> pd.DataFrame:
    try:
        return load_sales()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Data load error: {exc}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {"status": "ok"}


@app.get("/api/sales/historical", response_model=List[SalesPoint], tags=["Sales"])
def get_historical(
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    category: Optional[str] = Query(None, description="Product category"),
    region: Optional[str] = Query(None, description="Region"),
    granularity: str = Query("D", description="D=daily, W=weekly, ME=monthly"),
):
    """Return aggregated historical sales filtered by date range, category, and region."""
    df = _get_df()

    # Validate date range
    if start and end:
        try:
            if pd.to_datetime(start) > pd.to_datetime(end):
                raise HTTPException(
                    status_code=400, detail="start date must be before end date"
                )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid date: {exc}")

    agg = aggregate_sales(df, start=start, end=end, category=category,
                          region=region, granularity=granularity)

    if agg.empty:
        raise HTTPException(
            status_code=404,
            detail="No data found for the given filters. Try broadening your query.",
        )

    return [
        {
            "date": row["date"],
            "revenue": round(float(row["revenue"]), 2),
            "quantity": float(row["quantity"]),
            "orders": int(row["orders"]),
        }
        for _, row in agg.iterrows()
    ]


@app.get("/api/sales/forecast", response_model=List[ForecastPoint], tags=["Forecast"])
def get_forecast(
    horizon_days: int = Query(30, ge=1, le=365, description="Forecast horizon in days"),
    category: Optional[str] = Query(None, description="Product category filter"),
    region: Optional[str] = Query(None, description="Region filter"),
):
    """
    Train the forecasting model on the full historical dataset (filtered by
    category/region if provided) and return real predictions with confidence
    intervals for horizon_days into the future.
    """
    df = _get_df()

    # Aggregate to daily revenue for the chosen slice
    agg = aggregate_sales(df, category=category, region=region, granularity="D")

    if agg.empty or len(agg) < 14:
        raise HTTPException(
            status_code=422,
            detail="Insufficient data to build a forecast. Need at least 14 daily data points.",
        )

    # Rename for forecaster
    agg_for_model = agg.rename(columns={"date": "ds", "revenue": "y"})

    try:
        predictions = run_forecast(agg_for_model, horizon_days=horizon_days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Forecasting error: {exc}")

    return predictions


@app.get("/api/kpis/summary", response_model=KPISummary, tags=["KPIs"])
def get_kpi_summary():
    """
    Returns key business metrics computed from the full dataset:
      - total_revenue         : sum of all revenue
      - mom_growth_pct        : current-month vs previous-month revenue growth %
      - best_category         : category with highest total revenue
      - forecast_next_30_revenue : summed yhat of the 30-day forecast
    """
    df = _get_df()

    total_revenue = round(float(df["revenue"].sum()), 2)

    # Month-over-month growth
    df["month"] = df["order_date"].dt.to_period("M")
    monthly = df.groupby("month")["revenue"].sum().sort_index()
    if len(monthly) >= 2:
        prev_rev = float(monthly.iloc[-2])
        curr_rev = float(monthly.iloc[-1])
        mom_growth = round((curr_rev - prev_rev) / prev_rev * 100, 2) if prev_rev else 0.0
    else:
        mom_growth = 0.0

    # Best-selling category
    cat_rev = df.groupby("product_category")["revenue"].sum()
    best_category = str(cat_rev.idxmax())

    # Next-30-day forecast (full dataset, no filters)
    agg = aggregate_sales(df, granularity="D").rename(columns={"date": "ds", "revenue": "y"})
    try:
        fc = run_forecast(agg, horizon_days=30)
        forecast_next_30 = round(sum(p["yhat"] for p in fc), 2)
    except Exception:
        forecast_next_30 = 0.0

    return {
        "total_revenue": total_revenue,
        "mom_growth_pct": mom_growth,
        "best_category": best_category,
        "forecast_next_30_revenue": forecast_next_30,
    }


@app.get("/api/products/top", response_model=List[CategoryRevenue], tags=["Products"])
def get_top_products(
    n: int = Query(5, ge=1, le=20, description="Number of top categories to return"),
    start: Optional[str] = Query(None, description="Start date YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    region: Optional[str] = Query(None, description="Region filter"),
):
    """Return the top N product categories by total revenue."""
    df = _get_df()

    if start:
        df = df[df["order_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["order_date"] <= pd.to_datetime(end)]
    if region and region.lower() != "all":
        df = df[df["region"].str.lower() == region.lower()]

    if df.empty:
        raise HTTPException(status_code=404, detail="No data for the given filters.")

    top = (
        df.groupby("product_category")["revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )

    return [
        {"category": row["product_category"], "revenue": round(float(row["revenue"]), 2)}
        for _, row in top.iterrows()
    ]

"""
data_loader.py
--------------
Made BY DHRUV BATHLA
Single source of truth for reading and aggregating sales data.

Functions
---------
load_sales()        : Load (and auto-generate) the CSV. Returns a DataFrame.
aggregate_sales()   : Group by date granularity + optional filters.
"""

import os
import pandas as pd
from typing import Optional

RAW_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sales_data.csv")

REQUIRED_COLS = {"order_date", "product_category", "region", "quantity", "revenue"}


def load_sales() -> pd.DataFrame:
    """Load CSV; auto-generate if missing."""
    if not os.path.exists(RAW_CSV):
        print("[data_loader] CSV not found – generating synthetic data …")
        from backend.data_generator import generate_sales_data

        generate_sales_data(output_path=RAW_CSV)

    df = pd.read_csv(RAW_CSV, parse_dates=["order_date"])

    # Validate schema
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df = df.dropna(subset=["order_date", "revenue"])
    return df


def aggregate_sales(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
    category: Optional[str] = None,
    region: Optional[str] = None,
    granularity: str = "D",          # "D", "W", "ME" (month-end)
) -> pd.DataFrame:
    """
    Filter and aggregate a sales DataFrame.

    Returns a DataFrame with columns: date, revenue, quantity, orders.
    """
    # --- Date filters ---
    if start:
        df = df[df["order_date"] >= pd.to_datetime(start)]
    if end:
        df = df[df["order_date"] <= pd.to_datetime(end)]

    # --- Category / region filters (case-insensitive) ---
    if category and category.lower() != "all":
        df = df[df["product_category"].str.lower() == category.lower()]
    if region and region.lower() != "all":
        df = df[df["region"].str.lower() == region.lower()]

    if df.empty:
        return pd.DataFrame(columns=["date", "revenue", "quantity", "orders"])

    # --- Resample to chosen granularity ---
    df = df.set_index("order_date").sort_index()
    agg = df.resample(granularity).agg(
        revenue=("revenue", "sum"),
        quantity=("quantity", "sum"),
        orders=("revenue", "count"),
    ).reset_index()
    agg.rename(columns={"order_date": "date"}, inplace=True)
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")
    return agg

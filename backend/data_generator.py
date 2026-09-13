"""
data_generator.py
-----------------
Made BY DHRUV BATHLA
Generates two years of synthetic daily order-level sales data and saves it to
data/raw/sales_data.csv.

Design decisions:
  - Base revenue per order drawn from a gamma distribution so values are
    right-skewed (realistic for e-commerce).
  - Weekly seasonality: Saturday/Sunday see 30 % fewer orders than weekdays.
  - Yearly spike: November and December get a 2× multiplier (holiday season).
  - Gradual upward trend: ~20 % YoY revenue growth implemented as a linear
    scaling factor over the 2-year window.
  - Gaussian noise added to individual order revenues.
"""

import os
import numpy as np
import pandas as pd

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Beauty"]
REGIONS = ["North", "South", "East", "West"]

# Base orders per day per category (weekday average)
BASE_ORDERS = {
    "Electronics": 18,
    "Clothing": 25,
    "Home & Garden": 15,
    "Sports": 12,
    "Beauty": 20,
}

# Average revenue per order per category (USD)
AVG_REVENUE = {
    "Electronics": 150,
    "Clothing": 60,
    "Home & Garden": 80,
    "Sports": 70,
    "Beauty": 45,
}


def generate_sales_data(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    seed: int = 42,
    output_path: str = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    total_days = len(dates)

    rows = []
    for day_idx, date in enumerate(dates):
        # --- Trend factor: linear 1.0 → 1.20 over the whole window ---
        trend = 1.0 + 0.20 * (day_idx / (total_days - 1))

        # --- Weekly seasonality factor ---
        dow = date.dayofweek  # 0=Mon … 6=Sun
        week_factor = 0.70 if dow >= 5 else 1.0

        # --- Yearly spike factor ---
        month = date.month
        year_factor = 2.0 if month in (11, 12) else 1.0

        # --- Per-category, per-day orders ---
        for cat in CATEGORIES:
            n_orders = int(
                BASE_ORDERS[cat] * trend * week_factor * year_factor
                + rng.normal(0, 2)
            )
            n_orders = max(n_orders, 1)

            avg_rev = AVG_REVENUE[cat]
            for _ in range(n_orders):
                # Revenue: gamma distribution centred on avg_rev, then scale by trend
                revenue = rng.gamma(shape=4, scale=avg_rev / 4) * trend
                revenue = round(float(revenue), 2)
                qty = int(rng.integers(1, 5))
                region = rng.choice(REGIONS)
                rows.append(
                    {
                        "order_date": date.strftime("%Y-%m-%d"),
                        "product_category": cat,
                        "region": region,
                        "quantity": qty,
                        "revenue": revenue,
                    }
                )

    df = pd.DataFrame(rows)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"[data_generator] Saved {len(df):,} rows to {output_path}")

    return df


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "..")
    out = os.path.join(base_dir, "data", "raw", "sales_data.csv")
    generate_sales_data(output_path=out)
    print("Done.")

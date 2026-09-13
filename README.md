# E-Commerce Sales Forecasting Dashboard

A full-stack, end-to-end sales forecasting application:

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| Forecasting | Facebook Prophet (SARIMA fallback) |
| Data processing | pandas · numpy |
| Frontend | Streamlit + Plotly |
| Storage | Local CSV (`data/raw/sales_data.csv`) |

---

## Project Structure

```
.
├── backend/
│   ├── __init__.py
│   ├── main.py             ← FastAPI app (5 endpoints)
│   ├── data_generator.py   ← Synthetic dataset creator
│   ├── data_loader.py      ← CSV loader + aggregation helpers
│   └── forecaster.py       ← Prophet / SARIMA model wrapper
├── frontend/
│   └── app.py              ← Streamlit dashboard
├── data/
│   └── raw/
│       └── sales_data.csv  ← Auto-generated if missing
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv .venv

# Activate — Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate — macOS / Linux
source .venv/bin/activate

# Install all packages
pip install -r requirements.txt
```

> **Note on Prophet on Windows**: If `pip install prophet` fails, Prophet will
> be silently skipped and the app will use the SARIMA fallback automatically.
> Alternatively install `pystan` first:
> ```
> pip install pystan==2.19.1.1 prophet
> ```

---

### 2. Generate the dataset (optional — auto-runs on first API call)

```bash
python -m backend.data_generator
```

This creates `data/raw/sales_data.csv` (~150 000 rows, 2 years of daily
order-level data).

---

### 3. Start the FastAPI backend

Open a terminal in the project root and run:

```bash
uvicorn backend.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

### 4. Start the Streamlit dashboard

Open a **second terminal** in the project root and run:

```bash
streamlit run frontend/app.py
```

The dashboard opens at `http://localhost:8501`.

---

### 5. Using a custom CSV

Drop your own CSV into `data/raw/sales_data.csv`.
Required columns (case-sensitive):

| Column | Type | Description |
|---|---|---|
| `order_date` | date string (YYYY-MM-DD) | Order date |
| `product_category` | string | Category name |
| `region` | string | Geographic region |
| `quantity` | integer | Units ordered |
| `revenue` | float | Revenue in USD |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/sales/historical` | Filtered historical sales |
| `GET` | `/api/sales/forecast` | Real-time forecast |
| `GET` | `/api/kpis/summary` | KPI metrics |
| `GET` | `/api/products/top` | Top N categories |

Full interactive docs: **http://localhost:8000/docs**

### Example calls

```bash
# Health
curl http://localhost:8000/api/health

# Historical — all data
curl "http://localhost:8000/api/sales/historical?start=2023-01-01&end=2023-06-30"

# Historical — Electronics only
curl "http://localhost:8000/api/sales/historical?start=2023-01-01&end=2023-06-30&category=Electronics"

# 30-day forecast (full dataset)
curl "http://localhost:8000/api/sales/forecast?horizon_days=30"

# 90-day forecast (Clothing, North only)
curl "http://localhost:8000/api/sales/forecast?horizon_days=90&category=Clothing&region=North"

# KPI summary
curl http://localhost:8000/api/kpis/summary

# Top 5 categories
curl "http://localhost:8000/api/products/top?n=5"
```

---

## Forecasting Approach

### Model: Facebook Prophet

Prophet is a **decomposable additive time-series model**:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + εₜ
```

**Why Prophet for this project?**

1. **Dual seasonality out of the box** — The e-commerce dataset has both
   weekly patterns (weekday vs. weekend order volumes) and yearly patterns
   (Nov-Dec holiday spike). Prophet learns both simultaneously without
   manual feature engineering.

2. **Robust to missing data and outliers** — Real sales data is messy.
   Prophet handles gaps and sudden spikes (e.g., flash sales) without
   crashing.

3. **Interpretable confidence intervals** — Prophet samples from its
   posterior distribution (using Stan under the hood) to produce
   prediction intervals. The **80 % credible interval** (`yhat_lower`,
   `yhat_upper`) means: "given everything the model has learned, we are
   80 % confident the actual revenue will fall in this range."

4. **No manual hyperparameter tuning required** — `changepoint_prior_scale`
   controls trend flexibility; we set it to 0.05 (conservative), which
   prevents overfitting to short-term noise.

### Fallback: SARIMAX(1,1,1)(1,1,0)[7]

If Prophet's compiled dependencies (C++/Stan) are unavailable in the
environment, `forecaster.py` automatically falls back to `statsmodels`
SARIMAX:

- `(1,1,1)` — AR(1) captures autocorrelation; first-order differencing
  removes the long-term trend; MA(1) smooths residuals.
- `(1,1,0)[7]` — Seasonal AR(1) + seasonal differencing with period 7
  (weekly cycle).
- **80 % prediction intervals** come from the asymptotic distribution of
  the one-step-ahead forecast errors, propagated over the horizon.

### What do the confidence intervals mean?

The shaded band in the dashboard is the **80 % confidence / credible
interval**. It answers the question: *"How wide is our uncertainty about
the future?"*

- A **narrow band** = the model has high confidence (low historical
  variance, strong seasonal signal).
- A **wide band** = more uncertainty (e.g., forecasting 90 days out vs.
  30 days, or filtering to a small data slice with high noise).

As the forecast horizon grows, uncertainty compounds — so the band
naturally widens. This is statistically correct, not a bug.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Streamlit → backend URL |

```bash
# Example: backend on a different host
BACKEND_URL=http://192.168.1.10:8000 streamlit run frontend/app.py
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to backend` | Start `uvicorn backend.main:app --reload --port 8000` first |
| `Prophet install fails` | The app auto-falls back to SARIMA — no action needed |
| `Forecast takes >2 min` | Normal for Prophet on 2 years of data; SARIMA is faster |
| `No data found` error | Selected filters return 0 rows — broaden date range or remove filters |

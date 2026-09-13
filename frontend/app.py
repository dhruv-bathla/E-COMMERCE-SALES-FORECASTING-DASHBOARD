"""
app.py  —  Streamlit Dashboard for the E-Commerce Sales Forecasting Dashboard
------------------------------------------------------------------------------
Made BY DHRUV BATHLA
Every number displayed in this dashboard comes from a live HTTP call to the
FastAPI backend. No calculations are performed here.

Start with:
    streamlit run frontend/app.py
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

CATEGORIES = ["All", "Electronics", "Clothing", "Home & Garden", "Sports", "Beauty"]
REGIONS = ["All", "North", "South", "East", "West"]

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Helper: call backend
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict = None, timeout: int = 120):
    """Make a GET request to the backend; surface errors gracefully."""
    url = f"{BACKEND_URL}{endpoint}"
    try:
        resp = requests.get(url, params=params or {}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"❌ Cannot connect to the backend at **{BACKEND_URL}**. "
            "Make sure you started the FastAPI server:\n\n"
            "```\nuvicorn backend.main:app --reload --port 8000\n```"
        )
        st.stop()
    except requests.exceptions.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc))
        st.error(f"Backend error ({exc.response.status_code}): {detail}")
        return None
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar — Filters
# ---------------------------------------------------------------------------

st.sidebar.title("🛒 Filters")
st.sidebar.markdown("---")

default_start = date(2023, 1, 1)
default_end = date(2024, 12, 31)

date_start = st.sidebar.date_input("Start date", value=default_start, min_value=date(2020, 1, 1))
date_end = st.sidebar.date_input("End date", value=default_end)

if date_start > date_end:
    st.sidebar.error("Start date must be before end date.")

selected_category = st.sidebar.selectbox("Product Category", CATEGORIES)
selected_region = st.sidebar.selectbox("Region", REGIONS)

granularity_map = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
selected_gran_label = st.sidebar.radio("Chart Granularity", list(granularity_map.keys()), index=0)
granularity = granularity_map[selected_gran_label]

forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", min_value=7, max_value=180, value=30, step=7)

st.sidebar.markdown("---")
st.sidebar.caption(f"Backend: `{BACKEND_URL}`")

# Build filter params (pass None → "All" is omitted)
hist_params = {
    "start": str(date_start),
    "end": str(date_end),
    "granularity": granularity,
}
if selected_category != "All":
    hist_params["category"] = selected_category
if selected_region != "All":
    hist_params["region"] = selected_region

fc_params = {"horizon_days": forecast_horizon}
if selected_category != "All":
    fc_params["category"] = selected_category
if selected_region != "All":
    fc_params["region"] = selected_region

top_params = {"n": 5, "start": str(date_start), "end": str(date_end)}
if selected_region != "All":
    top_params["region"] = selected_region

# ---------------------------------------------------------------------------
# Main content — Title
# ---------------------------------------------------------------------------

st.title("📈 E-Commerce Sales Forecasting Dashboard")
st.markdown("### by DHRUV BATHLA")
st.markdown(
    "Real-time analytics powered by a **Prophet / SARIMA** time-series model. "
    "Adjust the sidebar filters to re-query the backend and update every chart."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# KPI Cards  (from /api/kpis/summary)
# ---------------------------------------------------------------------------

with st.spinner("Loading KPIs …"):
    kpis = _get("/api/kpis/summary")

if kpis:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 Total Revenue",
            value=f"${kpis['total_revenue']:,.0f}",
        )
    with col2:
        mom = kpis["mom_growth_pct"]
        st.metric(
            label="📊 MoM Growth",
            value=f"{mom:+.1f}%",
            delta=f"{mom:+.1f}%",
        )
    with col3:
        st.metric(
            label="🏆 Best Category",
            value=kpis["best_category"],
        )
    with col4:
        st.metric(
            label="🔮 Next-30-Day Forecast",
            value=f"${kpis['forecast_next_30_revenue']:,.0f}",
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Historical + Forecast Line Chart
# ---------------------------------------------------------------------------

st.subheader("📉 Historical Sales + Forecast")

col_hist, col_fc = st.columns([1, 1])

with col_hist:
    with st.spinner("Loading historical data …"):
        hist_data = _get("/api/sales/historical", params=hist_params)

with col_fc:
    with st.spinner(f"Running {forecast_horizon}-day forecast … (this may take ~30s)"):
        fc_data = _get("/api/sales/forecast", params=fc_params)

if hist_data and fc_data:
    hist_df = pd.DataFrame(hist_data)
    fc_df = pd.DataFrame(fc_data)

    fig = go.Figure()

    # Historical line
    fig.add_trace(
        go.Scatter(
            x=hist_df["date"],
            y=hist_df["revenue"],
            mode="lines",
            name="Historical Revenue",
            line=dict(color="#4C9BE8", width=2),
        )
    )

    # Confidence interval band (filled area)
    fig.add_trace(
        go.Scatter(
            x=list(fc_df["date"]) + list(fc_df["date"][::-1]),
            y=list(fc_df["yhat_upper"]) + list(fc_df["yhat_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(255, 140, 0, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="80% Confidence Interval",
            showlegend=True,
        )
    )

    # Forecast line
    fig.add_trace(
        go.Scatter(
            x=fc_df["date"],
            y=fc_df["yhat"],
            mode="lines",
            name="Forecast",
            line=dict(color="#FF8C00", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Raw data expanders
    with st.expander("📋 Historical data table"):
        st.dataframe(hist_df, use_container_width=True)
    with st.expander("📋 Forecast data table"):
        st.dataframe(fc_df, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Top Categories Bar Chart  (from /api/products/top)
# ---------------------------------------------------------------------------

st.subheader("🏅 Top Categories by Revenue")

with st.spinner("Loading top categories …"):
    top_data = _get("/api/products/top", params=top_params)

if top_data:
    top_df = pd.DataFrame(top_data)

    bar_fig = go.Figure(
        go.Bar(
            x=top_df["category"],
            y=top_df["revenue"],
            marker_color="#4C9BE8",
            text=[f"${v:,.0f}" for v in top_df["revenue"]],
            textposition="outside",
        )
    )
    bar_fig.update_layout(
        xaxis_title="Category",
        yaxis_title="Total Revenue (USD)",
        height=350,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    st.plotly_chart(bar_fig, use_container_width=True)

st.markdown("---")
st.caption(
    "Made BY DHRUV BATHLA · "
    "Data source: synthetic e-commerce dataset · Model: Facebook Prophet (SARIMA fallback) · "
    "Confidence intervals: 80% credible interval"
)

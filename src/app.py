"""Minimal Streamlit portfolio demo for forecasts and pricing recommendations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "powerbi_exports"
FORECAST_PATH = EXPORT_DIR / "sales_actuals_vs_forecast.csv"
RECOMMENDATION_PATH = EXPORT_DIR / "pricing_recommendations.csv"


st.set_page_config(
    page_title="Dynamic Pricing Platform",
    page_icon="📈",
    layout="wide",
)


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    """Load dashboard forecast data."""
    if not FORECAST_PATH.exists():
        return pd.DataFrame()
    data = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    required = {
        "date",
        "store",
        "actual",
        "prophet_pred",
        "xgboost_pred",
    }
    if not required.issubset(data.columns):
        return pd.DataFrame()
    return data.sort_values(["store", "date"])


@st.cache_data
def load_recommendations() -> pd.DataFrame:
    """Load segment-level pricing recommendations."""
    if not RECOMMENDATION_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RECOMMENDATION_PATH)


def forecast_panel(forecasts: pd.DataFrame) -> None:
    """Render store-level actual and forecast lines."""
    st.subheader("Demand forecast")
    if forecasts.empty:
        st.info(
            "Forecast exports are not available. Run "
            "`python src/export_for_powerbi.py` after model evaluation."
        )
        return

    stores = sorted(forecasts["store"].dropna().unique().tolist())
    selected_store = st.selectbox("Store", stores)
    selected = forecasts.loc[forecasts["store"].eq(selected_store)].copy()

    if "store_type" in selected and selected["store_type"].notna().any():
        st.caption(f"StoreType: {selected['store_type'].dropna().iloc[0]}")

    chart = (
        selected.set_index("date")[
            ["actual", "prophet_pred", "xgboost_pred"]
        ]
        .rename(
            columns={
                "actual": "Actual",
                "prophet_pred": "Prophet",
                "xgboost_pred": "XGBoost",
            }
        )
    )
    st.line_chart(chart, use_container_width=True)
    st.dataframe(
        selected[
            ["date", "actual", "prophet_pred", "xgboost_pred"]
        ].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )


def recommendation_panel(recommendations: pd.DataFrame) -> None:
    """Render one segment's recommended pricing action and expected effect."""
    st.subheader("Pricing recommendation")
    if recommendations.empty:
        st.info(
            "Recommendation exports are not available. Run "
            "`python src/recommend.py` and `python src/export_for_powerbi.py`."
        )
        return

    segment_column = (
        "segment" if "segment" in recommendations else "store_type"
    )
    segments = sorted(
        recommendations[segment_column].dropna().astype(str).unique().tolist()
    )
    segment = st.selectbox("Store segment", segments)
    row = recommendations.loc[
        recommendations[segment_column].astype(str).eq(segment)
    ].iloc[0]

    margin_lift = float(row.get("expected_margin_lift_pct", 0))
    revenue_change = float(row.get("revenue_change_pct", 0))
    col1, col2 = st.columns(2)
    col1.metric("Expected margin lift", f"{margin_lift:.2%}")
    col2.metric("Projected revenue change", f"{revenue_change:.2%}")

    st.markdown("**Recommended action**")
    st.write(row.get("recommended_action", "Not available"))
    st.markdown("**Current state**")
    st.write(row.get("current_state", "Not available"))
    if pd.notna(row.get("rationale")):
        st.caption(str(row["rationale"]))


st.title("Dynamic Pricing and Promotion Optimization")
st.caption(
    "Rossmann demand forecasting, promotion analysis, and pricing scenarios"
)

forecast_data = load_forecasts()
recommendation_data = load_recommendations()

left, right = st.columns([1.6, 1])
with left:
    forecast_panel(forecast_data)
with right:
    recommendation_panel(recommendation_data)

st.divider()
st.caption(
    "Scenario outputs rely on descriptive elasticity from an average-transaction-"
    "value proxy and should be validated experimentally before business use."
)


"""Streamlit dashboard for forecasts and pricing recommendations."""

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
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


def _demo_forecasts() -> pd.DataFrame:
    """Return a small built-in dataset for Streamlit Cloud demo mode."""
    dates = pd.date_range("2025-01-01", periods=12, freq="W")
    rows = []
    stores = [
        (101, "A", 7400, 210),
        (214, "B", 6200, 145),
        (389, "C", 5100, 120),
    ]
    weekly_shape = [0, 180, 420, 260, -120, -280, 90, 360, 210, -80, 140, 320]

    for store, store_type, base_sales, trend in stores:
        for idx, date in enumerate(dates):
            actual = base_sales + (idx * trend) + weekly_shape[idx]
            prophet_pred = actual * (0.97 + ((idx % 3) * 0.015))
            xgboost_pred = actual * (1.01 - ((idx % 4) * 0.01))
            rows.append(
                {
                    "date": date,
                    "store": store,
                    "store_type": store_type,
                    "actual": round(actual, 0),
                    "prophet_pred": round(prophet_pred, 0),
                    "xgboost_pred": round(xgboost_pred, 0),
                }
            )

    return pd.DataFrame(rows)


def _demo_recommendations() -> pd.DataFrame:
    """Return representative segment recommendations for demo mode."""
    return pd.DataFrame(
        [
            {
                "segment": "StoreType A",
                "expected_margin_lift_pct": 0.041,
                "revenue_change_pct": -0.006,
                "recommended_action": "Reduce blanket discounting and keep promotions for high-lift holiday periods.",
                "current_state": "High baseline demand with moderate promotion dependency.",
                "rationale": "Scenario search favors a smaller discount depth while staying above the revenue floor.",
            },
            {
                "segment": "StoreType B",
                "expected_margin_lift_pct": 0.027,
                "revenue_change_pct": 0.004,
                "recommended_action": "Keep the current price level and shift promotional budget to weekends.",
                "current_state": "Stable revenue and lower estimated price sensitivity.",
                "rationale": "The demo scenario preserves demand while improving contribution margin.",
            },
            {
                "segment": "StoreType C",
                "expected_margin_lift_pct": 0.018,
                "revenue_change_pct": -0.011,
                "recommended_action": "Review manually before changing price because elasticity evidence is weaker.",
                "current_state": "Lower volume segment with noisier transaction-value proxy.",
                "rationale": "The modeled lift is positive but close to the configured risk threshold.",
            },
        ]
    )


@st.cache_data
def load_forecasts() -> tuple[pd.DataFrame, bool]:
    """Load dashboard forecast data, falling back to demo data when needed."""
    if not FORECAST_PATH.exists():
        return _demo_forecasts(), True

    try:
        data = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    except Exception:
        return _demo_forecasts(), True

    required = {
        "date",
        "store",
        "actual",
        "prophet_pred",
        "xgboost_pred",
    }
    if not required.issubset(data.columns) or data.empty:
        return _demo_forecasts(), True

    return data.sort_values(["store", "date"]), False


@st.cache_data
def load_recommendations() -> tuple[pd.DataFrame, bool]:
    """Load segment-level pricing recommendations, or demo data."""
    if not RECOMMENDATION_PATH.exists():
        return _demo_recommendations(), True

    try:
        data = pd.read_csv(RECOMMENDATION_PATH)
    except Exception:
        return _demo_recommendations(), True

    if data.empty:
        return _demo_recommendations(), True

    return data, False


def forecast_panel(forecasts: pd.DataFrame) -> None:
    """Render store-level actual and forecast lines."""
    st.subheader("Demand forecast")

    stores = sorted(forecasts["store"].dropna().unique().tolist())
    selected_store = st.selectbox("Store", stores)
    selected = forecasts.loc[forecasts["store"].eq(selected_store)].copy()

    if "store_type" in selected and selected["store_type"].notna().any():
        st.caption(f"Store type: {selected['store_type'].dropna().iloc[0]}")

    chart = (
        selected.set_index("date")[["actual", "prophet_pred", "xgboost_pred"]]
        .rename(
            columns={
                "actual": "Actual",
                "prophet_pred": "Prophet",
                "xgboost_pred": "XGBoost",
            }
        )
    )
    st.line_chart(chart, width="stretch")
    st.dataframe(
        selected[["date", "actual", "prophet_pred", "xgboost_pred"]].sort_values(
            "date", ascending=False
        ),
        width="stretch",
        hide_index=True,
    )


def recommendation_panel(recommendations: pd.DataFrame) -> None:
    """Render one segment's recommended pricing action and expected effect."""
    st.subheader("Pricing recommendation")

    segment_column = "segment" if "segment" in recommendations else "store_type"
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
st.caption("Rossmann demand forecasting, promotion analysis, and pricing scenarios")

forecast_data, forecast_demo_mode = load_forecasts()
recommendation_data, recommendation_demo_mode = load_recommendations()

if forecast_demo_mode or recommendation_demo_mode:
    st.info(
        "Demo mode is active because generated dashboard exports were not found "
        "or could not be read. The full pipeline remains available for local "
        "execution with Kaggle data and PostgreSQL."
    )

total_actual_sales = forecast_data["actual"].sum()
avg_xgboost_gap = (
    (forecast_data["xgboost_pred"] - forecast_data["actual"]).abs()
    / forecast_data["actual"].replace(0, pd.NA)
).mean()
best_margin_lift = recommendation_data["expected_margin_lift_pct"].max()

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Dashboard sales", f"{total_actual_sales:,.0f}")
metric_2.metric("Avg XGBoost error", f"{avg_xgboost_gap:.1%}")
metric_3.metric("Best margin lift", f"{best_margin_lift:.1%}")

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

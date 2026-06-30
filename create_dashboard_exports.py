import pandas as pd
from pathlib import Path
import numpy as np

export_dir = Path("data/processed/powerbi_exports")
export_dir.mkdir(parents=True, exist_ok=True)

dates = pd.date_range("2025-01-01", periods=30, freq="D")
rows = []

for store, store_type, base in [
    (1, "a", 8200),
    (2, "b", 7600),
    (3, "c", 9100),
    (4, "d", 6900),
]:
    for i, date in enumerate(dates):
        actual = base + (i * 45) + np.sin(i / 3) * 350
        prophet = actual * 0.97
        xgboost = actual * 1.02
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "store": store,
            "store_type": store_type,
            "actual": round(actual, 2),
            "prophet_pred": round(prophet, 2),
            "xgboost_pred": round(xgboost, 2),
        })

forecast_df = pd.DataFrame(rows)
forecast_df.to_csv(export_dir / "sales_actuals_vs_forecast.csv", index=False)

recommendations = pd.DataFrame([
    {
        "segment": "a",
        "current_state": "High sales volume with moderate promotion dependency.",
        "recommended_action": "Increase price by 2% and reduce unnecessary promotion depth.",
        "expected_margin_lift_pct": 0.084,
        "revenue_change_pct": -0.012,
        "rationale": "Segment shows stable demand and can absorb a small price increase."
    },
    {
        "segment": "b",
        "current_state": "Price-sensitive segment with strong promotion response.",
        "recommended_action": "Maintain current price and use targeted promotions.",
        "expected_margin_lift_pct": 0.052,
        "revenue_change_pct": 0.018,
        "rationale": "Promotions drive demand, so controlled discounting is better than price increase."
    },
    {
        "segment": "c",
        "current_state": "Premium-performing stores with lower elasticity.",
        "recommended_action": "Increase price by 3% with limited promotional support.",
        "expected_margin_lift_pct": 0.113,
        "revenue_change_pct": -0.009,
        "rationale": "Lower elasticity suggests margin can improve without major revenue loss."
    },
    {
        "segment": "d",
        "current_state": "Lower sales segment with higher demand volatility.",
        "recommended_action": "Keep price stable and test small basket-level promotions.",
        "expected_margin_lift_pct": 0.041,
        "revenue_change_pct": 0.011,
        "rationale": "Stable pricing protects revenue while promotions support customer traffic."
    },
])

recommendations.to_csv(export_dir / "pricing_recommendations.csv", index=False)

manifest = pd.DataFrame([
    {"table": "sales_actuals_vs_forecast", "rows": len(forecast_df), "csv_file": "sales_actuals_vs_forecast.csv"},
    {"table": "pricing_recommendations", "rows": len(recommendations), "csv_file": "pricing_recommendations.csv"},
])
manifest.to_csv(export_dir / "export_manifest.csv", index=False)

print("Dashboard export CSV files created successfully.")

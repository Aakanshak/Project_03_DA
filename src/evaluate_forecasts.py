"""Evaluate Prophet, XGBoost, and seasonal-naive Rossmann forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from forecast_prophet import run_prophet_forecasts
from forecast_utils import (
    FIGURE_DIR,
    REPORT_DIR,
    ensure_output_directories,
    load_features,
    regression_metrics,
)
from forecast_xgboost import run_xgboost_forecast


def build_naive_predictions(
    df: pd.DataFrame, evaluation_keys: pd.DataFrame
) -> pd.DataFrame:
    """Predict each evaluation row with the same store's sales seven days earlier."""
    lag_lookup = df[["store_id", "sales_date", "sales"]].copy()
    lag_lookup["sales_date"] = lag_lookup["sales_date"] + pd.Timedelta(days=7)
    lag_lookup = lag_lookup.rename(columns={"sales": "prediction"})
    output = evaluation_keys.merge(
        lag_lookup, on=["store_id", "sales_date"], how="left", validate="one_to_one"
    )
    output["model"] = "Naive seasonal lag-7"
    return output.dropna(subset=["prediction"])


def _save_plots(predictions: pd.DataFrame, count: int = 3) -> None:
    """Save actual-vs-predicted charts for representative evaluation stores."""
    store_volume = (
        predictions.groupby("store_id")["actual"].sum().sort_values(ascending=False)
    )
    store_ids = store_volume.head(count).index.tolist()
    for position, store_id in enumerate(store_ids, start=1):
        store_data = predictions.loc[predictions["store_id"].eq(store_id)].copy()
        actual = (
            store_data[["sales_date", "actual"]]
            .drop_duplicates("sales_date")
            .sort_values("sales_date")
        )
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(
            actual["sales_date"],
            actual["actual"],
            color="black",
            linewidth=2,
            label="Actual",
        )
        for model, model_data in store_data.groupby("model"):
            model_data = model_data.sort_values("sales_date")
            ax.plot(
                model_data["sales_date"],
                model_data["prediction"],
                linewidth=1.5,
                label=model,
            )
        ax.set(
            title=f"Store {store_id}: Actual vs Predicted Sales",
            xlabel="",
            ylabel="Sales",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"forecast_actual_vs_predicted_{position}_store_{store_id}.png",
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(fig)


def evaluate(feature_path: Path | None = None) -> pd.DataFrame:
    """Run forecasts, evaluate a common store/date sample, and save artifacts."""
    ensure_output_directories()
    features = load_features(feature_path) if feature_path else load_features()
    prophet = run_prophet_forecasts(feature_path=feature_path)
    xgboost = run_xgboost_forecast(feature_path=feature_path)

    # Prophet covers the representative sample, so all models are compared on
    # exactly the same store/date rows.
    evaluation_keys = prophet[["store_id", "sales_date", "actual"]].copy()
    xgboost_common = evaluation_keys[["store_id", "sales_date"]].merge(
        xgboost.drop(columns=["actual"], errors="ignore"),
        on=["store_id", "sales_date"],
        how="inner",
        validate="one_to_one",
    )
    xgboost_common = xgboost_common.merge(
        evaluation_keys,
        on=["store_id", "sales_date"],
        how="left",
        validate="one_to_one",
    )
    naive = build_naive_predictions(features, evaluation_keys)

    common_keys = (
        prophet[["store_id", "sales_date"]]
        .merge(
            xgboost_common[["store_id", "sales_date"]],
            on=["store_id", "sales_date"],
            how="inner",
        )
        .merge(
            naive[["store_id", "sales_date"]],
            on=["store_id", "sales_date"],
            how="inner",
        )
        .drop_duplicates()
    )
    if common_keys.empty:
        raise RuntimeError("No common store/date rows exist across all forecasts.")

    prediction_sets = []
    for predictions in (prophet, xgboost_common, naive):
        aligned = common_keys.merge(
            predictions,
            on=["store_id", "sales_date"],
            how="left",
            validate="one_to_one",
        )
        prediction_sets.append(aligned)
    comparison_rows = []
    for predictions in prediction_sets:
        model_name = predictions["model"].iloc[0]
        metrics = regression_metrics(
            predictions["actual"], predictions["prediction"]
        )
        comparison_rows.append({"model": model_name, **metrics})

    comparison = pd.DataFrame(comparison_rows).sort_values("MAPE").reset_index(
        drop=True
    )
    naive_mape = comparison.loc[
        comparison["model"].eq("Naive seasonal lag-7"), "MAPE"
    ].iloc[0]
    model_mask = ~comparison["model"].eq("Naive seasonal lag-7")
    best_index = comparison.loc[model_mask, "MAPE"].idxmin()
    comparison["mape_improvement_vs_naive_pct"] = (
        (naive_mape - comparison["MAPE"]) / naive_mape * 100
    )
    comparison.to_csv(REPORT_DIR / "forecast_comparison.csv", index=False)

    combined = pd.concat(prediction_sets, ignore_index=True)
    _save_plots(combined)

    best = comparison.loc[best_index]
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:,.2f}"))
    improvement = best["mape_improvement_vs_naive_pct"]
    if improvement >= 0:
        headline = (
            f"{best['model']} improves MAPE by {improvement:.2f}% "
            "versus the naive baseline."
        )
    else:
        headline = (
            f"{best['model']} has {-improvement:.2f}% worse MAPE "
            "than the naive baseline."
        )
    print(f"\nHeadline: {headline}")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=None)
    args = parser.parse_args()
    evaluate(args.features)


if __name__ == "__main__":
    main()

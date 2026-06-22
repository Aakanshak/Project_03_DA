"""Create flat, validated Power BI exports from project analysis outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = PROJECT_ROOT / "reports" / "predictions"
PROPHET_PATH = PREDICTION_DIR / "prophet_predictions.csv"
XGBOOST_PATH = PREDICTION_DIR / "xgboost_predictions.csv"
ELASTICITY_PATH = PROJECT_ROOT / "reports" / "elasticity_by_segment.csv"
PROMO_PATH = PROJECT_ROOT / "reports" / "promo_impact_by_segment.csv"
RECOMMENDATION_PATH = PROJECT_ROOT / "data" / "processed" / "recommendations.csv"
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "powerbi_exports"


def _require_file(path: Path) -> None:
    """Raise a clear error when an upstream pipeline output is unavailable."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required input not found: {path}. Run its upstream analysis first."
        )


def _require_columns(
    dataframe: pd.DataFrame, required: set[str], source: Path
) -> None:
    """Validate columns expected from an upstream output."""
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"{source} is missing columns: {', '.join(sorted(missing))}"
        )


def _normalize_predictions(path: Path, prediction_name: str) -> pd.DataFrame:
    """Normalize one model's prediction file to the Power BI merge grain."""
    _require_file(path)
    predictions = pd.read_csv(path)
    _require_columns(
        predictions,
        {"store_id", "sales_date", "actual", "prediction"},
        path,
    )
    predictions = predictions.copy()
    predictions["sales_date"] = pd.to_datetime(
        predictions["sales_date"], errors="raise"
    )
    if predictions.duplicated(["store_id", "sales_date"]).any():
        raise ValueError(
            f"{path} contains duplicate store/date prediction rows."
        )

    columns = ["store_id", "sales_date", "actual", "prediction"]
    if "store_type" in predictions:
        columns.append("store_type")
    return predictions[columns].rename(
        columns={
            "actual": f"actual_{prediction_name}",
            "prediction": f"{prediction_name}_pred",
            "store_type": f"store_type_{prediction_name}",
        }
    )


def build_sales_actuals_vs_forecast(
    prophet_path: Path = PROPHET_PATH,
    xgboost_path: Path = XGBOOST_PATH,
) -> pd.DataFrame:
    """Merge Prophet and XGBoost predictions into one store-date fact table.

    An outer merge preserves the global XGBoost coverage while retaining
    Prophet values for its representative store sample. Blank Prophet values
    are expected for stores outside that sample.
    """
    prophet = _normalize_predictions(prophet_path, "prophet")
    xgboost = _normalize_predictions(xgboost_path, "xgboost")
    merged = prophet.merge(
        xgboost,
        on=["store_id", "sales_date"],
        how="outer",
        validate="one_to_one",
    )

    both_actuals = merged[
        ["actual_prophet", "actual_xgboost"]
    ].dropna()
    if not both_actuals.empty and not np.allclose(
        both_actuals["actual_prophet"],
        both_actuals["actual_xgboost"],
        rtol=0,
        atol=1e-6,
    ):
        raise ValueError(
            "Prophet and XGBoost files disagree on actual sales values."
        )

    merged["actual"] = merged["actual_xgboost"].combine_first(
        merged["actual_prophet"]
    )
    store_type_columns = [
        column
        for column in ("store_type_xgboost", "store_type_prophet")
        if column in merged
    ]
    if store_type_columns:
        merged["store_type"] = merged[store_type_columns[0]]
        for column in store_type_columns[1:]:
            merged["store_type"] = merged["store_type"].combine_first(
                merged[column]
            )
    else:
        merged["store_type"] = pd.NA

    output = merged[
        [
            "sales_date",
            "store_id",
            "store_type",
            "actual",
            "prophet_pred",
            "xgboost_pred",
        ]
    ].rename(columns={"sales_date": "date", "store_id": "store"})
    return output.sort_values(["date", "store"]).reset_index(drop=True)


def _write_csv_and_parquet(dataframe: pd.DataFrame, csv_path: Path) -> None:
    """Write matching CSV and Parquet versions of an export table."""
    dataframe.to_csv(csv_path, index=False, date_format="%Y-%m-%d")
    dataframe.to_parquet(csv_path.with_suffix(".parquet"), index=False)


def _copy_analysis_table(
    source: Path, destination_name: str, required_columns: set[str]
) -> pd.DataFrame:
    """Validate and copy an existing segment-level output."""
    _require_file(source)
    dataframe = pd.read_csv(source)
    _require_columns(dataframe, required_columns, source)
    destination = EXPORT_DIR / destination_name
    _write_csv_and_parquet(dataframe, destination)
    return dataframe


def export_for_powerbi(
    prophet_path: Path = PROPHET_PATH,
    xgboost_path: Path = XGBOOST_PATH,
    elasticity_path: Path = ELASTICITY_PATH,
    promo_path: Path = PROMO_PATH,
    recommendation_path: Path = RECOMMENDATION_PATH,
) -> dict[str, int]:
    """Build all dashboard-ready exports and return their row counts."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    forecasts = build_sales_actuals_vs_forecast(prophet_path, xgboost_path)
    _write_csv_and_parquet(
        forecasts, EXPORT_DIR / "sales_actuals_vs_forecast.csv"
    )

    elasticity = _copy_analysis_table(
        elasticity_path,
        "elasticity_by_segment.csv",
        {"store_type", "elasticity", "ci_lower", "ci_upper"},
    )
    promo = _copy_analysis_table(
        promo_path,
        "promo_impact_by_segment.csv",
        {"store_type", "promo_lift_pct", "net_promo_margin_impact"},
    )
    recommendations = _copy_analysis_table(
        recommendation_path,
        "pricing_recommendations.csv",
        {
            "segment",
            "current_state",
            "recommended_action",
            "expected_margin_lift_pct",
        },
    )

    row_counts = {
        "sales_actuals_vs_forecast": len(forecasts),
        "elasticity_by_segment": len(elasticity),
        "promo_impact_by_segment": len(promo),
        "pricing_recommendations": len(recommendations),
    }
    manifest = pd.DataFrame(
        [
            {
                "table": table,
                "rows": rows,
                "csv_file": f"{table}.csv",
                "parquet_file": f"{table}.parquet",
            }
            for table, rows in row_counts.items()
        ]
    )
    manifest.to_csv(EXPORT_DIR / "export_manifest.csv", index=False)
    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prophet", type=Path, default=PROPHET_PATH)
    parser.add_argument("--xgboost", type=Path, default=XGBOOST_PATH)
    parser.add_argument("--elasticity", type=Path, default=ELASTICITY_PATH)
    parser.add_argument("--promo", type=Path, default=PROMO_PATH)
    parser.add_argument(
        "--recommendations", type=Path, default=RECOMMENDATION_PATH
    )
    args = parser.parse_args()
    counts = export_for_powerbi(
        args.prophet,
        args.xgboost,
        args.elasticity,
        args.promo,
        args.recommendations,
    )
    for table, rows in counts.items():
        print(f"{table}: {rows:,} rows")
    print(f"Power BI exports saved to {EXPORT_DIR}")


if __name__ == "__main__":
    main()

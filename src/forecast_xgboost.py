"""Global leakage-safe XGBoost forecasting for Rossmann sales."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from forecast_utils import (
    HOLDOUT_DAYS,
    MODEL_DIR,
    PREDICTION_DIR,
    ensure_output_directories,
    holdout_cutoff,
    load_features,
    state_holiday_indicator,
)


LAG_COLUMNS = [
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "sales_rolling_mean_7",
    "sales_rolling_std_7",
    "sales_rolling_mean_28",
    "sales_rolling_std_28",
]

EXCLUDED_COLUMNS = {
    "sales",
    "sales_date",
    "customers",
    "avg_transaction_value",
    "avg_transaction_value_rolling_mean_7",
    "competition_open_date",
    "open",
}


def _prepare_known_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known-at-forecast-time columns."""
    prepared = df.copy()
    if "state_holiday" in prepared:
        prepared["state_holiday"] = (
            prepared["state_holiday"].fillna("0").astype(str).str.lower()
        )
    return prepared


def _feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Select numeric and categorical predictors without target leakage."""
    candidates = [
        column
        for column in df.columns
        if column not in EXCLUDED_COLUMNS and column != "prediction"
    ]
    categorical = [
        column
        for column in candidates
        if pd.api.types.is_object_dtype(df[column])
        or isinstance(df[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [column for column in candidates if column not in categorical]
    return numeric, categorical


def _recursive_lag_values(history: list[float]) -> dict[str, float]:
    """Calculate target-derived predictors from prior actual/predicted values only."""
    values: dict[str, float] = {}
    for lag in (7, 14, 28):
        values[f"sales_lag_{lag}"] = history[-lag] if len(history) >= lag else np.nan
    for window in (7, 28):
        prior = np.asarray(history[-window:], dtype=float)
        values[f"sales_rolling_mean_{window}"] = (
            float(prior.mean()) if len(prior) else np.nan
        )
        values[f"sales_rolling_std_{window}"] = (
            float(prior.std(ddof=1)) if len(prior) >= 2 else np.nan
        )
    return values


def run_xgboost_forecast(
    feature_path: Path | None = None,
    holdout_days: int = HOLDOUT_DAYS,
) -> pd.DataFrame:
    """Train globally and recursively forecast every row in the final holdout."""
    ensure_output_directories()
    df = load_features(feature_path) if feature_path else load_features()
    df = _prepare_known_features(df)
    cutoff = holdout_cutoff(df, holdout_days)
    train = df.loc[df["sales_date"] < cutoff].copy()
    test = df.loc[df["sales_date"] >= cutoff].copy()

    numeric, categorical = _feature_columns(train)
    missing_lags = set(LAG_COLUMNS).difference(numeric)
    if missing_lags:
        raise ValueError(
            "Missing engineered lag columns: " + ", ".join(sorted(missing_lags))
        )

    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    model = XGBRegressor(
        n_estimators=700,
        learning_rate=0.04,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", transformer), ("model", model)])
    pipeline.fit(train[numeric + categorical], train["sales"].astype(float))

    histories: dict[int, list[float]] = defaultdict(list)
    for store_id, values in train.groupby("store_id", sort=False)["sales"]:
        histories[int(store_id)] = values.astype(float).tolist()

    prediction_rows: list[dict[str, object]] = []
    for date in sorted(test["sales_date"].unique()):
        day_rows = test.loc[test["sales_date"].eq(date)].copy()
        for index, row in day_rows.iterrows():
            store_id = int(row["store_id"])
            lag_values = _recursive_lag_values(histories[store_id])
            for column, value in lag_values.items():
                day_rows.at[index, column] = value

        day_predictions = np.clip(
            pipeline.predict(day_rows[numeric + categorical]), 0, None
        )
        for (_, row), prediction in zip(day_rows.iterrows(), day_predictions):
            store_id = int(row["store_id"])
            histories[store_id].append(float(prediction))
            prediction_rows.append(
                {
                    "store_id": store_id,
                    "sales_date": row["sales_date"],
                    "actual": float(row["sales"]),
                    "prediction": float(prediction),
                    "model": "XGBoost",
                    **(
                        {"store_type": row["store_type"]}
                        if "store_type" in row.index
                        else {}
                    ),
                }
            )

    output = pd.DataFrame(prediction_rows)
    output.to_csv(PREDICTION_DIR / "xgboost_predictions.csv", index=False)
    joblib.dump(pipeline, MODEL_DIR / "xgboost_pipeline.joblib")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=None)
    args = parser.parse_args()
    output = run_xgboost_forecast(args.features)
    print(f"Saved {len(output):,} XGBoost predictions.")


if __name__ == "__main__":
    main()

"""Per-store Prophet demand forecasts for a representative Rossmann sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from forecast_utils import (
    HOLDOUT_DAYS,
    MODEL_DIR,
    PREDICTION_DIR,
    ensure_output_directories,
    holdout_cutoff,
    load_features,
    sample_representative_stores,
    state_holiday_indicator,
)


def _prophet_frame(store_df: pd.DataFrame) -> pd.DataFrame:
    """Map project columns to Prophet's expected input format."""
    frame = pd.DataFrame(
        {
            "ds": store_df["sales_date"],
            "y": store_df["sales"].astype(float),
            "promo": pd.to_numeric(store_df["promo"], errors="coerce").fillna(0),
            "school_holiday": pd.to_numeric(
                store_df.get("school_holiday", 0), errors="coerce"
            ).fillna(0),
            "state_holiday": state_holiday_indicator(
                store_df.get(
                    "state_holiday",
                    pd.Series("0", index=store_df.index),
                )
            ),
        }
    )
    return frame


def run_prophet_forecasts(
    feature_path: Path | None = None,
    sample_size: int = 20,
    holdout_days: int = HOLDOUT_DAYS,
) -> pd.DataFrame:
    """Fit one Prophet model per sampled store and predict the final six weeks."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise RuntimeError(
            "Prophet is not installed. Install requirements.txt before running."
        ) from exc

    ensure_output_directories()
    df = load_features(feature_path) if feature_path else load_features()
    cutoff = holdout_cutoff(df, holdout_days)
    store_ids = sample_representative_stores(df, sample_size=sample_size)
    predictions: list[pd.DataFrame] = []

    for store_id in store_ids:
        store_df = df.loc[df["store_id"].eq(store_id)].sort_values("sales_date")
        prophet_df = _prophet_frame(store_df)
        train = prophet_df.loc[prophet_df["ds"] < cutoff].copy()
        test = prophet_df.loc[prophet_df["ds"] >= cutoff].copy()
        if train.empty or test.empty:
            continue

        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            interval_width=0.80,
        )
        for regressor in ("promo", "school_holiday", "state_holiday"):
            model.add_regressor(regressor)
        model.fit(train)

        forecast = model.predict(
            test[["ds", "promo", "school_holiday", "state_holiday"]]
        )
        result = pd.DataFrame(
            {
                "store_id": store_id,
                "sales_date": test["ds"].to_numpy(),
                "actual": test["y"].to_numpy(),
                "prediction": forecast["yhat"].clip(lower=0).to_numpy(),
                "model": "Prophet",
            }
        )
        if "store_type" in store_df:
            result["store_type"] = store_df["store_type"].iloc[0]
        predictions.append(result)

    if not predictions:
        raise RuntimeError("No sampled stores contained both training and holdout rows.")

    output = pd.concat(predictions, ignore_index=True)
    output.to_csv(PREDICTION_DIR / "prophet_predictions.csv", index=False)
    metadata = {
        "holdout_start": str(cutoff.date()),
        "holdout_end": str(df["sales_date"].max().date()),
        "stores": store_ids,
    }
    (MODEL_DIR / "prophet_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    output = run_prophet_forecasts(args.features, args.sample_size)
    print(f"Saved {len(output):,} Prophet predictions.")


if __name__ == "__main__":
    main()


"""Shared utilities for Rossmann forecasting workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
PREDICTION_DIR = REPORT_DIR / "predictions"
MODEL_DIR = PROJECT_ROOT / "models"
HOLDOUT_DAYS = 42


def ensure_output_directories() -> None:
    """Create directories used by model and reporting artifacts."""
    for path in (REPORT_DIR, FIGURE_DIR, PREDICTION_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_features(path: Path = FEATURE_PATH) -> pd.DataFrame:
    """Load and validate the engineered feature table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. Run notebooks/01_eda.ipynb "
            "or save the engineered data there before forecasting."
        )

    df = pd.read_parquet(path)
    required = {"store_id", "sales_date", "sales", "promo"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Feature dataset is missing columns: " + ", ".join(sorted(missing))
        )

    df = df.copy()
    df["sales_date"] = pd.to_datetime(df["sales_date"], errors="raise")
    df = df.sort_values(["sales_date", "store_id"]).reset_index(drop=True)
    if df.duplicated(["store_id", "sales_date"]).any():
        raise ValueError("Duplicate store_id/sales_date rows found.")
    return df


def holdout_cutoff(df: pd.DataFrame, holdout_days: int = HOLDOUT_DAYS) -> pd.Timestamp:
    """Return the inclusive first date of the final holdout window."""
    max_date = pd.Timestamp(df["sales_date"].max()).normalize()
    cutoff = max_date - pd.Timedelta(days=holdout_days - 1)
    if not (df["sales_date"] < cutoff).any():
        raise ValueError("Dataset is too short for the requested holdout window.")
    return cutoff


def state_holiday_indicator(series: pd.Series) -> pd.Series:
    """Convert Rossmann state-holiday labels into a binary indicator."""
    return (
        series.fillna("0").astype(str).str.strip().str.lower().ne("0").astype("int8")
    )


def sample_representative_stores(
    df: pd.DataFrame, sample_size: int = 20, random_state: int = 42
) -> list[int]:
    """Sample stores across StoreType groups with deterministic allocation."""
    stores = df[["store_id"] + (["store_type"] if "store_type" in df else [])]
    stores = stores.drop_duplicates("store_id").copy()
    sample_size = min(sample_size, len(stores))

    if "store_type" not in stores or stores["store_type"].isna().all():
        return (
            stores.sample(sample_size, random_state=random_state)["store_id"]
            .astype(int)
            .sort_values()
            .tolist()
        )

    groups = list(stores.groupby("store_type", dropna=False, sort=True))
    allocations = {name: 1 for name, _ in groups}
    remaining = sample_size - len(allocations)
    if remaining < 0:
        selected_groups = groups[:sample_size]
        allocations = {name: 1 for name, _ in selected_groups}
        groups = selected_groups
        remaining = 0

    available = {name: max(len(group) - allocations[name], 0) for name, group in groups}
    while remaining > 0 and sum(available.values()) > 0:
        for name, _ in groups:
            if remaining == 0:
                break
            if available[name] > 0:
                allocations[name] += 1
                available[name] -= 1
                remaining -= 1

    selected: list[int] = []
    for offset, (name, group) in enumerate(groups):
        count = min(allocations[name], len(group))
        chosen = group.sample(count, random_state=random_state + offset)
        selected.extend(chosen["store_id"].astype(int).tolist())
    return sorted(selected)


def regression_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Calculate MAE, RMSE, and zero-safe MAPE."""
    actual_values = pd.to_numeric(actual, errors="coerce").to_numpy(dtype=float)
    predicted_values = pd.to_numeric(predicted, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[valid]
    predicted_values = predicted_values[valid]
    if len(actual_values) == 0:
        raise ValueError("No valid actual/predicted pairs were supplied.")

    errors = actual_values - predicted_values
    nonzero = actual_values != 0
    mape = (
        np.mean(np.abs(errors[nonzero] / actual_values[nonzero])) * 100
        if nonzero.any()
        else np.nan
    )
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(np.square(errors)))),
        "MAPE": float(mape),
        "observations": int(len(actual_values)),
    }


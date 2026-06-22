"""Feature engineering for Rossmann daily store sales."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "store_id",
    "sales_date",
    "sales",
    "customers",
    "promo",
    "state_holiday",
}


def _validate_input(df: pd.DataFrame) -> None:
    """Validate the columns and row grain required by the feature pipeline."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    duplicate_rows = df.duplicated(["store_id", "sales_date"])
    if duplicate_rows.any():
        raise ValueError(
            "Expected one row per store and sales_date; duplicate keys were found."
        )


def _holiday_distance_features(
    dates: pd.Series, state_holiday: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Return days since the previous and until the next observed holiday."""
    normalized_dates = pd.to_datetime(dates).dt.normalize()
    holiday_flag = (
        state_holiday.fillna("0").astype(str).str.strip().str.lower().ne("0")
    )
    holiday_dates = np.sort(normalized_dates.loc[holiday_flag].unique())

    if len(holiday_dates) == 0:
        empty = pd.Series(pd.NA, index=dates.index, dtype="Int64")
        return empty.copy(), empty.copy()

    date_values = normalized_dates.to_numpy(dtype="datetime64[D]")
    holiday_values = holiday_dates.astype("datetime64[D]")
    insertion_points = np.searchsorted(holiday_values, date_values)

    previous_index = insertion_points - 1
    exact_match = (
        (insertion_points < len(holiday_values))
        & (holiday_values[np.minimum(insertion_points, len(holiday_values) - 1)]
           == date_values)
    )
    previous_index = np.where(exact_match, insertion_points, previous_index)
    next_index = np.where(exact_match, insertion_points, insertion_points)

    days_from = np.full(len(dates), np.nan)
    has_previous = previous_index >= 0
    days_from[has_previous] = (
        date_values[has_previous] - holiday_values[previous_index[has_previous]]
    ).astype(int)

    days_to = np.full(len(dates), np.nan)
    has_next = next_index < len(holiday_values)
    days_to[has_next] = (
        holiday_values[next_index[has_next]] - date_values[has_next]
    ).astype(int)

    return (
        pd.Series(days_from, index=dates.index).astype("Int64"),
        pd.Series(days_to, index=dates.index).astype("Int64"),
    )


def _competition_open_date(df: pd.DataFrame) -> pd.Series:
    """Construct the first day of the recorded competition-open month."""
    if not {
        "competition_open_since_year",
        "competition_open_since_month",
    }.issubset(df.columns):
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    year = pd.to_numeric(
        df["competition_open_since_year"], errors="coerce"
    ).astype("Int64")
    month = pd.to_numeric(
        df["competition_open_since_month"], errors="coerce"
    ).astype("Int64")
    valid = year.notna() & month.between(1, 12)

    result = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    result.loc[valid] = pd.to_datetime(
        {
            "year": year.loc[valid].astype(int),
            "month": month.loc[valid].astype(int),
            "day": 1,
        },
        errors="coerce",
    ).to_numpy()
    return result


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe calendar, demand, promotion, and competition features.

    Rolling and lagged sales features use only observations strictly before the
    current row. Holiday distance is based on dates marked by ``state_holiday``
    in the supplied dataset.

    Args:
        df: Daily store-level sales joined to store metadata.

    Returns:
        A copy sorted by store/date with engineered feature columns.
    """
    _validate_input(df)
    features = df.copy()
    features["sales_date"] = pd.to_datetime(features["sales_date"], errors="raise")
    features = features.sort_values(["store_id", "sales_date"]).reset_index(drop=True)

    features["day_of_week"] = features["sales_date"].dt.dayofweek.astype("int8")
    features["month"] = features["sales_date"].dt.month.astype("int8")
    features["week_of_year"] = (
        features["sales_date"].dt.isocalendar().week.astype("int16")
    )
    features["is_weekend"] = features["day_of_week"].ge(5).astype("int8")
    features["days_from_holiday"], features["days_to_holiday"] = (
        _holiday_distance_features(
            features["sales_date"], features["state_holiday"]
        )
    )

    store_groups = features.groupby("store_id", sort=False)
    for lag in (7, 14, 28):
        features[f"sales_lag_{lag}"] = store_groups["sales"].shift(lag)

    shifted_sales = store_groups["sales"].shift(1)
    shifted_groups = shifted_sales.groupby(features["store_id"], sort=False)
    for window in (7, 28):
        features[f"sales_rolling_mean_{window}"] = shifted_groups.transform(
            lambda series: series.rolling(window, min_periods=1).mean()
        )
        features[f"sales_rolling_std_{window}"] = shifted_groups.transform(
            lambda series: series.rolling(window, min_periods=2).std()
        )

    features["promo"] = (
        pd.to_numeric(features["promo"], errors="coerce").fillna(0).astype("int8")
    )
    if "promo2" not in features:
        features["promo2"] = 0
    features["promo2"] = (
        pd.to_numeric(features["promo2"], errors="coerce").fillna(0).astype("int8")
    )

    promo_dates = features["sales_date"].where(features["promo"].eq(1))
    last_promo_date = promo_dates.groupby(features["store_id"]).ffill()
    features["days_since_last_promo"] = (
        features["sales_date"] - last_promo_date
    ).dt.days.astype("Int64")

    customers = pd.to_numeric(features["customers"], errors="coerce")
    sales = pd.to_numeric(features["sales"], errors="coerce")
    features["avg_transaction_value"] = sales.div(customers.where(customers.gt(0)))
    shifted_atv = features.groupby("store_id", sort=False)[
        "avg_transaction_value"
    ].shift(1)
    features["avg_transaction_value_rolling_mean_7"] = shifted_atv.groupby(
        features["store_id"], sort=False
    ).transform(lambda series: series.rolling(7, min_periods=1).mean())

    if "competition_distance" not in features:
        features["competition_distance"] = np.nan
    features["competition_distance"] = pd.to_numeric(
        features["competition_distance"], errors="coerce"
    )

    features["competition_open_date"] = _competition_open_date(features)
    competition_days = (
        features["sales_date"] - features["competition_open_date"]
    ).dt.days
    features["competition_open_since"] = (
        competition_days.clip(lower=0).div(30.4375).round().astype("Int64")
    )

    return features


"""Estimate descriptive price elasticity by Rossmann store segment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "analysis_config.yaml"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "elasticity_by_segment.csv"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load analysis assumptions and validation thresholds."""
    if not path.exists():
        raise FileNotFoundError(f"Analysis configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_feature_data(path: Path = DEFAULT_FEATURE_PATH) -> pd.DataFrame:
    """Load the model-ready feature table with required elasticity fields."""
    if not path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {path}. Build features.parquet first."
        )
    df = pd.read_parquet(path)
    required = {
        "store_id",
        "sales_date",
        "sales",
        "avg_transaction_value",
        "promo",
        "day_of_week",
        "month",
        "store_type",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Feature dataset is missing columns: " + ", ".join(sorted(missing))
        )
    return df


def prepare_elasticity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and transform observations used by the log-log regressions."""
    analysis = df.copy()
    analysis["sales"] = pd.to_numeric(analysis["sales"], errors="coerce")
    analysis["avg_transaction_value"] = pd.to_numeric(
        analysis["avg_transaction_value"], errors="coerce"
    )
    analysis["promo"] = (
        pd.to_numeric(analysis["promo"], errors="coerce").fillna(0).astype(int)
    )
    analysis = analysis.loc[
        analysis["sales"].gt(0) & analysis["avg_transaction_value"].gt(0)
    ].copy()
    analysis["log_sales"] = np.log(analysis["sales"])
    analysis["log_price_proxy"] = np.log(analysis["avg_transaction_value"])
    analysis["store_type"] = analysis["store_type"].astype(str)
    return analysis.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log_sales", "log_price_proxy", "store_type"]
    )


def estimate_elasticity_by_store_type(
    df: pd.DataFrame,
    confidence_level: float = 0.95,
    minimum_observations: int = 500,
) -> pd.DataFrame:
    """Fit segment OLS models with calendar controls and store fixed effects."""
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is not installed. Install requirements.txt before running."
        ) from exc

    analysis = prepare_elasticity_data(df)
    alpha = 1 - confidence_level
    rows: list[dict[str, object]] = []

    for store_type, segment in analysis.groupby("store_type", sort=True):
        if len(segment) < minimum_observations:
            rows.append(
                {
                    "store_type": store_type,
                    "observations": len(segment),
                    "stores": segment["store_id"].nunique(),
                    "elasticity": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "p_value": np.nan,
                    "r_squared": np.nan,
                    "elasticity_class": "insufficient_data",
                    "demand_direction": "unknown",
                }
            )
            continue

        formula = (
            "log_sales ~ log_price_proxy + promo "
            "+ C(day_of_week) + C(month) + C(store_id)"
        )
        fitted = smf.ols(formula, data=segment).fit(
            cov_type="cluster",
            cov_kwds={"groups": segment["store_id"]},
        )
        coefficient = float(fitted.params["log_price_proxy"])
        confidence_interval = fitted.conf_int(alpha=alpha).loc["log_price_proxy"]
        rows.append(
            {
                "store_type": store_type,
                "observations": int(fitted.nobs),
                "stores": int(segment["store_id"].nunique()),
                "elasticity": coefficient,
                "ci_lower": float(confidence_interval.iloc[0]),
                "ci_upper": float(confidence_interval.iloc[1]),
                "p_value": float(fitted.pvalues["log_price_proxy"]),
                "r_squared": float(fitted.rsquared),
                "elasticity_class": (
                    "elastic" if abs(coefficient) > 1 else "inelastic"
                ),
                "demand_direction": (
                    "negative" if coefficient < 0 else "positive_or_zero"
                ),
            }
        )

    return pd.DataFrame(rows).sort_values("store_type").reset_index(drop=True)


def run_elasticity_analysis(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Run elasticity estimation and persist the segment table."""
    config = load_config(config_path)
    settings = config.get("elasticity", {})
    df = load_feature_data(feature_path)
    output = estimate_elasticity_by_store_type(
        df,
        confidence_level=float(settings.get("confidence_level", 0.95)),
        minimum_observations=int(
            settings.get("minimum_observations_per_segment", 500)
        ),
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_PATH, index=False)
    print(output.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nSaved elasticity results to {OUTPUT_PATH}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    run_elasticity_analysis(args.features, args.config)


if __name__ == "__main__":
    main()


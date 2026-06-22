"""Matched promotion lift and simplified promotion economics by store type."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from elasticity import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FEATURE_PATH,
    OUTPUT_PATH as ELASTICITY_OUTPUT_PATH,
    load_config,
    load_feature_data,
    run_elasticity_analysis,
)
from config import CURRENT_PROMO_DEPTH, GROSS_MARGIN_PCT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMO_OUTPUT_PATH = PROJECT_ROOT / "reports" / "promo_impact_by_segment.csv"
FINDINGS_PATH = PROJECT_ROOT / "reports" / "elasticity_and_promo_findings.md"


def prepare_promotion_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize promotion-analysis columns and derive missing calendar fields."""
    analysis = df.copy()
    analysis["sales_date"] = pd.to_datetime(analysis["sales_date"], errors="raise")
    analysis["sales"] = pd.to_numeric(analysis["sales"], errors="coerce")
    analysis["promo"] = (
        pd.to_numeric(analysis["promo"], errors="coerce").fillna(0).astype(int)
    )
    if "day_of_week" not in analysis:
        analysis["day_of_week"] = analysis["sales_date"].dt.dayofweek
    if "month" not in analysis:
        analysis["month"] = analysis["sales_date"].dt.month
    analysis["year"] = analysis["sales_date"].dt.year
    analysis["store_type"] = analysis["store_type"].astype(str)
    return analysis.dropna(subset=["sales", "store_type"])


def estimate_matched_promo_lift(
    df: pd.DataFrame,
    discount_pct: float,
    gross_margin_pct: float,
    controls: list[str],
    minimum_cells: int = 10,
) -> pd.DataFrame:
    """Compare promo and non-promo sales within exact seasonality/store cells.

    Cell-level effects are weighted by the smaller of promo and non-promo
    observation counts, preventing large unmatched groups from dominating.
    """
    analysis = prepare_promotion_data(df)
    missing_controls = set(controls).difference(analysis.columns)
    if missing_controls:
        raise ValueError(
            "Promotion matching controls are missing: "
            + ", ".join(sorted(missing_controls))
        )

    cell = (
        analysis.groupby(["store_type", *controls, "promo"], dropna=False)["sales"]
        .agg(["mean", "count"])
        .reset_index()
    )
    means = cell.pivot(
        index=["store_type", *controls], columns="promo", values="mean"
    )
    counts = cell.pivot(
        index=["store_type", *controls], columns="promo", values="count"
    )
    matched = means.join(counts, lsuffix="_mean", rsuffix="_count").reset_index()

    required_columns = ["0_mean", "1_mean", "0_count", "1_count"]
    # Pivot column labels may be integers; normalize all non-key labels.
    matched.columns = [
        str(column) if column not in ["store_type", *controls] else column
        for column in matched.columns
    ]
    missing = set(required_columns).difference(matched.columns)
    if missing:
        raise ValueError(
            "Matched promotion analysis requires both promo and non-promo rows."
        )

    matched = matched.dropna(subset=["0_mean", "1_mean"]).copy()
    matched["weight"] = matched[["0_count", "1_count"]].min(axis=1)
    matched["baseline_sales"] = matched["0_mean"]
    matched["promo_sales"] = matched["1_mean"]
    matched["incremental_sales"] = matched["promo_sales"] - matched["baseline_sales"]

    rows: list[dict[str, object]] = []
    for store_type, segment in matched.groupby("store_type", sort=True):
        if len(segment) < minimum_cells:
            rows.append(
                {
                    "store_type": store_type,
                    "matched_cells": len(segment),
                    "matched_weight": segment["weight"].sum(),
                    "baseline_sales": np.nan,
                    "promo_sales": np.nan,
                    "promo_lift_pct": np.nan,
                    "incremental_revenue": np.nan,
                    "discount_cost": np.nan,
                    "incremental_gross_profit": np.nan,
                    "net_promo_margin_impact": np.nan,
                    "promo_roi_pct": np.nan,
                    "profitability": "insufficient_data",
                }
            )
            continue

        weights = segment["weight"].to_numpy(dtype=float)
        baseline = float(np.average(segment["baseline_sales"], weights=weights))
        promo_sales = float(np.average(segment["promo_sales"], weights=weights))
        incremental_revenue = promo_sales - baseline
        discount_cost = promo_sales * discount_pct
        incremental_gross_profit = incremental_revenue * gross_margin_pct
        net_impact = incremental_gross_profit - discount_cost
        promo_roi = (
            net_impact / discount_cost * 100 if discount_cost > 0 else np.nan
        )
        rows.append(
            {
                "store_type": store_type,
                "matched_cells": len(segment),
                "matched_weight": int(segment["weight"].sum()),
                "baseline_sales": baseline,
                "promo_sales": promo_sales,
                "promo_lift_pct": (
                    (promo_sales / baseline - 1) * 100 if baseline > 0 else np.nan
                ),
                "incremental_revenue": incremental_revenue,
                "discount_cost": discount_cost,
                "incremental_gross_profit": incremental_gross_profit,
                "net_promo_margin_impact": net_impact,
                "promo_roi_pct": promo_roi,
                "profitability": "profitable" if net_impact > 0 else "unprofitable",
            }
        )

    return pd.DataFrame(rows).sort_values("store_type").reset_index(drop=True)


def write_findings(
    elasticity: pd.DataFrame,
    promo: pd.DataFrame,
    discount_pct: float,
    gross_margin_pct: float,
) -> None:
    """Write a plain-English findings report using generated result values."""
    valid_elasticity = elasticity.dropna(subset=["elasticity"]).copy()
    valid_promo = promo.dropna(subset=["promo_lift_pct"]).copy()

    lines = [
        "# Price Elasticity and Promotion Impact Findings",
        "",
        "## Assumptions and interpretation",
        "",
        (
            f"Promotion economics assume an average discount of "
            f"{discount_pct:.1%} and a pre-discount gross margin of "
            f"{gross_margin_pct:.1%}."
        ),
        (
            "Elasticity uses average transaction value (sales divided by customers) "
            "as a price proxy. Because this proxy contains sales, the coefficient is "
            "descriptive and should not be interpreted as a causal price response."
        ),
        (
            "Promotion lift compares promo and non-promo observations within the same "
            "store, day of week, month, and year. Remaining differences may still "
            "reflect promotion targeting or other unobserved factors."
        ),
        "",
        "## Elasticity results",
        "",
    ]

    if valid_elasticity.empty:
        lines.append("No segment had enough valid observations for estimation.")
    else:
        for row in valid_elasticity.itertuples(index=False):
            lines.append(
                f"- StoreType {row.store_type}: elasticity {row.elasticity:.3f} "
                f"(95% CI {row.ci_lower:.3f} to {row.ci_upper:.3f}); "
                f"classified as {row.elasticity_class} by absolute magnitude."
            )

    lines.extend(["", "## Promotion results", ""])
    if valid_promo.empty:
        lines.append("No segment had enough matched promo/non-promo cells.")
    else:
        for row in valid_promo.itertuples(index=False):
            lines.append(
                f"- StoreType {row.store_type}: matched promo lift "
                f"{row.promo_lift_pct:.2f}%, with estimated net margin impact "
                f"{row.net_promo_margin_impact:,.2f} per matched store-day; "
                f"behavior is classified as {row.profitability}."
            )

        best = valid_promo.loc[valid_promo["net_promo_margin_impact"].idxmax()]
        lines.extend(
            [
                "",
                "## Key takeaway",
                "",
                (
                    f"StoreType {best['store_type']} has the strongest estimated "
                    f"promotion economics: {best['promo_lift_pct']:.2f}% sales lift "
                    f"and {best['net_promo_margin_impact']:,.2f} net margin impact "
                    "per matched store-day under the configured assumptions."
                ),
            ]
        )

    FINDINGS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_promotion_analysis(
    feature_path: Path = DEFAULT_FEATURE_PATH,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Estimate promotion impact, save outputs, and generate the findings report."""
    config = load_config(config_path)
    matching = config.get("promotion_matching", {})
    discount_pct = CURRENT_PROMO_DEPTH
    gross_margin_pct = GROSS_MARGIN_PCT
    if not 0 <= discount_pct < 1 or not 0 < gross_margin_pct < 1:
        raise ValueError("discount_pct and gross_margin_pct must be decimal rates.")

    df = load_feature_data(feature_path)
    output = estimate_matched_promo_lift(
        df,
        discount_pct=discount_pct,
        gross_margin_pct=gross_margin_pct,
        controls=list(
            matching.get(
                "controls", ["store_id", "day_of_week", "month", "year"]
            )
        ),
        minimum_cells=int(
            matching.get("minimum_matched_cells_per_segment", 10)
        ),
    )
    PROMO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(PROMO_OUTPUT_PATH, index=False)

    elasticity = (
        pd.read_csv(ELASTICITY_OUTPUT_PATH)
        if ELASTICITY_OUTPUT_PATH.exists()
        else run_elasticity_analysis(feature_path, config_path)
    )
    write_findings(elasticity, output, discount_pct, gross_margin_pct)
    print(output.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))
    print(f"\nSaved promotion results to {PROMO_OUTPUT_PATH}")
    print(f"Saved findings to {FINDINGS_PATH}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURE_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    run_promotion_analysis(args.features, args.config)


if __name__ == "__main__":
    main()

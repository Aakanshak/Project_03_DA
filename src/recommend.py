"""Select margin-maximizing price/promotion recommendations by segment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CURRENT_PROMO_DEPTH,
    MAX_REVENUE_DECLINE_PCT,
    REQUIRE_NEGATIVE_ELASTICITY,
)
from scenario_simulator import (
    ELASTICITY_PATH,
    FEATURE_PATH,
    PROMO_PATH,
    SCENARIO_OUTPUT_PATH,
    run_all_scenarios,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECOMMENDATION_PATH = (
    PROJECT_ROOT / "data" / "processed" / "recommendations.csv"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "pricing_recommendations.md"


def _action_text(price_change: float, promo_depth: float) -> str:
    """Format a compact business action from scenario rates."""
    if np.isclose(price_change, 0):
        price_action = "keep list price unchanged"
    elif price_change > 0:
        price_action = f"increase list price by {price_change:.1%}"
    else:
        price_action = f"decrease list price by {abs(price_change):.1%}"

    if np.isclose(promo_depth, 0):
        promo_action = "remove promotional discount"
    elif np.isclose(promo_depth, CURRENT_PROMO_DEPTH):
        promo_action = f"keep promo depth at {promo_depth:.1%}"
    else:
        promo_action = f"set promo depth to {promo_depth:.1%}"
    return f"{price_action}; {promo_action}"


def select_recommendations(
    scenarios: pd.DataFrame,
    max_revenue_decline_pct: float = MAX_REVENUE_DECLINE_PCT,
) -> pd.DataFrame:
    """Choose maximum-margin feasible scenarios for every segment."""
    required = {
        "segment",
        "elasticity",
        "price_change_pct",
        "promo_depth_pct",
        "current_net_price",
        "scenario_net_price",
        "current_revenue",
        "projected_revenue",
        "revenue_change_pct",
        "current_gross_margin",
        "projected_gross_margin",
        "margin_lift_pct",
        "is_current_state",
    }
    missing = required.difference(scenarios.columns)
    if missing:
        raise ValueError(
            "Scenario table is missing columns: " + ", ".join(sorted(missing))
        )
    if not 0 <= max_revenue_decline_pct < 1:
        raise ValueError("max_revenue_decline_pct must be in [0, 1).")

    rows: list[dict[str, object]] = []
    for segment, group in scenarios.groupby("segment", sort=True):
        elasticity = float(group["elasticity"].iloc[0])
        current = group.loc[group["is_current_state"].astype(bool)]
        if current.empty:
            nearest_index = (
                group["price_change_pct"].abs()
                + (group["promo_depth_pct"] - CURRENT_PROMO_DEPTH).abs()
            ).idxmin()
            current = group.loc[[nearest_index]]
        current_row = current.iloc[0]

        invalid_elasticity = (
            not np.isfinite(elasticity)
            or (REQUIRE_NEGATIVE_ELASTICITY and elasticity >= 0)
        )
        if invalid_elasticity:
            best = current_row
            status = "review_required"
            rationale = (
                "No change recommended because the estimated elasticity is "
                "missing or nonnegative."
            )
        else:
            revenue_floor = (
                float(current_row["current_revenue"])
                * (1 - max_revenue_decline_pct)
            )
            feasible = group.loc[
                group["projected_revenue"].ge(revenue_floor)
                & group["projected_gross_margin"].ge(0)
            ]
            if feasible.empty:
                best = current_row
                status = "current_state_retained"
                rationale = "No tested scenario satisfies the revenue floor."
            else:
                best = feasible.sort_values(
                    ["projected_gross_margin", "projected_revenue"],
                    ascending=False,
                ).iloc[0]
                status = (
                    "recommended"
                    if not bool(best["is_current_state"])
                    else "current_state_optimal"
                )
                rationale = (
                    f"Highest projected gross margin while keeping revenue "
                    f"above {(1 - max_revenue_decline_pct):.1%} of current."
                )

        rows.append(
            {
                "segment": segment,
                "elasticity": elasticity,
                "current_state": (
                    f"net price {current_row['current_net_price']:.2f}, "
                    f"promo depth {CURRENT_PROMO_DEPTH:.1%}, "
                    f"revenue {current_row['current_revenue']:,.2f}"
                ),
                "recommended_action": _action_text(
                    float(best["price_change_pct"]),
                    float(best["promo_depth_pct"]),
                ),
                "recommended_net_price": float(best["scenario_net_price"]),
                "projected_revenue": float(best["projected_revenue"]),
                "revenue_change_pct": float(best["revenue_change_pct"]),
                "projected_gross_margin": float(
                    best["projected_gross_margin"]
                ),
                "expected_margin_lift_pct": float(best["margin_lift_pct"]),
                "revenue_floor_pct": 1 - max_revenue_decline_pct,
                "status": status,
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def write_recommendation_report(recommendations: pd.DataFrame) -> None:
    """Write a plain-English recommendation summary."""
    lines = [
        "# Pricing and Promotion Recommendations",
        "",
        "## Decision rule",
        "",
        (
            "For each StoreType, select the tested scenario with the highest "
            f"projected gross margin while allowing at most a "
            f"{MAX_REVENUE_DECLINE_PCT:.1%} decline in revenue."
        ),
        (
            "These are scenario estimates based on descriptive elasticity and "
            "matched promotion lift. They should be validated with controlled "
            "price or promotion experiments before operational rollout."
        ),
        "",
        "## Segment recommendations",
        "",
    ]
    for row in recommendations.itertuples(index=False):
        lines.extend(
            [
                f"### StoreType {row.segment}",
                "",
                f"- Current state: {row.current_state}.",
                f"- Action: {row.recommended_action}.",
                (
                    f"- Expected result: revenue change {row.revenue_change_pct:.2%}; "
                    f"gross-margin lift {row.expected_margin_lift_pct:.2%}."
                ),
                f"- Status: {row.status}. {row.rationale}",
                "",
            ]
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_recommendations(
    scenario_path: Path = SCENARIO_OUTPUT_PATH,
    feature_path: Path = FEATURE_PATH,
    elasticity_path: Path = ELASTICITY_PATH,
    promo_path: Path = PROMO_PATH,
) -> pd.DataFrame:
    """Load or generate scenarios, choose recommendations, and save outputs."""
    scenarios = (
        pd.read_csv(scenario_path)
        if scenario_path.exists()
        else run_all_scenarios(feature_path, elasticity_path, promo_path)
    )
    recommendations = select_recommendations(scenarios)
    RECOMMENDATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(RECOMMENDATION_PATH, index=False)
    write_recommendation_report(recommendations)
    print(
        recommendations.to_string(
            index=False, float_format=lambda value: f"{value:,.4f}"
        )
    )
    print(f"\nSaved recommendations to {RECOMMENDATION_PATH}")
    print(f"Saved report to {REPORT_PATH}")
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=SCENARIO_OUTPUT_PATH)
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--elasticity", type=Path, default=ELASTICITY_PATH)
    parser.add_argument("--promo", type=Path, default=PROMO_PATH)
    args = parser.parse_args()
    run_recommendations(
        args.scenarios, args.features, args.elasticity, args.promo
    )


if __name__ == "__main__":
    main()

"""Price and promotion scenario simulation by Rossmann store segment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CURRENT_PROMO_DEPTH,
    GROSS_MARGIN_PCT,
    MAX_PROMO_DEMAND_MULTIPLIER,
    PRICE_CHANGE_MAX_PCT,
    PRICE_CHANGE_MIN_PCT,
    PRICE_CHANGE_STEP_PCT,
    PROMO_DEPTHS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
ELASTICITY_PATH = PROJECT_ROOT / "reports" / "elasticity_by_segment.csv"
PROMO_PATH = PROJECT_ROOT / "reports" / "promo_impact_by_segment.csv"
SCENARIO_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "scenario_results.csv"


@dataclass(frozen=True)
class SegmentState:
    """Current commercial state used to anchor a segment simulation."""

    segment: str
    elasticity: float
    current_revenue: float
    current_net_price: float
    current_promo_depth: float = CURRENT_PROMO_DEPTH
    gross_margin_pct: float = GROSS_MARGIN_PCT
    promo_lift_pct: float = 0.0

    def validate(self) -> None:
        """Validate required rates and monetary inputs."""
        if self.current_revenue <= 0:
            raise ValueError("current_revenue must be positive.")
        if self.current_net_price <= 0:
            raise ValueError("current_net_price must be positive.")
        if not 0 <= self.current_promo_depth < 1:
            raise ValueError("current_promo_depth must be in [0, 1).")
        if not 0 < self.gross_margin_pct < 1:
            raise ValueError("gross_margin_pct must be in (0, 1).")
        if self.promo_lift_pct <= -100:
            raise ValueError("promo_lift_pct must be greater than -100.")


class ScenarioSimulator:
    """Simulate constant-elasticity price and promotion responses."""

    def __init__(
        self,
        price_changes: tuple[float, ...] | list[float] | None = None,
        promo_depths: tuple[float, ...] | list[float] = PROMO_DEPTHS,
    ) -> None:
        if price_changes is None:
            steps = round(
                (PRICE_CHANGE_MAX_PCT - PRICE_CHANGE_MIN_PCT)
                / PRICE_CHANGE_STEP_PCT
            )
            price_changes = tuple(
                np.linspace(PRICE_CHANGE_MIN_PCT, PRICE_CHANGE_MAX_PCT, steps + 1)
            )
        self.price_changes = tuple(float(value) for value in price_changes)
        self.promo_depths = tuple(float(value) for value in promo_depths)
        if any(change <= -1 for change in self.price_changes):
            raise ValueError("Price changes must be greater than -100%.")
        if any(depth < 0 or depth >= 1 for depth in self.promo_depths):
            raise ValueError("Promo depths must be in [0, 1).")

    @staticmethod
    def _promo_multiplier(depth: float, state: SegmentState) -> float:
        """Scale matched promotion lift relative to the configured current depth."""
        lift_rate = state.promo_lift_pct / 100
        if state.current_promo_depth == 0 or lift_rate == 0:
            return 1.0
        multiplier = 1 + lift_rate * depth / state.current_promo_depth
        return float(np.clip(multiplier, 0.01, MAX_PROMO_DEMAND_MULTIPLIER))

    def simulate(self, state: SegmentState) -> pd.DataFrame:
        """Return projected units, revenue, and margin for every scenario.

        The current net price is treated as the realized price at the current
        promotion depth. Unit cost is inferred from current gross margin. The
        current state is reproduced exactly when price change and promo depth
        equal their current values.
        """
        state.validate()
        current_list_price = state.current_net_price / (
            1 - state.current_promo_depth
        )
        current_units = state.current_revenue / state.current_net_price
        unit_cost = state.current_net_price * (1 - state.gross_margin_pct)
        current_margin = state.current_revenue - current_units * unit_cost
        current_promo_multiplier = self._promo_multiplier(
            state.current_promo_depth, state
        )

        rows: list[dict[str, object]] = []
        for price_change in self.price_changes:
            for promo_depth in self.promo_depths:
                list_price = current_list_price * (1 + price_change)
                net_price = list_price * (1 - promo_depth)
                net_price_ratio = net_price / state.current_net_price
                promo_ratio = (
                    self._promo_multiplier(promo_depth, state)
                    / current_promo_multiplier
                )
                projected_units = (
                    current_units
                    * np.power(net_price_ratio, state.elasticity)
                    * promo_ratio
                )
                projected_revenue = projected_units * net_price
                projected_margin = projected_units * (net_price - unit_cost)
                rows.append(
                    {
                        "segment": state.segment,
                        "elasticity": state.elasticity,
                        "price_change_pct": price_change,
                        "promo_depth_pct": promo_depth,
                        "current_net_price": state.current_net_price,
                        "scenario_net_price": net_price,
                        "current_units": current_units,
                        "projected_units": projected_units,
                        "current_revenue": state.current_revenue,
                        "projected_revenue": projected_revenue,
                        "revenue_change_pct": (
                            projected_revenue / state.current_revenue - 1
                        ),
                        "current_gross_margin": current_margin,
                        "projected_gross_margin": projected_margin,
                        "margin_lift_pct": (
                            projected_margin / current_margin - 1
                            if current_margin != 0
                            else np.nan
                        ),
                        "unit_cost_assumption": unit_cost,
                        "is_current_state": bool(
                            np.isclose(price_change, 0)
                            and np.isclose(
                                promo_depth, state.current_promo_depth
                            )
                        ),
                    }
                )
        return pd.DataFrame(rows)


def load_segment_states(
    feature_path: Path = FEATURE_PATH,
    elasticity_path: Path = ELASTICITY_PATH,
    promo_path: Path = PROMO_PATH,
) -> list[SegmentState]:
    """Build segment baselines from features, elasticity, and promo outputs."""
    missing = [
        path
        for path in (feature_path, elasticity_path, promo_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Required analysis output(s) missing: "
            + ", ".join(str(path) for path in missing)
        )

    features = pd.read_parquet(feature_path)
    required_features = {
        "store_type",
        "avg_transaction_value",
        "promo",
        "sales",
    }
    missing_features = required_features.difference(features.columns)
    if missing_features:
        raise ValueError(
            "Feature data is missing columns: "
            + ", ".join(sorted(missing_features))
        )

    features = features.copy()
    features["store_type"] = features["store_type"].astype(str)
    features["avg_transaction_value"] = pd.to_numeric(
        features["avg_transaction_value"], errors="coerce"
    )
    features["promo"] = pd.to_numeric(
        features["promo"], errors="coerce"
    ).fillna(0)
    promo_prices = (
        features.loc[
            features["promo"].eq(1)
            & features["avg_transaction_value"].gt(0)
        ]
        .groupby("store_type")["avg_transaction_value"]
        .median()
        .rename("current_net_price")
    )
    all_prices = (
        features.loc[features["avg_transaction_value"].gt(0)]
        .groupby("store_type")["avg_transaction_value"]
        .median()
        .rename("fallback_net_price")
    )

    elasticity = pd.read_csv(elasticity_path)
    promo = pd.read_csv(promo_path)
    segments = (
        elasticity[["store_type", "elasticity"]]
        .merge(
            promo[["store_type", "promo_sales", "promo_lift_pct"]],
            on="store_type",
            how="inner",
            validate="one_to_one",
        )
        .merge(promo_prices, on="store_type", how="left")
        .merge(all_prices, on="store_type", how="left")
    )
    segments["current_net_price"] = segments["current_net_price"].fillna(
        segments["fallback_net_price"]
    )
    segments = segments.dropna(
        subset=[
            "elasticity",
            "promo_sales",
            "promo_lift_pct",
            "current_net_price",
        ]
    )

    return [
        SegmentState(
            segment=str(row.store_type),
            elasticity=float(row.elasticity),
            current_revenue=float(row.promo_sales),
            current_net_price=float(row.current_net_price),
            current_promo_depth=CURRENT_PROMO_DEPTH,
            gross_margin_pct=GROSS_MARGIN_PCT,
            promo_lift_pct=float(row.promo_lift_pct),
        )
        for row in segments.itertuples(index=False)
    ]


def run_all_scenarios(
    feature_path: Path = FEATURE_PATH,
    elasticity_path: Path = ELASTICITY_PATH,
    promo_path: Path = PROMO_PATH,
) -> pd.DataFrame:
    """Simulate and save scenarios for every eligible store segment."""
    states = load_segment_states(feature_path, elasticity_path, promo_path)
    if not states:
        raise RuntimeError("No segments had complete inputs for simulation.")
    simulator = ScenarioSimulator()
    output = pd.concat(
        [simulator.simulate(state) for state in states], ignore_index=True
    )
    SCENARIO_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(SCENARIO_OUTPUT_PATH, index=False)
    print(f"Saved {len(output):,} scenarios to {SCENARIO_OUTPUT_PATH}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURE_PATH)
    parser.add_argument("--elasticity", type=Path, default=ELASTICITY_PATH)
    parser.add_argument("--promo", type=Path, default=PROMO_PATH)
    args = parser.parse_args()
    run_all_scenarios(args.features, args.elasticity, args.promo)


if __name__ == "__main__":
    main()


"""Defensible commercial assumptions for pricing and promotion simulations.

Rates are expressed as decimals. This is the single source of truth for
revenue-floor, price-grid, promotion-depth, and gross-margin assumptions.
"""

from __future__ import annotations


# Current-state assumptions
CURRENT_PROMO_DEPTH = 0.10
GROSS_MARGIN_PCT = 0.30

# Recommendation constraint: projected revenue must remain at or above 95% of
# current segment revenue.
MAX_REVENUE_DECLINE_PCT = 0.05

# Scenario grid
PRICE_CHANGE_MIN_PCT = -0.15
PRICE_CHANGE_MAX_PCT = 0.15
PRICE_CHANGE_STEP_PCT = 0.025
PROMO_DEPTHS = (0.00, 0.05, 0.10, 0.15, 0.20)

# Promotion lift is scaled linearly by depth relative to CURRENT_PROMO_DEPTH.
# This assumption is intentionally explicit because Rossmann does not provide
# actual discount depth.
PROMO_LIFT_SCALING = "linear"
MAX_PROMO_DEMAND_MULTIPLIER = 2.50

# Avoid recommendations based on a price coefficient that contradicts normal
# demand behavior. Such segments retain the current state and are flagged.
REQUIRE_NEGATIVE_ELASTICITY = True


# Power BI Dashboard Build Guide

## Data files

Connect Power BI directly to the CSV or Parquet files in
`data/processed/powerbi_exports/`:

| Power BI table | File | Grain |
|---|---|---|
| Forecasts | `sales_actuals_vs_forecast.csv` | One row per date and store |
| Elasticity | `elasticity_by_segment.csv` | One row per StoreType |
| Promo Impact | `promo_impact_by_segment.csv` | One row per StoreType |
| Recommendations | `pricing_recommendations.csv` | One row per StoreType |

Parquet versions contain the same data and are preferable when supported by the
chosen Power BI connection method.

Prophet was trained on a representative store sample, so `prophet_pred` is
intentionally blank for stores outside that sample. XGBoost covers all eligible
holdout rows.

## Data model

Create a separate StoreType dimension:

```DAX
DimStoreType =
DISTINCT (
    UNION (
        SELECTCOLUMNS ( Forecasts, "StoreType", Forecasts[store_type] ),
        SELECTCOLUMNS ( Elasticity, "StoreType", Elasticity[store_type] ),
        SELECTCOLUMNS ( 'Promo Impact', "StoreType", 'Promo Impact'[store_type] ),
        SELECTCOLUMNS ( Recommendations, "StoreType", Recommendations[segment] )
    )
)
```

Create one-to-many relationships from `DimStoreType[StoreType]` to the
StoreType/segment field in each table. Use single-direction filtering from the
dimension to the fact tables.

Create a calendar table and relate `Calendar[Date]` one-to-many to
`Forecasts[date]`:

```DAX
Calendar =
CALENDAR ( MIN ( Forecasts[date] ), MAX ( Forecasts[date] ) )
```

Mark `Calendar` as the model's date table.

## Core measures

```DAX
Actual Sales =
SUM ( Forecasts[actual] )

Prophet Forecast =
SUM ( Forecasts[prophet_pred] )

XGBoost Forecast =
SUM ( Forecasts[xgboost_pred] )

Prophet MAPE % =
AVERAGEX (
    FILTER (
        Forecasts,
        Forecasts[actual] <> 0
            && NOT ISBLANK ( Forecasts[prophet_pred] )
    ),
    DIVIDE (
        ABS ( Forecasts[actual] - Forecasts[prophet_pred] ),
        Forecasts[actual]
    )
)

XGBoost MAPE % =
AVERAGEX (
    FILTER (
        Forecasts,
        Forecasts[actual] <> 0
            && NOT ISBLANK ( Forecasts[xgboost_pred] )
    ),
    DIVIDE (
        ABS ( Forecasts[actual] - Forecasts[xgboost_pred] ),
        Forecasts[actual]
    )
)

Best Model MAPE % =
MIN ( [Prophet MAPE %], [XGBoost MAPE %] )

Average Promo Lift % =
AVERAGE ( 'Promo Impact'[promo_lift_pct] ) / 100

Total Net Promo Margin Impact =
SUM ( 'Promo Impact'[net_promo_margin_impact] )

Average Expected Margin Lift % =
AVERAGE ( Recommendations[expected_margin_lift_pct] )

Projected Recommendation Revenue =
SUM ( Recommendations[projected_revenue] )
```

Format all MAPE, lift, elasticity-classification, and change measures
appropriately. Existing percentage fields ending in `_pct` use either decimal
rates or percentage points according to their source:

- `Recommendations[expected_margin_lift_pct]` and `revenue_change_pct` are
  decimal rates; format directly as Percentage.
- `Promo Impact[promo_lift_pct]` is stored in percentage points; divide by 100
  in DAX before percentage formatting.

## Page 1: Demand Forecast

Purpose: show forecast quality and where model error occurs.

### Visuals

1. **KPI cards**
   - Actual Sales
   - Prophet MAPE %
   - XGBoost MAPE %
   - Best Model MAPE %

2. **Actual vs predicted line chart**
   - X-axis: `Calendar[Date]`
   - Values: Actual Sales, Prophet Forecast, XGBoost Forecast
   - Use a continuous date axis.
   - Leave Prophet blanks unchanged; do not replace them with zero.

3. **Forecast error by store**
   - Clustered bar chart
   - Axis: `Forecasts[store]`
   - Values: Prophet MAPE %, XGBoost MAPE %
   - Apply a Top N filter to show the 15 stores with the largest actual sales
     or largest forecast error.

4. **Store/date detail table**
   - Date, store, StoreType, actual, Prophet prediction, XGBoost prediction
   - Add conditional formatting to prediction columns based on variance from
     actual sales.

### Filters

- Date range
- Store
- StoreType

## Page 2: Promotion and Elasticity

Purpose: identify promotion-responsive segments and price sensitivity.

### Visuals

1. **Promotion lift by segment**
   - Bar chart
   - Axis: `DimStoreType[StoreType]`
   - Value: promo lift %
   - Conditional color: positive lift in blue; negative lift in red.

2. **Promotion profitability**
   - Waterfall or column chart
   - Category: StoreType
   - Value: `net_promo_margin_impact`
   - Add `profitability` as legend or conditional formatting.

3. **Elasticity matrix**
   - Table or matrix
   - Rows: StoreType
   - Values: elasticity, lower confidence interval, upper confidence interval,
     p-value, elasticity class
   - Use icons to distinguish elastic, inelastic, and insufficient-data rows.

4. **Elasticity vs promotion-lift scatter plot**
   - X-axis: elasticity
   - Y-axis: promo lift %
   - Details: StoreType
   - Size: matched promotion weight or observations
   - Add a vertical constant line at elasticity `-1`.

5. **KPI cards**
   - Average Promo Lift %
   - Total Net Promo Margin Impact
   - Count of profitable segments

### Filters

- StoreType
- Elasticity class
- Promotion profitability

## Page 3: Pricing Recommendations

Purpose: communicate the recommended action and expected commercial impact.

### Visuals

1. **Recommendation table**
   - Segment
   - Current state
   - Recommended action
   - Recommended net price
   - Projected revenue
   - Revenue change %
   - Expected margin lift %
   - Status
   - Rationale

2. **Expected margin lift by segment**
   - Horizontal bar chart
   - Axis: segment
   - Value: expected margin lift %
   - Conditional colors: positive green, zero gray, negative red.

3. **Revenue change vs margin lift**
   - Scatter plot
   - X-axis: revenue change %
   - Y-axis: expected margin lift %
   - Details: segment
   - Add a vertical reference line at the configured revenue floor decline.

4. **KPI cards**
   - Average Expected Margin Lift %
   - Maximum Expected Margin Lift %
   - Projected Recommendation Revenue
   - Count of segments with status `recommended`

### Filters

- StoreType/segment
- Recommendation status
- Revenue-change range
- Expected-margin-lift range

## Formatting and interaction

- Use one consistent StoreType color across all pages.
- Synchronize the StoreType slicer across pages.
- Enable cross-highlighting between segment charts and detail tables.
- Add report-page tooltips showing observations, confidence intervals, matched
  cells, and model caveats.
- Display the configured gross margin, current promo depth, and revenue floor
  in an assumptions text box.
- Add a visible note that elasticity is descriptive because
  `avg_transaction_value = sales / customers` is a proxy rather than an
  observed posted price.

## Refresh order

Run the project outputs in this order before refreshing Power BI:

```powershell
python src/evaluate_forecasts.py
python src/elasticity.py
python src/promo_impact.py
python src/scenario_simulator.py
python src/recommend.py
python src/export_for_powerbi.py
```


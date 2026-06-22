# Price Elasticity and Promotion Impact Findings

Result generation is pending because `data/processed/features.parquet` is not
present in this workspace.

When the feature dataset is available, run:

```powershell
python src/elasticity.py
python src/promo_impact.py
```

The second command will replace this status note with a data-backed summary
containing the segment elasticity estimates, confidence intervals, matched
promotion lift, and promotion profitability results.

The elasticity analysis uses average transaction value (`sales / customers`) as
a price proxy. Because that proxy contains the target variable, its coefficient
is descriptive and must not be treated as a causal estimate of response to an
actual posted-price change.


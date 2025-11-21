# Testing Infrastructure

This folder contains utilities for testing individual features or feature groups without affecting the production `fx_features` table.

## Available Features

### Market Features (Group: `market`)
- `dxy` - US Dollar Index
- `gold` - Gold Futures (GC=F)
- `oil` - Crude Oil Futures (CL=F)
- `vix` - CBOE Volatility Index
- `sp500` - S&P 500 Index
- `set_index` - SET Index (Thailand)
- `usd_thb` - USD/THB Exchange Rate

### Thai Macro Features (Group: `th_macro`)
- `th_policy_rate` - Bank of Thailand Policy Rate
- `th_cpi` - Thailand CPI
- `th_10y` - Thailand 10-Year Bond Yield

### US Macro Features (Group: `us_macro`)
- `us_fed_rate` - Federal Funds Rate
- `us_cpi` - US CPI
- `us_10y` - US 10-Year Treasury Yield

### Sentiment Features (Group: `sentiment`)
- `news_sentiment` - News Sentiment Score

## Usage

### Test Single Feature

Extract and test a single feature:

```bash
# Test Gold data
python tests/test_single_feature.py --feature gold --start 2024-01-01 --end 2024-12-31

# Test USD/THB (end date defaults to today)
python tests/test_single_feature.py --feature usd_thb --start 2024-01-01
```

This creates a table named `test_{feature}` (e.g., `test_gold`).

### Test Multiple Features

Extract multiple specific features:

```bash
# Test multiple individual features
python tests/test_selected_features.py --features gold,oil,dxy --start 2024-01-01

# Test an entire group
python tests/test_selected_features.py --features market --start 2024-01-01

# Mix individual features and groups
python tests/test_selected_features.py --features gold,th_macro --start 2024-01-01

# Specify custom table name
python tests/test_selected_features.py --features gold,oil --start 2024-01-01 --table my_test_table
```

This creates a table named `test_custom_{timestamp}` or your custom name.

### List Test Tables

See all existing test tables:

```bash
python tests/cleanup.py --list
```

### Drop Test Tables

Drop specific test table(s):

```bash
# Drop single table
python tests/cleanup.py --tables test_gold

# Drop multiple tables
python tests/cleanup.py --tables test_gold,test_oil
```

Drop all test tables:

```bash
# With confirmation prompt
python tests/cleanup.py --all

# Without confirmation (use with caution!)
python tests/cleanup.py --all --no-confirm
```

## Best Practices

1. **Use short date ranges for testing** - Test with recent data (e.g., last 30-90 days) to speed up extraction
2. **Clean up regularly** - Drop test tables when done to avoid clutter
3. **Use descriptive table names** - When testing multiple features, use `--table` to give meaningful names
4. **Test before production** - Use this to validate data sources before adding to main pipeline

## Examples

### Quick Test Workflow

```bash
# 1. Test gold data for last 3 months
python tests/test_single_feature.py --feature gold --start 2024-08-01

# 2. Verify the data looks good (check output statistics)

# 3. Clean up when done
python tests/cleanup.py --tables test_gold
```

### Testing New Feature Combinations

```bash
# Test a combination of features you might use for modeling
python tests/test_selected_features.py --features gold,oil,usd_thb,th_policy_rate --start 2024-01-01 --table test_model_features

# Verify data quality, check correlations, etc.

# Clean up
python tests/cleanup.py --tables test_model_features
```

## Notes

- All test tables are prefixed with `test_` to avoid conflicts with production tables
- The production `fx_features` table is never modified by these test scripts
- Test tables use the same schema as production (DATE primary key, DOUBLE PRECISION columns)
- Missing values are handled using the same cleaning logic as the main pipeline

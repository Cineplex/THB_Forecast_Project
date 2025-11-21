# XGBoost USD/THB Forecasting Model

## Overview
This folder contains an XGBoost regression model for forecasting USD/THB exchange rate **30 days ahead**.

## Files
- `train_xgboost.py` - Model training script
- `saved_models/` - Directory for trained models and metadata
- `notebooks/usd_thb_forecasting.ipynb` - Jupyter notebook with visualizations

## Prerequisites

### Install Required Packages
```bash
pip install xgboost scikit-learn matplotlib seaborn jupyter notebook
```

Or install all at once:
```bash
pip install xgboost scikit-learn matplotlib seaborn jupyter notebook pandas numpy sqlalchemy psycopg2
```

## Usage

### 1. Train the Model
```powershell
$env:PYTHONPATH = "d:\University\CSS\Data Science & Data Engineer\Project\thb_forecast_project"
$env:PYTHONIOENCODING = "utf-8"
python models/train_xgboost.py
```

This will:
- Load data from `fx_features` table
- Create features (lags, rolling stats, returns)
- Train XGBoost model
- Save model to `saved_models/`
- Print evaluation metrics

### 2. Visualize Results (Jupyter Notebook)
```bash
jupyter notebook notebooks/usd_thb_forecasting.ipynb
```

The notebook includes:
- Data exploration
- Feature engineering
- Model training
- Performance evaluation
- Forecast visualization

## Model Details

### Target Variable
- **USD/THB 30 days ahead**

### Features
**Lag Features** (1, 5, 10, 20, 30 days):
- `usd_thb`, `gold`, `oil`, `dxy`, `vix`

**Rolling Statistics** (5, 10, 20, 30 day windows):
- Moving averages and standard deviations for `usd_thb`, `gold`, `oil`

**Returns** (daily % change):
- `gold`, `oil`, `sp500`

### Model Parameters
- Objective: Regression (squared error)
- Max depth: 6
- Learning rate: 0.1
- N estimators: 100
- Subsample: 0.8
- Colsample_bytree: 0.8

### Train/Test Split
- 80% train / 20% test
- Chronological split (preserves time-series order)

## Evaluation Metrics
- **MAE** (Mean Absolute Error) - Average prediction error
- **RMSE** (Root Mean Squared Error) - Penalizes large errors
- **MAPE** (Mean Absolute Percentage Error) - % error

**Good Performance**:
- MAPE < 2%: Excellent
- MAPE < 5%: Good
- MAPE > 5%: Needs improvement

## Next Steps

### Improve Model
1. **Hyperparameter Tuning**: Use GridSearchCV or RandomizedSearchCV
2. **Add More Features**: Technical indicators (RSI, MACD, Bollinger Bands)
3. **Try Different Horizons**: 7, 14, 60 days
4. **Ensemble Methods**: Combine with ARIMA, LSTM

### Deploy
1. Create prediction API
2. Schedule daily retraining
3. Monitor model drift

## Troubleshooting

### Missing Packages
```bash
pip install xgboost
```

### Database Connection Error
Check `config.py` for correct PostgreSQL credentials

### Not Enough Data
Ensure `fx_features` table has at least 6 months of data

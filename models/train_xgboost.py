"""
XGBoost model training for USD/THB forecasting (30-day ahead prediction).
"""

import os
from datetime import datetime
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb

import project_paths  # noqa: F401
from database.save_db import get_engine


def load_data(start_date: str = "2020-01-01", end_date: str = None) -> pd.DataFrame:
    """Load FX features from database."""
    engine = get_engine()
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    query = f"""
    SELECT * FROM fx_features
    WHERE date >= '{start_date}' AND date <= '{end_date}'
    ORDER BY date
    """
    
    df = pd.read_sql(query, engine, index_col="date", parse_dates=["date"])
    print(f"✅ Loaded {len(df)} rows from database")
    return df


def create_features(df: pd.DataFrame, target_shift: int = 30) -> pd.DataFrame:
    """
    Create features for forecasting.
    
    Args:
        df: DataFrame with FX features
        target_shift: Days ahead to forecast (default 30)
    
    Returns:
        DataFrame with features and target
    """
    result = df.copy()
    
    # Create target variable (30 days ahead)
    result['target'] = result['usd_thb'].shift(-target_shift)
    
    # Lag features
    lag_features = ['usd_thb', 'gold', 'oil', 'dxy', 'vix']
    lags = [1, 5, 10, 20, 30]
    
    for feature in lag_features:
        if feature in result.columns:
            for lag in lags:
                result[f'{feature}_lag{lag}'] = result[feature].shift(lag)
    
    # Rolling statistics
    rolling_features = ['usd_thb', 'gold', 'oil']
    windows = [5, 10, 20, 30]
    
    for feature in rolling_features:
        if feature in result.columns:
            for window in windows:
                result[f'{feature}_ma{window}'] = result[feature].rolling(window).mean()
                result[f'{feature}_std{window}'] = result[feature].rolling(window).std()
    
    # Returns (% change)
    price_features = ['gold', 'oil', 'sp500']
    for feature in price_features:
        if feature in result.columns:
            result[f'{feature}_return'] = result[feature].pct_change() * 100
    
    # Drop rows without target (last N days)
    result = result.dropna(subset=['target'])
    
    print(f"✅ Created {len(result.columns)} features")
    print(f"   Rows after target creation: {len(result)}")
    
    return result


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split data into train and test sets (time-series aware).
    
    Args:
        df: DataFrame with features and target
        test_size: Proportion of test set
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    # Remove non-feature columns
    feature_cols = [col for col in df.columns if col not in ['target', 'usd_thb']]
    
    # Drop any remaining NaN rows
    df_clean = df.dropna()
    
    # Time-series split (chronological)
    split_idx = int(len(df_clean) * (1 - test_size))
    
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train['target']
    X_test = test[feature_cols]
    y_test = test['target']
    
    print(f"\n📊 Train/Test Split:")
    print(f"   Train: {len(X_train)} samples ({train.index.min()} to {train.index.max()})")
    print(f"   Test:  {len(X_test)} samples ({test.index.min()} to {test.index.max()})")
    print(f"   Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Train XGBoost regression model.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
    
    Returns:
        Trained XGBoost model
    """
    print("\n🚀 Training XGBoost model...")
    
    # XGBoost parameters
    # XGBoost parameters (tuned to prevent overfitting)
    params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'gamma': 1.0,
    'reg_alpha': 2.0,
    'reg_lambda': 10.0,
    'random_state': 42,
    'eval_metric': 'rmse'
}

    
    model = xgb.XGBRegressor(**params)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=100
    )
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Evaluate
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100
    
    print(f"\n📈 Model Performance:")
    print(f"   Train MAE:  {train_mae:.4f}")
    print(f"   Test MAE:   {test_mae:.4f}")
    print(f"   Train RMSE: {train_rmse:.4f}")
    print(f"   Test RMSE:  {test_rmse:.4f}")
    print(f"   Test MAPE:  {test_mape:.2f}%")
    
    return model


def save_model(model, feature_cols, metrics: dict, save_dir: str = "models/saved_models"):
    """Save trained model and metadata."""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f"xgboost_usd_thb_{timestamp}.json")
    metadata_path = os.path.join(save_dir, f"metadata_{timestamp}.pkl")
    
    # Save model
    model.save_model(model_path)
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'metrics': metrics,
        'trained_at': timestamp
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"\n💾 Model saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")
    
    return model_path, metadata_path


def main():
    """Main training pipeline."""
    print("🚀 XGBoost USD/THB Forecasting Model Training\n")
    
    # 1. Load data
    df = load_data(start_date="2020-01-01")
    
    # 2. Create features
    df_features = create_features(df, target_shift=30)
    
    # 3. Prepare train/test
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test(df_features, test_size=0.2)
    
    # 4. Train model
    model = train_xgboost(X_train, y_train, X_test, y_test)
    
    # 5. Save model
    y_test_pred = model.predict(X_test)
    metrics = {
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'test_mape': mean_absolute_percentage_error(y_test, y_test_pred) * 100
    }
    
    save_model(model, feature_cols, metrics)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()

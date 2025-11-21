"""
XGBoost USD/THB 30-day ahead forecasting

ไฟล์นี้เป็นสคริปต์ Python ที่เตรียมข้อมูลจากตาราง `fx_features` ในฐานข้อมูล, สร้างฟีเจอร์,
เทรนโมเดล XGBoost (regressor) และทำการพยากรณ์ค่า USD/THB ล่วงหน้า 30 วัน

การใช้งาน (ตัวอย่าง):
- เปิดใน Jupyter Notebook (.ipynb) หรือรันเป็นสคริปต์ Python
- ตั้งค่าตัวแปรการเชื่อมต่อฐานข้อมูลตามส่วน CONFIG ด้านล่าง

Dependencies:
- pandas, numpy, sqlalchemy, psycopg2-binary (หรือ driver ที่ใช้), xgboost, scikit-learn, joblib

"""

# -------------------------- CONFIG --------------------------
# แก้เป็นค่า connection ของคุณ (หรือใช้ environment variables)
DB_CONN = "postgresql+psycopg2://postgres:168435729@localhost:5432/thb_forecast_project"  # ตัวอย่าง
TABLE_NAME = "fx_features"
TARGET_COL = "usd_thb"
FORECAST_HORIZON = 30  # พยากรณ์ 30 วันข้างหน้า
MODEL_OUTPUT = "xgb_usd_thb_30d.joblib"

# -------------------------- IMPORTS --------------------------
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
from datetime import timedelta

# -------------------------- UTILITIES --------------------------

def load_data(engine, table_name):
    query = f"SELECT * FROM {table_name} ORDER BY date"
    df = pd.read_sql(query, engine, parse_dates=["date"]) 
    return df


def create_features(df):
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure date is datetime and set index
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.set_index(pd.DatetimeIndex(df["date"]))

    # Fill/propagate missing values sensibly (forward fill then backward)
    df = df.ffill().bfill()

    # Lag features for target
    for lag in range(1, 8):  # 7-day lags
        df[f"usd_thb_lag_{lag}"] = df[TARGET_COL].shift(lag)

    # Rolling statistics
    df["usd_thb_roll_mean_7"] = df[TARGET_COL].shift(1).rolling(window=7).mean()
    df["usd_thb_roll_std_7"] = df[TARGET_COL].shift(1).rolling(window=7).std()

    # Use other series as is or with simple transforms
    other_cols = [c for c in df.columns if c not in ["date", TARGET_COL, "created_at"] and not c.startswith("usd_thb_lag_")]
    # create 1-day change for other series
    for c in other_cols:
        df[f"{c}_diff1"] = df[c].pct_change().fillna(0)

    # Date-based features
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    # Target shifted: level at horizon
    df[f"target_{FORECAST_HORIZON}d"] = df[TARGET_COL].shift(-FORECAST_HORIZON)

    df = df.dropna()
    return df


def train_xgb(X_train, y_train, X_val=None, y_val=None):
    xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)

    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 2, 5]
    }

    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        xgbr, param_distributions=param_dist, n_iter=20, cv=tscv,
        scoring='neg_mean_absolute_error', verbose=1, random_state=42
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    best = search.best_estimator_

    # Optionally refit on combined train+val
    if X_val is not None:
        best.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

    return best


def backtest_predict(model, X, y, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes = []
    rmses = []
    for train_index, test_index in tscv.split(X):
        X_tr, X_te = X.iloc[train_index], X.iloc[test_index]
        y_tr, y_te = y.iloc[train_index], y.iloc[test_index]
        m = model
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        maes.append(mean_absolute_error(y_te, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
    return np.mean(maes), np.mean(rmses)


def iterative_forecast(model, last_known_df, forecast_horizon, feature_cols):
    """
    last_known_df: DataFrame containing at least the last available date row with needed columns.
    We'll create new rows step by step and predict forward using previous predictions for lag features.
    """
    preds = []
    current = last_known_df.copy()

    for step in range(forecast_horizon):
        row = {}
        # Build features for next day
        next_date = current.index[-1] + pd.Timedelta(days=1)

        # date features
        row['dayofweek'] = next_date.dayofweek
        row['month'] = next_date.month

        # for lag features, use last values from current
        for lag in range(1, 8):
            col = f"usd_thb_lag_{lag}"
            # lag_1 is yesterday -> current last usd_thb
            if lag == 1:
                val = current[TARGET_COL].iloc[-1]
            else:
                # lag k becomes previous lag k-1
                prev_col = f"usd_thb_lag_{lag-1}"
                val = current[prev_col].iloc[-1]
            row[col] = val

        # rolling stats
        # compute from current TARGET_COL series
        last_series = np.append(current[TARGET_COL].values, preds) if preds else current[TARGET_COL].values
        row['usd_thb_roll_mean_7'] = pd.Series(last_series).shift(1).rolling(window=7).mean().iloc[-1]
        row['usd_thb_roll_std_7'] = pd.Series(last_series).shift(1).rolling(window=7).std().iloc[-1]

        # For other series diffs, we will assume they stay the same as last available (simple assumption)
        other_placeholder_cols = [c for c in current.columns if c.endswith('_diff1')]
        for c in other_placeholder_cols:
            row[c] = current[c].iloc[-1]

        # Assemble into DataFrame
        X_next = pd.DataFrame([row], index=[next_date])
        # reindex to ensure feature order
        X_next = X_next.reindex(columns=feature_cols)

        # fill na
        X_next = X_next.fillna(method='ffill').fillna(method='bfill').fillna(0)

        yhat = model.predict(X_next)[0]
        preds.append(yhat)

        # append to current for next iteration
        # create a new row with price (predicted) and other columns
        new_row = {}
        new_row[TARGET_COL] = yhat
        # copy other raw columns from last available row (non-diff) if exist
        for c in [col for col in current.columns if not col.endswith('_diff1') and not col.startswith('usd_thb_lag_') and col not in ['dayofweek','month','usd_thb_roll_mean_7','usd_thb_roll_std_7']]:
            try:
                new_row[c] = current[c].iloc[-1]
            except Exception:
                new_row[c] = np.nan
        # add diff cols
        for c in other_placeholder_cols:
            new_row[c] = current[c].iloc[-1]
        # add lag cols
        for lag in range(1,8):
            new_row[f"usd_thb_lag_{lag}"] = X_next[f"usd_thb_lag_{lag}"].iloc[0]
        new_row['usd_thb_roll_mean_7'] = X_next['usd_thb_roll_mean_7'].iloc[0]
        new_row['usd_thb_roll_std_7'] = X_next['usd_thb_roll_std_7'].iloc[0]
        new_row['dayofweek'] = X_next['dayofweek'].iloc[0]
        new_row['month'] = X_next['month'].iloc[0]

        new_df_row = pd.DataFrame(new_row, index=[next_date])
        # append
        current = pd.concat([current, new_df_row], axis=0)

    # return pandas Series with predictions indexed by date
    idx = [current.index[-forecast_horizon + i] for i in range(forecast_horizon)] if forecast_horizon>0 else []
    # Better construct index: last_known_date + 1..horizon
    start = last_known_df.index[-1] + pd.Timedelta(days=1)
    dates = [start + pd.Timedelta(days=i) for i in range(forecast_horizon)]
    return pd.Series(preds, index=dates)


# -------------------------- MAIN FLOW --------------------------
if __name__ == '__main__':
    print("Connecting to DB and loading data...")
    engine = create_engine(DB_CONN)
    df_raw = load_data(engine, TABLE_NAME)

    print(f"Loaded {len(df_raw)} rows")

    df = create_features(df_raw.copy())
    print(f"After feature creation: {df.shape}")

    target_col_name = f"target_{FORECAST_HORIZON}d"
    feature_cols = [c for c in df.columns if c not in [target_col_name, 'date', TARGET_COL]]

    X = df[feature_cols].copy()
    y = df[target_col_name].copy()

    # split train/validation by time (last 10% for validation)
    split_idx = int(len(X) * 0.9)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print("Training XGBoost with time-series CV and random search... (this may take a while)")
    model = train_xgb(X_train, y_train, X_val, y_val)

    # Evaluate on validation
    y_val_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"Validation MAE: {mae:.6f}, RMSE: {rmse:.6f}")

    # Backtest quick
    bt_mae, bt_rmse = backtest_predict(model, X, y, n_splits=3)
    print(f"Backtest mean MAE: {bt_mae:.6f}, mean RMSE: {bt_rmse:.6f}")

    # Save model
    joblib.dump(model, MODEL_OUTPUT)
    print(f"Saved model to {MODEL_OUTPUT}")

    # Forecast next 30 days (iterative)
    # For iterative we need the last available raw row(s) - use df_raw tail
    last_known = df_raw.set_index(pd.DatetimeIndex(pd.to_datetime(df_raw['date']))).sort_index()
    # ensure required feature columns exist in last_known
    # create same feature columns in last_known via create_features pipeline on the whole set and take last 30 rows
    df_all_features = create_features(df_raw.copy())
    last_rows = df_all_features.iloc[-30:].copy()

    preds = iterative_forecast(model, last_rows, FORECAST_HORIZON, feature_cols)
    print("30-day forecast (USD/THB):")
    print(preds)

    # Optional: persist forecast back to DB or CSV
    preds_df = preds.rename('usd_thb_forecast').to_frame()
    preds_df['forecast_horizon'] = FORECAST_HORIZON
    preds_df.to_csv('usd_thb_30d_forecast.csv')
    print("Saved forecast to usd_thb_30d_forecast.csv")

# -------------------------- NOTES --------------------------
# - ผลลัพธ์ขึ้นกับความสมบูรณ์ของข้อมูลและ feature engineering ที่ใช้
# - วิธีการ iterative forecast ที่ใช้ที่นี่ทำสมมติฐานเรียบง่าย (เช่น ค่าดัชนีอื่นคงที่)
#   ถ้าต้องการความแม่นยำขึ้น ควรมีโมเดลสำหรับแต่ละ series หรือใช้ multivariate time-series models
# - ปรับ param grid, เพิ่ม/ลด lags, สร้าง features เพิ่มเติม (momentum, seasonality, macro diffs) เพื่อปรับปรุง
# - หากต้องการให้รันใน ipynb ที่คุณอัพโหลด (/mnt/data/usd_thb_forecasting.ipynb) ให้คัดลอกโค้ดนี้ลงในเซลล์ของโน้ตบุ๊ก

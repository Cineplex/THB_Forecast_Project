from datetime import datetime
from typing import Optional

import pandas as pd

import project_paths  # noqa: F401
from apis.extract import extract_all_data
from apis.news_sentiment import get_news_sentiment_series
from config import DEFAULT_START_DATE
from database.save_db import save_fx_features
from features.cleaning import handle_missing_values


def _ensure_naive(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        return df
    copy = df.copy()
    copy.index = copy.index.tz_localize(None)
    return copy


def build_fx_dataset(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    base_df = extract_all_data(start_date, end_date)
    news_df = get_news_sentiment_series(start_date, end_date)

    base_df = _ensure_naive(base_df)
    news_df = _ensure_naive(news_df)

    combined = base_df.join(news_df, how="outer")
    combined = combined.sort_index()
    print(f"📊 Before cleaning: {len(combined)} rows, {combined.shape[1]} columns")
    print(f"   Missing values: {combined.isna().sum().sum()}")
    
    combined = handle_missing_values(combined)
    print(f"📊 After missing value handling: {len(combined)} rows")
    print(f"   Columns: {list(combined.columns)}")
    return combined


def run_all(start_date: str = DEFAULT_START_DATE, end_date: Optional[str] = None) -> None:
    print("🚀 Building FX feature dataset")
    dataset = build_fx_dataset(start_date, end_date or datetime.utcnow().strftime("%Y-%m-%d"))
    save_fx_features(dataset)
    print("🎉 Pipeline completed")


if __name__ == "__main__":
    run_all()

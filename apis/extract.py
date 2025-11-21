from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import requests
import yfinance as yf

import project_paths  # noqa: F401
from apis.th_macro import fetch_th_macro
from apis.us_macro import fetch_us_macro
from config import API_KEYS, REQUEST_TIMEOUT

YF_TICKERS: Dict[str, str] = {
    "dxy": "DX-Y.NYB",
    "gold": "GC=F",
    "oil": "CL=F",
    "vix": "^VIX",
    "sp500": "^GSPC",
    "set_index": "^SET.BK",
    "usd_thb": "THB=X",
}


def _normalize_market_series(series: pd.Series | pd.DataFrame, column: str) -> pd.DataFrame:
    if isinstance(series, pd.DataFrame):
        if series.shape[1] == 1:
            df = series.copy()
            df.columns = [column]
        else:
            df = series.iloc[:, [0]].copy()
            df.columns = [column]
    else:
        df = series.to_frame(name=column)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.dropna().sort_index()
    return df[[column]]


def fetch_market_assets(start_date: str, end_date: Optional[str]) -> pd.DataFrame:
    frames = []
    for column, ticker in YF_TICKERS.items():
        print(f"📈 Downloading {column} ({ticker})")
        data = yf.download(ticker, start=start_date, end=end_date)
        if "Close" not in data.columns:
            continue
        close = data["Close"]
        frames.append(_normalize_market_series(close, column))

    if not frames:
        return pd.DataFrame(columns=list(YF_TICKERS.keys()))

    return pd.concat(frames, axis=1).sort_index()





def extract_all_data(start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    end_date = end_date or datetime.utcnow().strftime("%Y-%m-%d")

    market = fetch_market_assets(start_date, end_date)
    th_macro = fetch_th_macro(start_date, end_date)
    us_macro = fetch_us_macro(start_date, end_date)

    frames = [market, th_macro, us_macro]
    combined = pd.concat(frames, axis=1).sort_index()
    
    # Convert string dates to Timestamp for proper slicing
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    combined = combined.loc[start_ts:end_ts]

    print(f"✅ Extracted dataset with shape {combined.shape}")
    return combined


FEATURE_GROUPS = {
    "market": ["dxy", "gold", "oil", "vix", "sp500", "set_index", "usd_thb"],
    "th_macro": ["th_policy_rate", "th_cpi", "th_10y"],
    "us_macro": ["us_fed_rate", "us_cpi", "us_10y"],
    "sentiment": ["news_sentiment"],
}


def extract_selected_features(
    start_date: str,
    end_date: Optional[str] = None,
    features: list[str] = None
) -> pd.DataFrame:
    """
    Extract only selected features instead of the full dataset.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        features: List of features or feature groups to extract
                 Examples: ["gold"], ["gold", "oil"], ["market"], ["market", "th_macro"]
    
    Returns:
        DataFrame with only selected features
    """
    end_date = end_date or datetime.utcnow().strftime("%Y-%m-%d")
    
    if not features:
        raise ValueError("Must specify at least one feature or feature group")
    
    # Expand feature groups
    expanded_features = set()
    for feature in features:
        if feature in FEATURE_GROUPS:
            expanded_features.update(FEATURE_GROUPS[feature])
        else:
            expanded_features.add(feature)
    
    frames = []
    
    # Fetch market features if any are requested
    market_features = set(YF_TICKERS.keys())
    requested_market = expanded_features & market_features
    if requested_market:
        market_df = fetch_market_assets(start_date, end_date)
        # Filter to only requested columns
        available_cols = [col for col in requested_market if col in market_df.columns]
        if available_cols:
            frames.append(market_df[available_cols])
    
    # Fetch Thai macro if requested
    th_macro_features = set(FEATURE_GROUPS["th_macro"])
    if expanded_features & th_macro_features:
        th_macro = fetch_th_macro(start_date, end_date)
        available_cols = [col for col in th_macro_features if col in th_macro.columns]
        if available_cols:
            frames.append(th_macro[available_cols])
    
    # Fetch US macro if requested
    us_macro_features = set(FEATURE_GROUPS["us_macro"])
    if expanded_features & us_macro_features:
        us_macro = fetch_us_macro(start_date, end_date)
        available_cols = [col for col in us_macro_features if col in us_macro.columns]
        if available_cols:
            frames.append(us_macro[available_cols])
    
    # Fetch sentiment if requested
    if "news_sentiment" in expanded_features:
        from apis.news_sentiment import get_news_sentiment_series
        sentiment = get_news_sentiment_series(start_date, end_date)
        if not sentiment.empty:
            frames.append(sentiment)
    
    if not frames:
        return pd.DataFrame()
    
    combined = pd.concat(frames, axis=1).sort_index()
    
    # Convert string dates to Timestamp for proper slicing
    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)
    combined = combined.loc[start_ts:end_ts]
    
    print(f"✅ Extracted {len(expanded_features)} features with shape {combined.shape}")
    return combined


__all__ = ["extract_all_data", "extract_selected_features", "FEATURE_GROUPS"]

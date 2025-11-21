"""
Trading Economics helpers for Thai macro series needed in the FX dataset.
"""

from datetime import datetime
from typing import Dict, Optional
import urllib.parse

import pandas as pd
import requests

import project_paths  # noqa: F401
from config import API_KEYS, REQUEST_TIMEOUT

TH_INDICATORS: Dict[str, str] = {
    "th_policy_rate": "Interest Rate",
    "th_cpi": "Inflation Rate",
    "th_10y": "Government Bond 10Y",
}

TE_BASE_URL = "https://api.tradingeconomics.com/historical/country/thailand/indicator"


def _fetch_indicator(indicator: str, api_key: str) -> pd.DataFrame:
    encoded = urllib.parse.quote(indicator)
    url = f"{TE_BASE_URL}/{encoded}?c={api_key}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = response.json()
    df = pd.DataFrame(payload)
    if "DateTime" not in df.columns or "Value" not in df.columns:
        return pd.DataFrame()
    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df = df.dropna(subset=["DateTime"])
    return df.set_index("DateTime")[["Value"]]


def fetch_th_macro(
    start_date: str,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download the Thai macro indicators defined in TH_INDICATORS.
    """
    key = api_key or API_KEYS["te"]
    if key is None:
        raise ValueError("Missing Trading Economics API key. Set TRADING_ECONOMICS_KEY in .env")

    frames = []
    for column, indicator_name in TH_INDICATORS.items():
        try:
            df_raw = _fetch_indicator(indicator_name, key)
        except Exception as exc:  # pragma: no cover
            print(f"⚠️  Skipping {indicator_name}: {exc}")
            continue

        if df_raw.empty:
            continue

        df_raw = df_raw.rename(columns={"Value": column})
        frames.append(df_raw)

    if not frames:
        return pd.DataFrame(columns=list(TH_INDICATORS.keys()))

    combined = pd.concat(frames, axis=1).sort_index()
    # Convert string dates to Timestamp for proper slicing
    start_ts = pd.to_datetime(start_date)
    if end_date:
        end_ts = pd.to_datetime(end_date)
        combined = combined.loc[start_ts:end_ts]
    else:
        combined = combined.loc[start_ts:]
    combined.index.name = "date"
    return combined


__all__ = ["fetch_th_macro", "TH_INDICATORS"]

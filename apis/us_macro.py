"""
Helper utilities for downloading US macro indicators needed by the FX feature table.
"""

from datetime import datetime
from typing import Dict, Optional

import pandas as pd
from fredapi import Fred

import project_paths  # noqa: F401
from config import API_KEYS

US_SERIES: Dict[str, str] = {
    "us_fed_rate": "FEDFUNDS",
    "us_cpi": "CPIAUCSL",
    "us_10y": "DGS10",
}


def _init_fred(api_key: Optional[str] = None) -> Fred:
    key = api_key or API_KEYS["fred"]
    if key is None:
        raise ValueError("Missing FRED API key. Set FRED_API_KEY in .env")
    return Fred(api_key=key)


def fetch_us_macro(
    start_date: str,
    end_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download and align US macroeconomic indicators defined in US_SERIES.
    """
    fred = _init_fred(api_key)
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    frames = []
    for column, code in US_SERIES.items():
        series = fred.get_series(code, observation_start=start_date, observation_end=end_date)
        df = series.to_frame(name=column)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.dropna().sort_index()
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=list(US_SERIES.keys()))

    combined = pd.concat(frames, axis=1).sort_index()
    return combined


__all__ = ["fetch_us_macro", "US_SERIES"]

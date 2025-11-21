from datetime import datetime
from typing import List, Optional

import pandas as pd
from gdeltdoc import Filters, GdeltDoc

import project_paths  # noqa: F401

KEYWORDS = [
    "Thai baht",
    "Thailand currency",
    "Thailand economy",
    "Thai economy",
    "Bank of Thailand",
    "Thailand inflation",
    "foreign exchange",
    "forex market",
    "Federal Reserve",
    "US inflation",
]

gd = GdeltDoc()


def _normalize_keyword_series(df: pd.DataFrame, keyword: str) -> Optional[pd.Series]:
    if df.empty or "Article Count" not in df.columns:
        return None

    for column_name in ["Date", "date", "datetime"]:
        if column_name in df.columns:
            df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
            df = df.dropna(subset=[column_name]).set_index(column_name)
            break
    else:
        return None

    series = df["Article Count"].astype(float)
    series.name = keyword.replace(" ", "_").lower()
    return series


def get_gdelt_keyword(keyword: str, start: str, end: Optional[str]) -> Optional[pd.Series]:
    filters = Filters(keyword=keyword, start_date=start, end_date=end or datetime.today().strftime("%Y-%m-%d"))
    df = gd.timeline_search("timelinevolraw", filters)
    return _normalize_keyword_series(df, keyword)


def _combine_keywords(series_list: List[pd.Series]) -> pd.DataFrame:
    df = pd.concat(series_list, axis=1).fillna(0)
    df = df.resample("D").sum()
    df["news_sentiment"] = df.sum(axis=1)
    return df[["news_sentiment"]]


def get_news_sentiment_series(start: str, end: Optional[str] = None) -> pd.DataFrame:
    items: List[pd.Series] = []
    for keyword in KEYWORDS:
        series = get_gdelt_keyword(keyword, start, end)
        if series is not None:
            items.append(series)

    if not items:
        return pd.DataFrame(columns=["news_sentiment"])

    return _combine_keywords(items)


__all__ = ["get_news_sentiment_series", "KEYWORDS"]

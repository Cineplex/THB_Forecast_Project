"""
Missing value handling and outlier mitigation utilities for the FX dataset.
"""

from typing import Optional

import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Fill missing numeric values using forward/backward fills.
    Only drop rows where ALL values are missing (not just any missing).
    """
    if df.empty:
        return df

    filled = df.sort_index().copy()
    numeric_cols = filled.select_dtypes(include=["number", "float64", "int64"]).columns
    filled[numeric_cols] = filled[numeric_cols].ffill(limit=limit).bfill(limit=limit)
    
    # Only drop rows where ALL numeric columns are missing
    # This preserves rows that have at least some data
    if len(numeric_cols) > 0:
        filled = filled.dropna(subset=numeric_cols, how="all")
    
    return filled


def clip_outliers(df: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """
    Winsorize extreme values using a robust z-score (MAD based).
    """
    if df.empty:
        return df

    clipped = df.copy()
    numeric_cols = clipped.select_dtypes(include=["number", "float64", "int64"]).columns
    for col in numeric_cols:
        series = clipped[col].dropna()
        if series.empty:
            continue
        median = series.median()
        mad = np.median(np.abs(series - median))
        if mad == 0:
            continue
        z_scores = 0.6745 * (series - median) / mad
        mask = z_scores.abs() > z_threshold
        if not mask.any():
            continue
        lower = series[~mask].quantile(0.01)
        upper = series[~mask].quantile(0.99)
        clipped.loc[mask.index, col] = series.clip(lower=lower, upper=upper)

    return clipped


__all__ = ["handle_missing_values", "clip_outliers"]


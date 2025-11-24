"""
Data loading utilities for fetching market data and computing returns.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class DataLoaderError(Exception):
    """Raised when historical data cannot be retrieved or validated."""


def get_historical_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker and period using yfinance.

    Parameters
    ----------
    ticker:
        Ticker symbol to fetch (e.g., ``"NVDA"``).
    period:
        Lookback window for the query (e.g., ``"5y"``, ``"1y"``).

    Returns
    -------
    pd.DataFrame
        Historical price DataFrame containing at least ``Adj Close``.

    Raises
    ------
    DataLoaderError
        If data is missing, invalid, or the ticker cannot be resolved.
    """
    if not ticker:
        raise DataLoaderError("Ticker symbol is required.")

    try:
        history = yf.Ticker(ticker).history(period=period)
    except Exception as exc:  # pragma: no cover - network/service error
        logger.exception("yfinance fetch failed for %s", ticker)
        raise DataLoaderError(f"Failed to fetch data for ticker '{ticker}'.") from exc

    if isinstance(history.columns, pd.MultiIndex):
        history.columns = history.columns.get_level_values(0)

    if history.empty:
        raise DataLoaderError(f"No historical data found for ticker '{ticker}'.")

    if "Adj Close" not in history.columns and "Close" in history.columns:
        # Fallback: some instruments do not provide adjusted close; use close as a proxy.
        history = history.copy()
        history["Adj Close"] = history["Close"]

    if "Adj Close" not in history.columns:
        raise DataLoaderError("Adjusted close column 'Adj Close' missing from data.")

    history = history.dropna(subset=["Adj Close"])
    if history.empty:
        raise DataLoaderError("Adjusted close data is empty after cleaning.")

    return history


def compute_log_returns(df: pd.DataFrame, col: str = "Adj Close") -> pd.Series:
    """
    Compute daily log returns from a price series.

    Parameters
    ----------
    df:
        DataFrame containing a price column.
    col:
        Column name to use for the return calculation.

    Returns
    -------
    pd.Series
        Series of log returns with name ``log_return``.

    Raises
    ------
    ValueError
        If the DataFrame is empty or the target column is missing.
    """
    if df is None or df.empty:
        raise ValueError("Price DataFrame is empty.")
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")

    prices = df[col].astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns.name = "log_return"
    if log_returns.empty:
        raise ValueError("Not enough data to compute log returns.")
    return log_returns

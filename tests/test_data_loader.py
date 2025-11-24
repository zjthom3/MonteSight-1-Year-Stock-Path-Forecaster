import numpy as np
import pandas as pd
import pytest

from core.data_loader import DataLoaderError, compute_log_returns, get_historical_data


class DummyTicker:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def history(self, period: str = "5y"):
        return self._df


def test_get_historical_data_success(monkeypatch):
    df = pd.DataFrame({"Adj Close": [100, 101, 102]})
    monkeypatch.setattr("yfinance.Ticker", lambda ticker: DummyTicker(df))
    result = get_historical_data("TEST", period="1y")
    assert not result.empty
    assert "Adj Close" in result.columns


def test_get_historical_data_fallback_close(monkeypatch):
    df = pd.DataFrame({"Close": [50, 51, 52]})
    monkeypatch.setattr("yfinance.Ticker", lambda ticker: DummyTicker(df))
    result = get_historical_data("TEST", period="1y")
    assert "Adj Close" in result.columns
    assert result["Adj Close"].iloc[-1] == 52


def test_get_historical_data_failure(monkeypatch):
    df_empty = pd.DataFrame()
    monkeypatch.setattr("yfinance.Ticker", lambda ticker: DummyTicker(df_empty))
    with pytest.raises(DataLoaderError):
        get_historical_data("TEST", period="1y")


def test_compute_log_returns():
    df = pd.DataFrame({"Adj Close": [100.0, 110.0, 121.0]})
    log_returns = compute_log_returns(df)
    assert log_returns.name == "log_return"
    expected_value = pytest.approx(np.log(1.1), rel=1e-3)
    assert log_returns.size == 2
    assert log_returns.iloc[0] == expected_value
    assert log_returns.size == 2

import pandas as pd
import pytest

from core.analytics import (
    compute_probability_for_levels,
    compute_path_hit_probability_for_levels,
    filter_by_probability,
    generate_price_grid,
    get_percentile_band,
    get_terminal_prices,
)
from core.data_loader import compute_log_returns, get_historical_data
from core.simulation import estimate_drift_vol, simulate_price_paths


class DummyTicker:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def history(self, period: str = "5y"):
        return self._df


def test_end_to_end_pipeline(monkeypatch):
    # Stub yfinance data retrieval
    df_prices = pd.DataFrame({"Adj Close": [100 + i for i in range(30)]})
    monkeypatch.setattr("yfinance.Ticker", lambda ticker: DummyTicker(df_prices))

    fetched = get_historical_data("TEST", period="1y")
    returns = compute_log_returns(fetched)
    mu, sigma = estimate_drift_vol(returns)
    paths = simulate_price_paths(
        s0=float(fetched["Adj Close"].iloc[-1]),
        drift=mu,
        vol=sigma,
        days=30,
        n_sims=200,
        seed=123,
    )

    terminal = get_terminal_prices(paths)
    levels = generate_price_grid(terminal, n_points=20, paths=paths)
    df_probs = compute_path_hit_probability_for_levels(
        paths, levels, current_price=float(fetched["Adj Close"].iloc[-1])
    )
    filtered = filter_by_probability(df_probs, threshold=0.5)
    band = get_percentile_band(terminal)

    assert paths.shape == (31, 200)
    assert terminal.shape[0] == 200
    assert not df_probs.empty
    assert band["p17"] <= band["p50"] <= band["p83"]
    assert filtered["prob_hit"].ge(0.5).all() or filtered.empty

Project Name:
MonteSight: 1-Year Stock Path Forecaster

1. Project Overview

A Streamlit app that:

Lets users input stock ticker(s)

Pulls historical data via yfinance

Runs Monte Carlo simulations for a 1-year horizon

Estimates probabilities of hitting specific price levels

Shows only price levels with ≥ 66% probability

Visualizes paths and distributions in a way non-experts can understand

2. Folder & File Structure
monte_sight/
├─ app.py                      # Main Streamlit entrypoint
├─ requirements.txt            # Python dependencies
├─ README.md                   # Project documentation
├─ config/
│  └─ settings.py              # App-wide settings (defaults, constants)
├─ data/
│  ├─ cache/                   # Optional: cached historical data
│  └─ samples/                 # Example outputs / demo configs
├─ core/
│  ├─ data_loader.py           # yfinance integration & preprocessing
│  ├─ simulation.py            # Monte Carlo engine
│  ├─ analytics.py             # Probability calculations, percentiles
│  └─ utils.py                 # Shared helpers
├─ ui/
│  ├─ layout.py                # Streamlit page layout & sections
│  ├─ components.py            # Reusable UI components (cards, tables, etc.)
│  └─ plots.py                 # Plotting functions (matplotlib/plotly)
└─ tests/
   ├─ test_data_loader.py
   ├─ test_simulation.py
   ├─ test_analytics.py
   └─ test_integration.py

3. Core Modules & Responsibilities
3.1 config/settings.py

Default ticker(s): e.g., "NVDA", "AAPL"

Default lookback period for historical data: e.g., 5y

Default number of simulations: e.g., 10_000

Default forecast horizon: 252 trading days (~1 year)

Probability cutoff: 0.66 (66%)

Random seed default (for reproducibility)

3.2 core/data_loader.py

Responsibilities:

Use yfinance to pull historical OHLC data

Clean and resample data if needed

Compute daily log returns

Key Functions:

get_historical_data(ticker: str, period: str = "5y") -> pd.DataFrame

compute_log_returns(df: pd.DataFrame) -> pd.Series

get_ticker_metadata(ticker: str) -> dict (name, exchange, currency)

3.3 core/simulation.py

Responsibilities:

Implement Monte Carlo engine for 1-year price paths

Derive drift and volatility from historical log returns

Simulate many paths (e.g., 10,000)

Key Functions:

estimate_drift_vol(log_returns: pd.Series) -> tuple[float, float]

Uses mean and standard deviation of log returns

simulate_price_paths( s0: float, drift: float, vol: float, days: int = 252, n_sims: int = 10_000, ) -> np.ndarray

Returns shape: (days + 1, n_sims)

3.4 core/analytics.py

Responsibilities:

Translate simulated paths into probability insights

Compute price level distribution at horizon

Filter to show only price targets with ≥ 66% probability

Key Functions:

get_terminal_prices(paths: np.ndarray) -> np.ndarray

compute_probability_for_levels( terminal_prices: np.ndarray, price_levels: np.ndarray ) -> pd.DataFrame

Returns a table with columns: price_level, prob_hit

get_percentile_band( terminal_prices: np.ndarray, lower: float = 0.17, upper: float = 0.83 ) -> dict

Returns { "p17": value, "p50": value, "p83": value }

generate_price_grid( terminal_prices: np.ndarray, n_points: int = 50 ) -> np.ndarray

Returns grid of prices over which to compute probabilities

filter_by_probability( df_probs: pd.DataFrame, threshold: float = 0.66 ) -> pd.DataFrame

3.5 core/utils.py

Responsibilities:

Shared helpers & formatting

Key Functions:

format_currency(value: float, currency: str = "USD") -> str

set_random_seed(seed: int | None) -> None

safe_float(val, default=None)

4. UI Layer
4.1 ui/layout.py

Responsibilities:

Define the Streamlit page structure:

Sidebar controls

Main content sections

Explanatory text

Sections:

Header

App title & short description

Sidebar

Ticker input

Historical lookback period selector

Simulation count slider

Horizon (days) selector (default 252)

Probability threshold (default 66%)

“Run Simulation” button

Main Panel

Current price & stats card

Simulation summary (mean, median, percentile band)

Price path visualization

Terminal distribution histogram

Probability table (filtered by threshold)

Plain-language explanation box

Key Functions:

render_header()

render_sidebar_controls() -> dict (returns user choices)

render_main_results(...)

4.2 ui/components.py

Responsibilities:

Small, reusable elements:

Examples:

metric_cards(current_price, drift, vol)

probability_table(df_probs_filtered)

simulation_summary_box(percentile_band: dict, horizon_days: int)

explanation_box(ticker: str, stats: dict)

4.3 ui/plots.py

Responsibilities:

Generate plots (matplotlib / plotly) for Streamlit

Key Plots:

plot_price_paths(paths: np.ndarray, n_sample_paths: int = 50)

Show a subset of paths to avoid clutter

plot_terminal_distribution(terminal_prices: np.ndarray)

Histogram + key percentiles as vertical lines

Optional:

plot_cumulative_probability_curve(...)

CDF of terminal prices

5. Data Flow

User Input (UI)

User enters ticker, selects period, n_sims, horizon, prob_threshold.

Historical Data (data_loader)

get_historical_data(ticker, period) → price series

compute_log_returns(prices) → log returns

Model Parameters (simulation)

estimate_drift_vol(log_returns) → (drift, vol)

Simulations (simulation)

simulate_price_paths(s0, drift, vol, days, n_sims) → paths

Analytics (analytics)

get_terminal_prices(paths) → terminal price array

generate_price_grid(terminal_prices) → price levels

compute_probability_for_levels(...) → prob table

filter_by_probability(..., threshold=0.66) → filtered table

get_percentile_band(terminal_prices) → band dict

UI Rendering (layout, components, plots)

Render metrics, plots, filtered tables, and explanation text.

6. Streamlit App Entry (app.py)

High-level pseudo-flow:

import streamlit as st
from config.settings import *
from core.data_loader import get_historical_data, compute_log_returns
from core.simulation import estimate_drift_vol, simulate_price_paths
from core.analytics import (
    get_terminal_prices,
    generate_price_grid,
    compute_probability_for_levels,
    filter_by_probability,
    get_percentile_band,
)
from ui.layout import render_header, render_sidebar_controls, render_main_results

def main():
    st.set_page_config(page_title="MonteSight", layout="wide")

    render_header()
    user_inputs = render_sidebar_controls()

    if user_inputs["run"]:
        df_prices = get_historical_data(user_inputs["ticker"], user_inputs["period"])
        log_returns = compute_log_returns(df_prices["Close"])

        drift, vol = estimate_drift_vol(log_returns)
        s0 = df_prices["Close"].iloc[-1]

        paths = simulate_price_paths(
            s0=s0,
            drift=drift,
            vol=vol,
            days=user_inputs["horizon_days"],
            n_sims=user_inputs["n_sims"],
        )

        terminal = get_terminal_prices(paths)
        grid = generate_price_grid(terminal)
        df_probs = compute_probability_for_levels(terminal, grid)
        df_filtered = filter_by_probability(df_probs, threshold=user_inputs["prob_threshold"])
        band = get_percentile_band(terminal)

        render_main_results(
            ticker=user_inputs["ticker"],
            s0=s0,
            drift=drift,
            vol=vol,
            paths=paths,
            terminal_prices=terminal,
            probs_table=df_filtered,
            percentile_band=band,
            horizon_days=user_inputs["horizon_days"],
        )

if __name__ == "__main__":
    main()

7. Dependencies (requirements.txt)

Minimal set:

streamlit
yfinance
pandas
numpy
matplotlib
plotly


(Optional: scipy for more advanced stats, python-dotenv for env config.)

8. UX & Copy Guidelines

Use plain-language labels:

“Chance this price is reached within 1 year”

“Most likely price range”

“Simulated future price paths based on past volatility”

Add short tooltips explaining:

What Monte Carlo simulation is (one sentence)

That probabilities are model-based estimates, not guarantees

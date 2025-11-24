ARCHITECTURE — MonteSight: 1-Year Monte Carlo Stock Forecaster

This document describes the high-level architecture, module boundaries, and responsibilities for the MonteSight Streamlit application.

All implementation must conform to the PRD, DATA_MODEL, PIPELINE_SPEC, and AGENT_INSTRUCTIONS.

1. System Overview

MonteSight is a single-page Streamlit app that:

Accepts a stock ticker and simulation parameters from the user.

Uses yfinance to pull historical price data.

Computes log returns, drift, and volatility.

Runs a Monte Carlo GBM simulation for a 1-year horizon.

Transforms simulated paths into:

A terminal price distribution

A probability table of price targets

A percentile band for “most likely” price range

Renders everything in a clean, intuitive UI (charts, tables, summary).

The system is organized into the following logical layers:

Config Layer — Global defaults and constants

Core Layer — Data, simulation, analytics, utilities

UI Layer — Layout, components, plots, and app entrypoint

Tests — Unit and integration tests

2. Directory Structure
monte_sight/
├─ app.py
├─ requirements.txt
├─ README.md
├─ PRD.md
├─ DATA_MODEL.md
├─ PIPELINE_SPEC.md
├─ AGENT_INSTRUCTIONS.md
├─ BACKLOG.md
├─ SPRINT_PLAN.md
├─ ARCHITECTURE.md
├─ config/
│  └─ settings.py
├─ core/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ data_loader.py
│  ├─ simulation.py
│  └─ analytics.py
├─ ui/
│  ├─ __init__.py
│  ├─ layout.py
│  ├─ components.py
│  └─ plots.py
└─ tests/
   ├─ __init__.py
   ├─ test_utils.py
   ├─ test_data_loader.py
   ├─ test_simulation.py
   ├─ test_analytics.py
   └─ test_integration.py

3. Modules & Responsibilities
3.1 Config Layer
config/settings.py

Responsibility: Centralized configuration and defaults.

Contents:

Default ticker (e.g., "NVDA")

Default historical period (e.g., "5y")

Default number of simulations (e.g., 10_000)

Default forecast horizon (e.g., 252 trading days)

Default probability threshold (e.g., 0.66)

Default random seed (optional)

Any other tunable app constants

Design Notes:

No business logic here — only constants.

Used across app.py, core, and ui.

3.2 Core Layer

The core layer is responsible for all non-UI logic:
data retrieval, transforms, simulation, and analytics.

3.2.1 core/utils.py

Responsibility: Small, shared helper functions.

Examples:

set_random_seed(seed: int | None) -> None

format_currency(value: float, currency: str = "USD") -> str

safe_float(val, default=None) -> float | None

Constraints:

No external side-effect besides random seed.

Reused by simulation, analytics, and UI.

3.2.2 core/data_loader.py

Responsibility: Fetch and preprocess historical data.

Key Functions:

get_historical_data(ticker: str, period: str = "5y") -> pd.DataFrame

Uses yfinance to fetch OHLCV data.

Ensures Adj Close column exists.

Handles invalid tickers / empty data gracefully.

compute_log_returns(df: pd.DataFrame, col: str = "Adj Close") -> pd.Series

Computes daily log returns:

log_return_t = log(price_t / price_{t-1})

Drops NaNs and returns a pd.Series.

Constraints:

No Streamlit or UI code in this file.

Must not hardcode tickers or periods.

3.2.3 core/simulation.py

Responsibility: Estimation of parameters and Monte Carlo GBM simulation.

Key Functions:

estimate_drift_vol(log_returns: pd.Series) -> tuple[float, float]

Computes daily mean & std of log returns.

Returns (mu, sigma).

simulate_price_paths( s0: float, drift: float, vol: float, days: int = 252, n_sims: int = 10_000, seed: int | None = None, ) -> np.ndarray

Vectorized GBM simulation.

Output shape: (days + 1, n_sims).

First row is all s0.

Constraints:

Must be efficient and fully vectorized (no Python loops over sims).

No UI or printing.

3.2.4 core/analytics.py

Responsibility: Convert simulated paths into probabilities and summaries.

Key Functions:

get_terminal_prices(paths: np.ndarray) -> np.ndarray

generate_price_grid(terminal_prices: np.ndarray, n_points: int = 50) -> np.ndarray

compute_probability_for_levels( terminal_prices: np.ndarray, price_levels: np.ndarray ) -> pd.DataFrame

Produces DataFrame with price_level, prob_hit.

filter_by_probability( df_probs: pd.DataFrame, threshold: float = 0.66 ) -> pd.DataFrame

get_percentile_band( terminal_prices: np.ndarray, lower: float = 0.17, upper: float = 0.83 ) -> dict

Returns { "p17": float, "p50": float, "p83": float }.

Constraints:

All logic should match the DATA_MODEL and PIPELINE_SPEC.

No Streamlit imports.

3.3 UI Layer

The UI layer is responsible for all Streamlit rendering and user interaction.

3.3.1 ui/layout.py

Responsibility: Overall page layout and main orchestration of UI.

Key Functions:

render_header() -> None

Renders title, subtitle, brief description.

render_sidebar_controls() -> dict

Renders sidebar inputs (ticker, period, n_sims, horizon, prob threshold, seed).

Returns a user_inputs dict matching the Data Model.

render_main_results( ticker: str, s0: float, mu: float, sigma: float, df_prices: pd.DataFrame, paths: np.ndarray, terminal_prices: np.ndarray, df_probs_filtered: pd.DataFrame, percentile_band: dict, horizon_days: int, prob_threshold: float, ) -> None

Calls components & plots to display:

Metrics

Charts

Probability table

Explanation box

Constraints:

No heavy numerical logic here — delegate to core and ui/plots.py / ui/components.py.

3.3.2 ui/components.py

Responsibility: Reusable UI building blocks.

Example Components:

metric_cards(...)

Shows current price, median forecast, etc.

probability_table(df_probs_filtered: pd.DataFrame)

Nicely formatted table of price levels & probabilities.

simulation_summary_box(percentile_band: dict, horizon_days: int)

Short textual summary of band.

explanation_box( ticker: str, s0: float, percentile_band: dict, prob_threshold: float, )

Natural-language summary of forecast and risk.

Constraints:

All components should be pure rendering based on passed data.

3.3.3 ui/plots.py

Responsibility: Plotting logic for Streamlit.

Key Functions:

plot_price_paths(paths: np.ndarray, n_sample_paths: int = 50)

Show a subset of paths to keep chart readable.

plot_terminal_distribution( terminal_prices: np.ndarray, percentile_band: dict, )

Histogram with vertical lines at p17, p50, p83.

Constraints:

Use matplotlib or plotly, embedded via Streamlit.

No simulation or analytics logic.

3.4 App Entrypoint
app.py

Responsibility: Tie all layers together and run the Streamlit app.

Key Behaviors:

Set page config (title, layout).

Render header and sidebar controls.

If user presses “Run Simulation”:

Call get_historical_data() and compute_log_returns().

Call estimate_drift_vol() and simulate_price_paths().

Call analytics functions to get:

terminal_prices

price_levels

df_probs / df_probs_filtered

percentile_band

Pass everything to render_main_results().

Handle errors gracefully with st.error() / st.warning().

Constraints:

app.py should contain no heavy data-science logic — just orchestration.

4. Tests
4.1 Unit Tests

Located under tests/:

test_utils.py

Random seed & formatting behavior.

test_data_loader.py

Presence of Adj Close, empty ticker behavior, etc.

test_simulation.py

Shapes, seed behavior, and first-row value.

test_analytics.py

Probability monotonicity, percentile ordering, grid ranges.

4.2 Integration Test

test_integration.py

Minimal end-to-end test:

Download data for a real ticker

Compute log returns

Estimate params

Simulate paths

Run analytics

Assert key structures have expected shapes and non-empty values.

5. Data Flow (Summary)

UI (Sidebar) → user_inputs dict

Data Loader → df_prices → log_returns

Simulation → (mu, sigma) → paths

Analytics → terminal_prices, price_levels, df_probs, df_probs_filtered, percentile_band

UI (Main) → plots + tables + explanation

6. Technology Stack

Language: Python 3.10+

Framework: Streamlit

Data: pandas, numpy

Market Data: yfinance

Plots: matplotlib or plotly

Tests: pytest (recommended, or built-in unittest)

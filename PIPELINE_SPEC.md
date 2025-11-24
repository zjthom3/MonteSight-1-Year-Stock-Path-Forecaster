Pipeline Spec — MonteSight: 1-Year Monte Carlo Stock Forecaster

This document defines the end-to-end processing pipeline from user input → data retrieval → simulation → analytics → UI rendering.

All functions should be pure where possible and return well-defined outputs (see DATA_MODEL.md).

1. High-Level Flow

Read user inputs from Streamlit sidebar.

Fetch historical price data from yfinance.

Compute log returns.

Estimate drift and volatility.

Run Monte Carlo simulations for 1-year horizon.

Compute terminal prices.

Generate price level grid.

Compute probabilities for each price level.

Filter price levels by probability threshold (e.g., ≥ 66%).

Compute percentile band (p17, p50, p83).

Render plots, tables, and explanations in Streamlit.

2. Module-Level Responsibilities

core/data_loader.py

Fetch historical data

Compute log returns

core/simulation.py

Estimate drift & volatility

Simulate price paths

core/analytics.py

Terminal prices

Price grid

Probability computations

Percentile bands

Filtering

ui/layout.py, ui/plots.py, ui/components.py

UI wiring, plots, tables, summaries

3. Detailed Pipeline Steps
Step 0: User Input Collection (UI Layer)

Location: ui/layout.py

Function (example):

def render_sidebar_controls() -> dict:
    """
    Render Streamlit sidebar and return user input configuration.
    """


Responsibilities:

Collect:

ticker (text input, default e.g. "NVDA")

period (selectbox: "1y", "3y", "5y", default "5y")

n_sims (slider, e.g. 1_000–50_000, default 10_000)

horizon_days (slider, default 252)

prob_threshold (slider 0.5–0.9, default 0.66)

Optional seed

Return user_inputs dict (see DATA_MODEL.md).

Step 1: Historical Data Retrieval

Location: core/data_loader.py

1.1 Fetch Historical Data
def get_historical_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker and period using yfinance.
    Returns a DataFrame with at least 'Adj Close'.
    Raises a custom exception or returns empty DataFrame on failure.
    """


Logic:

Use yfinance.Ticker(ticker).history(period=period)

Validate:

DataFrame non-empty

Column Adj Close exists

Optionally drop rows with NaNs in Adj Close.

1.2 Compute Log Returns
def compute_log_returns(df: pd.DataFrame, col: str = "Adj Close") -> pd.Series:
    """
    Compute daily log returns from adjusted close prices.
    log_return_t = log(price_t / price_{t-1})
    """


Logic:

Extract price series: prices = df[col]

Compute: log_returns = np.log(prices / prices.shift(1)).dropna()

Set name to "log_return".

Outputs:

df_prices: pd.DataFrame

log_returns: pd.Series

Step 2: Parameter Estimation (Drift & Volatility)

Location: core/simulation.py

def estimate_drift_vol(log_returns: pd.Series) -> tuple[float, float]:
    """
    Estimate drift (mu) and volatility (sigma) from historical log returns.
    """


Logic:

mu_daily = log_returns.mean()

sigma_daily = log_returns.std()

For now, keep them as daily parameters for the GBM model.

Optionally, could compute annualized values, but simulation will operate on daily.

Output:

mu: float

sigma: float

Step 3: Monte Carlo Simulation

Location: core/simulation.py

def simulate_price_paths(
    s0: float,
    drift: float,
    vol: float,
    days: int = 252,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM-based Monte Carlo price paths.
    Returns a (days + 1, n_sims) array.
    """


Logic (GBM-style):

If seed is not None, set numpy random seed.

Generate random shocks:

Z = np.random.normal(size=(days, n_sims))


Use discrete GBM step:

dt = 1.0  # 1 trading day
increments = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * Z
log_price_paths = np.vstack([
    np.zeros((1, n_sims)),
    np.cumsum(increments, axis=0)
])
paths = s0 * np.exp(log_price_paths)


Ensure result shape (days + 1, n_sims).

Output:

paths: np.ndarray

Step 4: Terminal Prices Extraction

Location: core/analytics.py

def get_terminal_prices(paths: np.ndarray) -> np.ndarray:
    """
    Extract terminal prices from simulated paths.
    Returns array of shape (n_sims,).
    """


Logic:

terminal_prices = paths[-1, :]

Step 5: Price Level Grid Generation

Location: core/analytics.py

def generate_price_grid(
    terminal_prices: np.ndarray,
    n_points: int = 50
) -> np.ndarray:
    """
    Generate an evenly spaced grid of price levels spanning the terminal prices.
    """


Logic:

min_price = terminal_prices.min()

max_price = terminal_prices.max()

Generate grid:

levels = np.linspace(min_price, max_price, n_points)


Output:

price_levels: np.ndarray

Step 6: Probability Computation for Price Levels

Location: core/analytics.py

def compute_probability_for_levels(
    terminal_prices: np.ndarray,
    price_levels: np.ndarray,
) -> pd.DataFrame:
    """
    For each price level, compute probability that terminal price >= level.
    Returns DataFrame with ['price_level', 'prob_hit'].
    """


Logic:

For each L in price_levels:

prob = (terminal_prices >= L).mean()


Build DataFrame:

df = pd.DataFrame({
    "price_level": price_levels,
    "prob_hit": probs_array
})


Sort by price_level ascending.

Output:

df_probs: pd.DataFrame

Step 7: Filter by Probability Threshold

Location: core/analytics.py

def filter_by_probability(
    df_probs: pd.DataFrame,
    threshold: float = 0.66
) -> pd.DataFrame:
    """
    Filter probability table to levels with prob_hit >= threshold.
    """


Logic:

df_filtered = df_probs[df_probs["prob_hit"] >= threshold].copy()

Sort by price_level ascending or descending (decide once).

Output:

df_probs_filtered: pd.DataFrame

Step 8: Percentile Band Computation

Location: core/analytics.py

def get_percentile_band(
    terminal_prices: np.ndarray,
    lower: float = 0.17,
    upper: float = 0.83,
) -> dict:
    """
    Compute percentile band for terminal prices.
    Returns dict with keys 'p17', 'p50', 'p83' by default.
    """


Logic:

p17 = np.percentile(terminal_prices, lower * 100)
p50 = np.percentile(terminal_prices, 50)
p83 = np.percentile(terminal_prices, upper * 100)

return {"p17": p17, "p50": p50, "p83": p83}


Output:

percentile_band: dict

4. UI Rendering Pipeline

Location: app.py, ui/layout.py, ui/plots.py, ui/components.py

4.1 High-Level app.py Flow
def main():
    st.set_page_config(page_title="MonteSight", layout="wide")

    render_header()
    user_inputs = render_sidebar_controls()

    if user_inputs["run"]:
        # 1) Data
        df_prices = get_historical_data(user_inputs["ticker"], user_inputs["period"])
        log_returns = compute_log_returns(df_prices)

        # 2) Parameters
        mu, sigma = estimate_drift_vol(log_returns)
        s0 = float(df_prices["Adj Close"].iloc[-1])

        # 3) Simulation
        paths = simulate_price_paths(
            s0=s0,
            drift=mu,
            vol=sigma,
            days=user_inputs["horizon_days"],
            n_sims=user_inputs["n_sims"],
            seed=user_inputs.get("seed"),
        )

        # 4) Analytics
        terminal = get_terminal_prices(paths)
        price_levels = generate_price_grid(terminal)
        df_probs = compute_probability_for_levels(terminal, price_levels)
        df_probs_filtered = filter_by_probability(df_probs, user_inputs["prob_threshold"])
        percentile_band = get_percentile_band(terminal)

        # 5) Render UI
        render_main_results(
            ticker=user_inputs["ticker"],
            s0=s0,
            mu=mu,
            sigma=sigma,
            df_prices=df_prices,
            paths=paths,
            terminal_prices=terminal,
            df_probs_filtered=df_probs_filtered,
            percentile_band=percentile_band,
            horizon_days=user_inputs["horizon_days"],
            prob_threshold=user_inputs["prob_threshold"],
        )

4.2 Plot Functions

Location: ui/plots.py

Examples:

def plot_price_paths(paths: np.ndarray, n_sample_paths: int = 50):
    """
    Plot a subset of simulated price paths over time.
    """

def plot_terminal_distribution(
    terminal_prices: np.ndarray,
    percentile_band: dict,
):
    """
    Plot histogram of terminal prices with vertical lines at p17, p50, p83.
    """

4.3 Components & Explanations

Location: ui/components.py

Key components:

Metric cards: current price, median forecast, 66% band

Probability table (from df_probs_filtered)

Explanation box:

def explanation_box(
    ticker: str,
    s0: float,
    percentile_band: dict,
    prob_threshold: float,
):
    """
    Render a natural-language explanation of forecast and probability band.
    """

5. Error Handling Pipeline

If get_historical_data fails or returns empty:

Show Streamlit error: “Unable to load data for ticker XYZ.”

Skip further steps.

If log_returns has too few points:

Show warning: “Not enough historical data to simulate.”

Catch unexpected exceptions at top-level in main() and show a generic error message to the user.

Data Model — MonteSight: 1-Year Monte Carlo Stock Forecaster

This document defines the core data structures used across the app.
All structures should be implemented using pandas, numpy, and native Python types.

1. Overview

MonteSight uses the following main data objects:

Historical Price Data (pd.DataFrame)

Log Returns (pd.Series)

Simulation Parameters (dict)

Simulated Price Paths (np.ndarray)

Terminal Prices (np.ndarray)

Price Level Grid (np.ndarray)

Probability Table (pd.DataFrame)

Percentile Band Summary (dict)

UI State / User Inputs (dict)

Each section below defines the schema and usage.

2. Historical Price Data

Type: pandas.DataFrame
Source: yfinance.download or yf.Ticker().history
Index: DatetimeIndex (trading days)

Schema:

Column	Type	Description
Open	float64	Opening price for the day
High	float64	Highest price of the day
Low	float64	Lowest price of the day
Close	float64	Closing price of the day
Adj Close	float64	Adjusted close price (splits/dividends accounted for)
Volume	int64	Trading volume

Notes:

The app should use Adj Close as the primary series for returns and simulation.

Column presence may vary, but Adj Close is required.

Example variable name:
df_prices: pd.DataFrame

3. Log Returns

Type: pandas.Series
Index: Aligns with df_prices.index (but one element shorter due to diff)

Schema:

Name: "log_return" (or similar)

Values: float64

Computed as: log(AdjClose_t / AdjClose_{t-1})

Example variable name:
log_returns: pd.Series

4. Simulation Parameters

Type: dict[str, Any]

Schema:

Key	Type	Description
ticker	str	Stock ticker symbol (e.g., "NVDA")
s0	float	Current price (last Adj Close)
mu	float	Estimated drift from historical log returns
sigma	float	Estimated volatility from historical log returns
days	int	Number of trading days to simulate (default 252)
n_sims	int	Number of simulated paths (default 10_000)
prob_threshold	float	Probability cutoff (default 0.66)
seed	int|None	Random seed for reproducibility
lookback_period	str	Historical lookback (e.g., "5y", "2y")

Example variable name:
sim_config: dict

5. Simulated Price Paths

Type: numpy.ndarray
Shape: (days + 1, n_sims)

Definition:

Each column = one simulated price path.

Each row = simulated price at a given time step.

Row 0 = s0 for all paths.

Example:

paths[0, :] → initial prices (all s0)

paths[-1, :] → terminal prices at forecast horizon

Example variable name:
paths: np.ndarray

6. Terminal Prices

Type: numpy.ndarray
Shape: (n_sims,)

Definition:

Extracted as the last row of paths:

terminal_prices = paths[-1, :]

Example variable name:
terminal_prices: np.ndarray

7. Price Level Grid

Type: numpy.ndarray
Shape: (n_levels,) (1D array)

Definition:

Represents the set of future price levels at which probabilities will be computed.

Typically generated as a range that spans the min–max of terminal_prices, with smoothing.

Generation approach (example):

min_price = np.min(terminal_prices)

max_price = np.max(terminal_prices)

levels = np.linspace(min_price, max_price, num=n_points)

n_points default: 50–100

Example variable name:
price_levels: np.ndarray

8. Probability Table

Type: pandas.DataFrame

Two main variants:

Raw Probability Table (all levels)

Filtered Probability Table (≥ threshold)

8.1 Raw Probability Table

Schema:

Column	Type	Description
price_level	float64	Specific target price level
prob_hit	float64	Probability (0–1) terminal price is ≥ price_level

Notes:

prob_hit is computed as:
prob_hit = (terminal_prices >= price_level).mean() for each level.

DataFrame is typically sorted by price_level ascending.

Example variable name:
df_probs: pd.DataFrame

8.2 Filtered Probability Table

Schema:

Same as above, but filtered to only include:

prob_hit >= prob_threshold (e.g., 0.66)

Example variable name:
df_probs_filtered: pd.DataFrame

9. Percentile Band Summary

Type: dict[str, float]

Definition:

Contains key summary statistics for the distribution of terminal prices.

Schema:

Key	Type	Description
p17	float	17th percentile terminal price
p50	float	50th percentile (median) terminal price
p83	float	83rd percentile terminal price

Notes:

The 17–83 band roughly represents a 1σ-like range for a normal-like distribution.

This range is used in UI to describe “most likely” price region.

Example variable name:
percentile_band: dict

10. Ticker Metadata (Optional but Recommended)

Type: dict[str, Any]

Schema:

Key	Type	Description
ticker	str	Ticker symbol
name	str	Company name
exchange	str	Exchange code
currency	str	Trading currency (e.g., "USD")

Example variable name:
ticker_info: dict

11. UI State / User Inputs

Type: dict[str, Any]
Source: Streamlit widgets

Schema:

Key	Type	Description
ticker	str	User-entered ticker
period	str	Historical lookback (e.g., "5y")
n_sims	int	Number of simulations
horizon_days	int	Number of trading days to simulate
prob_threshold	float	Probability cutoff (e.g., 0.66)
seed	int|None	Random seed for reproducibility
run	bool	Whether user pressed “Run Simulation” button

Example variable name:
user_inputs: dict

12. Plot Data Structures

Plots will typically reuse the same underlying data structures:

12.1 Price Paths Plot

Uses: paths: np.ndarray

Optional: Expose a paths_sample: np.ndarray for plotting limited subset

Shape: (days + 1, n_sample_paths)

12.2 Distribution Plot

Uses: terminal_prices: np.ndarray

Overlays:

Vertical lines at p17, p50, p83 from percentile_band

12.3 Probability Table Display

Uses: df_probs_filtered: pd.DataFrame

Sorted by price_level ascending or descending based on UI choice.

13. Error & Status Objects

Type: dict[str, Any] or simple strings handled in UI.

Examples:

error_message: str | None

status_message: str | None

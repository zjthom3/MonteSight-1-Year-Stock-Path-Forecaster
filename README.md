# MonteSight — 1-Year Monte Carlo Stock Forecaster

MonteSight is a Streamlit app that uses historical price data to run Monte Carlo simulations and surface probability-based price targets for the next year.

## Features
- Pulls historical prices via **yfinance**
- Computes log returns, drift, and volatility
- Simulates thousands of price paths with a GBM model
- Shows 66%+ probability price targets, percentile bands, and clear visuals

## Quickstart
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## How the simulation works
1. Download historical prices (default 5 years) and compute daily log returns.
2. Estimate daily drift (`mu`) and volatility (`sigma`) from those returns.
3. Simulate `n_sims` GBM price paths over `horizon_days` (default 252).
4. Extract terminal prices, build a price grid, and compute the probability of meeting/exceeding each level.
5. Filter for targets with probability ≥ chosen threshold (default 66%) and display percentile bands (p17, p50, p83).

## Usage tips
- Increase simulations for smoother distributions; decrease for faster runs.
- Adjust the probability threshold to see more or fewer targets.
- Provide a seed for reproducible runs.

## Notes and limitations
- Forecasts are purely statistical and do not incorporate fundamentals or news.
- Data availability depends on yfinance; invalid tickers are handled with clear errors.
- Performance assumes typical laptop CPU; extremely large simulation counts may slow down rendering.

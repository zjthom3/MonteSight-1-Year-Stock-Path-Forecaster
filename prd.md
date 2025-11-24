PRD — MonteSight: 1-Year Monte Carlo Stock Forecaster
1. Product Summary

MonteSight is a Streamlit-based application that lets users analyze a stock’s potential future performance using Monte Carlo simulations with a 1-year forecast window.
The app uses historical price data from yfinance, computes drift/volatility from log returns, and simulates thousands of price paths.
Users receive probabilities of hitting specific price levels, with a focus on levels with ≥ 66% probability.
The UI must be simple, interactive, and suitable for novice or expert users.

2. Goals
Primary Goals

Provide a fast, intuitive forecasting dashboard for any stock.

Display the probability distribution of future prices over 1 year.

Show only statistically meaningful price targets (≥ 66% probability).

Use clean visuals and natural-language summaries for accessibility.

Secondary Goals

Enable quick exploration of tickers.

Make simulations reproducible, efficient, and adjustable.

Offer deployable architecture (Streamlit Cloud, HuggingFace Spaces, etc.).

3. Users
Target Users

Retail investors

Financial analysts

Students learning quantitative finance

Data-driven traders

Curious beginners

User Needs

Fast, clear insights into future stock price odds

Easy exploration of tickers

Visual explanations

Probability-based price ranges

Minimal setup & effortless use

4. Key Features
4.1 Ticker Input & Data Retrieval

User enters a stock ticker (e.g., NVDA)

App retrieves:

Historical prices (default 5 years)

Adjusted close data

Log returns

Data source: yfinance

4.2 Monte Carlo Simulation Engine (1-Year Horizon)

Compute drift & volatility from historical log returns

Simulate N price paths (default: 10,000)

Simulation length: 252 trading days

Output:

Price paths (2D array)

Terminal price distribution

4.3 Probability Engine

Generate a grid of price levels

Compute probability each target is met or exceeded

Filter probabilities to show only ≥ 66%

4.4 Visual Outputs

Required visualizations:

Price path fan chart (subset of simulated paths)

Terminal price distribution histogram

Probability table with ≥ 66% likelihood

Percentile bands (p17, p50, p83)

4.5 Text Summaries

Plain-language explanation including:

Most likely price range

Median projected price

Volatility sentiment (high/medium/low)

Probability of increasing vs decreasing

5. System Requirements
5.1 Functional Requirements
FR-1: The system must retrieve historical data using yfinance.
FR-2: The system must compute daily log returns.
FR-3: The system must compute drift and volatility from historical returns.
FR-4: The system must simulate at least 10,000 Monte Carlo price paths.
FR-5: The system must compute the probability of terminal prices hitting predefined levels.
FR-6: The UI must display only price targets with ≥ 66% probability unless modified.
FR-7: The UI must allow adjusting:

Number of simulations

Historical window

Probability threshold

Forecast horizon

FR-8: All calculations must complete within 2 seconds on common hardware.
5.2 Non-Functional Requirements
NFR-1: Performance

All simulations should run within 2s on a laptop CPU.

NFR-2: Usability

Clear labels, simple explanations, minimal text

All visuals must be readable and clean

NFR-3: Reliability

Simulation outputs must be reproducible (optional random seed)

Must handle invalid tickers gracefully

NFR-4: Extensibility

Architecture must allow future expansion:

Multi-year forecasts

Other simulation models (Heston, GBM+Jumps)

Portfolio-level simulations

6. Technical Architecture Summary

MonteSight consists of 4 core layers:

Data Layer

yfinance → cleaned historical price data

Log return computations

Simulation Layer

Drift/vol estimation

Monte Carlo price paths

Terminal price distribution

Analytics Layer

Percentile bands

Probability calculations

Price grid generation

66%+ probability filtering

UI Layer (Streamlit)

Controls in sidebar

Plots & tables

Interpretation text

This maps directly to folder structure for modularity and clear boundaries.

7. User Flow
Step 1: Ticker Input

User enters a ticker (ex: “NVDA”).

Step 2: Historical Data

App loads 5y of adjusted close data.

Step 3: Simulation

Monte Carlo engine generates 10,000 paths.

Step 4: Analytics

Compute terminal prices

Compute price target probabilities

Filter for ≥ 66%

Step 5: Output

Show price path chart

Show terminal distribution

Show probability table

Explain results in plain language

8. Edge Cases & Error Handling

Invalid tickers → Show warning + instructions

No internet/data source unavailable → Retry message

Missing historical data → Fallback to shorter window

Extremely volatile assets → Clamp charts to avoid distortion

9. Dependencies

Python 3.10+

Streamlit

yfinance

pandas

numpy

matplotlib or plotly

10. Success Metrics

Simulation completes in <2 seconds

Visual outputs readable without explanation

Users understand probability table within 10 seconds

95% of tickers load without errors

11. Future Enhancements (Not in MVP)

Multi-year simulations

Scenario overlays (bull/base/bear drift shifts)

Options pricing simulation

Portfolio risk model (VaR/CVaR)

Export to PDF/Excel

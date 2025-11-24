Sprint Plan — MonteSight (1-Week MVP Sprint)

This sprint plan turns the backlog into a clear, day-by-day execution schedule designed for a Codex agent or engineering team to deliver a working MVP of the MonteSight Streamlit app.

Sprint Goal

Deliver a fully functional MonteSight MVP that:

Loads historical data via yfinance

Runs Monte Carlo simulations

Produces 1-year probability forecasts

Displays clean Streamlit UI

Shows the 66%+ probability price targets

Includes basic tests, README, and requirements

Sprint Duration

5 Days (One Work Week)
Target velocity: Complete all P0 items from the Backlog.

Team

Codex Agent — primary engineer

You (Jerel) — product owner, reviewer

Day-by-Day Plan
🔵 Day 1 — Core Foundation & Data Layer
Objectives

Build the utilities, data loader, and return computations.

Tasks

Create folder structure

Implement /core/utils.py

Implement /core/data_loader.py:

get_historical_data()

compute_log_returns()

Ensure error handling for:

Invalid tickers

Missing data

Write unit tests for utils + data loader

Validate with 1–2 tickers (NVDA, BTC-USD)

Deliverables

Fully working data loader

Log return calculation tested

Tests passing

🟢 Day 2 — Simulation Engine
Objectives

Implement the Monte Carlo engine and drift/volatility estimation.

Tasks

Implement:

estimate_drift_vol()

simulate_price_paths()

Vectorize simulation (no loops)

Add optional random seed

Create tests:

Drift/vol sanity check

Price paths shape: (days+1, n_sims)

Initial row = starting price

Deliverables

Simulation engine fully functional

Test suite passing

🟡 Day 3 — Analytics Layer
Objectives

Build mechanics that turn simulation paths into actionable insights.

Tasks

Implement:

get_terminal_prices()

generate_price_grid()

compute_probability_for_levels()

filter_by_probability()

get_percentile_band()

Add unit tests:

Terminal prices correctly extracted

Probabilities monotonic

Percentile band in correct order

Deliverables

Analytics module complete and tested

End-to-end pipeline working headless (CLI test)

🟣 Day 4 — Streamlit UI & Integration
Objectives

Create the full user interface and wire everything into app.py.

Tasks

Build Streamlit layout:

Sidebar controls

Header section

Implement UI components:

Metric cards

Summary boxes

Probability table

Explanation text

Implement plots:

Price path visualization

Terminal distribution histogram

Build app.py using the pipeline spec

Integration testing with several tickers

Deliverables

Streamlit app runs end-to-end with clean UI

Plots render correctly

Probability table filters to ≥ 66%

🟠 Day 5 — Final Polish, Tests, Docs, Deployment Prep
Objectives

Finish polish, testing, documentation, and deployment readiness.

Tasks

Write README:

Overview

Installation

Running the app

How the simulation works

Write requirements.txt

Add integration tests

Clean up unused imports & fix lints

Optional: prep for Streamlit Cloud deployment

Final run-through with multiple tickers, including crypto

Deliverables

Fully documented project

All P0 tasks complete

App ready for deployment or demo

Sprint Milestones
Milestone	Due	Description
M1 – Data Layer Ready	End of Day 1	Loader + returns working
M2 – Simulation Ready	End of Day 2	Core Monte Carlo engine complete
M3 – Analytics Ready	End of Day 3	Probability + percentiles working
M4 – UI Ready	End of Day 4	Streamlit app functional
M5 – Final MVP	End of Day 5	Tests, README, polish complete
Definition of Done

The sprint is complete when:

App loads, simulates, and displays results with no errors

UI renders:

Path plot

Terminal distribution

Probability table (≥ 66%)

Explanation summary

README + requirements.txt exist

Tests pass

Code matches PRD, Data Model, Pipeline Spec, and Agent Instructions

Backlog — MonteSight: 1-Year Monte Carlo Stock Forecaster

This backlog contains all tasks required to deliver the MVP described in the PRD, Data Model, and Pipeline Spec.
Tasks are written in a way a Codex agent can directly follow.

1. Epic: Core Data + Simulation Engine
1.1 Story: Implement Utility Functions

Tasks

 Create core/utils.py

 Add set_random_seed(seed)

 Add format_currency(value, currency="USD")

 Add safe_float(val, default=None)

 Add complete docstrings for all functions

 Add unit tests for all utils

1.2 Story: Implement Historical Data Loader

Tasks

 Create core/data_loader.py

 Implement get_historical_data(ticker, period="5y")

 Implement compute_log_returns(df, col="Adj Close")

 Add validation for empty DataFrame or missing Adj Close

 Add error handling for invalid tickers

 Create unit tests:

empty data

valid data

missing column

1.3 Story: Implement Drift & Volatility Estimates

Tasks

 Create core/simulation.py

 Implement estimate_drift_vol(log_returns)

 Validate minimum number of data points

 Add unit tests for estimation correctness

1.4 Story: Implement Monte Carlo Engine

Tasks

 Implement simulate_price_paths(...)

 Add GBM vectorized simulation (no loops)

 Validate output shapes

 Add seed support

 Implement test:

shape (days+1, n_sims)

initial value equals s0

2. Epic: Analytics Layer
2.1 Story: Terminal Price Extraction

Tasks

 Implement get_terminal_prices(paths)

 Test shape and value correctness

2.2 Story: Price Grid Generation

Tasks

 Implement generate_price_grid(terminal_prices, n_points=50)

 Validate sorted output

 Add test for correct range

2.3 Story: Probability Computation

Tasks

 Implement compute_probability_for_levels(...)

 Compute P(terminal >= level)

 Create DataFrame with price_level, prob_hit

 Add test for monotonic decreasing probabilities

2.4 Story: Filtering Logic

Tasks

 Implement filter_by_probability(df_probs, threshold)

 Add validation for empty result

 Add unit tests

2.5 Story: Percentile Band

Tasks

 Implement get_percentile_band(terminal_prices)

 Compute p17, p50, p83

 Tests: percentile ordering, values in terminal range

3. Epic: UI Layer (Streamlit)
3.1 Story: Build Core Layout

Tasks

 Create ui/layout.py

 Implement render_header()

 Implement render_sidebar_controls()

 Return user inputs as a dict

 Validate types of all inputs

3.2 Story: Build Main Output Area

Tasks

 Implement render_main_results(...)

 Display metric cards, charts, probability table

 Connect to analytics & simulation modules

 Add loading states and error messages

3.3 Story: Plotting Functions

Tasks

 Create ui/plots.py

 Implement plot_price_paths(paths, n_sample_paths=50)

 Implement plot_terminal_distribution(...)

 Add percentile band overlay lines

 Validate performance with large n_sims

3.4 Story: UI Components

Tasks

 Create ui/components.py

 Implement:

metric_cards

probability_table

simulation_summary_box

explanation_box

 Ensure all components can be reused

4. Epic: App Integration
4.1 Story: Build app.py

Tasks

 Create top-level app.py

 Set Streamlit page configuration

 Wire together sidebar → pipeline → UI

 Handle all exceptions gracefully

 Add clear user-facing explanations and tooltips

4.2 Story: End-to-End Validation

Tasks

 Run complete simulation for several tickers

 Validate:

UI loads

Plots render

Probability table filters correctly

Percentile band matches analytics

 Check performance meets NFR (< 2s simulation)

5. Epic: Testing
5.1 Story: Create Test Structure

Tasks

 Create /tests directory

 Add __init__.py files

5.2 Story: Module Unit Tests

Tasks

 Tests for utils

 Tests for data loader

 Tests for simulation engine

 Tests for analytics logic

5.3 Story: Integration Tests

Tasks

 Ensure pipeline runs from historical lookback → terminal prices

 Validate shapes and expected outputs

6. Epic: Documentation
6.1 Story: Requirements File

Tasks

 Create requirements.txt

 List all required Python packages

6.2 Story: README

Tasks

 Write README.md

 Include:

What the app does

How to install

How to run locally

How the Monte Carlo engine works

Example usage

Notes about forecasting limitations

7. Epic: Deployment (Optional After MVP)
7.1 Story: Prepare Streamlit Cloud Deployment (optional)

Tasks

 Validate directory structure

 Ensure requirements.txt is correct

 Add app link once deployed

8. Prioritization for MVP
P0 – Must Have

Core modules: data loader, simulation, analytics

UI layout and main results

All plots

Probability table with 66% filter

Percentile band

app.py fully runnable

Basic tests

Requirements + README

P1 – Should Have

Error-handling polish

Component abstractions

Additional UI enhancements

P2 – Nice to Have

Caching

Multi-ticker comparison feature

Export PDF option

Dark mode

"""
MonteSight Streamlit application entrypoint.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from core.analytics import (
    filter_by_probability,
    generate_price_grid,
    get_percentile_band,
    get_terminal_prices,
    compute_path_hit_probability_for_levels,
)
from core.data_loader import DataLoaderError, compute_log_returns, get_historical_data
from core.external_data import aggregate_external_signals
from core.scenarios import build_scenario_configs, run_scenarios
from core.simulation import estimate_drift_vol, simulate_price_paths
from ui.layout import render_header, render_main_results, render_sidebar_controls


def run_simulation_pipeline(user_inputs: dict) -> None:
    """
    Execute the end-to-end simulation pipeline and render results.
    """
    ticker = user_inputs["ticker"]
    period = user_inputs["period"]
    horizon_days = user_inputs["horizon_days"]
    n_sims = user_inputs["n_sims"]
    prob_threshold = user_inputs["prob_threshold"]
    seed = user_inputs.get("seed")
    openai_api_key = user_inputs.get("openai_api_key")
    enable_scenarios = user_inputs.get("enable_scenarios", settings.DEFAULT_ENABLE_SCENARIOS)
    scenario_aggressiveness = user_inputs.get(
        "scenario_aggressiveness", settings.DEFAULT_SCENARIO_AGGRESSIVENESS
    )
    use_external_signals = user_inputs.get("use_external_signals", settings.DEFAULT_USE_EXTERNAL_SIGNALS)

    df_prices = get_historical_data(ticker, period=period)
    log_returns = compute_log_returns(df_prices)
    mu, sigma = estimate_drift_vol(log_returns)
    s0 = float(df_prices["Adj Close"].iloc[-1])

    sim_config = {
        "ticker": ticker,
        "s0": s0,
        "mu": mu,
        "sigma": sigma,
        "days": horizon_days,
        "n_sims": n_sims,
        "prob_threshold": prob_threshold,
        "seed": seed,
        "lookback_period": period,
        "price_grid_points": settings.DEFAULT_PRICE_GRID_POINTS,
        "scenario_aggressiveness": scenario_aggressiveness,
    }

    scenario_configs = None
    scenario_results = None
    external_signals = None

    if enable_scenarios:
        external_signals = aggregate_external_signals(ticker) if use_external_signals else None
        scenario_configs = build_scenario_configs(sim_config, external_signals=external_signals)
        scenario_results = run_scenarios(
            s0=s0,
            base_sim_config=sim_config,
            scenario_configs=scenario_configs,
            days=horizon_days,
            n_sims=n_sims,
            seed=seed,
        )
        base_result = scenario_results.get("base") or next(iter(scenario_results.values()))
        paths = base_result["paths"]
        terminal = base_result["terminal_prices"]
        df_probs_filtered = base_result["df_probs_filtered"]
        percentile_band = base_result["percentile_band"]
    else:
        paths = simulate_price_paths(
            s0=s0,
            drift=mu,
            vol=sigma,
            days=horizon_days,
            n_sims=n_sims,
            seed=seed,
        )
        terminal = get_terminal_prices(paths)
        price_levels = generate_price_grid(
            terminal, n_points=settings.DEFAULT_PRICE_GRID_POINTS, paths=paths
        )
        df_probs = compute_path_hit_probability_for_levels(paths, price_levels, current_price=s0)
        df_probs_filtered = filter_by_probability(df_probs, prob_threshold)
        percentile_band = get_percentile_band(terminal)

    render_main_results(
        ticker=ticker,
        s0=s0,
        mu=mu,
        sigma=sigma,
        df_prices=df_prices,
        paths=paths,
        terminal_prices=terminal,
        df_probs_filtered=df_probs_filtered,
        percentile_band=percentile_band,
        horizon_days=horizon_days,
        prob_threshold=prob_threshold,
        openai_api_key=openai_api_key,
        scenarios_enabled=enable_scenarios,
        scenario_results=scenario_results,
        scenario_configs=scenario_configs,
        external_signals=external_signals,
    )


def main() -> None:
    """Streamlit app entrypoint."""
    st.set_page_config(page_title="MonteSight", layout="wide")
    render_header()
    user_inputs = render_sidebar_controls()

    if not user_inputs.get("run"):
        st.info("Set your parameters in the sidebar, then click 'Run Simulation'.")
        return

    try:
        with st.spinner("Running simulation..."):
            run_simulation_pipeline(user_inputs)
    except DataLoaderError as exc:
        st.error(str(exc))
    except ValueError as exc:
        st.error(f"Unable to complete simulation: {exc}")
    except Exception as exc:  # pragma: no cover - unexpected errors
        st.error("An unexpected error occurred. Please try again.")
        raise exc


if __name__ == "__main__":
    main()

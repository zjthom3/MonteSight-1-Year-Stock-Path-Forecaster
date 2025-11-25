"""
Page layout and orchestration for MonteSight.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from core.utils import safe_float
from ui import components, plots


def render_header() -> None:
    """Render the main app header."""
    st.title("MonteSight — 1-Year Monte Carlo Stock Forecaster")
    st.caption("Explore probability-based price targets using Monte Carlo simulations.")


def render_sidebar_controls() -> dict:
    """
    Render sidebar controls and return user input configuration.

    Returns
    -------
    dict
        User input dictionary matching the data model.
    """
    st.sidebar.header("Simulation Controls")
    ticker = st.sidebar.text_input("Ticker", settings.DEFAULT_TICKER).strip().upper()
    period = st.sidebar.selectbox(
        "Historical Lookback",
        options=["1y", "3y", "5y", "10y"],
        index=["1y", "3y", "5y", "10y"].index(settings.DEFAULT_PERIOD)
        if settings.DEFAULT_PERIOD in ["1y", "3y", "5y", "10y"]
        else 2,
    )
    n_sims = st.sidebar.slider(
        "Number of Simulations",
        min_value=1_000,
        max_value=50_000,
        value=settings.DEFAULT_N_SIMS,
        step=1_000,
    )
    horizon_days = st.sidebar.slider(
        "Forecast Horizon (trading days)",
        min_value=60,
        max_value=756,
        value=settings.DEFAULT_HORIZON_DAYS,
        step=21,
    )
    prob_threshold = st.sidebar.slider(
        "Probability Threshold",
        min_value=0.50,
        max_value=0.90,
        value=float(settings.DEFAULT_PROB_THRESHOLD),
        step=0.01,
    )
    enable_scenarios = st.sidebar.checkbox(
        "Enable bull/base/bear scenarios",
        value=settings.DEFAULT_ENABLE_SCENARIOS,
        help="Run bull, base, and bear Monte Carlo variations with adjusted drift/vol.",
    )
    scenario_aggressiveness = st.sidebar.slider(
        "Scenario aggressiveness (spread)",
        min_value=0,
        max_value=100,
        value=int(settings.DEFAULT_SCENARIO_AGGRESSIVENESS),
        step=5,
        help="Controls how far bull/bear drift/vol deviate from the base case.",
        disabled=not enable_scenarios,
    )
    use_external_signals = st.sidebar.checkbox(
        "Incorporate external signals (news/filings/fundamentals)",
        value=settings.DEFAULT_USE_EXTERNAL_SIGNALS,
        help="If enabled, scenario biases will consider any available external signals.",
        disabled=not enable_scenarios,
    )
    seed = st.sidebar.text_input("Random Seed (optional)", value="")
    seed_val = safe_float(seed, default=None)
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key (optional for AI summary)",
        value="",
        type="password",
        help="If provided, an AI-generated narrative summary will be shown.",
    ).strip() or None

    run = st.sidebar.button("Run Simulation", type="primary")

    return {
        "ticker": ticker or settings.DEFAULT_TICKER,
        "period": period,
        "n_sims": int(n_sims),
        "horizon_days": int(horizon_days),
        "prob_threshold": float(prob_threshold),
        "seed": int(seed_val) if seed_val is not None else None,
        "openai_api_key": openai_api_key,
        "enable_scenarios": bool(enable_scenarios),
        "scenario_aggressiveness": int(scenario_aggressiveness),
        "use_external_signals": bool(use_external_signals),
        "run": run,
    }


def render_main_results(
    ticker: str,
    s0: float,
    mu: float,
    sigma: float,
    df_prices,
    paths,
    terminal_prices,
    df_probs_filtered,
    percentile_band,
    horizon_days: int,
    prob_threshold: float,
    openai_api_key: str | None = None,
    scenarios_enabled: bool = False,
    scenario_results: dict | None = None,
    scenario_configs: dict | None = None,
    external_signals: dict | None = None,
) -> None:
    """
    Render the main dashboard content.
    """
    base_paths = paths
    base_terminal_prices = terminal_prices
    base_probs_filtered = df_probs_filtered
    base_percentile_band = percentile_band
    base_mu = mu
    base_sigma = sigma

    if scenarios_enabled and scenario_results:
        base_result = scenario_results.get("base") or next(iter(scenario_results.values()))
        if base_result:
            base_paths = base_result.get("paths", paths)
            base_terminal_prices = base_result.get("terminal_prices", terminal_prices)
            base_probs_filtered = base_result.get("df_probs_filtered", df_probs_filtered)
            base_percentile_band = base_result.get("percentile_band", percentile_band)
            if scenario_configs and "base" in scenario_configs:
                base_mu = scenario_configs["base"].get("mu", mu)
                base_sigma = scenario_configs["base"].get("sigma", sigma)

    components.metric_cards(s0=s0, percentile_band=base_percentile_band, horizon_days=horizon_days)
    st.markdown(f"Estimated daily drift: **{base_mu:.5f}**, volatility: **{base_sigma:.5f}**")

    col1, col2 = st.columns(2)
    with col1:
        plots.plot_price_paths(
            base_paths, n_sample_paths=settings.DEFAULT_PATH_SAMPLE_SIZE, key="main-price-paths"
        )
    with col2:
        plots.plot_terminal_distribution(
            base_terminal_prices, base_percentile_band, key="main-terminal-dist"
        )

    if scenarios_enabled and scenario_results:
        components.render_scenario_comparison(
            scenario_results=scenario_results,
            horizon_days=horizon_days,
            prob_threshold=prob_threshold,
            sample_paths=settings.DEFAULT_PATH_SAMPLE_SIZE,
            s0=s0,
        )
        components.render_ai_scenario_report(
            ticker=ticker,
            scenario_results=scenario_results,
            horizon_days=horizon_days,
            external_signals=external_signals,
            openai_api_key=openai_api_key,
        )
        

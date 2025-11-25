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
) -> None:
    """
    Render the main dashboard content.
    """
    components.metric_cards(s0=s0, percentile_band=percentile_band, horizon_days=horizon_days)
    st.markdown(f"Estimated daily drift: **{mu:.5f}**, volatility: **{sigma:.5f}**")

    col1, col2 = st.columns(2)
    with col1:
        plots.plot_price_paths(paths, n_sample_paths=settings.DEFAULT_PATH_SAMPLE_SIZE)
    with col2:
        plots.plot_terminal_distribution(terminal_prices, percentile_band)

    st.subheader("Probability Targets")
    components.probability_table(df_probs_filtered)

    st.subheader("Summary")
    components.render_summary_section(
        ticker=ticker,
        s0=s0,
        percentile_band=percentile_band,
        horizon_days=horizon_days,
        prob_threshold=prob_threshold,
        df_probs_filtered=df_probs_filtered,
        openai_api_key=openai_api_key,
    )
    components.explanation_box(
        ticker=ticker,
        s0=s0,
        percentile_band=percentile_band,
        prob_threshold=prob_threshold,
    )

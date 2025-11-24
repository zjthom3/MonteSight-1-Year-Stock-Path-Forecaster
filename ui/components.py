"""
Reusable Streamlit UI components for MonteSight.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.utils import format_currency


def metric_cards(
    s0: float, percentile_band: dict[str, float], horizon_days: int, currency: str = "USD"
) -> None:
    """
    Render key metrics for the forecast.

    Parameters
    ----------
    s0:
        Current price.
    percentile_band:
        Dictionary containing percentile values.
    horizon_days:
        Simulation horizon in days.
    currency:
        Currency code for formatting.
    """
    cols = st.columns(3)
    cols[0].metric("Current Price", format_currency(s0, currency))
    cols[1].metric("Median Forecast", format_currency(percentile_band.get("p50"), currency))
    band_text = f"{format_currency(percentile_band.get('p17'), currency)} - {format_currency(percentile_band.get('p83'), currency)}"
    cols[2].metric(f"~66% Range ({horizon_days}d)", band_text)


def probability_table(df_probs_filtered: pd.DataFrame) -> None:
    """
    Render the filtered probability table.

    Parameters
    ----------
    df_probs_filtered:
        DataFrame with price levels and hit probabilities.
    """
    if df_probs_filtered is None or df_probs_filtered.empty:
        st.info("No price levels meet the probability threshold. Try lowering the cutoff.")
        return

    df_display = df_probs_filtered.copy()
    df_display["probability_%"] = (df_display["prob_hit"] * 100).round(2)
    display_cols = ["price_level", "probability_%"]
    if "direction" in df_display.columns:
        display_cols.append("direction")
    df_display = df_display[display_cols]
    st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "price_level": "Price Level",
            "probability_%": "Hit Probability (%)",
            "direction": "Direction (up/down)",
        },
    )


def simulation_summary_box(percentile_band: dict[str, float], horizon_days: int) -> None:
    """
    Render a concise summary of the percentile band.

    Parameters
    ----------
    percentile_band:
        Dictionary with percentile values.
    horizon_days:
        Simulation horizon in days.
    """
    p17 = percentile_band.get("p17")
    p50 = percentile_band.get("p50")
    p83 = percentile_band.get("p83")
    st.success(
        f"Over the next {horizon_days} trading days, the median simulated price is {format_currency(p50)}. "
        f"About two-thirds of outcomes fall between {format_currency(p17)} and {format_currency(p83)}."
    )


def explanation_box(
    ticker: str,
    s0: float,
    percentile_band: dict[str, float],
    prob_threshold: float,
    currency: str = "USD",
) -> None:
    """
    Render a natural-language explanation of the forecast.

    Parameters
    ----------
    ticker:
        Ticker symbol.
    s0:
        Current price.
    percentile_band:
        Dictionary with percentile values.
    prob_threshold:
        Probability threshold used for filtering.
    currency:
        Currency code for formatting.
    """
    p17 = percentile_band.get("p17")
    p50 = percentile_band.get("p50")
    p83 = percentile_band.get("p83")
    st.markdown(
        f"""
**What this means for {ticker.upper()}**

- Starting price: {format_currency(s0, currency)}
- Median forecast: {format_currency(p50, currency)}
- Most likely range (~66% confidence): {format_currency(p17, currency)} to {format_currency(p83, currency)}
- Showing targets with ≥ {prob_threshold:.0%} probability of being hit or exceeded.
"""
    )

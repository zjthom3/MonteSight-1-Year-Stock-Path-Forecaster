"""
Reusable Streamlit UI components for MonteSight.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

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


def _build_summary_prompt(
    ticker: str,
    s0: float,
    percentile_band: dict[str, float],
    prob_threshold: float,
    df_probs_filtered: pd.DataFrame | None,
    horizon_days: int,
) -> str:
    """Construct a concise prompt for the AI summary."""
    top_targets = ""
    if df_probs_filtered is not None and not df_probs_filtered.empty:
        top = (
            df_probs_filtered.sort_values("prob_hit", ascending=False)
            .head(3)
            .assign(prob_pct=lambda df: (df["prob_hit"] * 100).round(1))
        )
        lines = [f"- {row.price_level:.2f} ({row.prob_pct}%, {row.get('direction', 'n/a')})" for _, row in top.iterrows()]
        top_targets = "\n".join(lines)

    return f"""
You are summarizing Monte Carlo simulation results for {ticker}.
Current price: {s0:.2f}.
Simulation horizon: {horizon_days} trading days.
Percentile band: p17={percentile_band.get('p17'):.2f}, p50={percentile_band.get('p50'):.2f}, p83={percentile_band.get('p83'):.2f}.
Targets shown have probability >= {prob_threshold:.0%}.
Top probability targets:
{top_targets if top_targets else '- none available -'}
Write 2-3 short sentences, plain English, no financial advice, highlight upside/downside and central range.
"""


def _resolve_openai_key(provided_key: str | None) -> str | None:
    """
    Resolve an OpenAI API key from user input or a .env file.
    """
    if provided_key:
        return provided_key
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def render_summary_section(
    ticker: str,
    s0: float,
    percentile_band: dict[str, float],
    horizon_days: int,
    prob_threshold: float,
    df_probs_filtered: pd.DataFrame | None,
    openai_api_key: str | None = None,
) -> None:
    """
    Render either an AI-generated summary (if API key is available) or the default summary.
    """
    resolved_key = _resolve_openai_key(openai_api_key)
    if not resolved_key:
        simulation_summary_box(percentile_band, horizon_days)
        return

    try:
        from openai import OpenAI

        client = OpenAI(api_key=resolved_key)
        prompt = _build_summary_prompt(
            ticker=ticker,
            s0=s0,
            percentile_band=percentile_band,
            prob_threshold=prob_threshold,
            df_probs_filtered=df_probs_filtered,
            horizon_days=horizon_days,
        )
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You summarize Monte Carlo stock simulations succinctly for investors. Avoid financial advice."},
                {"role": "user", "content": prompt},
            ],
        )
        text = response.choices[0].message.content or "AI summary unavailable."
        st.markdown(text)
    except Exception:
        st.warning("AI summary unavailable; showing baseline summary instead.")
        simulation_summary_box(percentile_band, horizon_days)


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


if __name__ == "__main__":
    """
    Quick manual check to exercise summary generation without launching Streamlit.
    Run: python ui/components.py
    """
    sample_percentile_band = {"p17": 90.0, "p50": 100.0, "p83": 110.0}
    sample_probs = pd.DataFrame(
        {"price_level": [95.0, 120.0], "prob_hit": [0.7, 0.4], "direction": ["up", "up"]}
    )

    key = _resolve_openai_key(None)
    print(f"Resolved OPENAI_API_KEY present: {bool(key)}")
    prompt = _build_summary_prompt(
        ticker="TEST",
        s0=100.0,
        percentile_band=sample_percentile_band,
        prob_threshold=0.66,
        df_probs_filtered=sample_probs,
        horizon_days=10,
    )
    print("---- Prompt Preview ----")
    print(prompt)

    if not key:
        print("No API key found. Set OPENAI_API_KEY in .env or environment to exercise the call.")
    else:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            resp = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You summarize Monte Carlo stock simulations succinctly for investors. Avoid financial advice."},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=180,
            )
            print("---- OpenAI Response ----")
            print(resp.choices[0].message.content or "[empty response]")
        except Exception as exc:  # pragma: no cover - manual diagnostic
            print(f"OpenAI call failed: {exc}")

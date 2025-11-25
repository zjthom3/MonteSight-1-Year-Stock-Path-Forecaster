"""
Reusable Streamlit UI components for MonteSight.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents import Agent, Runner, WebSearchTool
from core.analytics import summarize_scenario_percentiles
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


def _scenario_label(key: str, config: dict[str, float] | None) -> str:
    """Resolve a readable scenario label."""
    if config and config.get("label"):
        return str(config["label"])
    return key.title()


def _format_top_targets(df_probs_filtered: pd.DataFrame | None) -> str:
    if df_probs_filtered is None or df_probs_filtered.empty:
        return "No price levels meet the probability cutoff."
    top = (
        df_probs_filtered.sort_values("prob_hit", ascending=False)
        .head(3)
        .assign(prob_pct=lambda df: (df["prob_hit"] * 100).round(1))
    )
    lines = [
        f"- {row.price_level:.2f} ({row.prob_pct}%, {row.get('direction', 'n/a')})"
        for _, row in top.iterrows()
    ]
    return "\n".join(lines)


def scenario_text_summary(
    scenario_key: str,
    config: dict[str, float],
    percentile_band: dict[str, float],
    prob_threshold: float,
    df_probs_filtered: pd.DataFrame | None,
    s0: float,
    horizon_days: int,
) -> None:
    """
    Render a compact narrative for a scenario using deterministic logic.
    """
    mu_delta = (config.get("mu_multiplier", 1.0) - 1.0) * 100
    sigma_delta = (config.get("sigma_multiplier", 1.0) - 1.0) * 100
    p17 = percentile_band.get("p17")
    p50 = percentile_band.get("p50")
    p83 = percentile_band.get("p83")

    tone = "upside" if mu_delta >= 0 else "downside"
    risk_note = (
        "Key risks include any reversal of the recent positive tone and unexpected volatility spikes."
        if mu_delta >= 0
        else "Upside surprises or rapid volatility compression could invalidate this downside skew."
    )

    top_targets = _format_top_targets(df_probs_filtered)
    label = _scenario_label(scenario_key, config)
    paragraphs = [
        f"**{label} assumptions:** Drift shift {mu_delta:+.1f}% vs base, vol shift {sigma_delta:+.1f}% vs base. {config.get('description', '').strip()}",
        f"**Expected range:** Median outcome {format_currency(p50)} over {horizon_days} trading days with ~66% of paths between {format_currency(p17)} and {format_currency(p83)} (starting from {format_currency(s0)}). Targets ≥ {prob_threshold:.0%}:\n{top_targets}",
        f"**Risks/uncertainty:** {risk_note}",
    ]
    st.markdown("\n\n".join(paragraphs))


def render_scenario_comparison(
    scenario_results: dict,
    horizon_days: int,
    prob_threshold: float,
    sample_paths: int,
    s0: float,
) -> None:
    """
    Render bull/base/bear scenario tabs with charts, tables, and summaries.
    """
    from ui import plots  # local import to avoid circular dependency

    if not scenario_results:
        st.info("Scenarios are enabled, but no scenario results are available.")
        return

    st.subheader("Scenario Analysis")
    st.caption("Bull/base/bear cases adjust drift and volatility using historical estimates plus optional external signals.")

    summary_df = summarize_scenario_percentiles(scenario_results)
    if not summary_df.empty:
        summary_df = summary_df.copy()
        summary_df["scenario"] = summary_df["scenario"].apply(
            lambda key: _scenario_label(key, scenario_results.get(key, {}).get("config"))
        )
        summary_df = summary_df[["scenario", "p17", "p50", "p83", "median_delta"]]
        summary_df = summary_df.rename(columns={"median_delta": "median_delta_vs_base"})
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "scenario": "Scenario",
                "p17": "p17",
                "p50": "p50",
                "p83": "p83",
                "median_delta_vs_base": "Median Δ vs Base",
            },
        )

    ordered_keys = [k for k in ["bull", "base", "bear"] if k in scenario_results] or list(
        scenario_results.keys()
    )
    tabs = st.tabs([_scenario_label(k, scenario_results[k].get("config")) for k in ordered_keys])

    for tab, key in zip(tabs, ordered_keys):
        result = scenario_results[key]
        config = result.get("config", {})
        with tab:
            metric_cards(
                s0=s0,
                percentile_band=result.get("percentile_band", {}),
                horizon_days=horizon_days,
            )
            mu_delta = (config.get("mu_multiplier", 1.0) - 1.0) * 100
            sigma_delta = (config.get("sigma_multiplier", 1.0) - 1.0) * 100
            st.caption(
                f"Drift adj: {mu_delta:+.1f}% | Vol adj: {sigma_delta:+.1f}% | {config.get('description', '').strip()}"
            )

            col1, col2 = st.columns(2)
            with col1:
                plots.plot_price_paths(
                    result["paths"],
                    n_sample_paths=min(sample_paths, result["paths"].shape[1]),
                    key=f"{key}-paths",
                )
            with col2:
                plots.plot_terminal_distribution(
                    result["terminal_prices"],
                    result.get("percentile_band", {}),
                    key=f"{key}-terminal",
                )

            st.caption("Narratives and comparisons are consolidated in the AI report below.")


def _build_scenario_report_prompt(
    ticker: str,
    scenario_results: dict,
    horizon_days: int,
    external_signals: dict | None,
) -> str:
    """
    Construct a prompt summarizing bull/base/bear scenario outcomes.
    """
    scenario_lines = []
    for key, result in scenario_results.items():
        band = result.get("percentile_band", {})
        cfg = result.get("config", {})
        label = _scenario_label(key, cfg)
        mu_mult = cfg.get("mu_multiplier", 1.0)
        sigma_mult = cfg.get("sigma_multiplier", 1.0)
        scenario_lines.append(
            f"- {label}: median={band.get('p50'):.2f}, band={band.get('p17'):.2f}-{band.get('p83'):.2f}, "
            f"drift adj={mu_mult:+.2f}x, vol adj={sigma_mult:+.2f}x, desc={cfg.get('description', '')}"
        )

    signals_text = "No external signals applied."
    if external_signals:
        notes = "; ".join(external_signals.get("notes", [])) or "external signals provided"
        summaries = external_signals.get("summaries", {})
        signals_text = (
            f"External signals applied: sentiment_score={external_signals.get('sentiment_score', 0.0)}, "
            f"volatility_bias={external_signals.get('volatility_bias', 0.0)}; notes: {notes}. "
            f"Signal snapshots — news: {summaries.get('news', 'n/a')}; "
            f"filings: {summaries.get('filings', 'n/a')}; fundamentals: {summaries.get('fundamentals', 'n/a')}."
        )

    return (
        f"Summarize MonteSight bull/base/bear Monte Carlo scenarios for {ticker} over {horizon_days} trading days.\n"
        f"Provide a concise, markdown-formatted report (no financial advice) highlighting:\n"
        f"1) relative outlook and ranges\n"
        f"2) key differences between scenarios\n"
        f"3) how external signals (news/filings/fundamentals) influenced drift/volatility if applicable.\n"
        f"4) A recommendation on which scenario (bull/base/bear) to follow based on the evidence above; keep it cautious and rationale-driven.\n"
        f"Scenario snapshots:\n" + "\n".join(scenario_lines) + "\n"
        f"{signals_text}\n"
        "Keep it to 3-4 bullet points plus a short lead sentence."
    )


def render_ai_scenario_report(
    ticker: str,
    scenario_results: dict,
    horizon_days: int,
    external_signals: dict | None,
    openai_api_key: str | None,
) -> None:
    """
    Generate an AI report that consolidates bull/base/bear scenarios (with external signals if provided).
    """
    if not scenario_results:
        return

    resolved_key = _resolve_openai_key(openai_api_key)
    if not resolved_key:
        st.info("Add an OpenAI API key to generate an AI scenario report.")
        return

    prompt = _build_scenario_report_prompt(
        ticker=ticker,
        scenario_results=scenario_results,
        horizon_days=horizon_days,
        external_signals=external_signals,
    )

    try:
        os.environ["OPENAI_API_KEY"] = resolved_key
        agent = Agent(
            name="MonteSight Scenario Reporter",
            model="gpt-4o-mini",
            instructions="You are an analyst summarizing scenario-based Monte Carlo outputs. Avoid financial advice.",
            tools=[WebSearchTool()],
        )
        text = ""
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                run_result = Runner.run_sync(agent, prompt)
                final_result = getattr(run_result, "final_output", None)
                if isinstance(final_result, str):
                    text = final_result
                elif hasattr(run_result, "output_text"):
                    text = getattr(run_result, "output_text") or ""
                elif hasattr(run_result, "output"):
                    text = str(run_result.output)
                if text:
                    break
            except Exception as exc:  # pragma: no cover - defensive retry
                last_exc = exc
                text = ""
                if attempt < 2:
                    continue
        if not text and last_exc:
            raise last_exc
        text = text or "AI scenario report unavailable."
        st.markdown("### AI Scenario Report")
        st.markdown(text)
    except Exception as exc:  # pragma: no cover - UI feedback only
        st.warning(f"AI scenario report unavailable: {exc}")


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
    print("Agents SDK diagnostic not implemented for standalone run.")

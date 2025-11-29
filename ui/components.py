"""
Reusable Streamlit UI components for MonteSight.
"""

from __future__ import annotations

import os
import time

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
    """
    cols = st.columns(3)
    cols[0].metric("Current Price", format_currency(s0, currency))
    cols[1].metric("Median Forecast", format_currency(percentile_band.get("p50"), currency))
    band_text = f"{format_currency(percentile_band.get('p17'), currency)} - {format_currency(percentile_band.get('p83'), currency)}"
    cols[2].metric(f"~66% Range ({horizon_days}d)", band_text)


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


def _scenario_label(key: str, config: dict[str, float] | None) -> str:
    """Resolve a readable scenario label."""
    if config and config.get("label"):
        return str(config["label"])
    return key.title()


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

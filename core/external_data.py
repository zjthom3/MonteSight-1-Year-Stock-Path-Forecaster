"""
Optional external data ingestion using OpenAI personas to enrich scenarios.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, Dict

from dotenv import load_dotenv

try:  # Allow graceful degradation if agents package is absent
    from agents import Agent, Runner, WebSearchTool
except ImportError:  # pragma: no cover - environment guard
    Agent = None
    Runner = None
    WebSearchTool = None


def _resolve_openai_key(provided_key: str | None = None) -> str | None:
    """Resolve OpenAI API key from argument or .env/environment."""
    if provided_key:
        return provided_key
    load_dotenv()
    return os.getenv("OPENAI_API_KEY")


def _run_agent_task(system_prompt: str, user_prompt: str, api_key: str) -> str:
    """
    Execute an Agents SDK run with web search enabled. Falls back to empty string on error.
    """
    if Agent is None or Runner is None or WebSearchTool is None:
        print("Agents SDK not available; skipping agent task.")
        return ""

    os.environ["OPENAI_API_KEY"] = api_key
    agent = Agent(
        name="MonteSight External Signals Agent",
        model="gpt-4o-mini",
        instructions=system_prompt,
        tools=[WebSearchTool()],
    )

    for attempt in range(3):
        try:
            run_result = Runner.run_sync(agent, user_prompt)
            final_result = getattr(run_result, "final_output", None)
            if isinstance(final_result, str):
                return final_result
            if hasattr(run_result, "output_text"):
                return getattr(run_result, "output_text") or ""
            if hasattr(run_result, "output") and run_result.output:
                return str(run_result.output)
            return ""
        except Exception as exc:  # pragma: no cover - defensive retry
            if attempt == 2:
                print(f"Agent run failed after retries: {exc}")
                return ""
            time.sleep(1 + attempt)


def _safe_signal_payload(content: str, default_score: float = 0.0) -> Dict[str, Any]:
    """
    Build a defensive signal payload from LLM output.
    """
    return {
        "summary": content.strip() if content else "No signals found.",
        "sentiment_score": default_score,
        "volatility_bias": default_score,
    }


def fetch_news_signals(ticker: str, openai_api_key: str | None = None) -> Dict[str, Any]:
    """
    Use an OpenAI 'News Desk Analyst' persona to summarize recent news tone.
    """
    key = _resolve_openai_key(openai_api_key)
    if not key:
        return {}
    system_prompt = (
        "You are a cautious News Desk Analyst. You do NOT invent headlines or numbers. "
        "If you lack data, reply 'Insufficient news data; treat as neutral.' "
        "Otherwise, give a 2-3 bullet summary of recent tone (bullish/bearish/neutral) "
        "and a rough sentiment score in [-1,1]. Avoid dates and tickers beyond the one provided."
    )
    user_prompt = f"Ticker: {ticker}. Summarize news tone."
    content = _run_agent_task(system_prompt, user_prompt, key)
    print("News signals content:", content)
    return _safe_signal_payload(content)


def fetch_filings_signals(ticker: str, openai_api_key: str | None = None) -> Dict[str, Any]:
    """
    Use an OpenAI 'Filing Forensics' persona to summarize regulatory/filing tone.
    """
    key = _resolve_openai_key(openai_api_key)
    if not key:
        return {}
    system_prompt = (
        "You are a Filing Forensics analyst. Do NOT fabricate filings. "
        "If unsure, say 'Insufficient filing data; treat as neutral.' "
        "Otherwise, provide 2 bullets on tone (optimistic/cautious/risk factors) and "
        "a rough sentiment score in [-1,1]. No dates or speculative claims."
    )
    user_prompt = f"Ticker: {ticker}. Summarize recent filing tone."
    content = _run_agent_task(system_prompt, user_prompt, key)
    return _safe_signal_payload(content)


def fetch_fundamentals(ticker: str, openai_api_key: str | None = None) -> Dict[str, Any]:
    """
    Use an OpenAI 'Fundamentals Scout' persona to outline high-level fundamentals.
    """
    key = _resolve_openai_key(openai_api_key)
    if not key:
        return {}
    system_prompt = (
        "You are a Fundamentals Scout. Do NOT invent financials. "
        "If you lack data, answer 'Insufficient fundamentals; treat as neutral.' "
        "Otherwise, provide 2 bullets on valuation/quality signals and a "
        "neutral-leaning sentiment score in [-1,1] if confident."
    )
    user_prompt = f"Ticker: {ticker}. Provide a cautious fundamentals snapshot."
    content = _run_agent_task(system_prompt, user_prompt, key)
    return _safe_signal_payload(content)


def aggregate_external_signals(
    ticker: str, openai_api_key: str | None = None
) -> Dict[str, Any]:
    """
    Aggregate external signals into a normalized structure for scenario biasing and reporting.

    Returns
    -------
    dict
        Keys:
        - sentiment_score: float in [-1, 1] summarizing bullish/bearish tone.
        - volatility_bias: float in [-1, 1] indicating expected vol expansion.
        - notes: list[str] describing signal provenance.
        - summaries: dict with 'news', 'filings', 'fundamentals' text.
    """
    key = _resolve_openai_key(openai_api_key)
    if not key:
        return {}

    # Run agents concurrently to reduce latency.
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            "news": executor.submit(fetch_news_signals, ticker, key),
            "filings": executor.submit(fetch_filings_signals, ticker, key),
            "fundamentals": executor.submit(fetch_fundamentals, ticker, key),
        }
    news = futures["news"].result()
    filings = futures["filings"].result()
    fundamentals = futures["fundamentals"].result()

    sources = {
        "news": news,
        "filings": filings,
        "fundamentals": fundamentals,
    }

    # Simple aggregation: average non-zero signals; otherwise neutral.
    sentiments = [v.get("sentiment_score", 0.0) for v in sources.values() if v]
    vols = [v.get("volatility_bias", 0.0) for v in sources.values() if v]
    sentiment_score = float(sum(sentiments) / len(sentiments)) if sentiments else 0.0
    volatility_bias = float(sum(vols) / len(vols)) if vols else 0.0

    notes = []
    summaries = {}
    for name, payload in sources.items():
        if payload:
            notes.append(f"{name} signals incorporated")
            summaries[name] = payload.get("summary", "")

    return {
        "sentiment_score": sentiment_score,
        "volatility_bias": volatility_bias,
        "notes": notes,
        "summaries": summaries,
    }


if __name__ == "__main__":
    """
    Manual diagnostic for external signal agents.
    Run: python core/external_data.py
    Ensure OPENAI_API_KEY is set in environment or .env.
    """
    import json

    if Agent is None:
        print("Agents SDK not installed. Please install `openai-agents` or ensure it is on PYTHONPATH.")
        raise SystemExit(1)

    test_ticker = "AAPL"
    key_present = bool(_resolve_openai_key())
    print(f"Resolved OPENAI_API_KEY present: {key_present}")

    if not key_present:
        print("No API key found. Set OPENAI_API_KEY to exercise agent calls.")
    else:
        news = fetch_news_signals(test_ticker)
        filings = fetch_filings_signals(test_ticker)
        fundamentals = fetch_fundamentals(test_ticker)
        agg = aggregate_external_signals(test_ticker)

        print("\n--- News Signals ---")
        print(json.dumps(news, indent=2))
        print("\n--- Filings Signals ---")
        print(json.dumps(filings, indent=2))
        print("\n--- Fundamentals Signals ---")
        print(json.dumps(fundamentals, indent=2))
        print("\n--- Aggregated Signals ---")
        print(json.dumps(agg, indent=2))

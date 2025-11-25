"""
Scenario configuration and execution utilities for MonteSight.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from config import settings
from core.analytics import (
    compute_path_hit_probability_for_levels,
    filter_by_probability,
    generate_price_grid,
    get_percentile_band,
    get_terminal_prices,
)
from core.simulation import simulate_price_paths


def _derive_signal_bias(external_signals: dict[str, Any] | None) -> tuple[float, float]:
    """
    Convert optional external signals into mu/sigma bias terms.

    Returns
    -------
    tuple[float, float]
        (mu_bias, sigma_bias) multipliers where 0 is neutral. Values are clamped
        to keep adjustments small unless the caller explicitly scales.
    """
    if not external_signals:
        return 0.0, 0.0
    sentiment = float(external_signals.get("sentiment_score", 0.0))
    volatility_bias = float(external_signals.get("volatility_bias", 0.0))
    # Keep biases modest: sentiment nudges mu by up to +/-5%, volatility bias by +/-10%.
    mu_bias = float(np.clip(sentiment * 0.05, -0.05, 0.05))
    sigma_bias = float(np.clip(volatility_bias * 0.10, -0.10, 0.10))
    return mu_bias, sigma_bias


def build_scenario_configs(
    sim_config: dict, external_signals: dict | None = None
) -> Dict[str, dict]:
    """
    Build bull/base/bear scenario configurations by adjusting drift and volatility.

    Parameters
    ----------
    sim_config:
        Baseline simulation configuration containing at least ``mu`` and ``sigma``.
        Optional keys:
            - scenario_aggressiveness: 0-100 scale controlling deviation size.
            - lookback_period: string label for descriptions.
    external_signals:
        Optional dict of external signals (e.g., sentiment, filing tone). When
        provided, these nudges are applied symmetrically to bull/bear drift and
        volatility.

    Returns
    -------
    dict
        Mapping of scenario key -> scenario config dict. Each scenario contains:
        ``name``, ``mu``, ``sigma``, ``label``, ``description``, and multipliers.
    """
    base_mu = float(sim_config.get("mu"))
    base_sigma = float(sim_config.get("sigma"))
    lookback = sim_config.get("lookback_period", "historical window")

    aggressiveness_raw = sim_config.get("scenario_aggressiveness", 40)
    aggressiveness = float(np.clip(aggressiveness_raw, 0, 100)) / 100.0

    mu_bias, sigma_bias = _derive_signal_bias(external_signals)

    # Deviation magnitudes: higher aggressiveness widens the spread.
    mu_shift_pct = 0.15 + 0.35 * aggressiveness  # 15% to 50% swing
    sigma_shift_pct = 0.10 + 0.20 * aggressiveness  # 10% to 30% swing

    bull_mu_multiplier = 1.0 + mu_shift_pct + mu_bias
    bear_mu_multiplier = max(0.0, 1.0 - mu_shift_pct + mu_bias * 0.5)

    bull_sigma_multiplier = max(0.05, 1.0 - sigma_shift_pct + sigma_bias)
    bear_sigma_multiplier = 1.0 + sigma_shift_pct + sigma_bias

    base_config = {
        "name": "Base",
        "label": "Base case",
        "mu": base_mu,
        "sigma": base_sigma,
        "mu_multiplier": 1.0,
        "sigma_multiplier": 1.0,
        "description": f"Historical drift/vol from {lookback}.",
    }
    bull_config = {
        "name": "Bull",
        "label": "Bull case",
        "mu": base_mu * bull_mu_multiplier,
        "sigma": max(base_sigma * bull_sigma_multiplier, 1e-8),
        "mu_multiplier": bull_mu_multiplier,
        "sigma_multiplier": bull_sigma_multiplier,
        "description": "Optimistic scenario: positive drift uplift and volatility trimmed.",
    }
    bear_config = {
        "name": "Bear",
        "label": "Bear case",
        "mu": base_mu * bear_mu_multiplier,
        "sigma": max(base_sigma * bear_sigma_multiplier, 1e-8),
        "mu_multiplier": bear_mu_multiplier,
        "sigma_multiplier": bear_sigma_multiplier,
        "description": "Defensive scenario: drift pulled lower with fatter volatility tails.",
    }

    return {"bull": bull_config, "base": base_config, "bear": bear_config}


def run_scenarios(
    s0: float,
    base_sim_config: dict,
    scenario_configs: dict,
    days: int,
    n_sims: int,
    seed: int | None,
) -> Dict[str, dict]:
    """
    Execute simulations for each scenario and compute analytics artifacts.

    Parameters
    ----------
    s0:
        Starting price.
    base_sim_config:
        Baseline simulation configuration dict (ticker, mu, sigma, prob_threshold, etc).
    scenario_configs:
        Output of :func:`build_scenario_configs`.
    days:
        Trading days to simulate.
    n_sims:
        Number of Monte Carlo simulations per scenario.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    dict
        Mapping of scenario key -> results dict containing paths, terminal prices,
        probability tables (raw + filtered), and percentile bands.
    """
    prob_threshold = float(base_sim_config.get("prob_threshold", settings.DEFAULT_PROB_THRESHOLD))
    grid_points = int(base_sim_config.get("price_grid_points", settings.DEFAULT_PRICE_GRID_POINTS))

    results: Dict[str, dict] = {}
    for scenario_key, cfg in scenario_configs.items():
        paths = simulate_price_paths(
            s0=s0,
            drift=cfg["mu"],
            vol=cfg["sigma"],
            days=days,
            n_sims=n_sims,
            seed=seed,
        )
        terminal_prices = get_terminal_prices(paths)
        price_levels = generate_price_grid(terminal_prices, n_points=grid_points, paths=paths)
        df_probs = compute_path_hit_probability_for_levels(
            paths, price_levels, current_price=s0
        )
        df_probs_filtered = filter_by_probability(df_probs, prob_threshold)
        percentile_band = get_percentile_band(terminal_prices)

        results[scenario_key] = {
            "config": cfg,
            "paths": paths,
            "terminal_prices": terminal_prices,
            "price_levels": price_levels,
            "df_probs": df_probs,
            "df_probs_filtered": df_probs_filtered,
            "percentile_band": percentile_band,
        }

    return results

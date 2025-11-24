"""
Simulation utilities for estimating parameters and generating price paths.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.utils import set_random_seed


def estimate_drift_vol(log_returns: pd.Series) -> tuple[float, float]:
    """
    Estimate daily drift and volatility from log returns.

    Parameters
    ----------
    log_returns:
        Series of historical log returns.

    Returns
    -------
    tuple[float, float]
        Drift (mu) and volatility (sigma).

    Raises
    ------
    ValueError
        If not enough data is provided.
    """
    if log_returns is None or len(log_returns) < 2:
        raise ValueError("At least two log return observations are required.")
    mu = float(log_returns.mean())
    sigma = float(log_returns.std())
    return mu, sigma


def simulate_price_paths(
    s0: float,
    drift: float,
    vol: float,
    days: int = 252,
    n_sims: int = 10_000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM-based Monte Carlo price paths.

    Parameters
    ----------
    s0:
        Starting price.
    drift:
        Expected daily drift (mu).
    vol:
        Daily volatility (sigma).
    days:
        Number of trading days to simulate.
    n_sims:
        Number of Monte Carlo simulations.
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Simulated paths of shape ``(days + 1, n_sims)``.

    Raises
    ------
    ValueError
        If parameters are invalid.
    """
    if s0 <= 0:
        raise ValueError("Starting price s0 must be positive.")
    if days <= 0 or n_sims <= 0:
        raise ValueError("days and n_sims must be positive.")
    if vol < 0:
        raise ValueError("Volatility cannot be negative.")

    set_random_seed(seed)

    dt = 1.0
    shocks = np.random.normal(size=(days, n_sims))
    increments = (drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
    log_price_paths = np.vstack(
        (np.zeros((1, n_sims)), np.cumsum(increments, axis=0))
    )
    paths = s0 * np.exp(log_price_paths)
    return paths

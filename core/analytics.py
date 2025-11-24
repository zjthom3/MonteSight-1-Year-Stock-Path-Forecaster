"""
Analytics helpers for transforming simulated paths into insights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_terminal_prices(paths: np.ndarray) -> np.ndarray:
    """
    Extract terminal prices from simulated paths.

    Parameters
    ----------
    paths:
        Simulated price paths of shape ``(days + 1, n_sims)``.

    Returns
    -------
    np.ndarray
        Terminal prices of shape ``(n_sims,)``.
    """
    if paths is None or paths.size == 0:
        raise ValueError("Paths array is empty.")
    return paths[-1, :]


def generate_price_grid(
    terminal_prices: np.ndarray, n_points: int = 50, paths: np.ndarray | None = None
) -> np.ndarray:
    """
    Generate an evenly spaced grid of price levels.

    Parameters
    ----------
    terminal_prices:
        Array of terminal prices.
    n_points:
        Number of grid points to generate.
    paths:
        Optional full path matrix to derive min/max from intrahorizon moves.

    Parameters
    ----------
    terminal_prices:
        Array of terminal prices.
    n_points:
        Number of grid points to generate.
    paths:
        Optional full path matrix to derive min/max from intrahorizon moves.

    Returns
    -------
    np.ndarray
        Sorted price level grid.
    """
    if terminal_prices is None or terminal_prices.size == 0:
        raise ValueError("Terminal prices array is empty.")
    if n_points <= 1:
        raise ValueError("n_points must be greater than 1.")

    if paths is not None:
        min_price = float(np.min(paths))
        max_price = float(np.max(paths))
    else:
        min_price = float(np.min(terminal_prices))
        max_price = float(np.max(terminal_prices))

    if np.isclose(min_price, max_price):
        return np.full(shape=n_points, fill_value=min_price, dtype=float)
    return np.linspace(min_price, max_price, n_points)


def compute_probability_for_levels(
    terminal_prices: np.ndarray, price_levels: np.ndarray
) -> pd.DataFrame:
    """
    Compute probability of terminal prices meeting or exceeding each level.

    Parameters
    ----------
    terminal_prices:
        Array of terminal prices.
    price_levels:
        Array of price levels to evaluate.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``price_level`` and ``prob_hit``.
    """
    if terminal_prices is None or price_levels is None:
        raise ValueError("Inputs cannot be None.")
    if terminal_prices.size == 0 or price_levels.size == 0:
        raise ValueError("Inputs must be non-empty.")

    probs = (terminal_prices[:, None] >= price_levels[None, :]).mean(axis=0)
    df = pd.DataFrame({"price_level": price_levels, "prob_hit": probs})
    df = df.sort_values("price_level", ascending=True).reset_index(drop=True)
    return df


def compute_path_hit_probability_for_levels(
    paths: np.ndarray, price_levels: np.ndarray, current_price: float
) -> pd.DataFrame:
    """
    Compute probability of hitting each price level at any time during the path.

    For targets above the current price, this checks whether a path's maximum
    reaches or exceeds the level. For targets below, it checks whether the path's
    minimum falls to or below the level.

    Parameters
    ----------
    paths:
        Simulated price paths of shape ``(days + 1, n_sims)``.
    price_levels:
        Array of price levels to evaluate.
    current_price:
        Starting price at time 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``price_level``, ``prob_hit``, and ``direction``.
    """
    if paths is None or price_levels is None:
        raise ValueError("Inputs cannot be None.")
    if paths.size == 0 or price_levels.size == 0:
        raise ValueError("Inputs must be non-empty.")

    max_per_path = paths.max(axis=0)
    min_per_path = paths.min(axis=0)

    probs = []
    directions = []
    for level in price_levels:
        if level >= current_price:
            hit = max_per_path >= level
            directions.append("up")
        else:
            hit = min_per_path <= level
            directions.append("down")
        probs.append(hit.mean())

    df = pd.DataFrame(
        {"price_level": price_levels, "prob_hit": np.array(probs), "direction": directions}
    )
    return df.sort_values("price_level", ascending=True).reset_index(drop=True)


def filter_by_probability(
    df_probs: pd.DataFrame, threshold: float = 0.66
) -> pd.DataFrame:
    """
    Filter a probability table to entries meeting the threshold.

    Parameters
    ----------
    df_probs:
        DataFrame containing ``price_level`` and ``prob_hit`` columns.
    threshold:
        Minimum probability to include.

    Returns
    -------
    pd.DataFrame
        Filtered and sorted probability table.
    """
    if df_probs is None or df_probs.empty:
        return pd.DataFrame(columns=["price_level", "prob_hit"])
    filtered = df_probs[df_probs["prob_hit"] >= threshold].copy()
    return filtered.sort_values("price_level", ascending=True).reset_index(drop=True)


def get_percentile_band(
    terminal_prices: np.ndarray, lower: float = 0.17, upper: float = 0.83
) -> dict[str, float]:
    """
    Compute percentile band for terminal prices.

    Parameters
    ----------
    terminal_prices:
        Array of terminal prices.
    lower:
        Lower percentile as a fraction.
    upper:
        Upper percentile as a fraction.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``p17``, ``p50``, and ``p83``.
    """
    if terminal_prices is None or terminal_prices.size == 0:
        raise ValueError("Terminal prices array is empty.")

    p17 = float(np.percentile(terminal_prices, lower * 100))
    p50 = float(np.percentile(terminal_prices, 50))
    p83 = float(np.percentile(terminal_prices, upper * 100))
    return {"p17": p17, "p50": p50, "p83": p83}

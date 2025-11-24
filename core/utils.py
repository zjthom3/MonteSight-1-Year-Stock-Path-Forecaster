"""
Utility helpers shared across MonteSight modules.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def set_random_seed(seed: int | None) -> None:
    """
    Set the global numpy random seed if provided.

    Parameters
    ----------
    seed:
        Seed value to use. If ``None`` no change is made.
    """
    if seed is None:
        return
    np.random.seed(seed)


def format_currency(value: float, currency: str = "USD") -> str:
    """
    Format a numeric value as a currency-like string.

    Parameters
    ----------
    value:
        Numeric value to format.
    currency:
        Currency code prefix to display.

    Returns
    -------
    str
        Formatted currency string, e.g. ``"USD 1,234.56"``.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{currency} --"
    return f"{currency} {value:,.2f}"


def safe_float(val: Any, default: float | None = None) -> float | None:
    """
    Safely cast a value to ``float`` returning ``default`` on failure.

    Parameters
    ----------
    val:
        Value to attempt to convert.
    default:
        Value to return if conversion fails.

    Returns
    -------
    float | None
        Converted float or the provided default.
    """
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

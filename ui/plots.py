"""
Plotting utilities for MonteSight using Plotly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def _sample_paths(paths: np.ndarray, n_sample_paths: int) -> np.ndarray:
    """Sample columns from the full path matrix for plotting clarity."""
    n_available = paths.shape[1]
    if n_sample_paths >= n_available:
        return paths
    indices = np.linspace(0, n_available - 1, n_sample_paths, dtype=int)
    return paths[:, indices]


def plot_price_paths(paths: np.ndarray, n_sample_paths: int = 50) -> None:
    """
    Plot a subset of simulated price paths over time.

    Parameters
    ----------
    paths:
        Simulated price paths of shape ``(days + 1, n_sims)``.
    n_sample_paths:
        Number of paths to display for readability.
    """
    sampled = _sample_paths(paths, n_sample_paths)
    days_axis = np.arange(sampled.shape[0])
    fig = go.Figure()
    for col in range(sampled.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=days_axis,
                y=sampled[:, col],
                mode="lines",
                line=dict(width=1, color="rgba(52, 152, 219, 0.35)"),
                showlegend=False,
            )
        )
    fig.update_layout(
        title="Simulated Price Paths",
        xaxis_title="Trading Days",
        yaxis_title="Price",
        template="plotly_white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_terminal_distribution(terminal_prices: np.ndarray, percentile_band: dict[str, float]) -> None:
    """
    Plot histogram of terminal prices with percentile overlays.

    Parameters
    ----------
    terminal_prices:
        Array of terminal prices.
    percentile_band:
        Dictionary containing ``p17``, ``p50``, and ``p83`` keys.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=terminal_prices,
            nbinsx=40,
            marker_color="rgba(39, 174, 96, 0.6)",
            name="Terminal Prices",
        )
    )

    for key, color in zip(["p17", "p50", "p83"], ["#e67e22", "#34495e", "#e67e22"]):
        fig.add_vline(
            x=percentile_band.get(key),
            line_width=2,
            line_dash="dash",
            line_color=color,
        )

    fig.update_layout(
        title="Terminal Price Distribution",
        xaxis_title="Price",
        yaxis_title="Frequency",
        bargap=0.1,
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

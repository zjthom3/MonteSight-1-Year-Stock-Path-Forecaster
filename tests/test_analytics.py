import numpy as np
import pandas as pd

from core.analytics import (
    compute_probability_for_levels,
    compute_path_hit_probability_for_levels,
    filter_by_probability,
    generate_price_grid,
    get_percentile_band,
    get_terminal_prices,
)


def test_terminal_and_grid_generation():
    paths = np.array([[100, 100], [110, 90], [120, 80]], dtype=float)
    terminal = get_terminal_prices(paths)
    assert terminal.shape == (2,)
    grid = generate_price_grid(terminal, n_points=5)
    assert grid[0] == terminal.min()
    assert grid[-1] == terminal.max()


def test_probability_monotonic_and_filtering():
    terminal_prices = np.array([100, 110, 120, 130], dtype=float)
    levels = np.linspace(90, 140, 6)
    df_probs = compute_probability_for_levels(terminal_prices, levels)
    filtered = filter_by_probability(df_probs, threshold=0.5)
    assert (filtered["prob_hit"] >= 0.5).all()


def test_percentile_band_ordering():
    terminal_prices = np.linspace(80, 120, 100)
    band = get_percentile_band(terminal_prices)
    assert band["p17"] <= band["p50"] <= band["p83"]


def test_path_hit_probability():
    paths = np.array(
        [
            [100, 100],
            [120, 80],   # one path goes up, one goes down
            [130, 70],
        ],
        dtype=float,
    )
    levels = np.array([75, 100, 125], dtype=float)
    df_hit = compute_path_hit_probability_for_levels(paths, levels, current_price=100)
    # 75 should only be hit by the down path
    prob_75 = df_hit[df_hit["price_level"] == 75]["prob_hit"].iloc[0]
    prob_125 = df_hit[df_hit["price_level"] == 125]["prob_hit"].iloc[0]
    assert prob_75 == 0.5
    assert prob_125 == 0.5

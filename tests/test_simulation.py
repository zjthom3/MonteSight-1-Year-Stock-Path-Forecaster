import numpy as np
import pandas as pd

from core.simulation import estimate_drift_vol, simulate_price_paths


def test_estimate_drift_vol_basic():
    log_returns = pd.Series([0.01, 0.02, 0.03])
    mu, sigma = estimate_drift_vol(log_returns)
    assert mu > 0
    assert sigma > 0


def test_simulate_price_paths_shape_and_seed():
    s0 = 100.0
    mu = 0.001
    sigma = 0.02
    days = 10
    n_sims = 5
    paths_first = simulate_price_paths(s0, mu, sigma, days=days, n_sims=n_sims, seed=123)
    paths_second = simulate_price_paths(s0, mu, sigma, days=days, n_sims=n_sims, seed=123)
    assert paths_first.shape == (days + 1, n_sims)
    assert np.allclose(paths_first[0, :], s0)
    assert np.allclose(paths_first, paths_second)

import numpy as np

from core import utils


def test_set_random_seed_reproducible():
    utils.set_random_seed(42)
    first = np.random.rand()
    utils.set_random_seed(42)
    second = np.random.rand()
    assert first == second


def test_format_currency_and_safe_float():
    assert utils.format_currency(1234.56) == "USD 1,234.56"
    assert utils.format_currency(None) == "USD --"
    assert utils.safe_float("10.5") == 10.5
    assert utils.safe_float("bad", default=1.0) == 1.0

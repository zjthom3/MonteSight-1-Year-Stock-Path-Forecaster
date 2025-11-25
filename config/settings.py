"""
Global configuration and default values for MonteSight.
"""

DEFAULT_TICKER: str = "NVDA"
DEFAULT_PERIOD: str = "5y"
DEFAULT_N_SIMS: int = 10_000
DEFAULT_HORIZON_DAYS: int = 252
DEFAULT_PROB_THRESHOLD: float = 0.66
DEFAULT_SEED: int | None = None

# UI-specific defaults
DEFAULT_PRICE_GRID_POINTS: int = 60
DEFAULT_PATH_SAMPLE_SIZE: int = 80

# Scenario defaults
DEFAULT_ENABLE_SCENARIOS: bool = True
DEFAULT_SCENARIO_AGGRESSIVENESS: int = 35  # 0-100 scale controlling mu/sigma deviation
DEFAULT_USE_EXTERNAL_SIGNALS: bool = False

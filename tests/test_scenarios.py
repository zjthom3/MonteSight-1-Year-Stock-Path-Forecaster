import pandas as pd

from core.scenarios import build_scenario_configs, run_scenarios


def test_build_scenario_configs_spreads_mu_and_sigma():
    sim_config = {"mu": 0.001, "sigma": 0.02, "scenario_aggressiveness": 50}
    scenarios = build_scenario_configs(sim_config)

    assert set(scenarios.keys()) == {"bull", "base", "bear"}
    assert scenarios["base"]["mu"] == sim_config["mu"]
    assert scenarios["bull"]["mu"] > sim_config["mu"]
    assert scenarios["bear"]["mu"] < sim_config["mu"]
    assert scenarios["bear"]["sigma"] >= sim_config["sigma"]


def test_run_scenarios_produces_results():
    base_config = {
        "mu": 0.001,
        "sigma": 0.02,
        "prob_threshold": 0.5,
        "price_grid_points": 10,
        "scenario_aggressiveness": 30,
    }
    scenarios = build_scenario_configs(base_config)
    results = run_scenarios(
        s0=100.0,
        base_sim_config=base_config,
        scenario_configs=scenarios,
        days=5,
        n_sims=200,
        seed=123,
    )

    assert set(results.keys()) == {"bull", "base", "bear"}
    for res in results.values():
        assert res["paths"].shape == (6, 200)
        assert res["terminal_prices"].shape == (200,)
        assert isinstance(res["df_probs_filtered"], pd.DataFrame)
        assert isinstance(res["percentile_band"], dict)

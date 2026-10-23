"""Seeded scenario families, held-out splits and the Task 2 contingency catalog."""

from __future__ import annotations

import pytest

from server.scenarios import (
    EVAL_SEED_START,
    MAX_TICKS,
    build_scenario,
    cascade_config_check,
    load_cascade_catalog,
)


def _fingerprint(scenario):
    engine = scenario["engine"]
    return (
        tuple(sorted((k, round(v, 6) if isinstance(v, float) else str(v)) for k, v in scenario["params"].items()
                     if not isinstance(v, (list, dict)))),
        round(engine.get_total_demand(), 4),
        tuple(round(n["setpoint_mw"], 4) for n in engine.nodes.values()),
    )


@pytest.mark.parametrize("task_id", sorted(MAX_TICKS))
def test_same_seed_reproduces_scenario(task_id):
    assert _fingerprint(build_scenario(task_id, 123)) == _fingerprint(build_scenario(task_id, 123))


@pytest.mark.parametrize("task_id", sorted(MAX_TICKS))
def test_seeds_produce_different_scenarios(task_id):
    prints = {_fingerprint(build_scenario(task_id, seed)) for seed in range(12)}
    assert len(prints) >= 8


@pytest.mark.parametrize("task_id", sorted(MAX_TICKS))
def test_split_label(task_id):
    assert build_scenario(task_id, 5)["split"] == "train"
    assert build_scenario(task_id, EVAL_SEED_START + 5)["split"] == "eval"


def test_attack_targets_and_timing_vary():
    targets = {build_scenario(3, s)["attack_config"]["target_node"] for s in range(40)}
    starts = {build_scenario(3, s)["attack_config"]["start_tick"] for s in range(40)}
    modes = {build_scenario(4, s)["attack_config"]["mode_hz"] for s in range(40)}
    assert len(targets) >= 4 and len(starts) >= 2 and len(modes) >= 5


def test_black_start_unit_varies_and_grid_is_dark():
    units = set()
    for s in range(20):
        sc = build_scenario(5, s)
        engine = sc["engine"]
        unit = sc["params"]["black_start_unit"]
        units.add(unit)
        energized = [nid for nid, n in engine.nodes.items() if n["energized"]]
        assert energized == [unit]
        assert all(e["status"] == "TRIPPED" for e in engine.edges.values())
    assert units == {"NODE_01", "NODE_13"}


def test_cascade_catalog_is_split_between_train_and_eval():
    catalog = load_cascade_catalog()
    assert len(catalog) >= 100
    split_at = int(len(catalog) * 0.8)
    train_keys = {(c["trip_line"], tuple(c["outages"]), c["load_factor"]) for c in catalog[:split_at]}
    eval_keys = {(c["trip_line"], tuple(c["outages"]), c["load_factor"]) for c in catalog[split_at:]}
    assert not train_keys & eval_keys


def test_cascade_configs_start_overloaded_but_secure_before_the_trip():
    catalog = load_cascade_catalog()
    for config in catalog[:: max(1, len(catalog) // 12)]:
        info = cascade_config_check(config)
        assert info["valid"], config
        assert 1.06 <= info["max_loading"] <= 1.35


def test_cascade_scenario_locks_out_damaged_lines():
    sc = build_scenario(2, 7)
    engine = sc["engine"]
    locked = [eid for eid, e in engine.edges.items() if e["out_of_service"]]
    assert sc["params"]["trip_line"] in locked
    assert set(sc["params"]["outages"]) <= set(locked)
    assert engine.get_max_loading() > 1.0


def test_duck_curve_forecast_matches_realised_demand():
    sc = build_scenario(1, 9)
    engine, schedule = sc["engine"], sc["schedule"]
    forecast = list(engine.demand_forecast)
    for k, expected in enumerate(forecast[:4], start=1):
        schedule.apply(engine, k)
        assert engine.get_total_demand() == pytest.approx(expected, abs=0.2)

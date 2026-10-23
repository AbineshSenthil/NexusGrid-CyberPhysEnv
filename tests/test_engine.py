"""Physics of the grid engine: DC power flow, frequency, protection and actions."""

from __future__ import annotations

import pytest

from server.grid_engine import GENERATOR_TYPES, GridEngine
from server.scenarios import build_base_topology


def _net_injection(engine: GridEngine, nid: str) -> float:
    n = engine.nodes[nid]
    return n["generation_mw"] if n["node_type"] in GENERATOR_TYPES else -n["consumption_mw"]


def _line_balance(engine: GridEngine, nid: str) -> float:
    total = 0.0
    for e in engine.edges.values():
        if e["status"] != "LIVE":
            continue
        if e["source"] == nid:
            total += e["flow_mw"]
        elif e["target"] == nid:
            total -= e["flow_mw"]
    return total


class TestPowerFlow:
    def test_base_case_is_balanced_and_nominal(self, engine):
        assert len(engine.get_stable_islands()) == 1
        assert engine.frequency_hz == pytest.approx(60.0, abs=1e-6)
        assert engine.get_total_generation() == pytest.approx(engine.get_total_load(), abs=1e-6)

    def test_kirchhoff_current_law_holds_at_every_bus(self, engine):
        for nid in engine.nodes:
            assert _line_balance(engine, nid) == pytest.approx(_net_injection(engine, nid), abs=1e-6)

    def test_kirchhoff_voltage_law_holds_on_every_line(self, engine):
        import math
        for e in engine.edges.values():
            if e["status"] != "LIVE":
                continue
            d_theta = math.radians(engine.nodes[e["source"]]["phase_angle_deg"]
                                   - engine.nodes[e["target"]]["phase_angle_deg"])
            assert e["flow_mw"] == pytest.approx(d_theta / e["reactance"] * engine.BASE_MVA, abs=1e-6)

    def test_opening_a_line_redistributes_flow(self, engine):
        before = dict((eid, e["flow_mw"]) for eid, e in engine.edges.items())
        engine.toggle_circuit_breaker("LINE_28", "OPEN")
        assert engine.edges["LINE_28"]["flow_mw"] == 0.0
        changed = [eid for eid, e in engine.edges.items() if abs(e["flow_mw"] - before[eid]) > 1.0]
        assert len(changed) > 3
        for nid in engine.nodes:
            assert _line_balance(engine, nid) == pytest.approx(_net_injection(engine, nid), abs=1e-6)

    def test_reactance_scales_inversely_with_rating(self, engine):
        assert engine.edges["LINE_28"]["reactance"] < engine.edges["LINE_27"]["reactance"]


class TestFrequency:
    def test_generation_deficit_lowers_frequency_through_stiffness(self, engine):
        engine.nodes["NODE_17"]["setpoint_mw"] -= 150.0
        engine.compute_power_flow()
        target = engine._island_target_hz[0]
        stiffness = engine._island_stiffness(engine.get_stable_islands()[0])
        assert target == pytest.approx(60.0 - 150.0 / stiffness, abs=1e-6)
        f = engine.compute_frequency(smooth=True)
        assert 60.0 > f > target  # inertia closes only part of the gap per tick

    def test_governors_keep_load_served_during_small_deficit(self, engine):
        load_before = engine.get_total_load()
        engine.dispatch_generation("NODE_17", -150.0)
        assert engine.get_total_load() == pytest.approx(load_before, abs=1e-6)

    def test_ufls_sheds_non_critical_load_below_threshold(self, engine):
        for nid in ("NODE_01", "NODE_13", "NODE_17", "NODE_09"):
            engine.nodes[nid]["setpoint_mw"] = 0.0
        events = []
        for _ in range(4):
            events += engine.advance_tick()
        assert any(e["type"] == "ufls" for e in events)
        shed_nodes = {nid for e in events if e["type"] == "ufls" for nid in e["nodes"]}
        assert shed_nodes and all(not engine.nodes[nid]["critical"] for nid in shed_nodes)

    def test_weak_island_collapses_outside_hard_limits(self, engine):
        # Isolate NODE_19 (load) with the NODE_20 battery: far too little stiffness for the load.
        engine.nodes["NODE_20"]["setpoint_mw"] = 10.0  # running, so the island stays energised when split
        for eid in ("LINE_23", "LINE_25", "LINE_36", "LINE_24", "LINE_26", "LINE_40"):
            engine.toggle_circuit_breaker(eid, "OPEN")
        assert len(engine.get_stable_islands()) == 2
        events = engine.advance_tick()
        assert any(e["type"] == "island_collapse" and "NODE_19" in e["nodes"] for e in events)
        assert not engine.nodes["NODE_19"]["energized"]


class TestProtection:
    def test_instant_trip_at_140_percent(self, engine):
        line = engine.edges["LINE_02"]
        line["capacity_mw"] = max(1.0, abs(line["flow_mw"]) / 1.5)
        events = engine.advance_tick()
        assert any(e["type"] == "protection_trip" and e["edge"] == "LINE_02" for e in events)
        assert line["status"] == "TRIPPED"

    def test_sustained_overload_trips_after_three_ticks(self, engine):
        line = engine.edges["LINE_02"]
        line["capacity_mw"] = abs(line["flow_mw"]) / 1.1
        assert engine.advance_tick() == [] or line["status"] == "LIVE"
        engine.advance_tick()
        assert line["status"] == "LIVE"
        events = engine.advance_tick()
        assert any(e["type"] == "protection_trip" and e["edge"] == "LINE_02" for e in events)

    def test_out_of_service_line_cannot_be_closed(self, engine):
        engine.set_out_of_service("LINE_28", "storm damage")
        result = engine.toggle_circuit_breaker("LINE_28", "CLOSED")
        assert result["success"] is False
        assert "out of service" in result["error"]
        assert engine.edges["LINE_28"]["status"] == "TRIPPED"


class TestActions:
    def test_dispatch_is_ramp_limited(self, engine):
        old = engine.nodes["NODE_17"]["setpoint_mw"]
        result = engine.dispatch_generation("NODE_17", 1000.0)
        assert result["success"] and result["ramp_limited"]
        assert engine.nodes["NODE_17"]["setpoint_mw"] == pytest.approx(old + 150.0)

    def test_dispatch_rejects_loads_and_unknown_nodes(self, engine):
        assert engine.dispatch_generation("NODE_03", 50)["success"] is False
        assert engine.dispatch_generation("NODE_99", 50)["success"] is False

    def test_shed_and_restore_load(self, engine):
        served = engine.nodes["NODE_07"]["consumption_mw"]
        engine.shed_load("NODE_07", 100.0)
        assert engine.nodes["NODE_07"]["consumption_mw"] == pytest.approx(served - 100.0, abs=1e-6)
        engine.shed_load("NODE_07", -100.0)
        assert engine.nodes["NODE_07"]["consumption_mw"] == pytest.approx(served, abs=1e-6)

    def test_critical_load_cannot_be_shed(self, engine):
        assert engine.shed_load("NODE_03", 50.0)["success"] is False

    def test_restore_also_returns_ufls_shed_load(self, engine):
        node = engine.nodes["NODE_12"]
        node["ufls_fraction"] = 0.2
        engine.compute_power_flow()
        engine.shed_load("NODE_12", -node["demand_mw"])
        assert node["ufls_fraction"] == 0.0

    def test_state_estimation_finds_falsified_reading(self, engine):
        truth = engine.get_ground_truth_telemetry()
        clean = engine.run_state_estimation(["NODE_09", "NODE_10", "NODE_11"], truth)
        assert clean["consistent"] is True
        truth["NODE_09"]["generation_mw"] += 200.0
        dirty = engine.run_state_estimation(["NODE_09", "NODE_10", "NODE_11"], truth)
        assert dirty["consistent"] is False and dirty["violation_node"] == "NODE_09"
        assert dirty["residual_mw"] == pytest.approx(200.0, abs=3.0)

    def test_out_of_phase_merge_is_refused(self):
        eng = GridEngine(seed=3)
        build_base_topology(eng, load_factor=0.8, solar_irradiance=0.5)
        # Split NORTH from the rest and unbalance it so the islands drift apart.
        for eid in ("LINE_28", "LINE_29", "LINE_30", "LINE_38"):
            eng.toggle_circuit_breaker(eid, "OPEN")
        assert len(eng.get_stable_islands()) == 2
        eng.nodes["NODE_02"]["setpoint_mw"] += 300.0
        eng.compute_power_flow()
        eng.compute_frequency(smooth=False)
        result = eng.toggle_circuit_breaker("LINE_28", "CLOSED")
        assert result["success"] is False and result["out_of_phase"] is True
        assert eng.edges["LINE_28"]["status"] == "TRIPPED"

    def test_battery_depletes_and_stops(self, engine):
        battery = engine.nodes["NODE_20"]
        battery["energy_mwh"] = 10.0
        engine.dispatch_generation("NODE_20", 400.0)
        events = []
        for _ in range(3):
            events += engine.advance_tick()
        assert any(e["type"] == "battery_depleted" for e in events)
        assert battery["setpoint_mw"] == 0.0

    def test_identical_seeds_give_identical_telemetry(self):
        a, b = GridEngine(seed=11), GridEngine(seed=11)
        for eng in (a, b):
            build_base_topology(eng, load_factor=0.8, solar_irradiance=0.4)
        assert a.get_ground_truth_telemetry() == b.get_ground_truth_telemetry()

"""
NexusGrid — Seeded Scenario Generator.

Every task is a *family* of scenarios. The episode seed drives a dedicated
random stream (``SeedSequence([seed, task_id])``) that picks the disturbance,
its location, its timing and the operating point, so a policy cannot pass by
memorising one fixed action sequence.

Seeds ``>= EVAL_SEED_START`` draw from held-out parameter sets (for Task 2,
a disjoint part of the validated contingency catalog) so evaluation runs on
scenarios a policy never trained on.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .grid_engine import GridEngine

MAX_TICKS = {
    0: 3,    # Smoke test
    1: 15,   # Duck curve
    2: 20,   # Cascade overload
    3: 18,   # Phantom injection
    4: 12,   # Stuxnet resonance
    5: 50,   # Black start
}

TASK_NAMES = {
    0: "smoke_test",
    1: "duck_curve",
    2: "cascade_overload",
    3: "phantom_inject",
    4: "stuxnet_resonance",
    5: "black_start",
}

TASK_DIFFICULTIES = {
    0: "trivial",
    1: "easy",
    2: "medium",
    3: "hard",
    4: "very_hard",
    5: "expert",
}

EVAL_SEED_START = 10_000
FORECAST_HORIZON = 6
ZONES = ("NORTH", "SOUTH", "EAST", "WEST", "CENTRAL")
CATALOG_PATH = Path(__file__).with_name("data") / "cascade_catalog.json"

BACKBONE_LINES = [f"LINE_{i}" for i in range(28, 41)]
PHANTOM_TARGETS = ["NODE_02", "NODE_06", "NODE_09", "NODE_13", "NODE_14", "NODE_17"]
RESONANCE_TARGETS = ["NODE_17", "NODE_09", "NODE_13", "NODE_01"]
BLACK_START_UNITS = ["NODE_01", "NODE_13"]


def is_eval_seed(seed: int) -> bool:
    return int(seed) >= EVAL_SEED_START


def scenario_rng(task_id: int, seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed), int(task_id), 3])))


# ---------------------------------------------------------------------------
# Base network
# ---------------------------------------------------------------------------

def build_base_topology(engine: GridEngine, load_factor: float = 0.85, solar_irradiance: float = 0.6) -> None:
    """
    Build the 20-bus, 40-line network and balance it at ``load_factor``.

    Regions: NORTH, SOUTH, EAST, WEST, CENTRAL. Critical loads: NODE_03 and
    NODE_15 (hospitals), NODE_11 (water treatment), NODE_18 (data centre).
    """
    add = engine.add_node
    # NORTH
    add("NODE_01", "hydro", capacity_mw=1200, region="NORTH", black_start_capable=True)
    add("NODE_02", "gas", capacity_mw=600, region="NORTH")
    add("NODE_03", "load", capacity_mw=0, region="NORTH", critical=True, peak_load_mw=500)
    add("NODE_04", "battery", capacity_mw=300, region="NORTH")
    # SOUTH
    add("NODE_05", "solar", capacity_mw=800, region="SOUTH")
    add("NODE_06", "gas", capacity_mw=500, region="SOUTH")
    add("NODE_07", "load", capacity_mw=0, region="SOUTH", peak_load_mw=400)
    add("NODE_08", "battery", capacity_mw=250, region="SOUTH")
    # EAST
    add("NODE_09", "gas", capacity_mw=900, region="EAST")
    add("NODE_10", "solar", capacity_mw=400, region="EAST")
    add("NODE_11", "load", capacity_mw=0, region="EAST", critical=True, peak_load_mw=600)
    add("NODE_12", "load", capacity_mw=0, region="EAST", peak_load_mw=350)
    # WEST
    add("NODE_13", "hydro", capacity_mw=1000, region="WEST")
    add("NODE_14", "gas", capacity_mw=700, region="WEST")
    add("NODE_15", "load", capacity_mw=0, region="WEST", critical=True, peak_load_mw=450)
    add("NODE_16", "battery", capacity_mw=350, region="WEST")
    # CENTRAL
    add("NODE_17", "gas", capacity_mw=1500, region="CENTRAL")
    add("NODE_18", "load", capacity_mw=0, region="CENTRAL", critical=True, peak_load_mw=700)
    add("NODE_19", "load", capacity_mw=0, region="CENTRAL", peak_load_mw=300)
    add("NODE_20", "battery", capacity_mw=400, region="CENTRAL")

    line = engine.add_edge
    # Regional networks
    line("LINE_01", "NODE_01", "NODE_02", 800)
    line("LINE_02", "NODE_01", "NODE_03", 600)
    line("LINE_03", "NODE_02", "NODE_03", 500)
    line("LINE_04", "NODE_02", "NODE_04", 400)
    line("LINE_05", "NODE_03", "NODE_04", 350)
    line("LINE_06", "NODE_05", "NODE_06", 600)
    line("LINE_07", "NODE_05", "NODE_07", 500)
    line("LINE_08", "NODE_06", "NODE_07", 450)
    line("LINE_09", "NODE_06", "NODE_08", 350)
    line("LINE_10", "NODE_07", "NODE_08", 300)
    line("LINE_11", "NODE_09", "NODE_10", 600)
    line("LINE_12", "NODE_09", "NODE_11", 700)
    line("LINE_13", "NODE_09", "NODE_12", 500)
    line("LINE_14", "NODE_10", "NODE_11", 450)
    line("LINE_15", "NODE_10", "NODE_12", 350)
    line("LINE_16", "NODE_11", "NODE_12", 400)
    line("LINE_17", "NODE_13", "NODE_14", 700)
    line("LINE_18", "NODE_13", "NODE_15", 600)
    line("LINE_19", "NODE_14", "NODE_15", 500)
    line("LINE_20", "NODE_14", "NODE_16", 450)
    line("LINE_21", "NODE_15", "NODE_16", 350)
    line("LINE_22", "NODE_17", "NODE_18", 1000)
    line("LINE_23", "NODE_17", "NODE_19", 600)
    line("LINE_24", "NODE_17", "NODE_20", 500)
    line("LINE_25", "NODE_18", "NODE_19", 450)
    line("LINE_26", "NODE_18", "NODE_20", 400)
    line("LINE_27", "NODE_19", "NODE_20", 350)
    # Inter-regional backbone
    line("LINE_28", "NODE_01", "NODE_17", 1200)  # NORTH-CENTRAL (primary)
    line("LINE_29", "NODE_02", "NODE_09", 800)   # NORTH-EAST
    line("LINE_30", "NODE_04", "NODE_13", 500)   # NORTH-WEST
    line("LINE_31", "NODE_05", "NODE_17", 900)   # SOUTH-CENTRAL
    line("LINE_32", "NODE_06", "NODE_09", 600)   # SOUTH-EAST
    line("LINE_33", "NODE_08", "NODE_16", 400)   # SOUTH-WEST
    line("LINE_34", "NODE_13", "NODE_17", 800)   # WEST-CENTRAL
    line("LINE_35", "NODE_11", "NODE_18", 700)   # EAST-CENTRAL
    line("LINE_36", "NODE_12", "NODE_19", 500)   # EAST-CENTRAL 2
    line("LINE_37", "NODE_14", "NODE_09", 600)   # WEST-EAST
    line("LINE_38", "NODE_03", "NODE_15", 450)   # NORTH-WEST (hospitals)
    line("LINE_39", "NODE_07", "NODE_12", 400)   # SOUTH-EAST
    line("LINE_40", "NODE_16", "NODE_20", 500)   # WEST-CENTRAL batteries

    engine.weather = {
        zone: {"solar_irradiance": float(solar_irradiance), "wind_speed_ms": 6.0, "cloud_cover": 0.25}
        for zone in ZONES
    }
    set_load_factor(engine, load_factor)
    set_solar(engine, {zone: float(solar_irradiance) for zone in ZONES}, cloud=0.25)
    engine.balance_dispatch()
    engine.settle()


def set_load_factor(engine: GridEngine, load_factor: float, jitter: Optional[Dict[str, float]] = None) -> None:
    for nid, n in engine.nodes.items():
        if n["node_type"] != "load":
            continue
        mult = jitter.get(nid, 1.0) if jitter else 1.0
        demand = min(n["peak_load_mw"], n["peak_load_mw"] * load_factor * mult)
        n["base_consumption_mw"] = demand
        if n["energized"]:
            n["demand_mw"] = demand


def solar_output_factor(irradiance: float, cloud: float) -> float:
    return max(0.0, min(1.0, irradiance * (1.0 - 0.7 * cloud)))


def set_solar(engine: GridEngine, irradiance: Dict[str, float], cloud: float) -> None:
    for zone, irr in irradiance.items():
        if zone in engine.weather:
            engine.weather[zone]["solar_irradiance"] = float(irr)
            engine.weather[zone]["cloud_cover"] = float(cloud)
    for n in engine.nodes.values():
        if n["node_type"] == "solar":
            irr = irradiance.get(n["region"], 0.5)
            n["available_mw"] = n["capacity_mw"] * solar_output_factor(irr, cloud)


# ---------------------------------------------------------------------------
# Time-varying schedules (load and solar curves)
# ---------------------------------------------------------------------------

@dataclass
class Schedule:
    """Per-tick operating conditions: load factor and solar irradiance."""

    load_factor: List[float]
    irradiance: List[float]
    cloud: float = 0.25
    jitter: Dict[str, float] = field(default_factory=dict)
    wind: float = 6.0

    def at(self, values: List[float], tick: int) -> float:
        return values[min(max(tick, 0), len(values) - 1)]

    def apply(self, engine: GridEngine, tick: int) -> None:
        lf = self.at(self.load_factor, tick)
        irr = self.at(self.irradiance, tick)
        set_load_factor(engine, lf, self.jitter)
        set_solar(engine, {zone: irr for zone in ZONES}, self.cloud)
        for zone in ZONES:
            engine.weather[zone]["wind_speed_ms"] = self.wind
        horizon = range(tick + 1, tick + 1 + FORECAST_HORIZON)
        engine.weather_forecast = {zone: [self.at(self.irradiance, t) for t in horizon] for zone in ZONES}
        loads = [(n["peak_load_mw"], self.jitter.get(nid, 1.0)) for nid, n in engine.nodes.items()
                 if n["node_type"] == "load"]
        solar_cap = sum(n["capacity_mw"] for n in engine.nodes.values() if n["node_type"] == "solar")
        engine.demand_forecast = [
            round(sum(min(peak, peak * self.at(self.load_factor, t) * mult) for peak, mult in loads), 1)
            for t in horizon
        ]
        engine.solar_forecast = [round(solar_cap * solar_output_factor(self.at(self.irradiance, t), self.cloud), 1)
                                 for t in horizon]


def _constant(value: float, ticks: int) -> List[float]:
    return [float(value)] * (ticks + FORECAST_HORIZON + 2)


def _ramp(start: float, end: float, ramp_start: int, ramp_len: int, ticks: int) -> List[float]:
    values = []
    for t in range(ticks + FORECAST_HORIZON + 2):
        if t <= ramp_start:
            values.append(start)
        elif t >= ramp_start + ramp_len:
            values.append(end)
        else:
            frac = (t - ramp_start) / ramp_len
            frac = 0.5 - 0.5 * math.cos(math.pi * frac)  # smooth S-curve
            values.append(start + (end - start) * frac)
    return values


def _load_jitter(rng: np.random.Generator, engine: GridEngine) -> Dict[str, float]:
    return {nid: float(rng.uniform(0.94, 1.04)) for nid, n in engine.nodes.items() if n["node_type"] == "load"}


# ---------------------------------------------------------------------------
# Scenario container
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    task_id: int
    seed: int
    engine: GridEngine
    schedule: Schedule
    attack_config: Dict[str, Any]
    params: Dict[str, Any]
    brief: str

    @property
    def max_ticks(self) -> int:
        return MAX_TICKS[self.task_id]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "schedule": self.schedule,
            "attack_config": self.attack_config,
            "params": self.params,
            "brief": self.brief,
            "max_ticks": self.max_ticks,
            "task_name": TASK_NAMES[self.task_id],
            "difficulty": TASK_DIFFICULTIES[self.task_id],
            "split": "eval" if is_eval_seed(self.seed) else "train",
        }


def build_scenario(task_id: int, seed: int = 42) -> Dict[str, Any]:
    """Build the scenario for ``task_id`` drawn from the seed's random stream."""
    if task_id not in MAX_TICKS:
        raise ValueError(f"Unknown task_id: {task_id}")
    rng = scenario_rng(task_id, seed)
    builders = {
        0: _smoke_test,
        1: _duck_curve,
        2: _cascade_overload,
        3: _phantom_injection,
        4: _stuxnet_resonance,
        5: _black_start,
    }
    scenario = builders[task_id](rng, int(seed))
    scenario.engine.tick = 0
    scenario.schedule.apply(scenario.engine, 0)
    scenario.engine.settle()
    return scenario.as_dict()


def _new_engine(seed: int, load_factor: float, irradiance: float) -> GridEngine:
    engine = GridEngine(seed=seed)
    build_base_topology(engine, load_factor=load_factor, solar_irradiance=irradiance)
    return engine


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def _smoke_test(rng: np.random.Generator, seed: int) -> Scenario:
    lf = float(rng.uniform(0.70, 0.90))
    irr = float(rng.uniform(0.3, 0.8))
    engine = _new_engine(seed, lf, irr)
    schedule = Schedule(load_factor=_constant(lf, 3), irradiance=_constant(irr, 3), jitter=_load_jitter(rng, engine))
    set_load_factor(engine, lf, schedule.jitter)
    engine.balance_dispatch()
    return Scenario(0, seed, engine, schedule, {}, {"load_factor": lf, "irradiance": irr},
                    "Normal operations. Issue any safe positive dispatch to prove the control loop works.")


def _duck_curve(rng: np.random.Generator, seed: int) -> Scenario:
    ticks = MAX_TICKS[1]
    lf0 = float(rng.uniform(0.72, 0.80))
    lf1 = float(min(1.0, lf0 + rng.uniform(0.18, 0.26)))
    irr0 = float(rng.uniform(0.72, 0.92))
    irr1 = float(rng.uniform(0.0, 0.06))
    ramp_start = int(rng.integers(1, 3))
    ramp_len = int(rng.integers(5, 9))
    cloud = float(rng.uniform(0.05, 0.25))
    engine = _new_engine(seed, lf0, irr0)
    schedule = Schedule(
        load_factor=_ramp(lf0, lf1, ramp_start, ramp_len + 2, ticks),
        irradiance=_ramp(irr0, irr1, ramp_start, ramp_len, ticks),
        cloud=cloud,
        jitter=_load_jitter(rng, engine),
    )
    set_load_factor(engine, lf0, schedule.jitter)
    set_solar(engine, {z: irr0 for z in ZONES}, cloud)
    engine.balance_dispatch()
    params = {"load_start": lf0, "load_end": lf1, "irradiance_start": irr0, "irradiance_end": irr1,
              "ramp_start": ramp_start, "ramp_len": ramp_len, "cloud": cloud}
    brief = ("Evening ramp: solar output will fall over the next hour while demand climbs to the evening peak. "
             "Use the forecasts and keep frequency inside 59.8-60.2 Hz.")
    return Scenario(1, seed, engine, schedule, {}, params, brief)


def _phantom_injection(rng: np.random.Generator, seed: int) -> Scenario:
    lf = float(rng.uniform(0.80, 0.92))
    irr = float(rng.uniform(0.3, 0.6))
    engine = _new_engine(seed, lf, irr)
    schedule = Schedule(load_factor=_constant(lf, 18), irradiance=_constant(irr, 18), jitter=_load_jitter(rng, engine))
    set_load_factor(engine, lf, schedule.jitter)
    engine.balance_dispatch()
    target = str(rng.choice(PHANTOM_TARGETS))
    start = int(rng.integers(1, 4))
    rate = float(rng.uniform(70.0, 130.0))
    max_loss = float(min(engine.nodes[target]["setpoint_mw"] * 0.95, rng.uniform(350.0, 600.0)))
    others = [nid for nid in engine.nodes if nid != target]
    decoys = [str(x) for x in rng.choice(others, size=2, replace=False)]
    attack = {
        "type": "phantom_injection",
        "target_node": target,
        "start_tick": start,
        "loss_rate_mw": rate,
        "max_loss_mw": max_loss,
        "decoy_nodes": decoys,
        "decoy_tick": int(rng.integers(0, 6)),
        "active": True,
    }
    brief = ("SOC alert: possible SCADA compromise. Telemetry may be falsified. Triage packet logs, verify with "
             "state estimation (max 6 buses per call), quarantine only proven nodes, then restore frequency.")
    return Scenario(3, seed, engine, schedule, attack, {"load_factor": lf, **{k: v for k, v in attack.items() if k != "active"}}, brief)


def _stuxnet_resonance(rng: np.random.Generator, seed: int) -> Scenario:
    lf = float(rng.uniform(0.80, 0.92))
    irr = float(rng.uniform(0.3, 0.6))
    engine = _new_engine(seed, lf, irr)
    schedule = Schedule(load_factor=_constant(lf, 12), irradiance=_constant(irr, 12), jitter=_load_jitter(rng, engine))
    set_load_factor(engine, lf, schedule.jitter)
    engine.balance_dispatch()
    target = str(rng.choice(RESONANCE_TARGETS))
    mode = float(round(rng.uniform(0.30, 0.80) / 0.05) * 0.05)
    attack = {
        "type": "resonance_oscillation",
        "target_node": target,
        "mode_hz": round(mode, 2),
        "start_tick": int(rng.integers(0, 2)),
        "initial_amplitude_hz": float(rng.uniform(0.03, 0.05)),
        "growth": float(rng.uniform(1.30, 1.45)),
        "destruction_amplitude_hz": 0.35,
        "active": True,
    }
    brief = ("PMU alarm: a forced oscillation is building on a large generator. Do not disconnect the unit. "
             "Identify the oscillating unit and its mode from telemetry, damp it with a counter-signal from a "
             "nearby battery (within 2 hops), then detune the unit and rebalance.")
    return Scenario(4, seed, engine, schedule, attack, {"load_factor": lf, **{k: v for k, v in attack.items() if k != "active"}}, brief)


def _black_start(rng: np.random.Generator, seed: int) -> Scenario:
    lf = float(rng.uniform(0.65, 0.85))
    engine = _new_engine(seed, lf, 0.05)
    schedule = Schedule(load_factor=_constant(lf, 50), irradiance=_constant(float(rng.uniform(0.0, 0.08)), 50),
                        jitter=_load_jitter(rng, engine), cloud=0.6)
    set_load_factor(engine, lf, schedule.jitter)
    bs_unit = str(rng.choice(BLACK_START_UNITS))
    for nid, n in engine.nodes.items():
        n["black_start_capable"] = nid == bs_unit
        n["energized"] = nid == bs_unit
        n["setpoint_mw"] = 0.0
        n["generation_mw"] = 0.0
        n["consumption_mw"] = 0.0
        n["shed_mw"] = 0.0
        n["frequency_hz"] = 60.0 if nid == bs_unit else 0.0
        n["voltage_kv"] = 345.0 if nid == bs_unit else 0.0
    for e in engine.edges.values():
        e["status"] = "TRIPPED"
    brief = (f"System-wide blackout. Only {bs_unit} has black-start capability. Restart it, energise lines "
             "step by step, raise generation before picking up load, keep each island inside 59.5-60.5 Hz, "
             "and restore critical loads first.")
    return Scenario(5, seed, engine, schedule, {}, {"load_factor": lf, "black_start_unit": bs_unit}, brief)


# ---------------------------------------------------------------------------
# Task 2: validated contingency catalog
# ---------------------------------------------------------------------------

def apply_cascade_config(engine: GridEngine, config: Dict[str, Any]) -> None:
    """Apply a stressed dispatch, maintenance outages and the storm trip."""
    set_load_factor(engine, config["load_factor"])
    set_solar(engine, {z: config["irradiance"] for z in ZONES}, 0.25)
    for nid, setpoint in config["setpoints"].items():
        engine.nodes[nid]["setpoint_mw"] = float(setpoint)
    for nid, n in engine.nodes.items():
        if n["node_type"] == "battery":
            n["setpoint_mw"] = 0.0
    for line_id in config["outages"]:
        engine.set_out_of_service(line_id, "maintenance")
    engine.set_out_of_service(config["trip_line"], "storm damage")
    engine.settle()


def stressed_setpoints(engine: GridEngine, rng: np.random.Generator, exporters: List[str]) -> Optional[Dict[str, float]]:
    demand = sum(n["demand_mw"] for n in engine.nodes.values() if n["node_type"] == "load")
    solar = sum(min(n["available_mw"], n["capacity_mw"]) for n in engine.nodes.values() if n["node_type"] == "solar")
    need = demand - solar
    gens = [nid for nid, n in engine.nodes.items() if n["node_type"] in ("hydro", "gas")]
    setpoints = {nid: 0.0 for nid in gens}
    exp_units = [g for g in gens if engine.nodes[g]["region"] in exporters]
    for g in exp_units:
        setpoints[g] = engine.nodes[g]["capacity_mw"] * float(rng.uniform(0.80, 0.95))
    rest = need - sum(setpoints[g] for g in exp_units)
    others = [g for g in gens if g not in exp_units]
    cap_others = sum(engine.nodes[g]["capacity_mw"] for g in others)
    if rest < 0 or cap_others <= 0:
        return None
    for g in others:
        setpoints[g] = rest * engine.nodes[g]["capacity_mw"] / cap_others
    return {k: round(v, 2) for k, v in setpoints.items()}


def cascade_config_check(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return validity information for a Task 2 configuration."""
    engine = GridEngine(seed=0)
    build_base_topology(engine, load_factor=config["load_factor"], solar_irradiance=config["irradiance"])
    # Pre-contingency: outages only, must be secure.
    set_load_factor(engine, config["load_factor"])
    for nid, setpoint in config["setpoints"].items():
        engine.nodes[nid]["setpoint_mw"] = float(setpoint)
    for nid, n in engine.nodes.items():
        if n["node_type"] == "battery":
            n["setpoint_mw"] = 0.0
    for line_id in config["outages"]:
        engine.edges[line_id]["status"] = "TRIPPED"
    engine.settle()
    pre_ok = len(engine.get_stable_islands()) == 1 and engine.get_max_loading() <= 0.97
    engine.edges[config["trip_line"]]["status"] = "TRIPPED"
    engine.settle()
    islands = len(engine.get_stable_islands())
    max_loading = engine.get_max_loading()
    overloaded = engine.get_overloaded_edges(1.0)
    valid = pre_ok and islands == 1 and 1.06 <= max_loading <= 1.35 and abs(engine.frequency_hz - 60.0) < 0.05
    return {"valid": bool(valid), "max_loading": round(max_loading, 3), "overloaded": overloaded}


def generate_cascade_catalog(count: int = 240, seed: int = 2026, max_trials: int = 20000) -> List[Dict[str, Any]]:
    """Search for stressed operating points where one storm trip overloads the grid."""
    rng = np.random.Generator(np.random.PCG64(seed))
    catalog: List[Dict[str, Any]] = []
    seen = set()
    for _ in range(max_trials):
        if len(catalog) >= count:
            break
        lf = float(rng.uniform(0.88, 1.0))
        irr = float(rng.uniform(0.2, 0.5))
        engine = GridEngine(seed=0)
        build_base_topology(engine, load_factor=lf, solar_irradiance=irr)
        exporters = [str(x) for x in rng.choice(list(ZONES), size=2, replace=False)]
        setpoints = stressed_setpoints(engine, rng, exporters)
        if setpoints is None:
            continue
        outages = [str(x) for x in rng.choice(BACKBONE_LINES, size=int(rng.integers(1, 3)), replace=False)]
        candidates = [b for b in BACKBONE_LINES if b not in outages]
        rng.shuffle(candidates)
        for trip in candidates[:6]:
            config = {"load_factor": round(lf, 4), "irradiance": round(irr, 4), "exporters": sorted(exporters),
                      "setpoints": setpoints, "outages": sorted(outages), "trip_line": trip}
            key = (tuple(sorted(exporters)), tuple(sorted(outages)), trip, round(lf, 2))
            if key in seen:
                continue
            info = cascade_config_check(config)
            if info["valid"]:
                seen.add(key)
                config["max_loading"] = info["max_loading"]
                config["overloaded"] = info["overloaded"]
                catalog.append(config)
                break
    return catalog


_CATALOG_CACHE: Optional[List[Dict[str, Any]]] = None


def load_cascade_catalog() -> List[Dict[str, Any]]:
    global _CATALOG_CACHE
    if _CATALOG_CACHE is None:
        if CATALOG_PATH.exists():
            _CATALOG_CACHE = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["configs"]
        else:  # pragma: no cover - only when the data file is missing
            _CATALOG_CACHE = generate_cascade_catalog(count=60)
    return _CATALOG_CACHE


def _cascade_overload(rng: np.random.Generator, seed: int) -> Scenario:
    catalog = load_cascade_catalog()
    split_at = int(len(catalog) * 0.8)
    pool = catalog[split_at:] if is_eval_seed(seed) else catalog[:split_at]
    config = copy.deepcopy(pool[int(rng.integers(len(pool)))])
    engine = GridEngine(seed=seed)
    build_base_topology(engine, load_factor=config["load_factor"], solar_irradiance=config["irradiance"])
    apply_cascade_config(engine, config)
    schedule = Schedule(load_factor=_constant(config["load_factor"], 20), irradiance=_constant(config["irradiance"], 20))
    for e in engine.edges.values():
        if e["status"] == "LIVE" and e["loading"] > 1.0:
            e["overload_ticks"] = 0
    outages = ", ".join(config["outages"])
    brief = (f"Storm damage: {config['trip_line']} has tripped while {outages} is out for maintenance. "
             "Power is rerouting and some lines are above their thermal rating. Relays trip lines that stay above "
             "100% for 3 ticks (instantly above 140%). Relieve the overloads without losing critical loads.")
    params = {k: v for k, v in config.items() if k not in ("setpoints",)}
    return Scenario(2, seed, engine, schedule, {}, params, brief)

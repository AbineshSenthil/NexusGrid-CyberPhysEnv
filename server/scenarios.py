"""
NexusGrid — Scenario Definitions.

Six hardcoded scenarios (one per task), fully deterministic from seed.
Each scenario defines: topology, load profile, weather, attack vectors, max_ticks.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .grid_engine import GridEngine


# Max ticks per task — matches openenv.yaml
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


def build_base_topology(engine: GridEngine) -> None:
    """
    Build the standard 20-node grid topology.

    Regions: NORTH, SOUTH, EAST, WEST, CENTRAL
    Node types mix: hydro, solar, gas, battery, load
    """
    # ---- NORTH region (nodes 01-04) ----
    engine.add_node("NODE_01", "hydro",   capacity_mw=1200, region="NORTH", critical=False,
                    generation_mw=800, peak_load_mw=0)
    engine.add_node("NODE_02", "gas",     capacity_mw=600,  region="NORTH", critical=False,
                    generation_mw=400, peak_load_mw=0)
    engine.add_node("NODE_03", "load",    capacity_mw=0,    region="NORTH", critical=True,
                    peak_load_mw=500, consumption_mw=450)  # Hospital complex
    engine.add_node("NODE_04", "battery", capacity_mw=300,  region="NORTH", critical=False,
                    generation_mw=0, peak_load_mw=0)

    # ---- SOUTH region (nodes 05-08) ----
    engine.add_node("NODE_05", "solar",   capacity_mw=800,  region="SOUTH", critical=False,
                    generation_mw=600, peak_load_mw=0)
    engine.add_node("NODE_06", "gas",     capacity_mw=500,  region="SOUTH", critical=False,
                    generation_mw=350, peak_load_mw=0)
    engine.add_node("NODE_07", "load",    capacity_mw=0,    region="SOUTH", critical=False,
                    peak_load_mw=400, consumption_mw=380)
    engine.add_node("NODE_08", "battery", capacity_mw=250,  region="SOUTH", critical=False,
                    generation_mw=0, peak_load_mw=0)

    # ---- EAST region (nodes 09-12) ----
    engine.add_node("NODE_09", "gas",     capacity_mw=900,  region="EAST",  critical=False,
                    generation_mw=600, peak_load_mw=0)
    engine.add_node("NODE_10", "solar",   capacity_mw=400,  region="EAST",  critical=False,
                    generation_mw=300, peak_load_mw=0)
    engine.add_node("NODE_11", "load",    capacity_mw=0,    region="EAST",  critical=True,
                    peak_load_mw=600, consumption_mw=550)  # Water treatment
    engine.add_node("NODE_12", "load",    capacity_mw=0,    region="EAST",  critical=False,
                    peak_load_mw=350, consumption_mw=320)

    # ---- WEST region (nodes 13-16) ----
    engine.add_node("NODE_13", "hydro",   capacity_mw=1000, region="WEST",  critical=False,
                    generation_mw=700, peak_load_mw=0)
    engine.add_node("NODE_14", "gas",     capacity_mw=700,  region="WEST",  critical=False,
                    generation_mw=500, peak_load_mw=0)  # This gets spoofed in Task 3
    engine.add_node("NODE_15", "load",    capacity_mw=0,    region="WEST",  critical=True,
                    peak_load_mw=450, consumption_mw=420)  # Hospital
    engine.add_node("NODE_16", "battery", capacity_mw=350,  region="WEST",  critical=False,
                    generation_mw=0, peak_load_mw=0)

    # ---- CENTRAL region (nodes 17-20) ----
    engine.add_node("NODE_17", "gas",     capacity_mw=1500, region="CENTRAL", critical=False,
                    generation_mw=1000, peak_load_mw=0)  # Major turbine (Task 4 target)
    engine.add_node("NODE_18", "load",    capacity_mw=0,    region="CENTRAL", critical=True,
                    peak_load_mw=700, consumption_mw=650)  # Data center
    engine.add_node("NODE_19", "load",    capacity_mw=0,    region="CENTRAL", critical=False,
                    peak_load_mw=300, consumption_mw=280)
    engine.add_node("NODE_20", "battery", capacity_mw=400,  region="CENTRAL", critical=False,
                    generation_mw=0, peak_load_mw=0)

    # ---- Transmission lines (40 edges) ----
    # NORTH internal
    engine.add_edge("LINE_01", "NODE_01", "NODE_02", capacity_mw=800)
    engine.add_edge("LINE_02", "NODE_01", "NODE_03", capacity_mw=600)
    engine.add_edge("LINE_03", "NODE_02", "NODE_03", capacity_mw=500)
    engine.add_edge("LINE_04", "NODE_02", "NODE_04", capacity_mw=400)
    engine.add_edge("LINE_05", "NODE_03", "NODE_04", capacity_mw=350)

    # SOUTH internal
    engine.add_edge("LINE_06", "NODE_05", "NODE_06", capacity_mw=600)
    engine.add_edge("LINE_07", "NODE_05", "NODE_07", capacity_mw=500)
    engine.add_edge("LINE_08", "NODE_06", "NODE_07", capacity_mw=450)
    engine.add_edge("LINE_09", "NODE_06", "NODE_08", capacity_mw=350)
    engine.add_edge("LINE_10", "NODE_07", "NODE_08", capacity_mw=300)

    # EAST internal
    engine.add_edge("LINE_11", "NODE_09", "NODE_10", capacity_mw=600)
    engine.add_edge("LINE_12", "NODE_09", "NODE_11", capacity_mw=700)
    engine.add_edge("LINE_13", "NODE_09", "NODE_12", capacity_mw=500)
    engine.add_edge("LINE_14", "NODE_10", "NODE_11", capacity_mw=450)
    engine.add_edge("LINE_15", "NODE_10", "NODE_12", capacity_mw=350)
    engine.add_edge("LINE_16", "NODE_11", "NODE_12", capacity_mw=400)

    # WEST internal
    engine.add_edge("LINE_17", "NODE_13", "NODE_14", capacity_mw=700)
    engine.add_edge("LINE_18", "NODE_13", "NODE_15", capacity_mw=600)
    engine.add_edge("LINE_19", "NODE_14", "NODE_15", capacity_mw=500)
    engine.add_edge("LINE_20", "NODE_14", "NODE_16", capacity_mw=450)
    engine.add_edge("LINE_21", "NODE_15", "NODE_16", capacity_mw=350)

    # CENTRAL internal
    engine.add_edge("LINE_22", "NODE_17", "NODE_18", capacity_mw=1000)
    engine.add_edge("LINE_23", "NODE_17", "NODE_19", capacity_mw=600)
    engine.add_edge("LINE_24", "NODE_17", "NODE_20", capacity_mw=500)
    engine.add_edge("LINE_25", "NODE_18", "NODE_19", capacity_mw=450)
    engine.add_edge("LINE_26", "NODE_18", "NODE_20", capacity_mw=400)
    engine.add_edge("LINE_27", "NODE_19", "NODE_20", capacity_mw=350)

    # Inter-region backbone lines
    engine.add_edge("LINE_28", "NODE_01", "NODE_17", capacity_mw=1200)  # NORTH-CENTRAL (primary)
    engine.add_edge("LINE_29", "NODE_02", "NODE_09", capacity_mw=800)   # NORTH-EAST
    engine.add_edge("LINE_30", "NODE_04", "NODE_13", capacity_mw=500)   # NORTH-WEST
    engine.add_edge("LINE_31", "NODE_05", "NODE_17", capacity_mw=900)   # SOUTH-CENTRAL
    engine.add_edge("LINE_32", "NODE_06", "NODE_09", capacity_mw=600)   # SOUTH-EAST
    engine.add_edge("LINE_33", "NODE_08", "NODE_16", capacity_mw=400)   # SOUTH-WEST
    engine.add_edge("LINE_34", "NODE_13", "NODE_17", capacity_mw=800)   # WEST-CENTRAL
    engine.add_edge("LINE_35", "NODE_11", "NODE_18", capacity_mw=700)   # EAST-CENTRAL
    engine.add_edge("LINE_36", "NODE_12", "NODE_19", capacity_mw=500)   # EAST-CENTRAL2
    engine.add_edge("LINE_37", "NODE_14", "NODE_09", capacity_mw=600)   # WEST-EAST
    engine.add_edge("LINE_38", "NODE_03", "NODE_15", capacity_mw=450)   # NORTH-WEST (hospitals)
    engine.add_edge("LINE_39", "NODE_07", "NODE_12", capacity_mw=400)   # SOUTH-EAST
    engine.add_edge("LINE_40", "NODE_16", "NODE_20", capacity_mw=500)   # WEST-CENTRAL batteries

    # Set up weather zones
    engine.weather = {
        "NORTH":   {"solar_irradiance": 0.6, "wind_speed_ms": 8.0, "cloud_cover": 0.3},
        "SOUTH":   {"solar_irradiance": 0.8, "wind_speed_ms": 5.0, "cloud_cover": 0.2},
        "EAST":    {"solar_irradiance": 0.5, "wind_speed_ms": 6.0, "cloud_cover": 0.4},
        "WEST":    {"solar_irradiance": 0.7, "wind_speed_ms": 7.0, "cloud_cover": 0.25},
        "CENTRAL": {"solar_irradiance": 0.6, "wind_speed_ms": 4.0, "cloud_cover": 0.35},
    }

    # Initial power flow
    engine.compute_power_flow()
    engine.compute_frequency()
    engine._record_telemetry()


# ---------------------------------------------------------------------------
# Scenario-specific modifications
# ---------------------------------------------------------------------------

def build_scenario(task_id: int, seed: int = 42) -> Dict[str, Any]:
    """
    Build a complete scenario for the given task.

    Returns:
        Dict with keys: engine, attack_config, max_ticks, task_name, difficulty
    """
    engine = GridEngine(seed=seed)
    attack_config: Dict[str, Any] = {}

    if task_id == 0:
        # Task 0: Smoke test — basic topology, no modifications needed
        build_base_topology(engine)

    elif task_id == 1:
        # Task 1: Duck curve — solar ramp-down at sunset
        build_base_topology(engine)
        _setup_duck_curve(engine, seed)

    elif task_id == 2:
        # Task 2: Cascade overload — storm snaps primary line
        build_base_topology(engine)
        _setup_cascade_overload(engine, seed)

    elif task_id == 3:
        # Task 3: Phantom injection — NODE_14 spoofed
        build_base_topology(engine)
        attack_config = _setup_phantom_injection(engine, seed)

    elif task_id == 4:
        # Task 4: Stuxnet resonance — NODE_17 turbine under attack
        build_base_topology(engine)
        attack_config = _setup_stuxnet_resonance(engine, seed)

    elif task_id == 5:
        # Task 5: Black start — entire grid is dark
        build_base_topology(engine)
        _setup_black_start(engine, seed)

    else:
        raise ValueError(f"Unknown task_id: {task_id}")

    return {
        "engine": engine,
        "attack_config": attack_config,
        "max_ticks": MAX_TICKS[task_id],
        "task_name": TASK_NAMES[task_id],
        "difficulty": TASK_DIFFICULTIES[task_id],
    }


def _setup_duck_curve(engine: GridEngine, seed: int) -> None:
    """
    Task 1: Solar generation drops rapidly (sunset) while demand spikes.
    The agent must proactively dispatch battery reserves.
    """
    # Reduce solar irradiance to simulate sunset
    for zone in engine.weather:
        engine.weather[zone]["solar_irradiance"] = 0.1  # Sunset
        engine.weather[zone]["cloud_cover"] = 0.8

    # Solar nodes drop to near-zero
    for node in engine.nodes.values():
        if node["node_type"] == "solar":
            node["generation_mw"] = node["capacity_mw"] * 0.05  # 5% residual

    # Spike demand
    for node in engine.nodes.values():
        if node["node_type"] == "load":
            node["consumption_mw"] = node["peak_load_mw"] * 0.95

    # Recompute -- this will show a frequency drop
    engine.compute_power_flow()
    engine.compute_frequency()
    engine._record_telemetry()


def _setup_cascade_overload(engine: GridEngine, seed: int) -> None:
    """
    Task 2: A storm snaps the NORTH-CENTRAL primary line (LINE_28, 1200MW).
    Power reroutes through secondary paths and overloads them.
    """
    # Trip the primary backbone line
    engine.edges["LINE_28"]["status"] = "TRIPPED"
    engine.edges["LINE_28"]["current_load_mw"] = 0.0

    # Increase load to stress secondary paths
    for node in engine.nodes.values():
        if node["node_type"] == "load":
            node["consumption_mw"] = node["peak_load_mw"] * 0.90

    # Recompute -- secondary lines will be overloaded
    engine.compute_power_flow()
    engine.compute_frequency()
    engine._record_telemetry()


def _setup_phantom_injection(engine: GridEngine, seed: int) -> Dict[str, Any]:
    """
    Task 3: NODE_14 is spoofed to report receiving 100MW when actually 0MW.
    Returns attack configuration for the spoof engine.
    """
    # NODE_14 is the target: its actual generation is already set to 500MW
    # The spoof will make it appear to have 600MW (adding phantom 100MW)
    attack_config = {
        "type": "phantom_injection",
        "target_node": "NODE_14",
        "phantom_mw": 100.0,  # Extra MW injected into reports
        "start_tick": 0,
        "active": True,
    }

    # Slightly reduce overall generation to make the grid vulnerable
    engine.nodes["NODE_14"]["generation_mw"] = 400.0  # Actual is 400, spoof reports 500

    engine.compute_power_flow()
    engine.compute_frequency()
    engine._record_telemetry()

    return attack_config


def _setup_stuxnet_resonance(engine: GridEngine, seed: int) -> Dict[str, Any]:
    """
    Task 4: NODE_17 (1500MW turbine) is being oscillated at 0.5Hz intervals.
    Agent must inject counter-signal at correct hz_offset.
    """
    attack_config = {
        "type": "resonance_oscillation",
        "target_node": "NODE_17",
        "oscillation_hz": 0.5,  # Attack frequency offset
        "start_tick": 0,
        "destruction_tick": 10,  # Turbine destroyed if not countered by tick 10
        "active": True,
    }

    engine.compute_power_flow()
    engine.compute_frequency()
    engine._record_telemetry()

    return attack_config


def _setup_black_start(engine: GridEngine, seed: int) -> None:
    """
    Task 5: Entire grid is dark. All nodes de-energized except NODE_01 (hydro dam).
    Agent must restart the grid from scratch.
    """
    # De-energize everything
    for node in engine.nodes.values():
        node["energized"] = False
        node["generation_mw"] = 0.0
        node["consumption_mw"] = 0.0
        node["voltage_kv"] = 0.0

    # Trip all lines
    for edge in engine.edges.values():
        edge["status"] = "TRIPPED"
        edge["current_load_mw"] = 0.0

    # NODE_01 (hydro dam) can be energized — it has black-start capability
    engine.nodes["NODE_01"]["energized"] = True
    engine.nodes["NODE_01"]["voltage_kv"] = 345.0
    # But generation is still 0 — agent must dispatch

    # Set frequency to near-zero (grid is dead)
    engine.frequency_hz = 59.0

    engine._record_telemetry()

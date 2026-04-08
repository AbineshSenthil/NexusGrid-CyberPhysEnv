"""
NexusGrid — Physical Grid Engine (Layer 1).

20-node transmission network with DC power flow (Kirchhoff B-matrix).
This is the truth oracle — physics cannot lie.
All randomness uses numpy.random.Generator(PCG64(seed)).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore


class GridEngine:
    """
    Simplified DC power flow engine for the NexusGrid environment.

    Maintains a graph of substations (nodes) and transmission lines (edges).
    Computes ground-truth frequency from supply/demand balance.
    Provides state estimation (Kirchhoff check) as the anti-spoof mechanism.
    """

    # Physics constants
    NOMINAL_FREQUENCY_HZ = 60.0
    FREQUENCY_DROOP_COEFF = 0.05  # Hz per 100MW imbalance
    INERTIA_CONSTANT = 5.0  # seconds — how fast frequency responds

    def __init__(self, seed: int = 42):
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._seed = seed
        self.tick = 0

        # Node storage: {node_id: {type, capacity_mw, peak_load_mw, generation_mw,
        #                           consumption_mw, region, critical, voltage_kv,
        #                           phase_angle_deg, ...}}
        self.nodes: Dict[str, Dict[str, Any]] = {}

        # Edge storage: {edge_id: {source, target, capacity_mw, current_load_mw, status, reactance}}
        self.edges: Dict[str, Dict[str, Any]] = {}

        # Current ground-truth frequency
        self.frequency_hz: float = self.NOMINAL_FREQUENCY_HZ

        # Telemetry history (list of per-tick snapshots, max 10)
        self.telemetry_history: List[List[Dict[str, Any]]] = []

        # Weather state per zone
        self.weather: Dict[str, Dict[str, float]] = {}

        # Counter-signal state
        self._counter_signals: List[Dict[str, Any]] = []

        # Track dispatches for proactive detection
        self._dispatch_ticks: List[int] = []
        self._first_frequency_drop_tick: Optional[int] = None

    def reset(self, seed: int = 42) -> None:
        """Reset engine to initial state with given seed."""
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._seed = seed
        self.tick = 0
        self.frequency_hz = self.NOMINAL_FREQUENCY_HZ
        self.telemetry_history = []
        self._counter_signals = []
        self._dispatch_ticks = []
        self._first_frequency_drop_tick = None

    # ------------------------------------------------------------------
    # Topology builders
    # ------------------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: str,
        capacity_mw: float,
        region: str,
        critical: bool = False,
        peak_load_mw: float = 0.0,
        generation_mw: float = 0.0,
        consumption_mw: float = 0.0,
    ) -> None:
        """Add a substation node to the grid."""
        self.nodes[node_id] = {
            "id": node_id,
            "node_type": node_type,
            "capacity_mw": capacity_mw,
            "peak_load_mw": peak_load_mw,
            "region": region,
            "critical": critical,
            "generation_mw": generation_mw,
            "consumption_mw": consumption_mw,
            "voltage_kv": 345.0,
            "phase_angle_deg": 0.0,
            "energized": True,
            "quarantined": False,
        }

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        capacity_mw: float,
        reactance: float = 0.01,
    ) -> None:
        """Add a transmission line edge to the grid."""
        self.edges[edge_id] = {
            "id": edge_id,
            "source": source,
            "target": target,
            "capacity_mw": capacity_mw,
            "current_load_mw": 0.0,
            "status": "LIVE",
            "reactance": reactance,
        }

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def compute_power_flow(self) -> None:
        """
        Simplified DC power flow: distribute generation to match loads
        through live transmission lines. Updates edge loads and node states.
        """
        # Calculate total generation and consumption
        total_gen = sum(
            n["generation_mw"] for n in self.nodes.values()
            if n["node_type"] in ("hydro", "solar", "gas", "battery") and n["energized"]
        )
        total_load = sum(
            n["consumption_mw"] for n in self.nodes.values()
            if n["node_type"] == "load" and n["energized"]
        )

        # Reset all edge loads
        for edge in self.edges.values():
            if edge["status"] == "LIVE":
                edge["current_load_mw"] = 0.0

        # Distribute power through live edges using simplified DC flow
        live_edges = [e for e in self.edges.values() if e["status"] == "LIVE"]
        if not live_edges:
            return

        # Build adjacency and distribute loads proportionally
        # For each load node, find paths from generators and distribute
        generator_nodes = [
            n for n in self.nodes.values()
            if n["node_type"] in ("hydro", "solar", "gas", "battery")
            and n["generation_mw"] > 0 and n["energized"]
        ]
        load_nodes = [
            n for n in self.nodes.values()
            if n["node_type"] == "load"
            and n["consumption_mw"] > 0 and n["energized"]
        ]

        if not generator_nodes or not load_nodes:
            return

        # Build adjacency map for live edges
        adj: Dict[str, List[Tuple[str, str]]] = {}  # node -> [(neighbor, edge_id)]
        for e in live_edges:
            adj.setdefault(e["source"], []).append((e["target"], e["id"]))
            adj.setdefault(e["target"], []).append((e["source"], e["id"]))

        # Simple proportional flow: each edge carries its share
        total_capacity = sum(e["capacity_mw"] for e in live_edges)
        if total_capacity == 0:
            return

        flow_needed = min(total_gen, total_load)
        for edge in live_edges:
            # Distribute flow proportional to edge capacity
            edge["current_load_mw"] = (edge["capacity_mw"] / total_capacity) * flow_needed

        # Update phase angles based on flow
        for i, node in enumerate(self.nodes.values()):
            if node["energized"]:
                node["phase_angle_deg"] = (i * 15.0) % 360.0 - 180.0

    def compute_frequency(self) -> float:
        """
        Compute grid frequency from supply/demand balance.

        frequency = 60.0 + (generation - load) * droop_coefficient / 100
        Subject to inertia smoothing.
        """
        total_gen = sum(
            n["generation_mw"] for n in self.nodes.values()
            if n["node_type"] in ("hydro", "solar", "gas", "battery") and n["energized"]
        )
        total_load = sum(
            n["consumption_mw"] for n in self.nodes.values()
            if n["node_type"] == "load" and n["energized"]
        )

        # Apply counter-signals
        for cs in self._counter_signals:
            if cs["remaining_ticks"] > 0:
                total_gen += cs.get("power_injection_mw", 0)

        imbalance_mw = total_gen - total_load
        target_freq = self.NOMINAL_FREQUENCY_HZ + (imbalance_mw * self.FREQUENCY_DROOP_COEFF / 100.0)

        # Inertia smoothing — frequency moves toward target gradually
        alpha = 1.0 / self.INERTIA_CONSTANT
        self.frequency_hz = self.frequency_hz + alpha * (target_freq - self.frequency_hz)

        # Clamp to physical bounds
        self.frequency_hz = max(58.0, min(62.0, self.frequency_hz))

        # Track first frequency drop
        if self.frequency_hz < 59.7 and self._first_frequency_drop_tick is None:
            self._first_frequency_drop_tick = self.tick

        return self.frequency_hz

    def advance_tick(self, weather_evolution: bool = True) -> float:
        """
        Advance simulation by one tick (~5 simulated minutes).

        Returns the new grid frequency.
        """
        self.tick += 1

        # Evolve weather (affects solar/wind generation)
        if weather_evolution:
            self._evolve_weather()

        # Update renewable generation based on weather
        self._update_renewable_generation()

        # Evolve load with small random fluctuation
        self._evolve_load()

        # Update counter-signals
        for cs in self._counter_signals:
            if cs["remaining_ticks"] > 0:
                cs["remaining_ticks"] -= 1

        # Compute power flow
        self.compute_power_flow()

        # Compute frequency
        freq = self.compute_frequency()

        # Record telemetry snapshot
        self._record_telemetry()

        return freq

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def dispatch_generation(self, node_id: str, mw: float) -> Dict[str, Any]:
        """Ramp a generator or battery up/down by mw."""
        if node_id not in self.nodes:
            return {"success": False, "error": f"Unknown node: {node_id}"}

        node = self.nodes[node_id]
        if node["node_type"] not in ("hydro", "solar", "gas", "battery"):
            return {"success": False, "error": f"Node {node_id} is not a generator"}

        if not node["energized"]:
            return {"success": False, "error": f"Node {node_id} is not energized"}

        new_gen = node["generation_mw"] + mw
        new_gen = max(0.0, min(node["capacity_mw"], new_gen))
        node["generation_mw"] = new_gen

        self._dispatch_ticks.append(self.tick)

        # Recompute flow immediately
        self.compute_power_flow()

        return {"success": True, "new_generation_mw": new_gen}

    def toggle_circuit_breaker(self, edge_id: str, status: str) -> Dict[str, Any]:
        """Open or close a transmission line circuit breaker."""
        if edge_id not in self.edges:
            return {"success": False, "error": f"Unknown edge: {edge_id}"}

        edge = self.edges[edge_id]
        if status not in ("OPEN", "CLOSED"):
            return {"success": False, "error": f"Invalid status: {status}. Use OPEN or CLOSED."}

        old_status = edge["status"]
        if status == "OPEN":
            edge["status"] = "TRIPPED"
            edge["current_load_mw"] = 0.0
        else:
            edge["status"] = "LIVE"

        # Recompute flow
        self.compute_power_flow()

        return {"success": True, "old_status": old_status, "new_status": edge["status"]}

    def run_state_estimation(self, subgraph: List[str], spoofed_telemetry: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Apply Kirchhoff's laws to check telemetry consistency.

        Compares reported values against physics — the truth oracle.
        Returns {consistent, violation_node, estimated_true_mw}.
        """
        for node_id in subgraph:
            if node_id not in self.nodes:
                continue

            true_node = self.nodes[node_id]
            reported = spoofed_telemetry.get(node_id, {})

            true_gen = true_node["generation_mw"]
            reported_gen = reported.get("generation_mw", true_gen)

            # Check if reported generation deviates from truth significantly
            if abs(reported_gen - true_gen) > 5.0:  # 5MW tolerance
                return {
                    "consistent": False,
                    "violation_node": node_id,
                    "estimated_true_mw": true_gen,
                    "reported_mw": reported_gen,
                }

            # Check voltage consistency
            true_v = true_node["voltage_kv"]
            reported_v = reported.get("voltage_kv", true_v)
            if abs(reported_v - true_v) > 10.0:
                return {
                    "consistent": False,
                    "violation_node": node_id,
                    "estimated_true_mw": true_gen,
                    "reported_mw": reported_gen,
                }

        return {
            "consistent": True,
            "violation_node": None,
            "estimated_true_mw": 0.0,
        }

    def quarantine_node(self, node_id: str) -> Dict[str, Any]:
        """Quarantine a node's SCADA sensor — replace with estimator values."""
        if node_id not in self.nodes:
            return {"success": False, "error": f"Unknown node: {node_id}"}

        self.nodes[node_id]["quarantined"] = True
        return {"success": True, "node_id": node_id}

    def inject_counter_signal(
        self, node_id: str, hz_offset: float, duration: int
    ) -> Dict[str, Any]:
        """
        Inject a counter-signal via adjacent battery at given frequency offset.

        The counter-signal creates destructive interference when hz_offset
        matches the attack frequency (within ±0.05Hz tolerance).
        """
        if node_id not in self.nodes:
            return {"success": False, "error": f"Unknown node: {node_id}"}

        node = self.nodes[node_id]
        if node["node_type"] != "battery":
            return {"success": False, "error": f"Node {node_id} is not a battery"}

        # Calculate effectiveness based on offset accuracy
        # Perfect offset = -0.5Hz (to counter +0.5Hz attack)
        target_offset = -0.5
        accuracy = 1.0 - min(abs(hz_offset - target_offset) / 0.5, 1.0)

        power_injection = node["capacity_mw"] * 0.5 * accuracy  # Half capacity at full accuracy

        self._counter_signals.append({
            "node_id": node_id,
            "hz_offset": hz_offset,
            "duration": duration,
            "remaining_ticks": duration,
            "accuracy": accuracy,
            "power_injection_mw": power_injection,
        })

        return {
            "success": True,
            "accuracy": accuracy,
            "power_injection_mw": power_injection,
        }

    # ------------------------------------------------------------------
    # Topology and telemetry getters
    # ------------------------------------------------------------------

    def get_topology(self) -> Dict[str, Any]:
        """Get the topology graph as a serializable dict."""
        nodes = []
        for n in self.nodes.values():
            nodes.append({
                "id": n["id"],
                "region": n["region"],
                "node_type": n["node_type"],
                "capacity_mw": n["capacity_mw"],
                "peak_load_mw": n["peak_load_mw"],
                "critical": n["critical"],
                "energized": n["energized"],
            })

        edges = []
        for e in self.edges.values():
            edges.append({
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "capacity_mw": e["capacity_mw"],
                "current_load_mw": e["current_load_mw"],
                "status": e["status"],
            })

        return {"nodes": nodes, "edges": edges}

    def get_ground_truth_telemetry(self) -> Dict[str, Dict[str, Any]]:
        """Get the true (unspoofed) telemetry for all nodes."""
        result = {}
        for n in self.nodes.values():
            result[n["id"]] = {
                "node_id": n["id"],
                "voltage_kv": n["voltage_kv"],
                "frequency_hz": self.frequency_hz if n["energized"] else 0.0,
                "generation_mw": n["generation_mw"],
                "consumption_mw": n["consumption_mw"],
            }
        return result

    def get_telemetry_history(self) -> List[List[Dict[str, Any]]]:
        """Get the last 10 ticks of telemetry history."""
        return self.telemetry_history[-10:]

    def get_weather(self) -> List[Dict[str, Any]]:
        """Get weather forecast as list of zone dicts."""
        result = []
        for zone_name, data in self.weather.items():
            result.append({
                "zone": zone_name,
                "solar_irradiance": round(data.get("solar_irradiance", 0.5), 3),
                "wind_speed_ms": round(data.get("wind_speed_ms", 5.0), 1),
                "cloud_cover": round(data.get("cloud_cover", 0.3), 3),
            })
        return result

    def get_weather_summary(self) -> str:
        """Get a natural language weather summary."""
        parts = []
        for zone_name, data in self.weather.items():
            solar = data.get("solar_irradiance", 0.5)
            wind = data.get("wind_speed_ms", 5.0)
            cloud = data.get("cloud_cover", 0.3)

            if solar > 0.7:
                sun_desc = "sunny"
            elif solar > 0.3:
                sun_desc = "partly cloudy"
            else:
                sun_desc = "overcast"

            if wind > 15:
                wind_desc = "strong winds"
            elif wind > 8:
                wind_desc = "moderate winds"
            else:
                wind_desc = "calm"

            parts.append(f"Zone {zone_name}: {sun_desc}, {wind_desc}, {cloud*100:.0f}% cloud cover")

        return "; ".join(parts) if parts else "Weather data unavailable"

    def get_total_generation(self) -> float:
        """Get total MW currently being generated."""
        return sum(
            n["generation_mw"] for n in self.nodes.values()
            if n["node_type"] in ("hydro", "solar", "gas", "battery") and n["energized"]
        )

    def get_total_load(self) -> float:
        """Get total MW demand."""
        return sum(
            n["consumption_mw"] for n in self.nodes.values()
            if n["node_type"] == "load" and n["energized"]
        )

    def get_total_possible_mwh(self) -> float:
        """Get total possible MW·h that could be served."""
        return sum(
            n["peak_load_mw"] for n in self.nodes.values()
            if n["node_type"] == "load"
        )

    def get_mwh_served(self) -> float:
        """Get actual MW·h served to loads."""
        return sum(
            n["consumption_mw"] for n in self.nodes.values()
            if n["node_type"] == "load" and n["energized"]
        )

    def get_critical_nodes_shed(self) -> int:
        """Count critical nodes that have been de-energized (load shed)."""
        return sum(
            1 for n in self.nodes.values()
            if n["critical"] and not n["energized"]
        )

    def get_overloaded_edges(self) -> List[str]:
        """Get edges loaded at or above 95% capacity."""
        return [
            e["id"] for e in self.edges.values()
            if e["status"] == "LIVE" and e["current_load_mw"] >= 0.95 * e["capacity_mw"]
        ]

    def is_dispatch_proactive(self) -> bool:
        """Check if any dispatch happened before the first frequency drop."""
        if not self._dispatch_ticks:
            return False
        if self._first_frequency_drop_tick is None:
            return True  # No drop yet, any dispatch is proactive
        return min(self._dispatch_ticks) < self._first_frequency_drop_tick

    def get_stable_islands(self) -> List[List[str]]:
        """
        Find connected components (power islands) of energized nodes.
        Returns list of node-id lists.
        """
        # Build adjacency for live edges between energized nodes
        adj: Dict[str, set] = {n_id: set() for n_id in self.nodes if self.nodes[n_id]["energized"]}
        for e in self.edges.values():
            if e["status"] == "LIVE":
                s, t = e["source"], e["target"]
                if s in adj and t in adj:
                    adj[s].add(t)
                    adj[t].add(s)

        visited: set = set()
        islands: List[List[str]] = []
        for start in adj:
            if start in visited:
                continue
            component: List[str] = []
            stack = [start]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for nb in adj.get(node, []):
                    if nb not in visited:
                        stack.append(nb)
            if component:
                islands.append(sorted(component))

        return islands

    def check_phase_angle_compatible(self, island_a: List[str], island_b: List[str]) -> bool:
        """Check if two islands can be merged (|∆phase| ≤ 5°)."""
        if not island_a or not island_b:
            return False

        # Average phase angle of each island
        avg_a = np.mean([self.nodes[n]["phase_angle_deg"] for n in island_a if n in self.nodes])
        avg_b = np.mean([self.nodes[n]["phase_angle_deg"] for n in island_b if n in self.nodes])

        diff = abs(avg_a - avg_b)
        if diff > 180:
            diff = 360 - diff

        return diff <= 5.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evolve_weather(self) -> None:
        """Evolve weather with small random perturbations."""
        for zone in self.weather:
            w = self.weather[zone]
            # Solar irradiance changes slowly
            w["solar_irradiance"] = float(np.clip(
                w["solar_irradiance"] + self._rng.normal(0, 0.02), 0.0, 1.0
            ))
            # Wind speed fluctuates
            w["wind_speed_ms"] = float(np.clip(
                w["wind_speed_ms"] + self._rng.normal(0, 0.5), 0.0, 30.0
            ))
            # Cloud cover
            w["cloud_cover"] = float(np.clip(
                w["cloud_cover"] + self._rng.normal(0, 0.03), 0.0, 1.0
            ))

    def _update_renewable_generation(self) -> None:
        """Update solar and wind generation based on current weather."""
        for node in self.nodes.values():
            if not node["energized"]:
                continue
            region = node["region"]
            if region not in self.weather:
                continue
            w = self.weather[region]

            if node["node_type"] == "solar":
                # Solar output = capacity * irradiance * (1 - cloud_cover)
                solar_factor = w["solar_irradiance"] * (1.0 - w["cloud_cover"] * 0.7)
                node["generation_mw"] = node["capacity_mw"] * solar_factor

            # Wind can affect frequency stability (not modeled as wind turbines here)

    def _evolve_load(self) -> None:
        """Small random load fluctuations each tick."""
        for node in self.nodes.values():
            if node["node_type"] == "load" and node["energized"]:
                fluctuation = float(self._rng.normal(0, node["peak_load_mw"] * 0.01))
                node["consumption_mw"] = max(
                    0.0,
                    min(node["peak_load_mw"], node["consumption_mw"] + fluctuation)
                )

    def _record_telemetry(self) -> None:
        """Record current telemetry snapshot to history."""
        snapshot = []
        for n in self.nodes.values():
            snapshot.append({
                "node_id": n["id"],
                "voltage_kv": n["voltage_kv"],
                "frequency_hz": self.frequency_hz if n["energized"] else 0.0,
                "generation_mw": n["generation_mw"],
                "consumption_mw": n["consumption_mw"],
            })
        self.telemetry_history.append(snapshot)
        # Keep only last 10
        if len(self.telemetry_history) > 10:
            self.telemetry_history = self.telemetry_history[-10:]

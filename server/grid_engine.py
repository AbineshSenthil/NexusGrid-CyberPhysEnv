"""
NexusGrid — Physical Grid Engine (Layer 1).

A 20-bus transmission network solved with a linear DC power flow
(Kirchhoff B-matrix) on every electrical island.

Physics summary
---------------
* Every energized island is solved independently. Injections are balanced by
  primary frequency response: the setpoint mismatch is shared by governor-
  responsive units (hydro, gas, battery) in proportion to their capacity.
* Line flows come from solving ``B θ = P`` with a reference bus per island,
  so flows obey Kirchhoff's current and voltage laws.
* Island frequency follows the setpoint mismatch through an effective
  stiffness ``K = 0.25 MW/Hz per MW of online dispatchable capacity`` with
  first-order inertia smoothing once per tick.
* Protection relays trip lines that stay above 100 % loading for three ticks
  or exceed 140 % at the end of a tick. Under-frequency load shedding (UFLS)
  acts below 59.2 Hz and an island collapses outside 58.8–61.2 Hz.

All randomness comes from ``numpy.random.Generator(PCG64(seed))`` streams so
identical seeds give identical trajectories.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DISPATCHABLE_TYPES = ("hydro", "gas", "battery")
GENERATOR_TYPES = ("hydro", "solar", "gas", "battery")

DEFAULT_RAMP_MW = {"hydro": 250.0, "gas": 150.0, "battery": None, "solar": None}


class GridEngine:
    """DC power-flow grid simulator with islands, frequency and protection."""

    NOMINAL_FREQUENCY_HZ = 60.0
    STIFFNESS_PER_MW = 0.25         # MW/Hz of stiffness per MW of online dispatchable capacity
    MIN_STIFFNESS = 50.0            # MW/Hz floor for tiny islands
    INERTIA_ALPHA = 0.6             # fraction of the gap to the target closed per tick
    TICK_HOURS = 5.0 / 60.0         # one tick is five simulated minutes

    UFLS_THRESHOLD_HZ = 59.2
    UFLS_STEP = 0.10                # fraction of non-critical demand shed per UFLS stage
    UFLS_MAX = 0.50
    COLLAPSE_LOW_HZ = 58.8
    COLLAPSE_HIGH_HZ = 61.2

    OVERLOAD_TRIP_TICKS = 3         # consecutive ticks above 100 % before the relay trips
    INSTANT_TRIP_LOADING = 1.40     # loading that trips the relay at the end of a tick

    BASE_MVA = 100.0
    REACTANCE_800MW = 0.02          # per-unit reactance of an 800 MW line

    def __init__(self, seed: int = 42):
        self._seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(self._seed))
        self._noise_rng = np.random.Generator(np.random.PCG64(self._seed + 7_919))
        self.tick = 0

        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: Dict[str, Dict[str, Any]] = {}

        # System frequency (the island with the most demand), plus per-island state
        self.frequency_hz: float = self.NOMINAL_FREQUENCY_HZ
        self.frequency_offset_hz: float = 0.0   # external oscillation added on top (attacks)
        self._island_of: Dict[str, int] = {}
        self._islands: List[List[str]] = []
        self._island_target_hz: Dict[int, float] = {}
        self._island_mismatch_mw: Dict[int, float] = {}

        self.telemetry_history: deque = deque(maxlen=10)
        self.weather: Dict[str, Dict[str, float]] = {}
        self.weather_forecast: Dict[str, List[float]] = {}
        self.demand_forecast: List[float] = []
        self.solar_forecast: List[float] = []

        self._counter_signals: List[Dict[str, Any]] = []
        self._dispatch_ticks: List[int] = []
        self._first_frequency_drop_tick: Optional[int] = None
        self.events: List[Dict[str, Any]] = []
        self.event_log: List[Dict[str, Any]] = []
        self.measurement_noise_mw = 0.5

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
        black_start_capable: bool = False,
        ramp_mw_per_tick: Optional[float] = None,
        energy_mwh: Optional[float] = None,
    ) -> None:
        """Add a substation (bus) to the grid."""
        if ramp_mw_per_tick is None:
            ramp_mw_per_tick = DEFAULT_RAMP_MW.get(node_type) or capacity_mw
        is_battery = node_type == "battery"
        max_energy = energy_mwh if energy_mwh is not None else (2.0 * capacity_mw if is_battery else 0.0)
        self.nodes[node_id] = {
            "id": node_id,
            "node_type": node_type,
            "capacity_mw": float(capacity_mw),
            "peak_load_mw": float(peak_load_mw),
            "region": region,
            "critical": bool(critical),
            # Generation: setpoint is what the operator commanded, actual is what flows.
            "setpoint_mw": float(generation_mw),
            "generation_mw": float(generation_mw),
            "available_mw": float(capacity_mw),   # solar availability / derates
            "lost_mw": 0.0,                       # hidden physical loss (attacks, faults)
            "compromised": False,                 # controller under attacker influence
            "ramp_mw_per_tick": float(ramp_mw_per_tick),
            "energy_mwh": float(max_energy * 0.8) if is_battery else 0.0,
            "max_energy_mwh": float(max_energy),
            # Load
            "demand_mw": float(consumption_mw),
            "consumption_mw": float(consumption_mw),
            "base_consumption_mw": float(consumption_mw),
            "shed_mw": 0.0,
            "ufls_fraction": 0.0,
            # Electrical state
            "voltage_kv": 345.0,
            "phase_angle_deg": 0.0,
            "energized": True,
            "quarantined": False,
            "tripped": False,
            "black_start_capable": bool(black_start_capable),
            "frequency_hz": self.NOMINAL_FREQUENCY_HZ,
        }

    def add_edge(
        self,
        edge_id: str,
        source: str,
        target: str,
        capacity_mw: float,
        reactance: Optional[float] = None,
    ) -> None:
        """Add a transmission line. Reactance defaults to scale inversely with rating."""
        if reactance is None:
            reactance = self.REACTANCE_800MW * (800.0 / max(capacity_mw, 1.0))
        self.edges[edge_id] = {
            "id": edge_id,
            "source": source,
            "target": target,
            "capacity_mw": float(capacity_mw),
            "reactance": float(reactance),
            "flow_mw": 0.0,
            "current_load_mw": 0.0,
            "loading": 0.0,
            "status": "LIVE",
            "overload_ticks": 0,
            "out_of_service": False,      # damaged or under maintenance: cannot be closed
            "outage_reason": None,
        }

    # ------------------------------------------------------------------
    # Dispatch helpers
    # ------------------------------------------------------------------

    def balance_dispatch(self, reserve_fraction: float = 0.0) -> None:
        """
        Set dispatchable setpoints so generation matches demand (economic dispatch).

        Solar runs at availability, batteries stay on standby, and the remaining
        demand is shared by hydro and gas units in proportion to capacity.
        ``reserve_fraction`` leaves that share of every unit's capacity unused.
        """
        demand = sum(n["demand_mw"] for n in self.nodes.values() if n["node_type"] == "load" and n["energized"])
        solar = 0.0
        for n in self.nodes.values():
            if n["node_type"] == "solar" and n["energized"]:
                n["setpoint_mw"] = n["capacity_mw"]
                solar += min(n["available_mw"], n["capacity_mw"])
        thermal = [
            n for n in self.nodes.values()
            if n["node_type"] in ("hydro", "gas") and n["energized"] and not n["tripped"]
        ]
        for n in self.nodes.values():
            if n["node_type"] == "battery":
                n["setpoint_mw"] = 0.0
        remaining = max(0.0, demand - solar)
        usable = [(n, n["capacity_mw"] * (1.0 - reserve_fraction) - n["lost_mw"]) for n in thermal]
        total_usable = sum(max(u, 0.0) for _, u in usable)
        for n, usable_mw in usable:
            share = (max(usable_mw, 0.0) / total_usable) if total_usable > 0 else 0.0
            n["setpoint_mw"] = round(min(max(usable_mw, 0.0), remaining * share), 3)

    def _unit_output_cap(self, node: Dict[str, Any]) -> float:
        """Maximum physical output of a unit right now."""
        if node["node_type"] == "solar":
            return max(0.0, min(node["available_mw"], node["capacity_mw"]))
        if node["node_type"] == "battery" and node["energy_mwh"] <= 1e-6:
            return 0.0
        return node["capacity_mw"]

    # ------------------------------------------------------------------
    # Core physics
    # ------------------------------------------------------------------

    def _build_islands(self) -> List[List[str]]:
        """Connected components of the LIVE-line graph (all buses, energized or not)."""
        adjacency: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            if edge["status"] == "LIVE":
                adjacency[edge["source"]].append(edge["target"])
                adjacency[edge["target"]].append(edge["source"])
        visited: set = set()
        components: List[List[str]] = []
        for start in self.nodes:
            if start in visited:
                continue
            stack, comp = [start], []
            while stack:
                cur = stack.pop()
                if cur in visited:
                    continue
                visited.add(cur)
                comp.append(cur)
                stack.extend(nb for nb in adjacency[cur] if nb not in visited)
            components.append(sorted(comp))
        return components

    def _recompute_energization(self, components: List[List[str]]) -> List[str]:
        """A component is energized if it holds a running generator or a black-start unit."""
        previously = {nid: n["energized"] for nid, n in self.nodes.items()}
        newly: List[str] = []
        for comp in components:
            powered = any(
                self.nodes[nid]["node_type"] in GENERATOR_TYPES
                and not self.nodes[nid]["tripped"]
                and (self.nodes[nid]["setpoint_mw"] > 0.0 or self.nodes[nid]["black_start_capable"])
                and previously[nid]
                for nid in comp
            ) or any(
                self.nodes[nid]["black_start_capable"] and previously[nid] for nid in comp
            )
            for nid in comp:
                node = self.nodes[nid]
                was = previously[nid]
                node["energized"] = bool(powered)
                if node["energized"] and not was:
                    newly.append(nid)
                    if node["node_type"] == "load":
                        # Cold-load pickup: demand returns at its base value.
                        node["demand_mw"] = max(node["demand_mw"], node["base_consumption_mw"])
                        node["ufls_fraction"] = 0.0
                if not node["energized"]:
                    node["voltage_kv"] = 0.0
                    node["phase_angle_deg"] = 0.0
                    if node["node_type"] != "load":
                        node["generation_mw"] = 0.0
                    else:
                        node["consumption_mw"] = 0.0
                else:
                    node["voltage_kv"] = 345.0
        return newly

    def _island_stiffness(self, comp: List[str]) -> float:
        cap = sum(
            self.nodes[nid]["capacity_mw"]
            for nid in comp
            if self.nodes[nid]["node_type"] in DISPATCHABLE_TYPES
            and self.nodes[nid]["energized"]
            and not self.nodes[nid]["tripped"]
        )
        return max(self.MIN_STIFFNESS, self.STIFFNESS_PER_MW * cap)

    def compute_power_flow(self) -> List[str]:
        """
        Solve every island: energization, primary response, DC flows, angles.

        Returns the list of buses that became energized during this solve.
        """
        components = self._build_islands()
        newly = self._recompute_energization(components)

        self._islands = [comp for comp in components if self.nodes[comp[0]]["energized"]]
        self._island_of = {}
        self._island_target_hz = {}
        self._island_mismatch_mw = {}

        for edge in self.edges.values():
            edge["flow_mw"] = 0.0
            edge["current_load_mw"] = 0.0
            edge["loading"] = 0.0

        for idx, comp in enumerate(self._islands):
            for nid in comp:
                self._island_of[nid] = idx
            self._solve_island(idx, comp)
        return newly

    def _solve_island(self, idx: int, comp: List[str]) -> None:
        nodes = self.nodes
        # Loads consume demand minus manual shedding and UFLS.
        demand_total = 0.0
        for nid in comp:
            n = nodes[nid]
            if n["node_type"] == "load":
                served = max(0.0, n["demand_mw"] - n["shed_mw"]) * (1.0 - n["ufls_fraction"])
                n["consumption_mw"] = served
                demand_total += served

        # Setpoint-side generation (what the operators commanded, limited by physics).
        set_total = 0.0
        responsive: List[str] = []
        for nid in comp:
            n = nodes[nid]
            if n["node_type"] not in GENERATOR_TYPES or n["tripped"]:
                continue
            cap = self._unit_output_cap(n)
            n["setpoint_mw"] = max(0.0, min(n["setpoint_mw"], n["capacity_mw"]))
            effective = max(0.0, min(n["setpoint_mw"], cap) - n["lost_mw"])
            n["generation_mw"] = effective
            set_total += effective
            if n["node_type"] in DISPATCHABLE_TYPES and cap > 0 and not n["compromised"] and n["lost_mw"] <= 0:
                responsive.append(nid)

        mismatch = set_total - demand_total
        stiffness = self._island_stiffness(comp)
        self._island_mismatch_mw[idx] = mismatch
        self._island_target_hz[idx] = self.NOMINAL_FREQUENCY_HZ + mismatch / stiffness

        # Primary frequency response: responsive units absorb the mismatch.
        remaining = -mismatch
        active = list(responsive)
        for _ in range(6):
            if abs(remaining) < 1e-6 or not active:
                break
            weight = sum(nodes[nid]["capacity_mw"] for nid in active)
            next_active = []
            spent = 0.0
            for nid in active:
                n = nodes[nid]
                share = remaining * n["capacity_mw"] / weight
                new_out = n["generation_mw"] + share
                cap = self._unit_output_cap(n)
                clipped = min(max(new_out, 0.0), cap)
                spent += clipped - n["generation_mw"]
                n["generation_mw"] = clipped
                if 0.0 < clipped < cap:
                    next_active.append(nid)
            remaining -= spent
            active = next_active
        if remaining > 1e-6 and demand_total > 0:
            # Not enough upward response: the shortfall becomes unserved load (brownout).
            scale = max(0.0, (demand_total - remaining) / demand_total)
            for nid in comp:
                if nodes[nid]["node_type"] == "load":
                    nodes[nid]["consumption_mw"] *= scale
        elif remaining < -1e-6:
            # Surplus that governors cannot absorb: solar plants curtail automatically.
            solar = [nid for nid in comp if nodes[nid]["node_type"] == "solar" and nodes[nid]["generation_mw"] > 0]
            solar_total = sum(nodes[nid]["generation_mw"] for nid in solar)
            if solar_total > 0:
                cut = min(solar_total, -remaining)
                for nid in solar:
                    nodes[nid]["generation_mw"] -= cut * nodes[nid]["generation_mw"] / solar_total

        # DC power flow on the island.
        live_edges = [
            e for e in self.edges.values()
            if e["status"] == "LIVE" and e["source"] in comp and e["target"] in comp
        ]
        if len(comp) < 2 or not live_edges:
            nodes[comp[0]]["phase_angle_deg"] = 0.0
            return
        index = {nid: i for i, nid in enumerate(comp)}
        size = len(comp)
        B = np.zeros((size, size))
        for e in live_edges:
            i, j = index[e["source"]], index[e["target"]]
            b = 1.0 / e["reactance"]
            B[i, i] += b
            B[j, j] += b
            B[i, j] -= b
            B[j, i] -= b
        P = np.zeros(size)
        for nid in comp:
            n = nodes[nid]
            inj = n["generation_mw"] if n["node_type"] in GENERATOR_TYPES else -n["consumption_mw"]
            P[index[nid]] = inj / self.BASE_MVA
        P -= P.sum() / size  # remove numerical residue so the system is consistent
        ref = max(
            comp,
            key=lambda nid: (nodes[nid]["node_type"] in DISPATCHABLE_TYPES, nodes[nid]["capacity_mw"], nid),
        )
        keep = [index[nid] for nid in comp if nid != ref]
        theta = np.zeros(size)
        try:
            theta_red = np.linalg.solve(B[np.ix_(keep, keep)], P[keep])
        except np.linalg.LinAlgError:
            theta_red = np.linalg.lstsq(B[np.ix_(keep, keep)], P[keep], rcond=None)[0]
        theta[keep] = theta_red
        for nid in comp:
            deg = math.degrees(theta[index[nid]])
            nodes[nid]["phase_angle_deg"] = float(((deg + 180.0) % 360.0) - 180.0)
        for e in live_edges:
            flow_pu = (theta[index[e["source"]]] - theta[index[e["target"]]]) / e["reactance"]
            flow = float(flow_pu * self.BASE_MVA)
            e["flow_mw"] = flow
            e["current_load_mw"] = abs(flow)
            e["loading"] = abs(flow) / max(e["capacity_mw"], 1.0)

    def compute_frequency(self, smooth: bool = True) -> float:
        """
        Move each island's frequency toward its target (inertia) and update
        the system frequency, which is the frequency of the island serving
        the most demand.
        """
        main_idx = self.main_island_index()
        for idx, comp in enumerate(self._islands):
            target = self._island_target_hz.get(idx, self.NOMINAL_FREQUENCY_HZ)
            # Previous frequency: capacity-weighted over buses that were already running.
            weights, values = [], []
            for nid in comp:
                f_prev = self.nodes[nid]["frequency_hz"]
                if f_prev > 1.0:
                    weights.append(max(self.nodes[nid]["capacity_mw"], 1.0))
                    values.append(f_prev)
            prev = float(np.average(values, weights=weights)) if values else target
            new = prev + self.INERTIA_ALPHA * (target - prev) if smooth else target
            for nid in comp:
                self.nodes[nid]["frequency_hz"] = new
        for nid, n in self.nodes.items():
            if not n["energized"]:
                n["frequency_hz"] = 0.0
        if main_idx is not None:
            base = self.nodes[self._islands[main_idx][0]]["frequency_hz"]
            self.frequency_hz = float(min(62.0, max(58.0, base + self.frequency_offset_hz)))
        else:
            self.frequency_hz = self.NOMINAL_FREQUENCY_HZ
        if self.frequency_hz < 59.8 and self._first_frequency_drop_tick is None:
            self._first_frequency_drop_tick = self.tick
        return self.frequency_hz

    def settle(self) -> None:
        """Solve the network and snap island frequencies to their targets (used at reset)."""
        self.compute_power_flow()
        for idx, comp in enumerate(self._islands):
            target = self._island_target_hz.get(idx, self.NOMINAL_FREQUENCY_HZ)
            for nid in comp:
                self.nodes[nid]["frequency_hz"] = target
        self.compute_frequency(smooth=False)

    def advance_tick(self) -> List[Dict[str, Any]]:
        """
        Advance one tick: battery energy, flows, frequency, protection, UFLS,
        island collapse. Returns the events that happened during the tick.
        Scenario schedules (load and solar curves) are applied by the caller
        before this method runs.
        """
        self.tick += 1
        self.events = []
        self._update_batteries()
        self._update_counter_signals()
        self.compute_power_flow()
        self.compute_frequency(smooth=True)
        self._apply_protection()
        self._apply_frequency_protection()
        self.event_log.extend(self.events)
        return list(self.events)

    # ------------------------------------------------------------------
    # Protection and dynamics helpers
    # ------------------------------------------------------------------

    def _update_batteries(self) -> None:
        for n in self.nodes.values():
            if n["node_type"] != "battery" or not n["energized"]:
                continue
            used = n["generation_mw"] * self.TICK_HOURS
            n["energy_mwh"] = max(0.0, n["energy_mwh"] - used)
            if n["energy_mwh"] <= 1e-6 and n["setpoint_mw"] > 0:
                n["setpoint_mw"] = 0.0
                self._event("battery_depleted", node=n["id"])

    def _update_counter_signals(self) -> None:
        for cs in self._counter_signals:
            if cs["remaining_ticks"] > 0:
                cs["remaining_ticks"] -= 1

    def _apply_protection(self) -> None:
        tripped_any = False
        for e in self.edges.values():
            if e["status"] != "LIVE":
                e["overload_ticks"] = 0
                continue
            if e["loading"] > 1.0:
                e["overload_ticks"] += 1
            else:
                e["overload_ticks"] = 0
            if e["loading"] >= self.INSTANT_TRIP_LOADING or e["overload_ticks"] >= self.OVERLOAD_TRIP_TICKS:
                self._event("protection_trip", edge=e["id"], loading=round(e["loading"], 3))
                e["status"] = "TRIPPED"
                e["flow_mw"] = 0.0
                e["current_load_mw"] = 0.0
                e["loading"] = 0.0
                e["overload_ticks"] = 0
                tripped_any = True
        if tripped_any:
            self.compute_power_flow()

    def _apply_frequency_protection(self) -> None:
        changed = False
        for comp in list(self._islands):
            freq = self.nodes[comp[0]]["frequency_hz"]
            if freq < self.COLLAPSE_LOW_HZ or freq > self.COLLAPSE_HIGH_HZ:
                self._collapse_island(comp, freq)
                changed = True
                continue
            if freq < self.UFLS_THRESHOLD_HZ:
                shed_nodes = []
                for nid in comp:
                    n = self.nodes[nid]
                    if n["node_type"] == "load" and not n["critical"] and n["ufls_fraction"] < self.UFLS_MAX:
                        n["ufls_fraction"] = min(self.UFLS_MAX, n["ufls_fraction"] + self.UFLS_STEP)
                        shed_nodes.append(nid)
                if shed_nodes:
                    self._event("ufls", nodes=shed_nodes, frequency_hz=round(freq, 3))
                    changed = True
        if changed:
            self.compute_power_flow()

    def _collapse_island(self, comp: List[str], freq: float) -> None:
        self._event("island_collapse", nodes=list(comp), frequency_hz=round(freq, 3))
        for nid in comp:
            n = self.nodes[nid]
            if n["node_type"] in GENERATOR_TYPES:
                n["setpoint_mw"] = 0.0
                n["generation_mw"] = 0.0
            n["energized"] = bool(n["black_start_capable"])
            n["frequency_hz"] = self.NOMINAL_FREQUENCY_HZ if n["energized"] else 0.0
        for e in self.edges.values():
            if e["status"] == "LIVE" and e["source"] in comp and e["target"] in comp:
                e["status"] = "TRIPPED"

    def set_out_of_service(self, edge_id: str, reason: str) -> None:
        """Open a line and lock it out (storm damage, maintenance)."""
        edge = self.edges[edge_id]
        edge["status"] = "TRIPPED"
        edge["out_of_service"] = True
        edge["outage_reason"] = reason

    def _event(self, kind: str, **data: Any) -> None:
        self.events.append({"type": kind, "tick": self.tick, **data})

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def dispatch_generation(self, node_id: str, mw: float) -> Dict[str, Any]:
        """Change a unit's setpoint by ``mw`` (ramp-limited)."""
        if node_id not in self.nodes:
            return {"success": False, "error": f"Unknown node: {node_id}"}
        node = self.nodes[node_id]
        if node["node_type"] not in GENERATOR_TYPES:
            return {"success": False, "error": f"Node {node_id} is not a generator (use shed_load for loads)"}
        if not node["energized"]:
            return {"success": False, "error": f"Node {node_id} is not energized"}
        if node["tripped"]:
            return {"success": False, "error": f"Node {node_id} has tripped offline"}
        if node["quarantined"]:
            return {"success": False, "error": f"Node {node_id} control channel is quarantined"}
        mw = float(mw)
        ramp = node["ramp_mw_per_tick"]
        clipped = max(-ramp, min(ramp, mw))
        old = node["setpoint_mw"]
        new = max(0.0, min(node["capacity_mw"], old + clipped))
        node["setpoint_mw"] = new
        self._dispatch_ticks.append(self.tick)
        self.compute_power_flow()
        return {
            "success": True,
            "old_setpoint_mw": round(old, 3),
            "new_setpoint_mw": round(new, 3),
            "new_generation_mw": round(new, 3),
            "ramp_limited": abs(clipped - mw) > 1e-9,
        }

    def shed_load(self, node_id: str, mw: float) -> Dict[str, Any]:
        """
        Shed (mw > 0) or restore (mw < 0) demand at a load.

        Restoring first returns manually shed MW, then load that under-frequency
        relays (UFLS) disconnected. Critical loads cannot be shed manually.
        """
        if node_id not in self.nodes:
            return {"success": False, "error": f"Unknown node: {node_id}"}
        node = self.nodes[node_id]
        if node["node_type"] != "load":
            return {"success": False, "error": f"Node {node_id} is not a load (use dispatch_generation for generators)"}
        mw = float(mw)
        if mw > 0 and node["critical"]:
            return {"success": False, "error": f"Node {node_id} is critical infrastructure and cannot be shed"}
        old_shed, old_ufls = node["shed_mw"], node["ufls_fraction"]
        if mw >= 0:
            node["shed_mw"] = max(0.0, min(node["demand_mw"], old_shed + mw))
        else:
            restore = -mw
            from_manual = min(restore, node["shed_mw"])
            node["shed_mw"] -= from_manual
            restore -= from_manual
            if restore > 0 and node["ufls_fraction"] > 0 and node["demand_mw"] > 0:
                node["ufls_fraction"] = max(0.0, node["ufls_fraction"] - restore / node["demand_mw"])
        self.compute_power_flow()
        return {
            "success": True,
            "old_shed_mw": round(old_shed, 3),
            "new_shed_mw": round(node["shed_mw"], 3),
            "old_ufls_pct": round(100.0 * old_ufls, 1),
            "new_ufls_pct": round(100.0 * node["ufls_fraction"], 1),
        }

    def toggle_circuit_breaker(self, edge_id: str, status: str) -> Dict[str, Any]:
        """Open or close a line. Closing between islands requires synchronism."""
        if edge_id not in self.edges:
            return {"success": False, "error": f"Unknown edge: {edge_id}"}
        status = str(status).upper()
        if status not in ("OPEN", "CLOSED"):
            return {"success": False, "error": f"Invalid status: {status}. Use OPEN or CLOSED."}
        edge = self.edges[edge_id]
        old_status = edge["status"]
        pre_islands = self.get_stable_islands()
        result: Dict[str, Any] = {"success": True, "old_status": old_status}
        if status == "OPEN":
            edge["status"] = "TRIPPED"
            edge["overload_ticks"] = 0
        else:
            if edge["out_of_service"]:
                result.update({"success": False, "new_status": edge["status"],
                               "error": f"{edge_id} is out of service ({edge['outage_reason']}) and cannot be closed"})
                return result
            sync = self.check_synchronism(edge["source"], edge["target"])
            result["sync_check"] = sync
            if sync["merging"] and not sync["in_sync"]:
                # Out-of-phase closure: the breaker recloses open and damages equipment.
                self._event("out_of_phase_close", edge=edge_id, **{k: v for k, v in sync.items() if k != "merging"})
                result.update({"new_status": edge["status"], "out_of_phase": True,
                               "error": "Breaker reopened: islands were not synchronised"})
                result["success"] = False
                return result
            edge["status"] = "LIVE"
        newly = self.compute_power_flow()
        post_islands = self.get_stable_islands()
        result.update({
            "new_status": edge["status"],
            "newly_energized_nodes": newly,
            "island_count_before": len(pre_islands),
            "island_count_after": len(post_islands),
        })
        return result

    def check_synchronism(self, a: str, b: str) -> Dict[str, Any]:
        """Synchronism check across a breaker between buses ``a`` and ``b``."""
        ia, ib = self._island_of.get(a), self._island_of.get(b)
        if ia is None or ib is None or ia == ib:
            return {"merging": False, "in_sync": True}
        fa = self.nodes[a]["frequency_hz"]
        fb = self.nodes[b]["frequency_hz"]
        da = abs(self.nodes[a]["phase_angle_deg"] - self.nodes[b]["phase_angle_deg"])
        da = min(da, 360.0 - da)
        in_sync = abs(fa - fb) <= 0.20 and da <= 20.0
        return {"merging": True, "in_sync": in_sync, "delta_hz": round(abs(fa - fb), 4), "delta_deg": round(da, 3)}

    def run_state_estimation(self, subgraph: List[str], reported: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Kirchhoff current-law consistency check.

        For every bus in ``subgraph`` the injection implied by the metered line
        flows is compared with the injection reported by SCADA telemetry
        (generation - consumption). Residuals above 15 MW are violations.
        """
        residuals: Dict[str, float] = {}
        worst: Tuple[Optional[str], float] = (None, 0.0)
        for nid in subgraph:
            n = self.nodes.get(nid)
            if n is None or not n["energized"]:
                continue
            implied = 0.0
            for e in self.edges.values():
                if e["status"] != "LIVE":
                    continue
                if e["source"] == nid:
                    implied += e["flow_mw"]
                elif e["target"] == nid:
                    implied -= e["flow_mw"]
            rep = reported.get(nid, {})
            rep_inj = float(rep.get("generation_mw", 0.0)) - float(rep.get("consumption_mw", 0.0))
            if self._island_size(nid) < 2:
                true_inj = (n["generation_mw"] if n["node_type"] in GENERATOR_TYPES else -n["consumption_mw"])
                implied = true_inj
            residual = rep_inj - implied
            residuals[nid] = round(residual, 2)
            if abs(residual) > abs(worst[1]):
                worst = (nid, residual)
        violation = worst[0] if abs(worst[1]) > 15.0 else None
        if violation is None:
            return {"consistent": True, "violation_node": None, "estimated_true_mw": 0.0,
                    "residual_mw": round(worst[1], 2), "residuals": residuals, "checked_nodes": list(residuals)}
        n = self.nodes[violation]
        reported_mw = float(reported.get(violation, {}).get("generation_mw", n["generation_mw"]))
        return {
            "consistent": False,
            "violation_node": violation,
            "estimated_true_mw": round(n["generation_mw"], 2),
            "reported_mw": round(reported_mw, 2),
            "residual_mw": round(worst[1], 2),
            "residuals": residuals,
            "checked_nodes": list(residuals),
        }

    def _island_size(self, nid: str) -> int:
        idx = self._island_of.get(nid)
        return len(self._islands[idx]) if idx is not None else 0

    def quarantine_node(self, node_id: str) -> Dict[str, Any]:
        """Isolate a bus's SCADA channel; its telemetry is replaced by estimator values."""
        if node_id not in self.nodes:
            return {"success": False, "error": f"Unknown node: {node_id}"}
        if self.nodes[node_id]["quarantined"]:
            return {"success": False, "error": f"Node {node_id} is already quarantined", "already_quarantined": True}
        self.nodes[node_id]["quarantined"] = True
        return {"success": True, "node_id": node_id, "already_quarantined": False}

    def inject_counter_signal(self, node_id: str, hz_offset: float, duration: int) -> Dict[str, Any]:
        """Start a damping counter-signal from a battery (effect handled by the attack model)."""
        if node_id not in self.nodes:
            return {"success": False, "error": f"Unknown node: {node_id}"}
        node = self.nodes[node_id]
        if node["node_type"] != "battery":
            return {"success": False, "error": f"Node {node_id} is not a battery"}
        if not node["energized"]:
            return {"success": False, "error": f"Node {node_id} is not energized"}
        if node["energy_mwh"] <= 1e-6:
            return {"success": False, "error": f"Battery {node_id} is depleted"}
        signal = {
            "node_id": node_id,
            "hz_offset": float(hz_offset),
            "duration": int(duration),
            "remaining_ticks": int(duration),
            "start_tick": self.tick,
        }
        self._counter_signals.append(signal)
        return {"success": True, "node_id": node_id, "hz_offset": float(hz_offset), "duration": int(duration)}

    def active_counter_signals(self) -> List[Dict[str, Any]]:
        return [cs for cs in self._counter_signals if cs["remaining_ticks"] > 0]

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------

    def hop_distance(self, a: str, b: str, live_only: bool = True) -> int:
        """Number of line hops between two buses (large number if disconnected)."""
        if a == b:
            return 0
        adjacency: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for e in self.edges.values():
            if live_only and e["status"] != "LIVE":
                continue
            adjacency[e["source"]].append(e["target"])
            adjacency[e["target"]].append(e["source"])
        frontier, seen, dist = [a], {a}, 0
        while frontier:
            dist += 1
            nxt = []
            for cur in frontier:
                for nb in adjacency[cur]:
                    if nb == b:
                        return dist
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        return 99

    def main_island_index(self) -> Optional[int]:
        best, best_demand = None, -1.0
        for idx, comp in enumerate(self._islands):
            demand = sum(self.nodes[nid]["consumption_mw"] for nid in comp if self.nodes[nid]["node_type"] == "load")
            gen = sum(self.nodes[nid]["generation_mw"] for nid in comp if self.nodes[nid]["node_type"] in GENERATOR_TYPES)
            key = demand + 1e-3 * gen
            if key > best_demand:
                best, best_demand = idx, key
        return best

    # ------------------------------------------------------------------
    # Getters
    # ------------------------------------------------------------------

    def get_topology(self) -> Dict[str, Any]:
        """Operator's single-line view (never spoofed)."""
        nodes = []
        for n in self.nodes.values():
            item = {
                "id": n["id"],
                "region": n["region"],
                "node_type": n["node_type"],
                "capacity_mw": n["capacity_mw"],
                "peak_load_mw": n["peak_load_mw"],
                "critical": n["critical"],
                "energized": n["energized"],
                "black_start_capable": n["black_start_capable"],
                "quarantined": n["quarantined"],
            }
            if n["node_type"] in GENERATOR_TYPES:
                item["setpoint_mw"] = round(n["setpoint_mw"], 2)
                item["ramp_mw_per_tick"] = n["ramp_mw_per_tick"]
                item["tripped"] = n["tripped"]
            if n["node_type"] == "battery":
                item["soc_pct"] = round(100.0 * n["energy_mwh"] / n["max_energy_mwh"], 1) if n["max_energy_mwh"] else 0.0
            if n["node_type"] == "load":
                item["shed_mw"] = round(n["shed_mw"], 2)
                item["ufls_shed_pct"] = round(100.0 * n["ufls_fraction"], 1)
            nodes.append(item)
        edges = []
        for e in self.edges.values():
            edges.append({
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "capacity_mw": e["capacity_mw"],
                "reactance_pu": round(e["reactance"], 5),
                "current_load_mw": round(min(e["current_load_mw"], 5000.0), 2),
                "flow_mw": round(e["flow_mw"], 2),
                "loading_pct": round(100.0 * e["loading"], 1),
                "status": e["status"],
                "out_of_service": e["out_of_service"],
            })
        return {"nodes": nodes, "edges": edges}

    def get_ground_truth_telemetry(self) -> Dict[str, Dict[str, Any]]:
        """True SCADA readings with small measurement noise (deterministic per seed)."""
        result = {}
        for n in self.nodes.values():
            noise_g = float(self._noise_rng.normal(0.0, self.measurement_noise_mw))
            noise_c = float(self._noise_rng.normal(0.0, self.measurement_noise_mw))
            noise_f = float(self._noise_rng.normal(0.0, 0.002))
            gen = n["generation_mw"] if n["node_type"] in GENERATOR_TYPES else 0.0
            cons = n["consumption_mw"] if n["node_type"] == "load" else 0.0
            energized = n["energized"]
            freq = n["frequency_hz"] + self.frequency_offset_hz if energized else 0.0
            result[n["id"]] = {
                "node_id": n["id"],
                "voltage_kv": 345.0 if energized else 0.0,
                "frequency_hz": round(min(62.0, max(0.0, freq + (noise_f if energized else 0.0))), 4),
                "phase_angle_deg": round(n["phase_angle_deg"], 2) if energized else 0.0,
                "generation_mw": round(max(0.0, gen + (noise_g if gen > 0 else 0.0)), 2),
                "consumption_mw": round(max(0.0, cons + (noise_c if cons > 0 else 0.0)), 2),
            }
        return result

    def get_telemetry_history(self) -> List[List[Dict[str, Any]]]:
        return [list(snapshot) for snapshot in self.telemetry_history]

    def push_telemetry_snapshot(self, snapshot: List[Dict[str, Any]]) -> None:
        """Record a (possibly spoofed) telemetry snapshot for the observation history."""
        self.telemetry_history.append(snapshot)

    def _record_telemetry(self) -> None:
        """Kept for compatibility; snapshots are recorded by the environment."""
        return None

    def get_weather(self) -> List[Dict[str, Any]]:
        result = []
        for zone, data in self.weather.items():
            item = {
                "zone": zone,
                "solar_irradiance": round(data.get("solar_irradiance", 0.5), 3),
                "wind_speed_ms": round(data.get("wind_speed_ms", 5.0), 1),
                "cloud_cover": round(data.get("cloud_cover", 0.3), 3),
            }
            if zone in self.weather_forecast:
                item["irradiance_forecast"] = [round(v, 3) for v in self.weather_forecast[zone]]
            result.append(item)
        return result

    def get_weather_summary(self) -> str:
        parts = []
        for zone, data in self.weather.items():
            solar = data.get("solar_irradiance", 0.5)
            wind = data.get("wind_speed_ms", 5.0)
            cloud = data.get("cloud_cover", 0.3)
            sun = "sunny" if solar > 0.7 else ("partly cloudy" if solar > 0.3 else "overcast")
            wind_desc = "strong winds" if wind > 15 else ("moderate winds" if wind > 8 else "calm")
            parts.append(f"Zone {zone}: {sun}, {wind_desc}, {cloud * 100:.0f}% cloud cover")
        return "; ".join(parts) if parts else "Weather data unavailable"

    def get_total_generation(self) -> float:
        return sum(n["generation_mw"] for n in self.nodes.values() if n["node_type"] in GENERATOR_TYPES and n["energized"])

    def get_total_setpoint(self) -> float:
        return sum(
            max(0.0, min(n["setpoint_mw"], self._unit_output_cap(n)) - n["lost_mw"])
            for n in self.nodes.values()
            if n["node_type"] in GENERATOR_TYPES and n["energized"] and not n["tripped"]
        )

    def get_total_load(self) -> float:
        return sum(n["consumption_mw"] for n in self.nodes.values() if n["node_type"] == "load" and n["energized"])

    def get_total_demand(self) -> float:
        """Demand of all loads, energized or not (what customers want)."""
        return sum(n["demand_mw"] if n["energized"] else n["base_consumption_mw"]
                   for n in self.nodes.values() if n["node_type"] == "load")

    def get_total_possible_mwh(self) -> float:
        return sum(n["peak_load_mw"] for n in self.nodes.values() if n["node_type"] == "load")

    def get_mwh_served(self) -> float:
        return self.get_total_load()

    def get_served_fraction(self) -> float:
        demand = self.get_total_demand()
        return self.get_total_load() / demand if demand > 0 else 0.0

    def get_critical_nodes_shed(self) -> int:
        return sum(1 for n in self.nodes.values() if n["critical"] and not n["energized"])

    def get_overloaded_edges(self, threshold: float = 1.0) -> List[str]:
        return [e["id"] for e in self.edges.values() if e["status"] == "LIVE" and e["loading"] > threshold]

    def get_max_loading(self) -> float:
        live = [e["loading"] for e in self.edges.values() if e["status"] == "LIVE"]
        return max(live) if live else 0.0

    def is_dispatch_proactive(self) -> bool:
        if not self._dispatch_ticks:
            return False
        if self._first_frequency_drop_tick is None:
            return True
        return min(self._dispatch_ticks) <= self._first_frequency_drop_tick

    def get_stable_islands(self) -> List[List[str]]:
        """Energized islands (lists of bus IDs)."""
        return [list(comp) for comp in self._islands]

    def check_phase_angle_compatible(self, island_a: List[str], island_b: List[str]) -> bool:
        if not island_a or not island_b:
            return False
        fa = self.nodes[island_a[0]]["frequency_hz"]
        fb = self.nodes[island_b[0]]["frequency_hz"]
        return abs(fa - fb) <= 0.20

    def get_island_summary(self) -> List[Dict[str, Any]]:
        summary = []
        for idx, comp in enumerate(self._islands):
            gen = sum(self.nodes[n]["generation_mw"] for n in comp if self.nodes[n]["node_type"] in GENERATOR_TYPES)
            load = sum(self.nodes[n]["consumption_mw"] for n in comp if self.nodes[n]["node_type"] == "load")
            summary.append({
                "island": idx,
                "nodes": list(comp),
                "frequency_hz": round(self.nodes[comp[0]]["frequency_hz"], 4),
                "generation_mw": round(gen, 2),
                "load_mw": round(load, 2),
                "setpoint_mismatch_mw": round(self._island_mismatch_mw.get(idx, 0.0), 2),
            })
        return summary

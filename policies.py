"""
Baseline policies for NexusGrid.

Every policy acts from the observation alone (no access to hidden simulator
state), so the same code can drive a local environment or a remote server.

* :class:`NoopPolicy`    - always ``advance_tick``; the do-nothing baseline.
* :class:`RandomPolicy`  - a random action type with random arguments.
* :class:`ExpertPolicy`  - a hand-written operator: forecast-aware frequency
  control, DC power-flow checks before every redispatch or switching action,
  evidence-first incident response, oscillation damping and staged
  black-start restoration.

The expert is a reference point for what the scenarios demand, not an upper
bound: it only uses information that is in the observation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from models import ActionType, GridAction
except ImportError:  # pragma: no cover - package-relative import
    from .models import ActionType, GridAction

DISPATCHABLE = ("hydro", "gas", "battery")
GENERATORS = ("hydro", "solar", "gas", "battery")
ACTION_TYPES = [a.value for a in ActionType]


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def observation_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return dict(obs)


class GridView:
    """Convenient read-only view of one observation."""

    def __init__(self, obs: Any):
        d = observation_dict(obs)
        self.raw = d
        topo = d.get("topology_graph") or {}
        self.nodes: Dict[str, Dict[str, Any]] = {n["id"]: n for n in topo.get("nodes", [])}
        self.edges: Dict[str, Dict[str, Any]] = {e["id"]: e for e in topo.get("edges", [])}
        stream = d.get("telemetry_stream") or []
        self.telemetry: Dict[str, Dict[str, Any]] = {r["node_id"]: r for r in (stream[-1] if stream else [])}
        self.frequency = float(d.get("grid_frequency_hz", 60.0))
        self.tick = int(d.get("tick", 0))
        self.max_ticks = int(d.get("max_ticks", 0) or 0)
        self.task_id = int(d.get("task_id", 0))
        self.islands: List[Dict[str, Any]] = d.get("islands") or []
        self.forecast: Dict[str, List[float]] = d.get("forecast") or {}
        self.packets: List[Dict[str, Any]] = d.get("network_packet_logs") or []
        self.last_se: Optional[Dict[str, Any]] = d.get("last_state_estimation")
        self.counter_signals: List[Dict[str, Any]] = d.get("active_counter_signals") or []
        self.island_of = {nid: isl["island"] for isl in self.islands for nid in isl["nodes"]}
        self.adjacency: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        for e in self.edges.values():
            if e["status"] == "LIVE":
                self.adjacency[e["source"]].append(e["target"])
                self.adjacency[e["target"]].append(e["source"])

    def main_island(self) -> Optional[Dict[str, Any]]:
        if not self.islands:
            return None
        return max(self.islands, key=lambda isl: (isl["load_mw"], isl["generation_mw"], len(isl["nodes"])))

    def stiffness(self, island: Dict[str, Any], extra_capacity: float = 0.0) -> float:
        cap = sum(self.nodes[n]["capacity_mw"] for n in island["nodes"]
                  if self.nodes[n]["node_type"] in DISPATCHABLE and not self.nodes[n].get("tripped"))
        return max(50.0, 0.25 * (cap + extra_capacity))

    def hop_distance(self, a: str, b: str) -> int:
        if a == b:
            return 0
        seen, frontier, dist = {a}, [a], 0
        while frontier:
            dist += 1
            nxt = []
            for cur in frontier:
                for nb in self.adjacency.get(cur, []):
                    if nb == b:
                        return dist
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        return 99

    def dispatchable_units(self, island_nodes: Iterable[str], exclude: Set[str] = frozenset()) -> List[str]:
        units = []
        for nid in island_nodes:
            n = self.nodes[nid]
            if (n["node_type"] in DISPATCHABLE and n.get("energized") and not n.get("tripped")
                    and not n.get("quarantined") and nid not in exclude and n["capacity_mw"] > 0):
                units.append(nid)
        return units

    def latest_packets(self) -> List[Dict[str, Any]]:
        if not self.packets:
            return []
        last = max(int(p["timestamp"]) // 300 for p in self.packets)
        return [p for p in self.packets if int(p["timestamp"]) // 300 == last]

    def served_mw(self, node_id: str) -> float:
        return float(self.telemetry.get(node_id, {}).get("consumption_mw", 0.0))


def _reactance(edge: Dict[str, Any]) -> float:
    x = edge.get("reactance_pu")
    if not x:
        x = 0.02 * 800.0 / max(float(edge["capacity_mw"]), 1.0)
    return max(float(x), 1e-6)


def dc_flows(nodes: Sequence[str], edges: Sequence[Dict[str, Any]], injections: Dict[str, float]) -> Dict[str, float]:
    """Solve a DC power flow on a connected set of buses; returns MW per edge (source -> target)."""
    if len(nodes) < 2 or not edges:
        return {e["id"]: 0.0 for e in edges}
    index = {nid: i for i, nid in enumerate(nodes)}
    size = len(nodes)
    B = np.zeros((size, size))
    for e in edges:
        i, j = index[e["source"]], index[e["target"]]
        b = 1.0 / _reactance(e)
        B[i, i] += b
        B[j, j] += b
        B[i, j] -= b
        B[j, i] -= b
    P = np.array([injections.get(nid, 0.0) for nid in nodes], dtype=float)
    P -= P.sum() / size
    theta = np.zeros(size)
    try:
        theta[1:] = np.linalg.solve(B[1:, 1:], P[1:])
    except np.linalg.LinAlgError:
        theta[1:] = np.linalg.lstsq(B[1:, 1:], P[1:], rcond=None)[0]
    return {e["id"]: float((theta[index[e["source"]]] - theta[index[e["target"]]]) / _reactance(e)) for e in edges}


class FlowModel:
    """
    DC power-flow predictor for one island, rebuilt from the observed topology.

    Net bus injections are recovered from the metered line flows, so predictions
    use only what the operator can see. Governors redistribute any imbalance in
    proportion to the capacity of responsive units, as the simulator does.
    """

    def __init__(self, view: GridView, island_nodes: Iterable[str]):
        self.view = view
        self.nodes = list(island_nodes)
        members = set(self.nodes)
        self.edges = [e for e in view.edges.values()
                      if e["status"] == "LIVE" and e["source"] in members and e["target"] in members]
        self.p0 = {nid: 0.0 for nid in self.nodes}
        for e in self.edges:
            self.p0[e["source"]] += e["flow_mw"]
            self.p0[e["target"]] -= e["flow_mw"]
        self.flows = {e["id"]: float(e["flow_mw"]) for e in self.edges}

    def _weights(self, nodes: Iterable[str], imbalance: float) -> Dict[str, float]:
        """Governor participation: units that can still move in the required direction, by capacity."""
        weights = {}
        for nid in nodes:
            n = self.view.nodes[nid]
            if n["node_type"] not in DISPATCHABLE or n.get("tripped") or n["capacity_mw"] <= 0:
                continue
            if not n.get("energized", True) and nid in self.p0:
                continue
            output = float(self.view.telemetry.get(nid, {}).get("generation_mw", n.get("setpoint_mw", 0.0)))
            if imbalance > 0 and output <= 1.0 and nid in self.p0:
                continue  # surplus: units already at zero cannot back down further
            if imbalance < 0:
                if n["node_type"] == "battery" and n.get("soc_pct", 100.0) <= 0.5:
                    continue
                if n["capacity_mw"] - output <= 1.0:
                    continue  # deficit: units at their limit cannot pick up more
            weights[nid] = float(n["capacity_mw"])
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    def predict(self, deltas: Optional[Dict[str, float]] = None, close: Sequence[str] = (),
                new_buses: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Flows after setpoint changes, line closures and newly energised buses (MW injections)."""
        deltas = deltas or {}
        new_buses = new_buses or {}
        nodes = self.nodes + [b for b in new_buses if b not in self.p0]
        edges = self.edges + [self.view.edges[eid] for eid in close]
        injections = dict(self.p0)
        imbalance = 0.0
        for nid, mw in list(deltas.items()) + list(new_buses.items()):
            injections[nid] = injections.get(nid, 0.0) + mw
            imbalance += mw
        weights = self._weights(nodes, imbalance)
        for nid, w in weights.items():
            injections[nid] = injections.get(nid, 0.0) - imbalance * w
        if not deltas and not new_buses and not close:
            return dict(self.flows)
        return dc_flows(nodes, edges, injections)

    def loading(self, flows: Dict[str, float]) -> Dict[str, float]:
        return {eid: abs(f) / max(float(self.view.edges[eid]["capacity_mw"]), 1.0) for eid, f in flows.items()}

    def max_loading(self, flows: Optional[Dict[str, float]] = None) -> float:
        loads = self.loading(self.flows if flows is None else flows)
        return max(loads.values()) if loads else 0.0

    def overload_mw(self, flows: Dict[str, float], margin: float = 0.97) -> float:
        total = 0.0
        for eid, f in flows.items():
            cap = float(self.view.edges[eid]["capacity_mw"])
            total += max(0.0, abs(f) - margin * cap)
        return total


def advance() -> GridAction:
    return GridAction(action_type=ActionType.ADVANCE_TICK)


def dispatch(node_id: str, mw: float) -> GridAction:
    return GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id=node_id, mw=round(float(mw), 1))


def shed(node_id: str, mw: float) -> GridAction:
    return GridAction(action_type=ActionType.SHED_LOAD, node_id=node_id, mw=round(float(mw), 1))


def close_line(edge_id: str) -> GridAction:
    return GridAction(action_type=ActionType.TOGGLE_CIRCUIT_BREAKER, edge_id=edge_id, status="CLOSED")


# ---------------------------------------------------------------------------
# Simple baselines
# ---------------------------------------------------------------------------

class NoopPolicy:
    """Always advances time without acting."""

    name = "noop"

    def reset(self, seed: Optional[int] = None) -> None:
        return None

    def act(self, obs: Any) -> GridAction:
        return advance()


class RandomPolicy:
    """Uniformly random action types with random (but well-formed) arguments."""

    name = "random"

    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)

    def act(self, obs: Any) -> GridAction:
        view = GridView(obs)
        rng = self._rng
        nodes = sorted(view.nodes) or ["NODE_01"]
        edges = sorted(view.edges) or ["LINE_01"]
        kind = rng.choice(ACTION_TYPES)
        if kind == "dispatch_generation":
            return dispatch(rng.choice(nodes), rng.uniform(-300.0, 300.0))
        if kind == "shed_load":
            return shed(rng.choice(nodes), rng.uniform(-100.0, 200.0))
        if kind == "toggle_circuit_breaker":
            return GridAction(action_type=ActionType.TOGGLE_CIRCUIT_BREAKER, edge_id=rng.choice(edges),
                              status=rng.choice(["OPEN", "CLOSED"]))
        if kind == "run_state_estimation":
            return GridAction(action_type=ActionType.RUN_STATE_ESTIMATION,
                              subgraph=rng.sample(nodes, k=rng.randint(1, min(6, len(nodes)))))
        if kind == "quarantine_scada_node":
            return GridAction(action_type=ActionType.QUARANTINE_SCADA_NODE, node_id=rng.choice(nodes))
        if kind == "inject_counter_signal":
            return GridAction(action_type=ActionType.INJECT_COUNTER_SIGNAL, node_id=rng.choice(nodes),
                              hz_offset=round(rng.uniform(-1.0, 1.0), 2), duration=rng.randint(1, 6))
        return advance()


# ---------------------------------------------------------------------------
# Expert policy
# ---------------------------------------------------------------------------

@dataclass
class _Candidate:
    action: GridAction
    deltas: Dict[str, float] = field(default_factory=dict)  # setpoint-side injection change per bus
    mismatch_delta: float = 0.0                              # change of island setpoint mismatch
    shed_delta: float = 0.0                                  # change of manually shed MW
    cost: float = 0.0                                        # preference cost
    close: Tuple[str, ...] = ()                              # lines closed by the action


class ExpertPolicy:
    """
    Observation-only reference operator.

    Frequency control works like an operator running AGC with the published
    forecasts: it drives the island's setpoint mismatch (generation setpoints
    minus demand) to zero for the next tick. Every candidate action is checked
    with a DC power flow so redispatch and switching never create overloads.
    """

    name = "expert"

    ACTION_COST = 3.0
    OVERLOAD_WEIGHT = 10.0
    FREQUENCY_WEIGHT = 500.0     # per Hz of predicted deviation
    SHED_WEIGHT = 2.0            # per MW of manually shed load
    BATTERY_COST = 0.05          # per MW: keep batteries for fast or emergency support

    def __init__(self) -> None:
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self._task: Optional[int] = None
        self._prev_demand_forecast: Optional[float] = None
        self._prev_solar_forecast: Optional[float] = None
        self._initial_setpoints: Dict[str, float] = {}
        self._se_calls = 0
        self._cleared: Dict[str, int] = {}
        self._dispatched_once = False

    def act(self, obs: Any) -> GridAction:
        view = GridView(obs)
        if self._task != view.task_id or view.tick == 0:
            self.reset()
            self._task = view.task_id
            self._initial_setpoints = {nid: float(n.get("setpoint_mw", 0.0)) for nid, n in view.nodes.items()}
        handlers = {
            0: self._smoke_test,
            1: self._operate,
            2: self._operate,
            3: self._phantom,
            4: self._resonance,
            5: self._black_start,
        }
        action = handlers.get(view.task_id, self._operate)(view)
        demand = view.forecast.get("system_demand_mw") or []
        solar = view.forecast.get("solar_available_mw") or []
        self._prev_demand_forecast = float(demand[0]) if demand else None
        self._prev_solar_forecast = float(solar[0]) if solar else None
        return action

    # ------------------------------------------------------------------
    # Shared: forecast-aware, flow-checked frequency control
    # ------------------------------------------------------------------

    def _net_change_next_tick(self, view: GridView) -> float:
        """Expected change of the setpoint mismatch at the next tick from the forecasts."""
        demand = view.forecast.get("system_demand_mw") or []
        solar = view.forecast.get("solar_available_mw") or []
        change = 0.0
        if demand:
            if self._prev_demand_forecast is not None:
                now = self._prev_demand_forecast
            else:
                now = sum(view.served_mw(nid) + n.get("shed_mw", 0.0)
                          for nid, n in view.nodes.items() if n["node_type"] == "load" and n.get("energized"))
            change -= float(demand[0]) - now
        if solar:
            if self._prev_solar_forecast is not None:
                now = self._prev_solar_forecast
            else:
                now = sum(view.telemetry.get(nid, {}).get("generation_mw", 0.0)
                          for nid, n in view.nodes.items() if n["node_type"] == "solar")
            change += float(solar[0]) - now
        return change

    def _operate(self, view: GridView, exclude: Set[str] = frozenset(), island: Optional[Dict[str, Any]] = None,
                 allow_shed: bool = True, include_forecast: bool = True,
                 extra: Optional[List[_Candidate]] = None) -> GridAction:
        island = island or view.main_island()
        if island is None:
            return advance()
        nodes = island["nodes"]
        model = FlowModel(view, nodes)
        stiffness = view.stiffness(island)
        mismatch_next = float(island.get("setpoint_mismatch_mw", 0.0))
        if include_forecast:
            mismatch_next += self._net_change_next_tick(view)
        loads = [nid for nid in nodes if view.nodes[nid]["node_type"] == "load"]
        shed_total = sum(view.nodes[nid].get("shed_mw", 0.0) for nid in loads)

        candidates = [_Candidate(advance())]
        for unit in view.dispatchable_units(nodes, exclude):
            n = view.nodes[unit]
            ramp = float(n.get("ramp_mw_per_tick") or n["capacity_mw"])
            setpoint = float(n.get("setpoint_mw", 0.0))
            up = max(0.0, min(ramp, n["capacity_mw"] - setpoint))
            down = max(0.0, min(ramp, setpoint))
            if n["node_type"] == "battery" and n.get("soc_pct", 0.0) < 20.0:
                up = 0.0
            amounts = {round(min(up, max(-down, -mismatch_next)), 1)}
            for frac in (1.0, 0.5, 0.25):
                amounts.add(round(up * frac, 1))
                amounts.add(round(-down * frac, 1))
            for mw in amounts:
                if abs(mw) < 1.0:
                    continue
                cost = self.ACTION_COST + (self.BATTERY_COST * abs(mw) if n["node_type"] == "battery" else 0.0)
                candidates.append(_Candidate(dispatch(unit, mw), {unit: mw}, mw, 0.0, cost))

        overloaded = model.overload_mw(model.flows, margin=1.0) > 0.0
        if allow_shed:
            for nid in loads:
                n = view.nodes[nid]
                served = view.served_mw(nid)
                if overloaded and not n.get("critical") and served > 1.0:
                    for frac in (0.25, 0.5, 1.0):
                        mw = round(served * frac, 1)
                        candidates.append(_Candidate(shed(nid, mw), {nid: mw}, mw, mw, self.ACTION_COST))
                ufls = n.get("ufls_shed_pct", 0.0) / 100.0
                restorable = n.get("shed_mw", 0.0) + (served * ufls / max(1e-6, 1.0 - ufls))
                if restorable > 1.0:
                    mw = round(restorable, 1)
                    candidates.append(_Candidate(shed(nid, -mw), {nid: -mw}, -mw,
                                                 -min(mw, n.get("shed_mw", 0.0)), self.ACTION_COST - 2.0))
        if extra:
            candidates.extend(extra)

        def objective(c: _Candidate) -> float:
            flows = model.predict(c.deltas, close=c.close) if (c.deltas or c.close) else model.flows
            freq_dev = abs(mismatch_next + c.mismatch_delta) / stiffness
            return (self.OVERLOAD_WEIGHT * model.overload_mw(flows) + self.FREQUENCY_WEIGHT * freq_dev
                    + self.SHED_WEIGHT * (shed_total + c.shed_delta) + c.cost)

        return min(candidates, key=objective).action

    # ------------------------------------------------------------------
    # Task 0
    # ------------------------------------------------------------------

    def _smoke_test(self, view: GridView) -> GridAction:
        if not self._dispatched_once:
            island = view.main_island()
            if island:
                units = [u for u in view.dispatchable_units(island["nodes"]) if view.nodes[u]["node_type"] != "battery"]
                if units:
                    unit = max(units, key=lambda u: view.nodes[u]["capacity_mw"] - view.nodes[u].get("setpoint_mw", 0.0))
                    self._dispatched_once = True
                    return dispatch(unit, 40.0)
        return self._operate(view)

    # ------------------------------------------------------------------
    # Task 3: false data injection
    # ------------------------------------------------------------------

    def _phantom(self, view: GridView) -> GridAction:
        quarantined = {nid for nid, n in view.nodes.items() if n.get("quarantined")}
        se = view.last_se or {}
        violation = se.get("violation_node")
        if not se.get("consistent", True) and violation and violation not in quarantined:
            return GridAction(action_type=ActionType.QUARANTINE_SCADA_NODE, node_id=violation)

        flagged = [p["source_node"] for p in view.latest_packets() if p.get("anomaly_flag")]
        suspects = [nid for nid in dict.fromkeys(flagged)
                    if nid not in quarantined and view.tick - self._cleared.get(nid, -99) > 3]
        if suspects and self._se_calls < 8:
            subgraph = suspects[:6]
            for nid in suspects:
                for nb in view.adjacency.get(nid, []):
                    if len(subgraph) >= 6:
                        break
                    if nb not in subgraph and view.nodes[nb]["node_type"] in ("hydro", "gas"):
                        subgraph.append(nb)
            self._se_calls += 1
            for nid in subgraph:
                self._cleared[nid] = view.tick
            return GridAction(action_type=ActionType.RUN_STATE_ESTIMATION, subgraph=subgraph)
        return self._operate(view, exclude=set(flagged) | quarantined)

    # ------------------------------------------------------------------
    # Task 4: forced oscillation
    # ------------------------------------------------------------------

    def _resonance(self, view: GridView) -> GridAction:
        osc = {nid: r.get("oscillation_amplitude_hz", 0.0) for nid, r in view.telemetry.items()
               if r.get("oscillation_amplitude_hz", 0.0) > 0.0}
        if not osc:
            return self._operate(view)
        source = max(osc, key=osc.get)
        mode = float(view.telemetry[source].get("oscillation_hz", 0.0))
        matched = any(abs(cs["hz_offset"] + mode) <= 0.05 and cs.get("remaining_ticks", 0) >= 1
                      and view.hop_distance(cs["node_id"], source) <= 2 for cs in view.counter_signals)
        if not matched and osc[source] >= 0.02:
            batteries = [nid for nid, n in view.nodes.items()
                         if n["node_type"] == "battery" and n.get("energized") and n.get("soc_pct", 0.0) > 5.0
                         and view.hop_distance(nid, source) <= 2]
            if batteries:
                battery = min(batteries, key=lambda b: (view.hop_distance(b, source), -view.nodes[b]["capacity_mw"]))
                return GridAction(action_type=ActionType.INJECT_COUNTER_SIGNAL, node_id=battery,
                                  hz_offset=round(-mode, 3), duration=4)
        node = view.nodes[source]
        initial = self._initial_setpoints.get(source, float(node.get("setpoint_mw", 0.0)))
        if node["node_type"] in ("hydro", "gas") and initial > 0 and node.get("setpoint_mw", 0.0) > 0.78 * initial:
            ramp = float(node.get("ramp_mw_per_tick") or node["capacity_mw"])
            return dispatch(source, -min(ramp, node["setpoint_mw"] - 0.75 * initial))
        return self._operate(view, exclude={source})

    # ------------------------------------------------------------------
    # Task 5: black start
    # ------------------------------------------------------------------

    def _black_start(self, view: GridView) -> GridAction:
        if not view.islands:
            return advance()
        anchors = [isl for isl in view.islands if any(view.nodes[n].get("black_start_capable") for n in isl["nodes"])]
        island = max(anchors or view.islands, key=lambda isl: len(isl["nodes"]))
        members = set(island["nodes"])
        model = FlowModel(view, island["nodes"])
        stiffness = view.stiffness(island)
        mismatch = float(island.get("setpoint_mismatch_mw", 0.0))
        freq = float(island.get("frequency_hz", 60.0))
        target_hz = 60.0 + mismatch / stiffness

        total_peak = sum(n.get("peak_load_mw", 0.0) for n in view.nodes.values() if n["node_type"] == "load")
        demand_fc = view.forecast.get("system_demand_mw") or []
        load_factor = (float(demand_fc[0]) / total_peak) if demand_fc and total_peak else 0.75

        frontier: List[Tuple[str, str]] = []
        mesh: List[str] = []
        merges: List[Tuple[str, str, str]] = []
        for e in view.edges.values():
            if e["status"] == "LIVE" or e.get("out_of_service"):
                continue
            a, b = e["source"], e["target"]
            a_in, b_in = a in members, b in members
            if a_in and b_in:
                mesh.append(e["id"])
            elif a_in or b_in:
                inside, outside = (a, b) if a_in else (b, a)
                if not view.nodes[outside].get("energized"):
                    frontier.append((e["id"], outside))
                else:
                    merges.append((e["id"], inside, outside))

        mesh_candidates = [_Candidate(close_line(eid), close=(eid,), cost=self.ACTION_COST) for eid in mesh]

        # 0. Relieve overloads first (redispatch or re-meshing, checked with a power flow).
        if model.max_loading() > 0.95:
            return self._operate(view, island=island, allow_shed=False, include_forecast=False, extra=mesh_candidates)

        # 1. Frequency safety.
        if target_hz < 59.75 or freq < 59.6 or target_hz > 60.45 or freq > 60.45:
            pickup = self._pickup(view, model, frontier, mismatch, stiffness, load_factor, min_target=59.8)
            if target_hz > 60.2 and pickup is not None:
                return close_line(pickup[0])
            return self._operate(view, island=island, allow_shed=False, include_forecast=False)

        # 2. Energise adjacent dark generators: more stiffness and more units, no new load.
        gens = [(eid, nid) for eid, nid in frontier if view.nodes[nid]["node_type"] in ("hydro", "gas")]
        if gens:
            return close_line(max(gens, key=lambda item: view.nodes[item[1]]["capacity_mw"])[0])

        # 3. Pick up loads (critical first) once generation can carry them.
        loads = [(eid, nid) for eid, nid in frontier if view.nodes[nid]["node_type"] == "load"]
        loads.sort(key=lambda item: (not view.nodes[item[1]].get("critical", False),
                                     -view.nodes[item[1]].get("peak_load_mw", 0.0)))
        if loads:
            pickup = self._pickup(view, model, loads, mismatch, stiffness, load_factor, min_target=59.7)
            if pickup is not None:
                return close_line(pickup[0])
            eid, nid = loads[0]
            demand = view.nodes[nid].get("peak_load_mw", 0.0) * load_factor * 1.04
            room = 0.40 * stiffness - mismatch
            if room > 20.0:
                raise_action = self._raise_for_pickup(view, model, island, eid, nid, demand, min(room, demand - mismatch))
                if raise_action is not None:
                    return raise_action
            best_mesh = self._best_mesh(model, mesh, (eid,), {nid: -demand})
            if best_mesh is not None:
                return close_line(best_mesh)

        # 4. Energise remaining buses (solar plants, batteries) that lead further into the dark grid.
        rest = [(eid, nid) for eid, nid in frontier if view.nodes[nid]["node_type"] in ("solar", "battery")]
        if rest:
            return close_line(rest[0][0])

        # 5. Merge other energised islands when synchronised.
        for eid, inside, outside in merges:
            if self._in_sync(view, inside, outside):
                return close_line(eid)

        # 6. Re-mesh the energised network when it lowers the peak loading.
        best_mesh = self._best_mesh(model, mesh)
        if best_mesh is not None and not loads:
            return close_line(best_mesh)

        # 7. Frequency control on the main island (also restores UFLS-shed load).
        return self._operate(view, island=island, include_forecast=False)

    def _pickup(self, view: GridView, model: FlowModel, loads: List[Tuple[str, str]], mismatch: float,
                stiffness: float, load_factor: float, min_target: float) -> Optional[Tuple[str, str]]:
        for eid, nid in loads:
            if view.nodes[nid]["node_type"] != "load":
                continue
            demand = view.nodes[nid].get("peak_load_mw", 0.0) * load_factor * 1.04
            if 60.0 + (mismatch - demand) / stiffness < min_target:
                continue
            flows = model.predict(close=(eid,), new_buses={nid: -demand})
            if model.max_loading(flows) <= 0.90:
                return eid, nid
        return None

    def _raise_for_pickup(self, view: GridView, model: FlowModel, island: Dict[str, Any], eid: str, nid: str,
                          demand: float, amount: float) -> Optional[GridAction]:
        best, best_key = None, None
        for unit in view.dispatchable_units(island["nodes"]):
            n = view.nodes[unit]
            if n["node_type"] not in ("hydro", "gas"):
                continue
            mw = min(float(n.get("ramp_mw_per_tick") or n["capacity_mw"]), n["capacity_mw"] - n.get("setpoint_mw", 0.0),
                     amount)
            if mw < 10.0:
                continue
            flows = model.predict(deltas={unit: mw}, close=(eid,), new_buses={nid: -demand})
            key = (round(model.max_loading(flows), 2), -mw)
            if best_key is None or key < best_key:
                best, best_key = dispatch(unit, mw), key
        return best

    def _best_mesh(self, model: FlowModel, mesh: List[str], close: Tuple[str, ...] = (),
                   new_buses: Optional[Dict[str, float]] = None) -> Optional[str]:
        base = model.max_loading(model.predict(close=close, new_buses=new_buses)) if (close or new_buses) \
            else model.max_loading()
        best, best_loading = None, base - 0.02
        for eid in mesh:
            loading = model.max_loading(model.predict(close=close + (eid,), new_buses=new_buses))
            if loading < best_loading and loading <= 0.95:
                best, best_loading = eid, loading
        return best

    @staticmethod
    def _in_sync(view: GridView, a: str, b: str) -> bool:
        ta, tb = view.telemetry.get(a, {}), view.telemetry.get(b, {})
        if not ta or not tb:
            return False
        delta_angle = abs(ta.get("phase_angle_deg", 0.0) - tb.get("phase_angle_deg", 0.0))
        delta_angle = min(delta_angle, 360.0 - delta_angle)
        return abs(ta.get("frequency_hz", 0.0) - tb.get("frequency_hz", 0.0)) <= 0.15 and delta_angle <= 15.0


POLICIES = {
    "noop": NoopPolicy,
    "random": RandomPolicy,
    "expert": ExpertPolicy,
}


def make_policy(name: str, seed: int = 0):
    """Create a baseline policy by name (``noop``, ``random`` or ``expert``)."""
    if name == "random":
        return RandomPolicy(seed)
    if name not in POLICIES:
        raise ValueError(f"Unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name]()


__all__ = [
    "ExpertPolicy",
    "FlowModel",
    "GridView",
    "NoopPolicy",
    "RandomPolicy",
    "dc_flows",
    "make_policy",
    "observation_dict",
]

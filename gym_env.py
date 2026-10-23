"""
Gymnasium wrapper for reinforcement-learning agents.

* Observation: a fixed-size ``float32`` feature vector built only from the
  :class:`GridObservation` (topology, telemetry, forecasts, PMU and SCADA
  signals). Hidden scenario parameters are never used.
* Action: ``Discrete(237)`` over a fixed catalogue (advance, dispatch steps per
  unit, load shedding, breaker switching, state estimation around each bus,
  quarantine, counter-signals) with action masks for ``sb3-contrib``'s
  ``MaskablePPO``.
* Reward: the environment's dense reward (potential-based task progress plus
  frequency and safety terms), so episode return tracks the graded score.

Training seeds are drawn below 10 000; seeds from 10 000 up are held out.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - gymnasium is an optional dependency
    gym = None
    spaces = None

from models import ActionType, GridAction
from policies import DISPATCHABLE, GridView
from server.nexusgrid_environment import NexusgridEnvironment
from server.scenarios import EVAL_SEED_START, MAX_TICKS

NODE_IDS = [f"NODE_{i:02d}" for i in range(1, 21)]
EDGE_IDS = [f"LINE_{i:02d}" for i in range(1, 41)]
NODE_TYPES = ("hydro", "solar", "gas", "battery", "load")
DISPATCH_UNITS = ["NODE_01", "NODE_02", "NODE_04", "NODE_06", "NODE_08", "NODE_09",
                  "NODE_13", "NODE_14", "NODE_16", "NODE_17", "NODE_20"]
BATTERIES = ["NODE_04", "NODE_08", "NODE_16", "NODE_20"]
SHEDDABLE_LOADS = ["NODE_07", "NODE_12", "NODE_19"]
DISPATCH_STEPS = (-250.0, -150.0, -50.0, 50.0, 150.0, 250.0)
COUNTER_MODES = tuple(round(0.30 + 0.05 * i, 2) for i in range(11))
SE_NEIGHBOURHOODS = {
    # each bus plus its neighbours in the full network (at most six buses)
}


def _base_neighbourhoods() -> Dict[str, List[str]]:
    from server.grid_engine import GridEngine
    from server.scenarios import build_base_topology

    engine = GridEngine(seed=0)
    build_base_topology(engine)
    adjacency: Dict[str, List[str]] = {nid: [] for nid in engine.nodes}
    for e in engine.edges.values():
        adjacency[e["source"]].append(e["target"])
        adjacency[e["target"]].append(e["source"])
    return {nid: [nid] + sorted(nbs)[:5] for nid, nbs in adjacency.items()}


SE_NEIGHBOURHOODS.update(_base_neighbourhoods())


class ActionCatalog:
    """Fixed discrete action set shared by every task."""

    def __init__(self) -> None:
        entries: List[Tuple[str, Tuple[Any, ...]]] = [("advance", ())]
        entries += [("dispatch", (u, d)) for u in DISPATCH_UNITS for d in DISPATCH_STEPS]
        entries += [("shed", (n, kind)) for n in SHEDDABLE_LOADS for kind in ("shed25", "restore")]
        entries += [("breaker", (e, s)) for e in EDGE_IDS for s in ("OPEN", "CLOSED")]
        entries += [("estimate", (n,)) for n in NODE_IDS]
        entries += [("quarantine", (n,)) for n in NODE_IDS]
        entries += [("counter", (b, m)) for b in BATTERIES for m in COUNTER_MODES]
        self.entries = entries
        self.index = {entry: i for i, entry in enumerate(entries)}

    def __len__(self) -> int:
        return len(self.entries)

    def describe(self, idx: int) -> str:
        kind, args = self.entries[idx]
        return f"{kind}({', '.join(str(a) for a in args)})"

    def to_action(self, idx: int, view: GridView) -> GridAction:
        kind, args = self.entries[int(idx)]
        if kind == "dispatch":
            return GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id=args[0], mw=args[1])
        if kind == "shed":
            node = view.nodes.get(args[0], {})
            served = view.served_mw(args[0])
            if args[1] == "shed25":
                return GridAction(action_type=ActionType.SHED_LOAD, node_id=args[0], mw=round(max(1.0, 0.25 * served), 1))
            ufls = node.get("ufls_shed_pct", 0.0) / 100.0
            restorable = node.get("shed_mw", 0.0) + served * ufls / max(1e-6, 1.0 - ufls)
            return GridAction(action_type=ActionType.SHED_LOAD, node_id=args[0], mw=-round(max(1.0, restorable), 1))
        if kind == "breaker":
            return GridAction(action_type=ActionType.TOGGLE_CIRCUIT_BREAKER, edge_id=args[0], status=args[1])
        if kind == "estimate":
            return GridAction(action_type=ActionType.RUN_STATE_ESTIMATION, subgraph=SE_NEIGHBOURHOODS[args[0]])
        if kind == "quarantine":
            return GridAction(action_type=ActionType.QUARANTINE_SCADA_NODE, node_id=args[0])
        if kind == "counter":
            return GridAction(action_type=ActionType.INJECT_COUNTER_SIGNAL, node_id=args[0], hz_offset=-args[1], duration=4)
        return GridAction(action_type=ActionType.ADVANCE_TICK)

    def mask(self, view: GridView) -> np.ndarray:
        mask = np.zeros(len(self.entries), dtype=bool)
        violation = None
        se = view.last_se or {}
        if not se.get("consistent", True):
            violation = se.get("violation_node")
        for i, (kind, args) in enumerate(self.entries):
            if kind == "advance":
                ok = True
            elif kind == "dispatch":
                n = view.nodes.get(args[0], {})
                usable = n.get("energized") and not n.get("tripped") and not n.get("quarantined")
                if args[1] > 0:
                    ok = usable and n.get("capacity_mw", 0) - n.get("setpoint_mw", 0) > 1.0
                    if n.get("node_type") == "battery":
                        ok = ok and n.get("soc_pct", 0.0) > 5.0
                else:
                    ok = usable and n.get("setpoint_mw", 0) > 1.0
            elif kind == "shed":
                n = view.nodes.get(args[0], {})
                if args[1] == "shed25":
                    ok = bool(n.get("energized")) and view.served_mw(args[0]) > 1.0
                else:
                    ok = n.get("shed_mw", 0.0) > 1.0 or n.get("ufls_shed_pct", 0.0) > 0.0
            elif kind == "breaker":
                e = view.edges.get(args[0], {})
                if args[1] == "OPEN":
                    ok = e.get("status") == "LIVE"
                else:
                    ends = (view.nodes.get(e.get("source"), {}), view.nodes.get(e.get("target"), {}))
                    ok = (e.get("status") != "LIVE" and not e.get("out_of_service")
                          and any(x.get("energized") for x in ends))
            elif kind == "estimate":
                ok = bool(view.nodes.get(args[0], {}).get("energized"))
            elif kind == "quarantine":
                ok = args[0] == violation and not view.nodes.get(args[0], {}).get("quarantined")
            else:  # counter
                n = view.nodes.get(args[0], {})
                ok = bool(n.get("energized")) and n.get("soc_pct", 0.0) > 5.0
            mask[i] = bool(ok)
        return mask

    def nearest(self, action: GridAction, view: GridView) -> int:
        """Closest catalogue entry to an arbitrary action (used to imitate the expert)."""
        t = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        if t == "dispatch_generation" and action.node_id in DISPATCH_UNITS and action.mw:
            step = min(DISPATCH_STEPS, key=lambda d: (abs(d - action.mw), -abs(d)))
            return self.index[("dispatch", (action.node_id, step))]
        if t == "shed_load" and action.node_id in SHEDDABLE_LOADS and action.mw is not None:
            return self.index[("shed", (action.node_id, "shed25" if action.mw > 0 else "restore"))]
        if t == "toggle_circuit_breaker" and action.edge_id in EDGE_IDS and action.status in ("OPEN", "CLOSED"):
            return self.index[("breaker", (action.edge_id, action.status))]
        if t == "run_state_estimation" and action.subgraph:
            target = action.subgraph[0]
            return self.index[("estimate", (target,))] if target in NODE_IDS else 0
        if t == "quarantine_scada_node" and action.node_id in NODE_IDS:
            return self.index[("quarantine", (action.node_id,))]
        if t == "inject_counter_signal" and action.node_id in BATTERIES and action.hz_offset is not None:
            mode = min(COUNTER_MODES, key=lambda m: abs(m + action.hz_offset))
            return self.index[("counter", (action.node_id, mode))]
        return 0


CATALOG = ActionCatalog()


def featurize(view: GridView) -> np.ndarray:
    """Fixed-size feature vector (float32) from an observation."""
    feats: List[float] = []
    task = [0.0] * 6
    task[view.task_id] = 1.0
    feats += task
    feats.append(view.tick / max(1, view.max_ticks or MAX_TICKS.get(view.task_id, 20)))
    feats.append(float(np.clip((view.frequency - 60.0) / 0.5, -3, 3)))
    feats.append(len(view.islands) / 5.0)
    main = view.main_island()
    if main:
        feats.append(float(np.clip(main["setpoint_mismatch_mw"] / 1000.0, -3, 3)))
        feats.append(view.stiffness(main) / 2000.0)
        feats.append(len(main["nodes"]) / 20.0)
    else:
        feats += [0.0, 0.0, 0.0]
    demand = view.forecast.get("system_demand_mw") or []
    solar = view.forecast.get("solar_available_mw") or []
    served_now = sum(view.served_mw(n) for n, x in view.nodes.items() if x["node_type"] == "load")
    solar_now = sum(view.telemetry.get(n, {}).get("generation_mw", 0.0)
                    for n, x in view.nodes.items() if x["node_type"] == "solar")
    for k in range(3):
        feats.append(float(np.clip(((demand[k] if k < len(demand) else served_now) - served_now) / 500.0, -4, 4)))
    for k in range(3):
        feats.append(float(np.clip(((solar[k] if k < len(solar) else solar_now) - solar_now) / 500.0, -4, 4)))
    se = view.last_se or {}
    feats.append(1.0 if se else 0.0)
    feats.append(1.0 if se and not se.get("consistent", True) else 0.0)
    feats.append(len(view.counter_signals) / 4.0)
    feats.append(1.0 if view.raw.get("last_action_error") else 0.0)

    flagged = {p["source_node"] for p in view.latest_packets() if p.get("anomaly_flag")}
    violation = se.get("violation_node") if se and not se.get("consistent", True) else None
    signal_nodes = {cs["node_id"] for cs in view.counter_signals}
    for nid in NODE_IDS:
        n = view.nodes.get(nid, {})
        r = view.telemetry.get(nid, {})
        onehot = [1.0 if n.get("node_type") == t else 0.0 for t in NODE_TYPES]
        cap = float(n.get("capacity_mw", 0.0))
        feats += onehot
        feats += [
            1.0 if n.get("energized") else 0.0,
            1.0 if n.get("critical") else 0.0,
            1.0 if n.get("quarantined") else 0.0,
            1.0 if n.get("black_start_capable") else 0.0,
            1.0 if n.get("tripped") else 0.0,
            cap / 1500.0,
            float(n.get("setpoint_mw", 0.0)) / 1500.0,
            float(r.get("generation_mw", 0.0)) / 1500.0,
            float(r.get("consumption_mw", 0.0)) / 700.0,
            float(n.get("peak_load_mw", 0.0)) / 700.0,
            float(n.get("shed_mw", 0.0)) / 500.0,
            float(n.get("ufls_shed_pct", 0.0)) / 100.0,
            float(n.get("soc_pct", 0.0)) / 100.0,
            float(n.get("ramp_mw_per_tick", 0.0) or 0.0) / 400.0,
            float(np.clip((r.get("frequency_hz", 60.0) - 60.0) / 0.5, -3, 3)) if n.get("energized") else 0.0,
            float(r.get("oscillation_amplitude_hz", 0.0)) * 10.0,
            float(r.get("oscillation_hz", 0.0)),
            1.0 if nid in flagged else 0.0,
            1.0 if nid == violation else 0.0,
            1.0 if nid in signal_nodes else 0.0,
        ]
    for eid in EDGE_IDS:
        e = view.edges.get(eid, {})
        cap = max(float(e.get("capacity_mw", 1.0)), 1.0)
        live = e.get("status") == "LIVE"
        feats += [
            1.0 if live else 0.0,
            1.0 if e.get("out_of_service") else 0.0,
            float(np.clip(e.get("flow_mw", 0.0) / cap, -2, 2)) if live else 0.0,
            float(np.clip(abs(e.get("flow_mw", 0.0)) / cap, 0, 2)) if live else 0.0,
            cap / 1200.0,
        ]
    return np.asarray(feats, dtype=np.float32)


FEATURE_SIZE = int(featurize(GridView(NexusgridEnvironment().reset(seed=0, task_id=1))).shape[0])


if gym is not None:

    class NexusGridGymEnv(gym.Env):
        """Single-agent Gymnasium environment over one or more NexusGrid tasks."""

        metadata = {"render_modes": []}

        def __init__(self, tasks: Sequence[int] = (1,), seed_range: Tuple[int, int] = (0, EVAL_SEED_START),
                     fixed_seeds: Optional[Sequence[int]] = None):
            super().__init__()
            self.tasks = list(tasks)
            self.seed_range = seed_range
            self.fixed_seeds = list(fixed_seeds) if fixed_seeds else None
            self._fixed_cursor = 0
            self.env = NexusgridEnvironment()
            self.action_space = spaces.Discrete(len(CATALOG))
            self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(FEATURE_SIZE,), dtype=np.float32)
            self._view: Optional[GridView] = None
            self._rng = np.random.default_rng()

        def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            options = options or {}
            task_id = int(options.get("task_id", self._rng.choice(self.tasks)))
            if "episode_seed" in options:
                episode_seed = int(options["episode_seed"])
            elif self.fixed_seeds:
                episode_seed = self.fixed_seeds[self._fixed_cursor % len(self.fixed_seeds)]
                self._fixed_cursor += 1
            else:
                episode_seed = int(self._rng.integers(*self.seed_range))
            obs = self.env.reset(seed=episode_seed, task_id=task_id)
            self._view = GridView(obs)
            return featurize(self._view), {"task_id": task_id, "episode_seed": episode_seed}

        def step(self, action: int):
            grid_action = CATALOG.to_action(int(action), self._view)
            obs = self.env.step(grid_action)
            self._view = GridView(obs)
            info: Dict[str, Any] = {"error": obs.last_action_error}
            if obs.done:
                info["score"] = self.env.get_score()
                info["task_id"] = self.env.task_id
            return featurize(self._view), float(obs.reward), bool(obs.done), False, info

        def action_masks(self) -> np.ndarray:
            return CATALOG.mask(self._view)


class PPOPolicy:
    """Wraps a trained MaskablePPO model as a NexusGrid policy (for scripts/evaluate.py)."""

    name = "ppo"

    def __init__(self, path: str, deterministic: bool = True):
        from sb3_contrib import MaskablePPO

        self.model = MaskablePPO.load(path, device="cpu")
        self.deterministic = deterministic

    def reset(self, seed: Optional[int] = None) -> None:
        return None

    def act(self, obs: Any) -> GridAction:
        view = GridView(obs)
        action, _ = self.model.predict(featurize(view), action_masks=CATALOG.mask(view),
                                       deterministic=self.deterministic)
        return CATALOG.to_action(int(action), view)


class DiscretizedExpert:
    """The expert restricted to the RL action catalogue (upper reference for PPO)."""

    name = "expert_discrete"

    def __init__(self) -> None:
        from policies import ExpertPolicy

        self.expert = ExpertPolicy()

    def reset(self, seed: Optional[int] = None) -> None:
        self.expert.reset(seed)

    def act_index(self, obs: Any) -> int:
        view = GridView(obs)
        idx = CATALOG.nearest(self.expert.act(obs), view)
        return idx if CATALOG.mask(view)[idx] else 0

    def act(self, obs: Any) -> GridAction:
        view = GridView(obs)
        return CATALOG.to_action(self.act_index(obs), view)


__all__ = ["ActionCatalog", "CATALOG", "DISPATCHABLE", "DiscretizedExpert", "FEATURE_SIZE", "NexusGridGymEnv",
           "PPOPolicy", "featurize"]

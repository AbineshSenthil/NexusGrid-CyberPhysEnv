"""
NexusGrid-CyberPhysEnv — Core Environment.

OpenEnv-compliant ``reset()`` / ``step()`` / ``state``. Every ``step`` applies
one operator action and then advances the simulation by one tick (five
minutes): scenario schedule, attack physics, DC power flow, frequency,
protection relays and SCADA telemetry.

Hidden information (attack target, oscillation mode, hidden MW) never appears
in observations. It is used for grading and is available to local tooling
through :meth:`NexusgridEnvironment.debug_state`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:  # pragma: no cover - lightweight fallback without openenv-core
    class Environment:  # type: ignore[no-redef]
        """Fallback base class for local use without openenv-core."""

    class State:  # type: ignore[no-redef]
        def __init__(self, episode_id: str = "", step_count: int = 0, **kwargs: Any):
            self.episode_id = episode_id
            self.step_count = step_count
            for key, value in kwargs.items():
                setattr(self, key, value)

try:
    from openenv.core.env_server.types import EnvironmentMetadata
except ImportError:  # pragma: no cover - older openenv-core releases
    EnvironmentMetadata = None  # type: ignore[assignment]

try:
    from ..models import ActionType, GridAction, GridObservation
except ImportError:
    from models import ActionType, GridAction, GridObservation

from .graders import grade_task
from .grid_engine import GridEngine
from .reward import RewardCalculator
from .rubric import evaluate_task_rubrics
from .scenarios import MAX_TICKS, TASK_DIFFICULTIES, TASK_NAMES, build_scenario
from .spoof_engine import SpoofEngine
from .training_logger import TrainingLogger

ENV_VERSION = "3.0.0"
MAX_ESTIMATION_BUSES = 6
COLLAPSE_TERMINATION_HZ = 59.0

# Forced-oscillation model (Task 4)
DAMPING_CORRECT = 0.40       # amplitude multiplier per tick with a matched counter-signal
DAMPING_PARTIAL = 0.90       # amplitude multiplier with a roughly matched counter-signal
DECAY_NEUTRALIZED = 0.30     # amplitude multiplier once the mode is neutralised
NEUTRALIZE_BELOW_HZ = 0.10   # amplitude below which a detuned unit is neutralised
DETUNE_FRACTION = 0.80       # setpoint must be at or below 80 % of the pre-attack value
COUNTER_SIGNAL_MAX_HOPS = 2
PMU_PROPAGATION = 0.5        # share of the amplitude seen one hop further away


class NexusgridEnvironment(Environment):
    """National power-grid defence under cyber-physical attack."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._episode_id = str(uuid4())
        self._seed = 42
        self._task_id = 0
        self._tick = 0
        self._done = False
        self._max_ticks = MAX_TICKS[0]
        self._engine: Optional[GridEngine] = None
        self._spoof: Optional[SpoofEngine] = None
        self._scenario: Dict[str, Any] = {}
        self._reward_calc = RewardCalculator()
        self._action_history: List[Dict[str, Any]] = []
        self._reset_trackers()
        self._state = State(episode_id=self._episode_id, step_count=0)

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              task_id: Optional[int] = None, **kwargs: Any) -> GridObservation:
        """Start a new episode of ``task_id`` (0-5) drawn from ``seed``."""
        task_id = self._task_id if task_id is None else int(task_id)
        if task_id not in MAX_TICKS:
            raise ValueError(f"task_id must be 0-5, got {task_id}")
        self._seed = 42 if seed is None else int(seed)
        self._task_id = task_id
        self._tick = 0
        self._done = False
        self._episode_id = episode_id or str(uuid4())
        self._action_history = []
        self._reset_trackers()

        self._scenario = build_scenario(task_id, self._seed)
        self._engine = self._scenario["engine"]
        self._max_ticks = int(self._scenario["max_ticks"])
        self._attack = dict(self._scenario.get("attack_config") or {})
        self._spoof = SpoofEngine(self._seed)
        if self._attack:
            self._spoof.configure_attack(self._attack)
        if self._attack.get("type") == "resonance_oscillation":
            self._pre_attack_setpoint = float(self._engine.nodes[self._attack["target_node"]]["setpoint_mw"])

        self._reward_calc = RewardCalculator()
        self._capture_telemetry()
        self._spoof.generate_packet_logs(list(self._engine.nodes), self._tick)
        self._record_histories()
        self._score_prev = self._current_score()
        self._state = self._make_state()
        return self._build_observation()

    def step(self, action: GridAction, timeout_s: Optional[float] = None, **kwargs: Any) -> GridObservation:
        """Apply one action, then advance the simulation by one tick."""
        if self._engine is None:
            return GridObservation(done=True, last_action_error="Environment not initialised. Call reset() first.")
        if self._done:
            return self._build_observation(error="Episode already finished")

        record, error, action_events = self._apply_action(action)
        self._action_history.append(record)

        self._tick += 1
        self._spoof.advance_tick()
        self._scenario["schedule"].apply(self._engine, self._tick)
        attack_events = self._update_attacks()
        physics_events = self._engine.advance_tick()
        self._tick_events = action_events + attack_events + physics_events
        self._all_events.extend(self._tick_events)
        self._capture_telemetry()
        self._spoof.generate_packet_logs(list(self._engine.nodes), self._tick)
        self._record_histories()
        self._check_termination()

        score_now = self._current_score()
        breakdown = self._reward_calc.compute_tick_reward(
            score_now=score_now,
            score_prev=self._score_prev,
            frequency_hz=self._engine.frequency_hz,
            overloaded_count=len(self._engine.get_overloaded_edges(1.0)),
            events=self._tick_events,
            invalid_action=error is not None,
        )
        self._score_prev = score_now
        self._state = self._make_state()
        return self._build_observation(reward=breakdown["total"], error=error, reward_breakdown=breakdown)

    @property
    def state(self) -> State:
        return self._state

    def get_metadata(self):  # type: ignore[override]
        if EnvironmentMetadata is None:
            return {"name": "NexusGrid-CyberPhysEnv", "version": ENV_VERSION}
        return EnvironmentMetadata(
            name="NexusGrid-CyberPhysEnv",
            description=("Power-grid operations benchmark: DC power flow, protection relays, seeded "
                         "cyber-physical attacks and six graded operator tasks."),
            version=ENV_VERSION,
            author="Abinesh Senthilkumar",
            documentation_url="https://github.com/AbineshSenthil/NexusGrid-CyberPhysEnv",
        )

    def close(self) -> None:  # pragma: no cover - nothing to release
        return None

    # ------------------------------------------------------------------
    # Grading API (in-process)
    # ------------------------------------------------------------------

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def done(self) -> bool:
        return self._done

    @property
    def engine(self) -> Optional[GridEngine]:
        return self._engine

    @property
    def action_history(self) -> List[Dict[str, Any]]:
        return list(self._action_history)

    def get_score(self) -> float:
        """Final grade for the episode so far, treating it as finished."""
        return grade_task(self._task_id, self._action_history, self._build_episode_state(force_done=True))

    def get_rubric_breakdown(self, force_done: bool = False) -> Dict[str, Any]:
        return evaluate_task_rubrics(self._task_id, self._action_history, self._build_episode_state(force_done))

    def get_episode_summary(self) -> Dict[str, Any]:
        rubric_eval = self.get_rubric_breakdown(force_done=True)
        freqs = self._frequency_history or [60.0]
        return {
            "task_id": self._task_id,
            "task_name": TASK_NAMES[self._task_id],
            "seed": self._seed,
            "split": self._scenario.get("split"),
            "score": self.get_score(),
            "rubrics": rubric_eval["rubrics"],
            "weighted_rubric_score": rubric_eval["weighted_score"],
            "total_reward": round(self._reward_calc.total_reward, 6),
            "actions_taken": [a.get("action_type", "unknown") for a in self._action_history],
            "invalid_actions": sum(1 for a in self._action_history if a.get("error")),
            "frequency_min": min(freqs),
            "frequency_max": max(freqs),
            "ticks_used": self._tick,
            "events": [e["type"] for e in self._all_events],
            "done": self._done,
        }

    def emit_training_log(self, episode: int, logger: Optional[TrainingLogger] = None) -> str:
        summary = self.get_episode_summary()
        training_logger = logger or TrainingLogger()
        record = training_logger.build_record(
            episode=episode,
            task_id=self._task_id,
            seed=self._seed,
            score=summary["score"],
            rubrics=summary["rubrics"],
            actions_taken=summary["actions_taken"],
            frequency_min=summary["frequency_min"],
            ticks_used=summary["ticks_used"],
            extra={"task_name": summary["task_name"], "done": summary["done"]},
        )
        return training_logger.write_episode(record)

    def debug_state(self) -> Dict[str, Any]:
        """Hidden scenario details for dashboards and tests (never sent to agents)."""
        return {
            "task_id": self._task_id,
            "seed": self._seed,
            "split": self._scenario.get("split"),
            "params": dict(self._scenario.get("params", {})),
            "attack": dict(self._attack),
            "active_spoofs": self._spoof.get_active_spoofs(self._tick) if self._spoof else [],
            "hidden_loss_mw": round(self._hidden_loss_mw, 3),
            "resonance_amplitude_hz": round(self._resonance_amplitude, 5),
            "attack_neutralized": self._attack_neutralized,
            "turbine_destroyed": self._turbine_destroyed,
            "quarantine_tick": self._quarantine_tick,
            "refused_quarantines": self._refused_quarantines,
        }

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply_action(self, action: GridAction) -> Tuple[Dict[str, Any], Optional[str], List[Dict[str, Any]]]:
        engine = self._engine
        action_type = action.action_type.value if isinstance(action.action_type, ActionType) else str(action.action_type)
        record: Dict[str, Any] = {
            "action_type": action_type,
            "tick": self._tick,
            "node_id": action.node_id,
            "edge_id": action.edge_id,
            "mw": action.mw,
            "status": action.status,
            "subgraph": list(action.subgraph) if action.subgraph else None,
            "hz_offset": action.hz_offset,
            "duration": action.duration,
        }
        events_before = len(engine.events)
        error: Optional[str] = None
        result: Optional[Dict[str, Any]] = None

        if action_type == "dispatch_generation":
            if not action.node_id or action.mw is None:
                error = "dispatch_generation requires node_id and mw"
            else:
                result = engine.dispatch_generation(action.node_id, action.mw)
        elif action_type == "shed_load":
            if not action.node_id or action.mw is None:
                error = "shed_load requires node_id and mw"
            else:
                result = engine.shed_load(action.node_id, action.mw)
        elif action_type == "toggle_circuit_breaker":
            if not action.edge_id or not action.status:
                error = "toggle_circuit_breaker requires edge_id and status"
            else:
                result = engine.toggle_circuit_breaker(action.edge_id, action.status)
        elif action_type == "run_state_estimation":
            subgraph = list(dict.fromkeys(action.subgraph or []))
            if not subgraph:
                error = "run_state_estimation requires subgraph (a list of node IDs)"
            elif len(subgraph) > MAX_ESTIMATION_BUSES:
                error = f"run_state_estimation is limited to {MAX_ESTIMATION_BUSES} buses per call"
            else:
                unknown = [nid for nid in subgraph if nid not in engine.nodes]
                if unknown:
                    error = f"Unknown node(s) in subgraph: {', '.join(unknown)}"
                else:
                    result = engine.run_state_estimation(subgraph, self._reported)
                    self._last_estimation = result
                    if not result.get("consistent", True) and result.get("violation_node"):
                        self._flagged_nodes.add(result["violation_node"])
        elif action_type == "quarantine_scada_node":
            result, error = self._quarantine(action.node_id)
        elif action_type == "inject_counter_signal":
            if not action.node_id or action.hz_offset is None or action.duration is None:
                error = "inject_counter_signal requires node_id, hz_offset and duration"
            else:
                result = engine.inject_counter_signal(action.node_id, action.hz_offset, action.duration)
        elif action_type == "advance_tick":
            result = {"success": True}
        else:
            error = f"Unknown action type: {action_type}"

        if result is not None:
            record["result"] = result
            if not result.get("success", True) and error is None:
                error = result.get("error", "action failed")
        if error:
            record["error"] = error
        action_events = list(engine.events[events_before:])
        engine.event_log.extend(action_events)
        return record, error, action_events

    def _quarantine(self, node_id: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not node_id:
            return None, "quarantine_scada_node requires node_id"
        if node_id not in self._engine.nodes:
            return None, f"Unknown node: {node_id}"
        if node_id not in self._flagged_nodes:
            self._refused_quarantines += 1
            error = (f"Quarantine refused: no state-estimation evidence against {node_id} yet. "
                     "Run run_state_estimation on a subgraph containing it first.")
            return {"success": False, "refused": True, "error": error}, error
        result = self._engine.quarantine_node(node_id)
        if result.get("success"):
            self._spoof.quarantine_node(node_id)
            if node_id == self._attack.get("target_node") and self._quarantine_tick is None:
                self._quarantine_tick = self._tick
                self._engine.nodes[node_id]["compromised"] = False
        return result, None

    # ------------------------------------------------------------------
    # Attack physics
    # ------------------------------------------------------------------

    def _update_attacks(self) -> List[Dict[str, Any]]:
        kind = self._attack.get("type")
        if kind == "phantom_injection":
            return self._update_phantom()
        if kind == "resonance_oscillation":
            return self._update_resonance()
        return []

    def _update_phantom(self) -> List[Dict[str, Any]]:
        """The compromised plant loses real output while SCADA keeps reporting it."""
        attack = self._attack
        target = attack["target_node"]
        node = self._engine.nodes[target]
        if self._tick < int(attack["start_tick"]) or self._quarantine_tick is not None:
            return []
        self._hidden_loss_mw = min(float(attack["max_loss_mw"]),
                                   self._hidden_loss_mw + float(attack["loss_rate_mw"]))
        node["lost_mw"] = self._hidden_loss_mw
        node["compromised"] = True
        self._spoof.set_hidden_mw(target, self._hidden_loss_mw)
        return []

    def _update_resonance(self) -> List[Dict[str, Any]]:
        """Forced oscillation: grows each tick unless damped by a matched counter-signal."""
        attack = self._attack
        engine = self._engine
        target = attack["target_node"]
        node = engine.nodes[target]
        events: List[Dict[str, Any]] = []
        amp = self._resonance_amplitude

        if self._turbine_destroyed or self._attack_neutralized:
            amp *= DECAY_NEUTRALIZED
        elif self._tick >= int(attack["start_tick"]):
            if amp <= 0.0:
                amp = float(attack["initial_amplitude_hz"])
            else:
                quality = self._counter_signal_quality(target, float(attack["mode_hz"]))
                if quality == "correct":
                    amp *= DAMPING_CORRECT
                elif quality == "partial":
                    amp *= DAMPING_PARTIAL
                else:
                    amp *= float(attack["growth"])
                if quality != "none":
                    self._damping_seen = True
            detuned = node["setpoint_mw"] <= DETUNE_FRACTION * self._pre_attack_setpoint
            if self._damping_seen and amp < NEUTRALIZE_BELOW_HZ and detuned and not node["tripped"]:
                self._attack_neutralized = True
                self._neutralized_tick = self._tick
                events.append({"type": "oscillation_neutralized", "tick": self._tick, "node": target})
            elif amp >= float(attack["destruction_amplitude_hz"]):
                self._turbine_destroyed = True
                node["tripped"] = True
                node["setpoint_mw"] = 0.0
                events.append({"type": "turbine_destroyed", "tick": self._tick, "node": target})
        if amp < 1e-4:
            amp = 0.0
        self._resonance_amplitude = amp
        # The oscillation shows up as an alternating deviation of the measured frequency.
        engine.frequency_offset_hz = amp * (1.0 if self._tick % 2 == 0 else -1.0)
        return events

    def _counter_signal_quality(self, target: str, mode_hz: float) -> str:
        best = "none"
        for cs in self._engine.active_counter_signals():
            if self._engine.hop_distance(cs["node_id"], target) > COUNTER_SIGNAL_MAX_HOPS:
                continue
            error = abs(cs["hz_offset"] + mode_hz)
            if error <= 0.05:
                best = "correct"
            elif error <= 0.15 and best != "correct":
                best = "partial"
        rank = {"none": 0, "partial": 1, "correct": 2}
        if rank[best] > rank[self._best_counter_signal]:
            self._best_counter_signal = best
        return best

    # ------------------------------------------------------------------
    # Telemetry and histories
    # ------------------------------------------------------------------

    def _capture_telemetry(self) -> None:
        truth = self._engine.get_ground_truth_telemetry()
        amplitudes = self._pmu_oscillation()
        mode = float(self._attack.get("mode_hz", 0.0))
        for node_id, reading in truth.items():
            amp = amplitudes.get(node_id, 0.0)
            reading["oscillation_hz"] = round(mode, 3) if amp >= 0.005 else 0.0
            reading["oscillation_amplitude_hz"] = round(amp, 4) if amp >= 0.005 else 0.0
        self._reported = self._spoof.apply_spoofs(truth, self._tick)
        self._engine.push_telemetry_snapshot([self._reported[nid] for nid in self._engine.nodes])

    def _pmu_oscillation(self) -> Dict[str, float]:
        """Oscillation amplitude seen by each PMU: strongest at the source, weaker with distance."""
        if self._attack.get("type") != "resonance_oscillation" or self._resonance_amplitude <= 0.0:
            return {}
        target = self._attack["target_node"]
        if self._engine.nodes[target]["tripped"]:
            return {}
        result = {}
        for nid, n in self._engine.nodes.items():
            if not n["energized"]:
                continue
            hops = self._engine.hop_distance(target, nid)
            if hops <= 3:
                result[nid] = self._resonance_amplitude * (PMU_PROPAGATION ** hops)
        return result

    def _record_histories(self) -> None:
        engine = self._engine
        self._frequency_history.append(round(engine.frequency_hz, 5))
        self._served_history.append(round(engine.get_served_fraction(), 5))
        self._max_loading_history.append(round(engine.get_max_loading(), 5))
        if self._tick > 0 and self._task_id != 5 and engine.get_critical_nodes_shed() > 0:
            self._critical_lost_ever = True

    def _check_termination(self) -> None:
        if self._tick >= self._max_ticks:
            self._done = True
            return
        if self._task_id != 5 and self._engine.frequency_hz < COLLAPSE_TERMINATION_HZ:
            self._done = True
            return
        if self._task_id == 5:
            state = self._build_episode_state()
            recent = self._frequency_history[-3:]
            if (state["load_restored_fraction"] >= 0.98 and state["critical_energized_fraction"] >= 1.0
                    and len(recent) == 3 and all(59.5 <= f <= 60.5 for f in recent)):
                self._done = True

    def _current_score(self) -> float:
        return grade_task(self._task_id, self._action_history, self._build_episode_state())

    def _build_episode_state(self, force_done: bool = False) -> Dict[str, Any]:
        engine = self._engine
        if engine is None:
            return {}
        loads = [n for n in engine.nodes.values() if n["node_type"] == "load"]
        critical = [n for n in loads if n["critical"]]
        total_base = sum(n["base_consumption_mw"] for n in loads) or 1.0
        served = sum(n["consumption_mw"] for n in loads if n["energized"])
        target = self._attack.get("target_node")
        target_disconnected = False
        if self._attack.get("type") == "resonance_oscillation" and target and not self._turbine_destroyed:
            live = [e for e in engine.edges.values() if e["status"] == "LIVE" and target in (e["source"], e["target"])]
            zeroed = engine.nodes[target]["setpoint_mw"] < 0.1 * self._pre_attack_setpoint
            target_disconnected = (not live) or zeroed
        return {
            "done": self._done or force_done,
            "task_id": self._task_id,
            "tick": self._tick,
            "max_ticks": self._max_ticks,
            "frequency_history": list(self._frequency_history),
            "served_history": list(self._served_history),
            "max_loading_history": list(self._max_loading_history),
            "events": list(self._all_events),
            "critical_lost_ever": self._critical_lost_ever,
            "critical_energized_fraction": (sum(1 for n in critical if n["energized"]) / len(critical)) if critical else 1.0,
            "load_restored_fraction": min(1.0, served / total_base),
            "attack_target": target,
            "attack_start_tick": int(self._attack.get("start_tick", 0)),
            "refused_quarantines": self._refused_quarantines,
            "best_counter_signal": self._best_counter_signal,
            "turbine_destroyed": self._turbine_destroyed,
            "target_disconnected": target_disconnected,
            "attack_neutralized": self._attack_neutralized,
            "neutralized_tick": self._neutralized_tick,
            "black_start_unit": self._scenario.get("params", {}).get("black_start_unit"),
        }

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _build_observation(self, reward: float = 0.0, error: Optional[str] = None,
                           reward_breakdown: Optional[Dict[str, float]] = None) -> GridObservation:
        engine = self._engine
        if engine is None:
            return GridObservation(done=True, last_action_error="Not initialised")
        rubric = self.get_rubric_breakdown()
        metadata: Dict[str, Any] = {
            "version": ENV_VERSION,
            "task_name": TASK_NAMES[self._task_id],
            "difficulty": TASK_DIFFICULTIES[self._task_id],
            "episode_seed": self._seed,
            "split": self._scenario.get("split"),
            "flagged_nodes": sorted(self._flagged_nodes),
            "quarantined_nodes": self._spoof.get_quarantined() if self._spoof else [],
            "rubric_breakdown": rubric,
        }
        if self._done:
            metadata["episode_summary"] = self.get_episode_summary()
        return GridObservation(
            topology_graph=engine.get_topology(),
            telemetry_stream=engine.get_telemetry_history(),
            weather_forecast_matrix=engine.get_weather(),
            network_packet_logs=self._spoof.get_recent_packet_logs(20) if self._spoof else [],
            grid_frequency_hz=float(min(62.0, max(58.0, engine.frequency_hz))),
            tick=self._tick,
            task_id=self._task_id,
            done=self._done,
            reward=float(reward),
            last_action_error=error,
            last_state_estimation=self._last_estimation,
            weather_summary=engine.get_weather_summary(),
            task_brief=str(self._scenario.get("brief", "")),
            max_ticks=self._max_ticks,
            forecast={
                "system_demand_mw": list(engine.demand_forecast),
                "solar_available_mw": list(engine.solar_forecast),
            },
            islands=engine.get_island_summary(),
            events=list(self._tick_events) if self._tick > 0 else [],
            active_counter_signals=[
                {"node_id": cs["node_id"], "hz_offset": cs["hz_offset"], "remaining_ticks": cs["remaining_ticks"]}
                for cs in engine.active_counter_signals()
            ],
            score=round(max(0.0, min(1.0, self._score_prev)), 6),
            rubric_scores={k: round(v, 4) for k, v in rubric["rubrics"].items()},
            reward_breakdown=dict(reward_breakdown or {}),
            metadata=metadata,
        )

    def _make_state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._tick,
            task_id=self._task_id,
            task_name=TASK_NAMES[self._task_id],
            seed=self._seed,
            split=self._scenario.get("split"),
            tick=self._tick,
            max_ticks=self._max_ticks,
            done=self._done,
            score=round(max(0.0, min(1.0, self._score_prev)), 6),
            total_reward=round(self._reward_calc.total_reward, 6),
        )

    def _reset_trackers(self) -> None:
        self._frequency_history: List[float] = []
        self._served_history: List[float] = []
        self._max_loading_history: List[float] = []
        self._all_events: List[Dict[str, Any]] = []
        self._tick_events: List[Dict[str, Any]] = []
        self._critical_lost_ever = False
        self._flagged_nodes: set = set()
        self._refused_quarantines = 0
        self._last_estimation: Optional[Dict[str, Any]] = None
        self._reported: Dict[str, Dict[str, Any]] = {}
        self._attack: Dict[str, Any] = {}
        self._hidden_loss_mw = 0.0
        self._quarantine_tick: Optional[int] = None
        self._resonance_amplitude = 0.0
        self._pre_attack_setpoint = 0.0
        self._damping_seen = False
        self._attack_neutralized = False
        self._neutralized_tick: Optional[int] = None
        self._turbine_destroyed = False
        self._best_counter_signal = "none"
        self._score_prev = 0.0

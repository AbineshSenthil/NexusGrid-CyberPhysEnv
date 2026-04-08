"""
NexusGrid-CyberPhysEnv — Core Environment Implementation.

Full OpenEnv-compliant environment: step() / reset() / state().
Wires together: GridEngine + SpoofEngine + RewardCalculator + Graders.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import GridAction, GridObservation, ActionType
except ImportError:
    from models import GridAction, GridObservation, ActionType

from .grid_engine import GridEngine
from .scenarios import build_scenario, MAX_TICKS, TASK_NAMES
from .spoof_engine import SpoofEngine
from .reward import RewardCalculator
from .graders import grade_task


class NexusgridEnvironment(Environment):
    """
    NexusGrid-CyberPhysEnv: National power grid defense under cyber-physical attack.

    A 20-node transmission network with DC power flow physics running alongside
    a deterministic SCADA sensor spoofing engine. The agent must distinguish
    real grid failures from adversarially fabricated telemetry.

    OpenEnv API:
        reset(seed) → GridObservation (initial state)
        step(action) → GridObservation (with reward, done, info embedded)
        state → GridState (full internal state for debugging)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize with default state."""
        self._episode_id = str(uuid4())
        self._seed = 42
        self._task_id = 0
        self._tick = 0
        self._done = False
        self._total_reward = 0.0
        self._max_ticks = 3

        # Core engines
        self._engine: Optional[GridEngine] = None
        self._spoof: Optional[SpoofEngine] = None
        self._reward_calc = RewardCalculator()

        # Action history for grading
        self._action_history: List[Dict[str, Any]] = []

        # Frequency history for grading
        self._frequency_history: List[float] = []

        # Spoofed telemetry cache
        self._current_spoofed_telemetry: Dict[str, Dict] = {}

        # Attack config from scenario
        self._attack_config: Dict[str, Any] = {}

        # State estimation tracking
        self._state_estimation_run = False
        self._last_estimation_result: Optional[Dict] = None

        # Black Start tracking
        self._hydro_stable_ticks = 0
        self._max_island_count = 0
        self._successful_mergers = 0
        self._premature_mergers = 0
        self._transformer_failures = 0

        # Track if packet logs have been "read" (agent called advance_tick to observe)
        self._logs_read_this_episode = False

        # Internal OpenEnv state
        self._state = State(episode_id=self._episode_id, step_count=0)

    def reset(self, seed: int = None, task_id: int = None, **kwargs) -> GridObservation:
        """
        Reset the environment for a new episode.

        Args:
            seed: Episode seed for reproducibility. Defaults to 42.
            task_id: Which task to run (0-5). Can also be passed in kwargs.

        Returns:
            Initial GridObservation with all fields freshly computed.
        """
        # Handle task_id from kwargs (OpenEnv may pass it there)
        if task_id is None:
            task_id = kwargs.get("task_id", self._task_id)
        if seed is None:
            seed = kwargs.get("seed", 42)

        self._seed = seed
        self._task_id = task_id
        self._tick = 0
        self._done = False
        self._total_reward = 0.0
        self._episode_id = str(uuid4())
        self._action_history = []
        self._frequency_history = []
        self._state_estimation_run = False
        self._last_estimation_result = None
        self._logs_read_this_episode = False

        # Black Start tracking
        self._hydro_stable_ticks = 0
        self._max_island_count = 0
        self._successful_mergers = 0
        self._premature_mergers = 0
        self._transformer_failures = 0

        # Build scenario
        scenario = build_scenario(task_id, seed)
        self._engine = scenario["engine"]
        self._max_ticks = scenario["max_ticks"]
        self._attack_config = scenario.get("attack_config", {})

        # Initialize spoof engine
        self._spoof = SpoofEngine(seed)
        if self._attack_config:
            self._spoof.configure_attack(self._attack_config)

        # Initialize reward calculator
        self._reward_calc = RewardCalculator()

        # Record initial frequency
        self._frequency_history.append(self._engine.frequency_hz)

        # Generate initial telemetry (may be spoofed)
        ground_truth = self._engine.get_ground_truth_telemetry()
        self._current_spoofed_telemetry = self._spoof.apply_spoofs(ground_truth, self._tick)

        # Update OpenEnv state
        self._state = State(episode_id=self._episode_id, step_count=0)

        return self._build_observation()

    def step(self, action: GridAction) -> GridObservation:
        """
        Execute an action in the environment.

        Returns GridObservation with reward, done, and info embedded.
        """
        if self._done:
            return self._build_observation(error="Episode already finished")

        if self._engine is None:
            return self._build_observation(error="Environment not initialized. Call reset() first.")

        # Parse the action
        action_type = action.action_type.value if isinstance(action.action_type, ActionType) else str(action.action_type)
        action_record = {
            "action_type": action_type,
            "tick": self._tick,
            "node_id": action.node_id,
            "edge_id": action.edge_id,
            "mw": action.mw,
            "status": action.status,
            "subgraph": action.subgraph,
            "hz_offset": action.hz_offset,
            "duration": action.duration,
        }

        error_msg = None
        fault_isolated = False
        spoof_detected = False

        # Execute the action
        if action_type == "dispatch_generation":
            if not action.node_id or action.mw is None:
                error_msg = "dispatch_generation requires node_id and mw"
            else:
                result = self._engine.dispatch_generation(action.node_id, action.mw)
                if not result["success"]:
                    error_msg = result.get("error", "dispatch failed")
                action_record["result"] = result

        elif action_type == "toggle_circuit_breaker":
            if not action.edge_id or not action.status:
                error_msg = "toggle_circuit_breaker requires edge_id and status"
            else:
                result = self._engine.toggle_circuit_breaker(action.edge_id, action.status)
                if not result["success"]:
                    error_msg = result.get("error", "toggle failed")
                else:
                    if action.status == "OPEN":
                        fault_isolated = True
                action_record["result"] = result

        elif action_type == "run_state_estimation":
            if not action.subgraph:
                error_msg = "run_state_estimation requires subgraph (list of node IDs)"
            else:
                result = self._engine.run_state_estimation(
                    action.subgraph,
                    self._current_spoofed_telemetry,
                )
                self._state_estimation_run = True
                self._last_estimation_result = result
                action_record["result"] = result

                if not result.get("consistent", True):
                    spoof_detected = True

        elif action_type == "quarantine_scada_node":
            if not action.node_id:
                error_msg = "quarantine_scada_node requires node_id"
            else:
                # Check anti-hallucination gate
                if not self._state_estimation_run:
                    error_msg = "Must run state_estimation before quarantine (anti-hallucination penalty applies)"

                result = self._engine.quarantine_node(action.node_id)
                self._spoof.quarantine_node(action.node_id)
                action_record["result"] = result

                if self._state_estimation_run and self._last_estimation_result:
                    if not self._last_estimation_result.get("consistent", True):
                        spoof_detected = True

        elif action_type == "inject_counter_signal":
            if not action.node_id or action.hz_offset is None or action.duration is None:
                error_msg = "inject_counter_signal requires node_id, hz_offset, duration"
            else:
                result = self._engine.inject_counter_signal(
                    action.node_id, action.hz_offset, action.duration
                )
                if not result["success"]:
                    error_msg = result.get("error", "injection failed")
                action_record["result"] = result

        elif action_type == "advance_tick":
            # Advance simulation (weather, load, frequency)
            self._engine.advance_tick()
            self._spoof.advance_tick()
            self._logs_read_this_episode = True  # Agent observes packet logs

            # Apply resonance effect if active
            if self._spoof.is_resonance_active():
                resonance_effect = self._spoof.get_resonance_effect(self._tick)
                self._engine.frequency_hz += resonance_effect
                self._engine.frequency_hz = max(58.0, min(62.0, self._engine.frequency_hz))

        else:
            error_msg = f"Unknown action type: {action_type}"

        # Record action
        self._action_history.append(action_record)

        # Advance tick counter
        self._tick += 1

        # Update spoofed telemetry
        ground_truth = self._engine.get_ground_truth_telemetry()
        self._current_spoofed_telemetry = self._spoof.apply_spoofs(ground_truth, self._tick)

        # Generate packet logs
        all_node_ids = list(self._engine.nodes.keys())
        self._spoof.generate_packet_logs(all_node_ids, self._tick)

        # Compute reward
        reward_breakdown = self._reward_calc.compute_tick_reward(
            action_type=action_type,
            action_params={
                "subgraph": action.subgraph or [],
                "node_id": action.node_id,
                "mw": action.mw,
            },
            frequency_hz=self._engine.frequency_hz,
            overloaded_edges=self._engine.get_overloaded_edges(),
            critical_nodes_shed=self._engine.get_critical_nodes_shed(),
            is_proactive=self._engine.is_dispatch_proactive(),
            spoof_detected=spoof_detected,
            fault_isolated=fault_isolated,
            has_read_logs_before_estimation=(
                self._logs_read_this_episode
                and action_type == "run_state_estimation"
            ),
        )

        tick_reward = reward_breakdown["total"]
        self._total_reward += tick_reward

        # Record frequency
        self._frequency_history.append(self._engine.frequency_hz)

        # Update Black Start tracking
        self._update_black_start_tracking()

        # Check termination conditions
        if self._engine.frequency_hz < 59.0:
            self._done = True
        if self._tick >= self._max_ticks:
            self._done = True

        # Task 4 special: turbine destruction
        if self._task_id == 4:
            destruction_tick = self._attack_config.get("destruction_tick", 10)
            if self._tick >= destruction_tick and self._spoof.is_resonance_active():
                self._done = True  # Turbine destroyed

        # Update OpenEnv state
        self._state = State(episode_id=self._episode_id, step_count=self._tick)

        return self._build_observation(
            reward=tick_reward,
            error=error_msg,
            reward_breakdown=reward_breakdown,
        )

    @property
    def state(self) -> State:
        """Get current OpenEnv state."""
        return self._state

    # ------------------------------------------------------------------
    # Grading
    # ------------------------------------------------------------------

    def get_score(self) -> float:
        """Get the final grader score for the current episode."""
        episode_state = self._build_episode_state()
        return grade_task(self._task_id, self._action_history, episode_state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float = 0.0,
        error: Optional[str] = None,
        reward_breakdown: Optional[Dict] = None,
    ) -> GridObservation:
        """Build a GridObservation from current engine state."""
        if self._engine is None:
            return GridObservation(done=True, last_action_error="Not initialized")

        # Build telemetry from spoofed readings for current tick
        current_telemetry = []
        for node_id, readings in self._current_spoofed_telemetry.items():
            current_telemetry.append(readings)

        # Telemetry history (includes spoofed values)
        telemetry_stream = self._engine.get_telemetry_history()

        # Add info dict fields
        info = {
            "task_id": self._task_id,
            "tick": self._tick,
            "grid_frequency_hz": self._engine.frequency_hz,
            "active_spoofs": self._spoof.get_active_spoofs() if self._spoof else [],
            "last_kirchhoff_result": self._last_estimation_result,
            "episode_seed": self._seed,
            "reward_breakdown": reward_breakdown or {},
        }

        obs = GridObservation(
            topology_graph=self._engine.get_topology(),
            telemetry_stream=telemetry_stream,
            weather_forecast_matrix=self._engine.get_weather(),
            network_packet_logs=self._spoof.get_recent_packet_logs() if self._spoof else [],
            grid_frequency_hz=self._engine.frequency_hz,
            tick=self._tick,
            task_id=self._task_id,
            done=self._done,
            reward=reward,
            last_action_error=error,
            last_state_estimation=self._last_estimation_result,
            weather_summary=self._engine.get_weather_summary(),
            metadata=info,
        )

        return obs

    def _build_episode_state(self) -> Dict[str, Any]:
        """Build episode state dict for grading."""
        if self._engine is None:
            return {}

        # Compute load restoration fraction
        total_possible = self._engine.get_total_possible_mwh()
        total_served = self._engine.get_mwh_served()
        load_fraction = total_served / total_possible if total_possible > 0 else 0.0

        # Check if all critical nodes are restored
        critical_nodes = [n for n in self._engine.nodes.values() if n["critical"]]
        critical_restored = all(n["energized"] for n in critical_nodes)

        return {
            "frequency_history": self._frequency_history,
            "is_proactive_dispatch": self._engine.is_dispatch_proactive(),
            "critical_nodes_shed": self._engine.get_critical_nodes_shed(),
            "full_restoration_tick": self._get_full_restoration_tick(),
            "load_restored_fraction": load_fraction,
            "hydro_stable_ticks": self._hydro_stable_ticks,
            "energized_node_count": sum(
                1 for n in self._engine.nodes.values() if n["energized"]
            ),
            "max_island_count": self._max_island_count,
            "successful_mergers": self._successful_mergers,
            "premature_mergers": self._premature_mergers,
            "transformer_failures": self._transformer_failures,
            "critical_nodes_restored": critical_restored,
        }

    def _update_black_start_tracking(self) -> None:
        """Track Black Start milestones for Task 5 grading."""
        if self._task_id != 5 or self._engine is None:
            return

        # Track hydro stability
        hydro = self._engine.nodes.get("NODE_01")
        if hydro and hydro["energized"] and hydro["generation_mw"] > 0:
            self._hydro_stable_ticks += 1
        else:
            self._hydro_stable_ticks = 0

        # Track island count
        islands = self._engine.get_stable_islands()
        self._max_island_count = max(self._max_island_count, len(islands))

    def _get_full_restoration_tick(self) -> Optional[int]:
        """Get the tick at which full load was restored (for Task 2)."""
        if self._engine is None:
            return None
        total_possible = self._engine.get_total_possible_mwh()
        total_served = self._engine.get_mwh_served()
        if total_possible > 0 and total_served >= total_possible * 0.95:
            return self._tick
        return None

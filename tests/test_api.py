"""
Test API — proves step()/reset()/state() return correctly typed models.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import GridAction, GridObservation, ActionType
from server.nexusgrid_environment import NexusgridEnvironment


class TestEnvironmentAPI:
    """Test the OpenEnv API interface."""

    def _make_env(self) -> NexusgridEnvironment:
        return NexusgridEnvironment()

    def test_reset_returns_observation(self):
        """reset() should return a GridObservation."""
        env = self._make_env()
        obs = env.reset(seed=42, task_id=0)

        assert isinstance(obs, GridObservation)
        assert obs.tick == 0
        assert obs.task_id == 0
        assert obs.done is False
        assert 58.0 <= obs.grid_frequency_hz <= 62.0

    def test_reset_topology_present(self):
        """reset() observation should contain topology with nodes and edges."""
        env = self._make_env()
        obs = env.reset(seed=42, task_id=0)

        topo = obs.topology_graph
        assert "nodes" in topo
        assert "edges" in topo
        assert len(topo["nodes"]) == 20  # 20-node grid
        assert len(topo["edges"]) == 40  # 40 edges

    def test_step_returns_observation(self):
        """step() should return a GridObservation with reward and done."""
        env = self._make_env()
        env.reset(seed=42, task_id=0)

        action = GridAction(
            action_type=ActionType.DISPATCH_GENERATION,
            node_id="NODE_01",
            mw=100.0,
        )
        obs = env.step(action)

        assert isinstance(obs, GridObservation)
        assert obs.tick == 1

    def test_step_advance_tick(self):
        """advance_tick should increment the tick counter."""
        env = self._make_env()
        env.reset(seed=42, task_id=1)

        action = GridAction(action_type=ActionType.ADVANCE_TICK)
        obs = env.step(action)

        assert obs.tick == 1

    def test_state_returns_state(self):
        """state property should return an OpenEnv State."""
        env = self._make_env()
        env.reset(seed=42, task_id=0)

        state = env.state
        assert state.episode_id is not None
        assert state.step_count == 0

    def test_done_when_frequency_below_59(self):
        """Episode should terminate when frequency drops below 59.0Hz."""
        env = self._make_env()
        env.reset(seed=42, task_id=5)  # Black start — frequency starts at 59.0

        # In black start, frequency can drop. Let's force it by doing nothing.
        # The initial frequency is 59.0 for black start
        action = GridAction(action_type=ActionType.ADVANCE_TICK)

        done = False
        for _ in range(5):
            obs = env.step(action)
            if obs.done:
                done = True
                break

        # Black start starts at 59.0 which is already at termination boundary
        # The env should detect this
        assert done or obs.grid_frequency_hz >= 59.0

    def test_smoke_test_scores_1(self):
        """Task 0 with valid dispatch should score 1.0."""
        env = self._make_env()
        env.reset(seed=42, task_id=0)

        action = GridAction(
            action_type=ActionType.DISPATCH_GENERATION,
            node_id="NODE_01",
            mw=100.0,
        )
        env.step(action)

        score = env.get_score()
        assert score == 1.0, f"Smoke test should score 1.0, got {score}"

    def test_reset_idempotent(self):
        """Calling reset(42) multiple times should produce identical observations."""
        env = self._make_env()

        obs1 = env.reset(seed=42, task_id=0)
        obs2 = env.reset(seed=42, task_id=0)

        assert obs1.grid_frequency_hz == obs2.grid_frequency_hz
        assert obs1.tick == obs2.tick
        assert obs1.task_id == obs2.task_id

    def test_all_action_types_accepted(self):
        """All action types should be accepted without crashing."""
        env = self._make_env()
        env.reset(seed=42, task_id=1)

        actions = [
            GridAction(action_type=ActionType.ADVANCE_TICK),
            GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id="NODE_01", mw=50),
            GridAction(action_type=ActionType.TOGGLE_CIRCUIT_BREAKER, edge_id="LINE_01", status="OPEN"),
            GridAction(action_type=ActionType.RUN_STATE_ESTIMATION, subgraph=["NODE_01", "NODE_02"]),
            GridAction(action_type=ActionType.QUARANTINE_SCADA_NODE, node_id="NODE_01"),
        ]

        for action in actions:
            obs = env.step(action)
            assert isinstance(obs, GridObservation)

    def test_weather_present(self):
        """Observation should include weather data."""
        env = self._make_env()
        obs = env.reset(seed=42, task_id=0)

        assert len(obs.weather_forecast_matrix) > 0
        assert obs.weather_summary != ""

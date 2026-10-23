"""OpenEnv environment behaviour: API, hidden information, rewards and determinism."""

from __future__ import annotations

import json

import pytest

from models import ActionType, GridAction, GridObservation
from policies import ExpertPolicy
from server.nexusgrid_environment import NexusgridEnvironment
from server.scenarios import MAX_TICKS
from tests.conftest import noop


class TestApi:
    def test_reset_returns_typed_observation(self, env):
        obs = env.reset(seed=1, task_id=1)
        assert isinstance(obs, GridObservation)
        assert obs.tick == 0 and obs.task_id == 1 and not obs.done
        assert len(obs.topology_graph["nodes"]) == 20 and len(obs.topology_graph["edges"]) == 40
        assert obs.max_ticks == MAX_TICKS[1]
        assert obs.task_brief and obs.forecast["system_demand_mw"]
        assert obs.islands and obs.telemetry_stream

    def test_step_advances_one_tick(self, env):
        env.reset(seed=1, task_id=1)
        obs = env.step(GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id="NODE_01", mw=50))
        assert obs.tick == 1
        assert set(obs.reward_breakdown) == {"progress", "frequency_band", "overload", "events", "invalid_action", "total"}
        assert obs.reward == pytest.approx(obs.reward_breakdown["total"])

    def test_episode_ends_at_horizon_and_rejects_more_steps(self, env):
        obs = env.reset(seed=1, task_id=0)
        while not obs.done:
            obs = env.step(noop())
        assert obs.tick == MAX_TICKS[0]
        again = env.step(noop())
        assert again.done and again.last_action_error == "Episode already finished"

    def test_step_before_reset_is_handled(self):
        obs = NexusgridEnvironment().step(noop())
        assert obs.done and "reset" in obs.last_action_error

    def test_invalid_task_rejected(self, env):
        with pytest.raises(ValueError):
            env.reset(seed=1, task_id=9)

    def test_state_exposes_score_and_progress(self, env):
        env.reset(seed=3, task_id=2)
        env.step(noop())
        state = env.state
        assert state.step_count == 1 and state.task_id == 2
        assert 0.0 <= state.score <= 1.0

    def test_invalid_actions_are_reported_and_penalised(self, env):
        env.reset(seed=1, task_id=1)
        obs = env.step(GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id="NODE_03", mw=50))
        assert obs.last_action_error and "not a generator" in obs.last_action_error
        assert obs.reward_breakdown["invalid_action"] < 0
        obs = env.step(GridAction(action_type=ActionType.RUN_STATE_ESTIMATION,
                                  subgraph=[f"NODE_{i:02d}" for i in range(1, 9)]))
        assert "limited to 6" in obs.last_action_error


class TestHiddenInformation:
    @pytest.mark.parametrize("task_id", [3, 4])
    def test_attack_target_never_serialised(self, env, task_id):
        obs = env.reset(seed=4, task_id=task_id)
        for _ in range(3):
            obs = env.step(noop())
        payload = obs.model_dump(exclude={"metadata"})
        text = json.dumps(payload)
        assert "active_spoofs" not in text and "attack_target" not in text
        assert "hidden_loss" not in text and "target_node" not in text
        assert "attack" in env.debug_state()

    def test_phantom_attack_falsifies_telemetry_but_not_physics(self, env):
        env.reset(seed=4, task_id=3)
        target = env.debug_state()["attack"]["target_node"]
        for _ in range(6):
            obs = env.step(noop())
        reported = obs.telemetry_stream[-1]
        reading = next(r for r in reported if r["node_id"] == target)
        true_gen = env.engine.nodes[target]["generation_mw"]
        assert env.debug_state()["hidden_loss_mw"] > 100
        assert reading["generation_mw"] - true_gen > 100

    def test_quarantine_requires_evidence(self, env):
        env.reset(seed=4, task_id=3)
        target = env.debug_state()["attack"]["target_node"]
        obs = env.step(GridAction(action_type=ActionType.QUARANTINE_SCADA_NODE, node_id=target))
        assert "refused" in obs.last_action_error.lower()
        assert env.debug_state()["refused_quarantines"] == 1


class TestRewards:
    @pytest.mark.parametrize("task_id", sorted(MAX_TICKS))
    def test_progress_rewards_telescope_to_final_score(self, env, task_id):
        policy = ExpertPolicy()
        obs = env.reset(seed=10_001, task_id=task_id)
        progress = 0.0
        while not obs.done:
            obs = env.step(policy.act(obs))
            progress += obs.reward_breakdown["progress"]
        assert progress == pytest.approx(env.get_score(), abs=1e-4)
        assert obs.score == pytest.approx(env.get_score(), abs=1e-4)

    def test_relay_trip_is_penalised(self, env):
        env.reset(seed=2, task_id=2)
        penalties = []
        obs = None
        for _ in range(5):
            obs = env.step(noop())
            penalties.append(obs.reward_breakdown["events"])
        assert min(penalties) < 0


class TestDeterminism:
    @pytest.mark.parametrize("task_id", sorted(MAX_TICKS))
    def test_same_seed_and_actions_give_same_trajectory(self, task_id):
        def trajectory():
            env = NexusgridEnvironment()
            policy = ExpertPolicy()
            obs = env.reset(seed=77, task_id=task_id)
            out = [obs.grid_frequency_hz]
            while not obs.done:
                obs = env.step(policy.act(obs))
                out.append((obs.grid_frequency_hz, obs.reward, json.dumps(obs.network_packet_logs[-3:])))
            return out, env.get_score()

        assert trajectory() == trajectory()


def test_early_completion_pays_remaining_band_bonus():
    """Finishing the black start early must not earn less than stalling in band until the horizon."""
    env = NexusgridEnvironment()
    policy = ExpertPolicy()
    obs = env.reset(seed=10_000, task_id=5)
    while not obs.done:
        obs = env.step(policy.act(obs))
    assert obs.tick < obs.max_ticks  # completed early
    assert obs.reward_breakdown["frequency_band"] >= 0.02 * (obs.max_ticks - obs.tick)

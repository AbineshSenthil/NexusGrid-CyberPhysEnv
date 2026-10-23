"""Training utilities: the RL action catalogue/features and the GRPO reward."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gym_env import CATALOG, FEATURE_SIZE, DiscretizedExpert, featurize
from policies import GridView
from server.nexusgrid_environment import NexusgridEnvironment
from training.grpo_common import INVALID_REWARD, collect_states, score_completion


class TestCatalog:
    def test_catalog_size_and_advance_first(self):
        assert len(CATALOG) == 237
        assert CATALOG.entries[0] == ("advance", ())

    def test_masks_follow_observation(self, env):
        obs = env.reset(seed=1, task_id=5)
        mask = CATALOG.mask(GridView(obs))
        assert mask[0]
        closable = [CATALOG.describe(i) for i, ok in enumerate(mask) if ok and CATALOG.entries[i][0] == "breaker"]
        assert closable and all("CLOSED" in c for c in closable)  # dark grid: only closing lines makes sense
        assert not any(ok for i, ok in enumerate(mask) if CATALOG.entries[i][0] == "quarantine")

    def test_every_unmasked_action_is_accepted_by_the_environment(self, env):
        obs = env.reset(seed=2, task_id=2)
        view = GridView(obs)
        for idx in np.flatnonzero(CATALOG.mask(view))[:40]:
            action = CATALOG.to_action(int(idx), view)
            assert action.action_type is not None

    def test_features_have_fixed_size(self, env):
        for task in range(6):
            obs = env.reset(seed=3, task_id=task)
            feats = featurize(GridView(obs))
            assert feats.shape == (FEATURE_SIZE,) and np.isfinite(feats).all()

    @pytest.mark.parametrize("task_id", [1, 3, 4])
    def test_discretised_expert_still_solves_tasks(self, task_id):
        env = NexusgridEnvironment()
        policy = DiscretizedExpert()
        obs = env.reset(seed=10_002, task_id=task_id)
        while not obs.done:
            obs = env.step(policy.act(obs))
        assert env.get_score() >= 0.8


class TestGrpoReward:
    def test_collect_states_records_rebuildable_prompts(self):
        records = collect_states([1, 3], episodes=2, seed=0)
        assert records
        for r in records:
            assert r["messages"][0]["role"] == "system"
            assert isinstance(json.loads(r["prefix"]), list)

    def test_invalid_output_is_penalised(self):
        r = collect_states([1], episodes=1, seed=0)[0]
        assert score_completion("I think we should wait", r["task_id"], r["seed"], r["prefix"])["reward"] == INVALID_REWARD

    def test_reward_prefers_evidence_over_waiting_in_phantom_attack(self):
        env = NexusgridEnvironment()
        obs = env.reset(seed=11, task_id=3)
        prefix = []
        target = env.debug_state()["attack"]["target_node"]
        while not any(p["anomaly_flag"] and p["source_node"] == target for p in obs.network_packet_logs[-20:]):
            prefix.append({"action_type": "advance_tick"})
            obs = env.step(__import__("models").GridAction(action_type="advance_tick"))
        se = json.dumps({"action_type": "run_state_estimation", "subgraph": [target]})
        wait = json.dumps({"action_type": "advance_tick"})
        good = score_completion(se, 3, 11, prefix)["reward"]
        idle = score_completion(wait, 3, 11, prefix)["reward"]
        assert good > idle + 0.2

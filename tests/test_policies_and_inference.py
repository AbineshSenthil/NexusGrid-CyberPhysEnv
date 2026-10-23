"""Baseline policies, the observation-only flow model and the LLM runner helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference import LLMPolicy, parse_action, render_observation, to_grid_action
from models import ActionType, GridAction
from policies import FlowModel, GridView, RandomPolicy, make_policy
from server.scenarios import MAX_TICKS
from tests.conftest import noop


class TestFlowModel:
    def test_reconstructs_engine_flows_from_observation(self, env):
        obs = env.reset(seed=5, task_id=2)
        view = GridView(obs)
        island = view.main_island()
        model = FlowModel(view, island["nodes"])
        predicted = model.predict(deltas={"NODE_17": 0.0}, close=())
        for eid, flow in predicted.items():
            assert flow == pytest.approx(env.engine.edges[eid]["flow_mw"], abs=0.5)

    def test_predicts_flows_after_redispatch(self, env):
        obs = env.reset(seed=5, task_id=1)
        view = GridView(obs)
        model = FlowModel(view, view.main_island()["nodes"])
        predicted = model.predict(deltas={"NODE_13": 120.0})
        env.engine.dispatch_generation("NODE_13", 120.0)
        for eid, flow in predicted.items():
            assert flow == pytest.approx(env.engine.edges[eid]["flow_mw"], abs=3.0)


@pytest.mark.parametrize("name", ["noop", "random", "expert"])
@pytest.mark.parametrize("task_id", sorted(MAX_TICKS))
def test_policies_emit_valid_actions(env, name, task_id):
    policy = make_policy(name, seed=3)
    obs = env.reset(seed=3, task_id=task_id)
    for _ in range(4):
        action = policy.act(obs)
        assert isinstance(action, GridAction)
        obs = env.step(action)
        if obs.done:
            break


def test_random_policy_is_seeded(env):
    obs = env.reset(seed=1, task_id=1)
    a = [RandomPolicy(9).act(obs).model_dump() for _ in range(3)]
    b = [RandomPolicy(9).act(obs).model_dump() for _ in range(3)]
    assert a == b


class TestParseAction:
    def test_reasoning_and_code_fence(self):
        text = '<think>raise hydro</think>\n```json\n{"action_type":"dispatch_generation","node_id":"NODE_01","mw":120}\n```'
        assert parse_action(text) == {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 120}

    def test_aliases_and_single_quotes(self):
        assert parse_action("{'action': 'shed', 'node': 'node_07', 'amount': '80'}") == {
            "action_type": "shed_load", "node_id": "NODE_07", "mw": 80.0}
        assert parse_action("{'action_type': 'toggle_breaker', 'edge': 'line_28', 'status': 'closed'}") == {
            "action_type": "toggle_circuit_breaker", "edge_id": "LINE_28", "status": "CLOSED"}

    def test_list_wrapper_and_garbage(self):
        assert parse_action('[{"action_type":"advance_tick"}]') == {"action_type": "advance_tick"}
        assert parse_action("I would dispatch something") is None
        assert to_grid_action({"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 99999}) is None


def test_render_observation_covers_key_signals(env):
    obs = env.reset(seed=4, task_id=4)
    for _ in range(2):
        obs = env.step(noop())
    text = render_observation(obs, ["t0 {\"action_type\":\"advance_tick\"}"])
    for marker in ("T4 stuxnet_resonance", "Island 0", "Units", "Loads MW", "Live links", "PMU oscillation",
                   "Your recent actions"):
        assert marker in text
    assert "target_node" not in text


class _FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        content = self.replies.pop(0) if self.replies else "nonsense"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_llm_policy_pure_mode_counts_invalid_outputs(env):
    policy = LLMPolicy(model="fake", api_base="http://example.invalid/v1",
                       client=_FakeClient(['{"action_type":"dispatch_generation","node_id":"NODE_01","mw":50}',
                                           "no idea"]))
    obs = env.reset(seed=1, task_id=1)
    first = policy.act(obs)
    second = policy.act(env.step(first))
    assert first.action_type == ActionType.DISPATCH_GENERATION
    assert second.action_type == ActionType.ADVANCE_TICK
    assert policy.stats()["invalid_outputs"] == 1 and policy.stats()["fallbacks"] == 0


def test_llm_policy_fallback_is_reported(env):
    policy = LLMPolicy(model="fake", api_base="http://example.invalid/v1", fallback=True,
                       client=_FakeClient(["???"]))
    obs = env.reset(seed=1, task_id=1)
    policy.act(obs)
    assert policy.stats()["fallbacks"] == 1

"""Graders separate competent operation from doing nothing or acting randomly."""

from __future__ import annotations

import statistics

import pytest

from policies import ExpertPolicy, NoopPolicy, RandomPolicy
from server.graders import grade_task, served_factor
from server.rubric import TASK_RUBRICS, get_task_rubric_specs
from server.scenarios import EVAL_SEED_START, MAX_TICKS
from tests.conftest import run_policy

SEEDS = [EVAL_SEED_START + i for i in range(6)]


@pytest.mark.parametrize("task_id", sorted(TASK_RUBRICS))
def test_rubric_weights_sum_to_one(task_id):
    assert sum(r["weight"] for r in get_task_rubric_specs(task_id)) == pytest.approx(1.0)


@pytest.mark.parametrize("task_id", sorted(MAX_TICKS))
def test_trivial_policies_score_low_and_expert_high(env, task_id):
    noop = statistics.mean(run_policy(env, NoopPolicy(), task_id, s) for s in SEEDS)
    rand = statistics.mean(run_policy(env, RandomPolicy(s), task_id, s) for s in SEEDS)
    expert = statistics.mean(run_policy(env, ExpertPolicy(), task_id, s) for s in SEEDS)
    assert noop <= 0.25, f"do-nothing scores {noop:.2f} on task {task_id}"
    assert rand <= 0.35, f"random scores {rand:.2f} on task {task_id}"
    assert expert >= 0.85, f"expert scores {expert:.2f} on task {task_id}"


def test_scores_are_clipped(env):
    assert grade_task(99, [], {}) == 0.0
    state = {"done": True, "frequency_history": [60.0] * 4, "max_ticks": 3}
    assert 0.0 <= grade_task(0, [], state) <= 1.0


def test_served_factor_penalises_load_shedding():
    assert served_factor({"served_history": [1.0, 1.0, 1.0]}) == 1.0
    assert served_factor({"served_history": [1.0, 0.89, 0.89]}) == pytest.approx(0.5, abs=1e-6)


def test_duck_curve_collapse_scores_zero():
    state = {"done": True, "frequency_history": [60.0, 59.5, 58.9], "max_ticks": 15}
    assert grade_task(1, [], state) == 0.0


def test_quarantine_without_evidence_costs_points():
    actions = [
        {"action_type": "run_state_estimation", "tick": 2,
         "result": {"consistent": False, "violation_node": "NODE_09"}},
        {"action_type": "quarantine_scada_node", "tick": 3, "node_id": "NODE_09", "result": {"success": True}},
    ]
    state = {"done": True, "attack_target": "NODE_09", "attack_start_tick": 2,
             "frequency_history": [60.0] * 19, "served_history": [1.0] * 19, "max_ticks": 18}
    clean = grade_task(3, actions, {**state, "refused_quarantines": 0})
    sloppy = grade_task(3, actions, {**state, "refused_quarantines": 2})
    assert clean == pytest.approx(1.0)
    assert sloppy == pytest.approx(clean - 0.2)


def test_disconnecting_the_oscillating_unit_scores_zero():
    state = {"done": True, "target_disconnected": True, "best_counter_signal": "correct",
             "attack_neutralized": True, "frequency_history": [60.0] * 13}
    assert grade_task(4, [], state) == 0.0

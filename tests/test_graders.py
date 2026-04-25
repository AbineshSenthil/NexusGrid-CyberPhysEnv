"""
Test Graders — proves each grader returns expected scores
for known inputs (correct action, wrong order, timeout, zero action).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.graders import (
    grade_smoke_test,
    grade_duck_curve,
    grade_cascade_overload,
    grade_phantom_injection,
    grade_stuxnet_resonance,
    grade_black_start,
    grade_task,
)
from server.reward import RewardCalculator


class TestSmokeTestGrader:
    def test_valid_dispatch_scores_1(self):
        actions = [{"action_type": "dispatch_generation", "mw": 100}]
        assert grade_smoke_test(actions, {}) == 1.0

    def test_no_action_scores_0(self):
        assert grade_smoke_test([], {}) == 0.0

    def test_wrong_action_scores_0(self):
        actions = [{"action_type": "advance_tick"}]
        assert grade_smoke_test(actions, {}) == 0.0

    def test_negative_mw_scores_0(self):
        actions = [{"action_type": "dispatch_generation", "mw": -100}]
        assert grade_smoke_test(actions, {}) == 0.0


class TestDuckCurveGrader:
    def test_perfect_frequency_scores_high(self):
        state = {"frequency_history": [60.0, 59.8, 59.7, 59.8, 60.0]}
        actions = [
            {"action_type": "dispatch_generation", "tick": 1},
            {"action_type": "dispatch_generation", "tick": 2},
        ]
        score = grade_duck_curve(actions, state)
        assert score >= 0.5  # Frequency stayed above 59.5

    def test_frequency_crash_scores_0(self):
        state = {"frequency_history": [60.0, 59.5, 59.0, 58.5]}
        actions = []
        assert grade_duck_curve(actions, state) == 0.0


class TestCascadeOverloadGrader:
    def test_full_correct_response(self):
        actions = [
            {"action_type": "toggle_circuit_breaker", "status": "OPEN"},
        ]
        state = {"critical_nodes_shed": 0, "full_restoration_tick": 5}
        score = grade_cascade_overload(actions, state)
        assert score >= 0.8

    def test_no_action_scores_0(self):
        state = {"critical_nodes_shed": 2}
        assert grade_cascade_overload([], state) == 0.0


class TestPhantomInjectionGrader:
    def test_correct_order_full_score(self):
        actions = [
            {"action_type": "advance_tick", "tick": 0},
            {"action_type": "run_state_estimation", "tick": 1, "result": {"consistent": False}},
            {"action_type": "quarantine_scada_node", "tick": 2, "node_id": "NODE_14"},
            {"action_type": "dispatch_generation", "tick": 3, "node_id": "NODE_09", "mw": 100},
        ]
        score = grade_phantom_injection(actions, {"spoof_target": "NODE_14"})
        assert score == 1.0

    def test_quarantine_without_estimation_scores_0(self):
        actions = [
            {"action_type": "quarantine_scada_node", "tick": 1},
        ]
        score = grade_phantom_injection(actions, {})
        assert score == 0.0

    def test_estimation_only_scores_partial(self):
        actions = [
            {"action_type": "run_state_estimation", "tick": 1, "result": {"consistent": False}},
        ]
        score = grade_phantom_injection(actions, {})
        assert score == 0.2


class TestStuxnetResonanceGrader:
    def test_correct_injection_requires_ramp_down_and_reroute_for_full_score(self):
        actions = [
            {"action_type": "inject_counter_signal", "tick": 0, "hz_offset": -0.5},
            {"action_type": "dispatch_generation", "tick": 1, "node_id": "NODE_17", "mw": -200},
            {"action_type": "dispatch_generation", "tick": 2, "node_id": "NODE_09", "mw": 300},
        ]
        score = grade_stuxnet_resonance(actions, {})
        assert score == 1.0

    def test_correct_injection_without_ramp_down_stays_partial(self):
        actions = [
            {"action_type": "inject_counter_signal", "tick": 0, "hz_offset": -0.5},
        ]
        score = grade_stuxnet_resonance(actions, {})
        assert score == 0.4

    def test_wrong_offset_partial(self):
        actions = [
            {"action_type": "inject_counter_signal", "hz_offset": -0.3},
        ]
        score = grade_stuxnet_resonance(actions, {})
        assert score == 0.2

    def test_cutting_turbine_scores_0(self):
        actions = [
            {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_22", "status": "OPEN"},
        ]
        assert grade_stuxnet_resonance(actions, {}) == 0.0


class TestBlackStartGrader:
    def test_hydro_dispatch_checkpoint_a(self):
        actions = [
            {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 500},
        ]
        state = {"hydro_stable_ticks": 0, "energized_node_count": 1}
        score = grade_black_start(actions, state)
        assert score >= 0.02  # Checkpoint A × load_fraction

    def test_no_action_scores_0(self):
        assert grade_black_start([], {}) == 0.0

    def test_critical_restore_requires_checkpoint_c(self):
        actions = [
            {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 500},
        ]
        state = {
            "hydro_stable_ticks": 3,
            "energized_node_count": 10,
            "max_island_count": 1,
            "successful_mergers": 0,
            "premature_mergers": 0,
            "critical_nodes_restored": True,
            "load_restored_fraction": 0.84,
            "transformer_failures": 0,
        }
        assert grade_black_start(actions, state) == 0.42


class TestRewardCalculator:
    def test_fault_isolation_reward_only_paid_once_per_episode(self):
        reward_calc = RewardCalculator()
        base_kwargs = {
            "action_type": "toggle_circuit_breaker",
            "action_params": {},
            "frequency_hz": 60.0,
            "overloaded_edges": [],
            "critical_nodes_shed": 0,
            "is_proactive": False,
            "spoof_detected": False,
            "has_read_logs_before_estimation": False,
        }

        first = reward_calc.compute_tick_reward(fault_isolated=True, **base_kwargs)
        second = reward_calc.compute_tick_reward(fault_isolated=True, **base_kwargs)

        assert first["fault_isolation"] == 0.20
        assert second["fault_isolation"] == 0.0


class TestGradeTaskRouter:
    def test_routes_correctly(self):
        """grade_task should route to the correct grader."""
        # Task 0 with valid action
        actions = [{"action_type": "dispatch_generation", "mw": 100}]
        assert grade_task(0, actions, {}) == 1.0

        # Invalid task ID
        assert grade_task(99, [], {}) == 0.0

    def test_score_clamped(self):
        """Scores should always be in [0, 1]."""
        for task_id in range(6):
            score = grade_task(task_id, [], {})
            assert 0.0 <= score <= 1.0

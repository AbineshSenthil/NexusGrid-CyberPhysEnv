"""
NexusGrid — Task Graders.

Pure functions ``(action_history, episode_state) -> score in [0, 1]``.
Each grader is the weighted rubric sum plus task-specific gates and penalties,
designed so that doing nothing or acting randomly scores low.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .rubric import evaluate_task_rubrics, mean_served

SERVED_TARGET = 0.99
SERVED_PENALTY = 5.0


def grade_task(task_id: int, action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    graders = {
        0: grade_smoke_test,
        1: grade_duck_curve,
        2: grade_cascade_overload,
        3: grade_phantom_injection,
        4: grade_stuxnet_resonance,
        5: grade_black_start,
    }
    grader = graders.get(task_id)
    if grader is None:
        return 0.0
    return max(0.0, min(1.0, float(grader(action_history, episode_state))))


def _score(task_id: int, actions: List[Dict[str, Any]], state: Dict[str, Any]) -> Dict[str, Any]:
    state = {"done": True, **state}
    return evaluate_task_rubrics(task_id, actions, state)


def served_factor(state: Dict[str, Any]) -> float:
    """
    Multiplier that stops agents from "fixing" frequency by cutting customers:
    every percentage point of average demand left unserved below 99 % costs
    5 % of the score.
    """
    shortfall = max(0.0, SERVED_TARGET - mean_served(state))
    return max(0.0, 1.0 - SERVED_PENALTY * shortfall)


def grade_smoke_test(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """Any valid positive dispatch (0.7) that leaves frequency in band (0.3)."""
    return _score(0, action_history, episode_state)["weighted_score"]


def grade_duck_curve(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """Frequency regulation through the evening ramp; a collapse below 59.0 Hz scores 0."""
    freqs = episode_state.get("frequency_history", [])
    if freqs and min(freqs) < 59.0:
        return 0.0
    return _score(1, action_history, episode_state)["weighted_score"] * served_factor(episode_state)


def grade_cascade_overload(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Relieve the post-contingency overloads. Without relief, the remaining
    rubrics are capped at 40% so riding out the cascade cannot score well.
    """
    result = _score(2, action_history, episode_state)
    if result["rubrics"].get("relieve_overload", 0.0) == 0.0:
        return 0.4 * result["weighted_score"]
    return result["weighted_score"]


def grade_phantom_injection(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Evidence-first incident response. Every refused quarantine attempt
    (no prior proof for that bus) costs 0.1; unserved demand is penalised.
    """
    result = _score(3, action_history, episode_state)
    refused = int(episode_state.get("refused_quarantines", 0))
    return max(0.0, result["weighted_score"] * served_factor(episode_state) - 0.1 * refused)


def grade_stuxnet_resonance(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """Damp, detune and rebalance; disconnecting or zeroing the unit scores 0."""
    if episode_state.get("target_disconnected"):
        return 0.0
    return _score(4, action_history, episode_state)["weighted_score"] * served_factor(episode_state)


def grade_black_start(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """Restore critical loads and demand without collapsing islands."""
    return _score(5, action_history, episode_state)["weighted_score"]

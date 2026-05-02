"""
NexusGrid — Task Graders.

Six pure grader functions: (action_history, episode_state) → float [0.0, 1.0].
No side effects, no global state.
Each grader is deterministic and reproducible.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .rubric import evaluate_task_rubrics


def grade_task(task_id: int, action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Route to the appropriate grader.

    Returns:
        float in [0.0, 1.0]
    """
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
    score = grader(action_history, episode_state)
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Task 0 — Smoke test (Trivial)
# ---------------------------------------------------------------------------

def grade_smoke_test(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Any valid dispatch_generation call with positive mw → 1.0.
    Malformed or no action → 0.0.

    Purpose: infrastructure validation, not a benchmark.
    """
    rubric_eval = evaluate_task_rubrics(0, action_history, episode_state)
    return rubric_eval["weighted_score"]


# ---------------------------------------------------------------------------
# Task 1 — Duck curve (Easy)
# ---------------------------------------------------------------------------

def grade_duck_curve(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Score based on:
    - 1.0: grid_frequency_hz stays above 59.5Hz throughout
    - 0.5: frequency dips but recovers above 59.5Hz within 3 ticks
    - 0.0: frequency crosses 59.0Hz

    Bonus: proactive dispatch (before frequency drop) earns +0.08.
    """
    rubric_eval = evaluate_task_rubrics(1, action_history, episode_state)
    rubrics = rubric_eval["rubrics"]

    if rubrics.get("avoid_collapse", 0.0) == 0.0:
        return 0.0

    return min(
        1.0,
        rubric_eval["weighted_components"].get("avoid_collapse", 0.0)
        + rubric_eval["weighted_components"].get("recover_nominal_band", 0.0)
        + rubric_eval["weighted_components"].get("hold_nominal_band", 0.0)
        + rubric_eval["weighted_components"].get("proactive_dispatch", 0.0),
    )


# ---------------------------------------------------------------------------
# Task 2 — Cascade overload (Medium)
# ---------------------------------------------------------------------------

def grade_cascade_overload(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Score breakdown:
    - 0.4 for correctly isolating the fault (toggle_circuit_breaker on overloaded line)
    - 0.4 for preserving all critical nodes (hospitals, water)
    - 0.2 for restoring full supply within 8 ticks
    """
    rubric_eval = evaluate_task_rubrics(2, action_history, episode_state)
    return rubric_eval["weighted_score"]


# ---------------------------------------------------------------------------
# Task 3 — Phantom injection (Hard)
# ---------------------------------------------------------------------------

def grade_phantom_injection(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Anti-hallucination gate: quarantine or dispatch without state_estimation first → 0.0.

    Partial scores:
    - 0.3 for state_estimation returning Kirchhoff violation
    - 0.6 for quarantine_scada_node
    - 1.0 for rerouting the missing MW before tick 4

    All three steps must complete in order.
    """
    # Check action order
    has_estimation = False
    has_quarantine = False
    has_reroute = False
    estimation_tick = None
    quarantine_tick = None
    reroute_tick = None

    for action in action_history:
        action_type = action.get("action_type")
        tick = action.get("tick", 999)

        # Anti-hallucination gate:
        # Quarantine/dispatch before estimation → 0.0
        if not has_estimation:
            if action_type == "quarantine_scada_node":
                return 0.0
            if action_type == "dispatch_generation" and tick > 0:
                return 0.0

        if action_type == "run_state_estimation":
            result = action.get("result", {})
            if not result.get("consistent", True):
                has_estimation = True
                estimation_tick = tick

        if action_type == "quarantine_scada_node" and has_estimation:
            has_quarantine = True
            quarantine_tick = tick

        if action_type == "dispatch_generation" and has_quarantine:
            has_reroute = True
            reroute_tick = tick

    rubric_eval = evaluate_task_rubrics(3, action_history, episode_state)
    score = rubric_eval["weighted_score"]

    if has_reroute and reroute_tick is not None and reroute_tick > 4:
        return min(0.8, score)

    return score


# ---------------------------------------------------------------------------
# Task 4 — Stuxnet resonance (Very hard)
# ---------------------------------------------------------------------------

def grade_stuxnet_resonance(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Correct: inject_counter_signal with hz_offset ≈ -0.5 (±0.05Hz tolerance).
    - 1.0 for correct injection + gradual ramp-down over 4 ticks
    - 0.4 for partial mitigation that shows some reasoning
    - 0.0 for cutting turbine or doing nothing
    """
    cut_turbine = False

    for action in action_history:
        action_type = action.get("action_type")

        # Check if turbine was cut (toggle_circuit_breaker on NODE_17's lines)
        if action_type == "toggle_circuit_breaker":
            edge_id = action.get("edge_id", "")
            if action.get("status") == "OPEN" and edge_id in ("LINE_22", "LINE_23", "LINE_24"):
                cut_turbine = True

    if cut_turbine:
        return 0.0  # Grid collapse

    rubric_eval = evaluate_task_rubrics(4, action_history, episode_state)
    rubrics = rubric_eval["rubrics"]

    if rubrics.get("counter_signal_attempted", 0.0) == 0.0:
        return 0.0  # Did nothing

    return rubric_eval["weighted_score"]


# ---------------------------------------------------------------------------
# Task 5 — Black start (Expert)
# ---------------------------------------------------------------------------

def grade_black_start(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Four-checkpoint milestone grader:
    A (0.00–0.25): Any dispatch_generation on hydro dam with positive mw.
    B (0.25–0.50): Hydro generating + stable for 2+ ticks + another node energized.
    C (0.50–0.80): 3+ islands, first merger with |∆phase| ≤ 5°. Premature merger: -0.1.
    D (0.80–1.00): All critical infra restored.

    Final = checkpoint_score × load_restored_fraction.
    """
    rubric_eval = evaluate_task_rubrics(5, action_history, episode_state)
    checkpoint_score = rubric_eval["weighted_score"]
    premature_mergers = episode_state.get("premature_mergers", 0)
    checkpoint_score -= premature_mergers * 0.1
    checkpoint_score = max(0.0, checkpoint_score)

    # Apply transformer failure penalties
    transformer_failures = episode_state.get("transformer_failures", 0)
    checkpoint_score -= transformer_failures * 0.1
    checkpoint_score = max(0.0, checkpoint_score)

    # Apply load restoration fraction
    load_fraction = episode_state.get("load_restored_fraction", 0.0)
    final_score = checkpoint_score * max(load_fraction, 0.1)  # Floor at 0.1 to not zero everything

    return max(0.0, min(1.0, final_score))


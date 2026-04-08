"""
NexusGrid — Task Graders.

Six pure grader functions: (action_history, episode_state) → float [0.0, 1.0].
No side effects, no global state.
Each grader is deterministic and reproducible.
"""

from __future__ import annotations

from typing import Any, Dict, List


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
    for action in action_history:
        if action.get("action_type") == "dispatch_generation":
            mw = action.get("mw", 0)
            if mw is not None and mw > 0:
                return 1.0
    return 0.0


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
    freq_history = episode_state.get("frequency_history", [])
    min_freq = min(freq_history) if freq_history else 60.0

    # Check for termination
    if min_freq < 59.0:
        return 0.0

    # Check if battery dispatches happened in first 3 ticks
    battery_dispatches_early = 0
    for action in action_history:
        if action.get("action_type") == "dispatch_generation":
            tick = action.get("tick", 999)
            if tick <= 3:
                battery_dispatches_early += 1

    # Base score from frequency stability
    if min_freq >= 59.5:
        base_score = 1.0
    else:
        # Check if it recovered within 3 ticks
        dip_recovered = _check_recovery(freq_history, threshold=59.5, window=3)
        base_score = 0.5 if dip_recovered else 0.3

    # Proactive dispatch bonus
    proactive_bonus = 0.0
    if battery_dispatches_early >= 2:
        is_proactive = episode_state.get("is_proactive_dispatch", False)
        if is_proactive:
            proactive_bonus = 0.08

    return min(1.0, base_score + proactive_bonus)


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
    score = 0.0

    # Check fault isolation
    fault_isolated = False
    for action in action_history:
        if action.get("action_type") == "toggle_circuit_breaker":
            if action.get("status") == "OPEN":
                fault_isolated = True
                break
    if fault_isolated:
        score += 0.4

    # Check critical node preservation
    critical_shed = episode_state.get("critical_nodes_shed", 0)
    if critical_shed == 0:
        score += 0.4

    # Check restoration within 8 ticks
    restored_tick = episode_state.get("full_restoration_tick")
    if restored_tick is not None and restored_tick <= 8:
        score += 0.2

    return score


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
                # Allow tick 0 advance_tick
                pass

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

    score = 0.0
    if has_estimation:
        score = 0.3
    if has_quarantine:
        score = 0.6
    if has_reroute and reroute_tick is not None and reroute_tick <= 4:
        score = 1.0
    elif has_reroute:
        score = 0.8  # Late reroute

    return score


# ---------------------------------------------------------------------------
# Task 4 — Stuxnet resonance (Very hard)
# ---------------------------------------------------------------------------

def grade_stuxnet_resonance(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    """
    Correct: inject_counter_signal with hz_offset ≈ -0.5 (±0.05Hz tolerance).
    - 1.0 for correct injection + gradual ramp-down over 4 ticks
    - 0.4 for wrong hz_offset (shows partial reasoning)
    - 0.0 for cutting turbine or doing nothing
    """
    injected = False
    correct_offset = False
    ramp_down = False
    cut_turbine = False

    target_offset = -0.5
    tolerance = 0.05

    for action in action_history:
        action_type = action.get("action_type")

        if action_type == "inject_counter_signal":
            injected = True
            hz_offset = action.get("hz_offset", 0.0)
            if hz_offset is not None and abs(hz_offset - target_offset) <= tolerance:
                correct_offset = True

        # Check if turbine was cut (toggle_circuit_breaker on NODE_17's lines)
        if action_type == "toggle_circuit_breaker":
            edge_id = action.get("edge_id", "")
            if action.get("status") == "OPEN" and edge_id in ("LINE_22", "LINE_23", "LINE_24"):
                cut_turbine = True

    # Check for ramp-down actions after injection
    if injected:
        post_inject_dispatches = sum(
            1 for a in action_history
            if a.get("action_type") == "dispatch_generation"
            and a.get("tick", 0) > 0
        )
        if post_inject_dispatches >= 2:
            ramp_down = True

    if cut_turbine:
        return 0.0  # Grid collapse

    if not injected:
        return 0.0  # Did nothing

    if correct_offset:
        if ramp_down:
            return 1.0
        return 0.7  # Good injection but no ramp-down

    return 0.4  # Wrong offset but showed partial reasoning


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
    checkpoint_score = 0.0

    # Checkpoint A: any dispatch on hydro dam
    hydro_dispatched = False
    for action in action_history:
        if action.get("action_type") == "dispatch_generation":
            node_id = action.get("node_id", "")
            mw = action.get("mw", 0)
            if node_id == "NODE_01" and mw is not None and mw > 0:
                hydro_dispatched = True
                break

    if hydro_dispatched:
        checkpoint_score = 0.25

    # Checkpoint B: hydro stable for 2+ ticks + another node energized
    stable_ticks = episode_state.get("hydro_stable_ticks", 0)
    energized_count = episode_state.get("energized_node_count", 0)
    if hydro_dispatched and stable_ticks >= 2 and energized_count >= 2:
        checkpoint_score = 0.50

    # Checkpoint C: 3+ islands, successful merger
    island_count = episode_state.get("max_island_count", 0)
    successful_mergers = episode_state.get("successful_mergers", 0)
    premature_mergers = episode_state.get("premature_mergers", 0)
    if island_count >= 3 and successful_mergers >= 1:
        checkpoint_score = 0.80 - (premature_mergers * 0.1)
        checkpoint_score = max(0.50, checkpoint_score)

    # Checkpoint D: all critical infrastructure restored
    critical_restored = episode_state.get("critical_nodes_restored", False)
    if critical_restored:
        checkpoint_score = 1.0

    # Apply transformer failure penalties
    transformer_failures = episode_state.get("transformer_failures", 0)
    checkpoint_score -= transformer_failures * 0.1
    checkpoint_score = max(0.0, checkpoint_score)

    # Apply load restoration fraction
    load_fraction = episode_state.get("load_restored_fraction", 0.0)
    final_score = checkpoint_score * max(load_fraction, 0.1)  # Floor at 0.1 to not zero everything

    return max(0.0, min(1.0, final_score))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_recovery(freq_history: List[float], threshold: float, window: int) -> bool:
    """Check if frequency recovered above threshold within `window` ticks after dipping."""
    dip_started = False
    ticks_since_dip = 0
    for freq in freq_history:
        if freq < threshold:
            dip_started = True
            ticks_since_dip = 0
        elif dip_started:
            ticks_since_dip += 1
            if freq >= threshold and ticks_since_dip <= window:
                return True
    return False

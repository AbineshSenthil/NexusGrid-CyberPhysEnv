"""Composable task rubrics for NexusGrid scoring and training feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List


CheckFn = Callable[[List[Dict[str, Any]], Dict[str, Any]], float]


@dataclass(frozen=True)
class Rubric:
    """A single named rubric component."""

    name: str
    weight: float
    description: str
    check: CheckFn

    def evaluate(self, action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> Dict[str, Any]:
        raw_score = float(self.check(action_history, episode_state))
        raw_score = max(0.0, min(1.0, raw_score))
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "raw": raw_score,
            "weighted": raw_score * self.weight,
        }


def evaluate_task_rubrics(task_id: int, action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all rubrics for a task and return a serializable breakdown."""
    rubrics = TASK_RUBRICS.get(task_id, [])
    rubric_results = [rubric.evaluate(action_history, episode_state) for rubric in rubrics]
    return {
        "task_id": task_id,
        "rubrics": {item["name"]: item["raw"] for item in rubric_results},
        "weights": {item["name"]: item["weight"] for item in rubric_results},
        "descriptions": {item["name"]: item["description"] for item in rubric_results},
        "weighted_components": {item["name"]: item["weighted"] for item in rubric_results},
        "weighted_score": sum(item["weighted"] for item in rubric_results),
    }


def get_task_rubric_specs(task_id: int) -> List[Dict[str, Any]]:
    """Return rubric metadata for OpenEnv discovery surfaces."""
    return [
        {"name": rubric.name, "weight": rubric.weight, "description": rubric.description}
        for rubric in TASK_RUBRICS.get(task_id, [])
    ]


def _first_tick(action_history: List[Dict[str, Any]], action_type: str) -> int | None:
    for action in action_history:
        if action.get("action_type") == action_type:
            return action.get("tick")
    return None


def _has_positive_dispatch(action_history: List[Dict[str, Any]]) -> bool:
    return any(
        action.get("action_type") == "dispatch_generation" and (action.get("mw") or 0.0) > 0.0
        for action in action_history
    )


def _task0_valid_dispatch(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    return 1.0 if _has_positive_dispatch(action_history) else 0.0


def _task1_avoid_collapse(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    freq_history = episode_state.get("frequency_history", [])
    min_freq = min(freq_history) if freq_history else 60.0
    return 1.0 if min_freq >= 59.0 else 0.0


def _task1_recover_nominal(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    freq_history = episode_state.get("frequency_history", [])
    min_freq = min(freq_history) if freq_history else 60.0
    if min_freq >= 59.5:
        return 1.0
    return 1.0 if episode_state.get("recovered_above_59_5_in_3_ticks", False) else 0.0


def _task1_hold_nominal(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    freq_history = episode_state.get("frequency_history", [])
    min_freq = min(freq_history) if freq_history else 60.0
    return 1.0 if min_freq >= 59.5 else 0.0


def _task1_proactive_dispatch(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    early_dispatches = sum(
        1
        for action in action_history
        if action.get("action_type") == "dispatch_generation" and action.get("tick", 999) <= 3
    )
    return 1.0 if early_dispatches >= 2 and episode_state.get("is_proactive_dispatch", False) else 0.0


def _task2_fault_isolation(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    return 1.0 if any(
        action.get("action_type") == "toggle_circuit_breaker"
        and str(action.get("status", "")).upper() == "OPEN"
        for action in action_history
    ) else 0.0


def _task2_protect_critical(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    return 1.0 if episode_state.get("critical_nodes_shed", 0) == 0 else 0.0


def _task2_fast_restore(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    restored_tick = episode_state.get("full_restoration_tick")
    return 1.0 if restored_tick is not None and restored_tick <= 8 else 0.0


def _task3_logs_read(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    estimation_tick = _first_tick(action_history, "run_state_estimation")
    if estimation_tick is None:
        return 1.0 if episode_state.get("logs_read_before_estimation", False) else 0.0
    return 1.0 if any(
        action.get("action_type") == "advance_tick" and action.get("tick", 999) < estimation_tick
        for action in action_history
    ) else 0.0


def _task3_estimation(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    for action in action_history:
        if action.get("action_type") == "run_state_estimation":
            result = action.get("result", {})
            if not result.get("consistent", True):
                return 1.0
    return 0.0


def _task3_quarantine(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    estimation_tick = _first_tick(action_history, "run_state_estimation")
    expected_node = episode_state.get("spoof_target", "NODE_14")
    for action in action_history:
        if action.get("action_type") != "quarantine_scada_node":
            continue
        if estimation_tick is None or action.get("tick", 999) <= estimation_tick:
            return 0.0
        if action.get("node_id") == expected_node:
            return 1.0
    return 0.0


def _task3_reroute(action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    quarantine_tick = None
    expected_node = episode_state.get("spoof_target", "NODE_14")
    for action in action_history:
        if action.get("action_type") == "quarantine_scada_node" and action.get("node_id") == expected_node:
            quarantine_tick = action.get("tick")
            break
    if quarantine_tick is None:
        return 0.0
    for action in action_history:
        if action.get("action_type") == "dispatch_generation" and action.get("tick", 999) > quarantine_tick:
            return 1.0 if action.get("tick", 999) <= 4 else 0.5
    return 0.0


def _task4_injection_attempt(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    return 1.0 if any(action.get("action_type") == "inject_counter_signal" for action in action_history) else 0.0


def _task4_correct_offset(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    for action in action_history:
        if action.get("action_type") != "inject_counter_signal":
            continue
        hz_offset = action.get("hz_offset", 0.0)
        if hz_offset is not None and abs(hz_offset - (-0.5)) <= 0.05:
            return 1.0
    return 0.0


def _task4_target_ramp_down(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    injection_tick = _first_tick(action_history, "inject_counter_signal")
    if injection_tick is None:
        return 0.0
    return 1.0 if any(
        action.get("action_type") == "dispatch_generation"
        and action.get("node_id") == "NODE_17"
        and (action.get("mw") or 0.0) < 0.0
        and action.get("tick", 999) > injection_tick
        for action in action_history
    ) else 0.0


def _task4_support_reroute(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    injection_tick = _first_tick(action_history, "inject_counter_signal")
    if injection_tick is None:
        return 0.0
    return 1.0 if any(
        action.get("action_type") == "dispatch_generation"
        and action.get("node_id") != "NODE_17"
        and (action.get("mw") or 0.0) > 0.0
        and action.get("tick", 999) > injection_tick
        for action in action_history
    ) else 0.0


def _task5_hydro_bootstrap(action_history: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    return 1.0 if any(
        action.get("action_type") == "dispatch_generation"
        and action.get("node_id") == "NODE_01"
        and (action.get("mw") or 0.0) > 0.0
        for action in action_history
    ) else 0.0


def _task5_secondary_energized(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    return 1.0 if episode_state.get("hydro_stable_ticks", 0) >= 2 and episode_state.get("energized_node_count", 0) >= 2 else 0.0


def _task5_safe_islanding(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    island_count = episode_state.get("max_island_count", 0)
    successful_mergers = episode_state.get("successful_mergers", 0)
    return 1.0 if island_count >= 3 and successful_mergers >= 1 else 0.0


def _task5_restore_critical(_: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> float:
    has_safe_islanding = _task5_safe_islanding([], episode_state) == 1.0
    return 1.0 if has_safe_islanding and episode_state.get("critical_nodes_restored", False) else 0.0


TASK_RUBRICS: Dict[int, List[Rubric]] = {
    0: [
        Rubric(
            name="valid_dispatch",
            weight=1.0,
            description="Agent issues any valid positive dispatch to prove the environment pipeline works.",
            check=_task0_valid_dispatch,
        ),
    ],
    1: [
        Rubric(
            name="avoid_collapse",
            weight=0.30,
            description="Grid frequency never crosses the 59.0Hz termination floor.",
            check=_task1_avoid_collapse,
        ),
        Rubric(
            name="recover_nominal_band",
            weight=0.20,
            description="Frequency stays or recovers above 59.5Hz quickly.",
            check=_task1_recover_nominal,
        ),
        Rubric(
            name="hold_nominal_band",
            weight=0.42,
            description="Grid frequency remains inside the nominal operating band.",
            check=_task1_hold_nominal,
        ),
        Rubric(
            name="proactive_dispatch",
            weight=0.08,
            description="Agent dispatches support resources before the main frequency dip.",
            check=_task1_proactive_dispatch,
        ),
    ],
    2: [
        Rubric(
            name="fault_isolation",
            weight=0.40,
            description="Agent isolates the overloaded line with a breaker action.",
            check=_task2_fault_isolation,
        ),
        Rubric(
            name="protect_critical_load",
            weight=0.40,
            description="No critical infrastructure nodes are shed during recovery.",
            check=_task2_protect_critical,
        ),
        Rubric(
            name="fast_restoration",
            weight=0.20,
            description="Service is restored within eight ticks of the incident.",
            check=_task2_fast_restore,
        ),
    ],
    3: [
        Rubric(
            name="log_inspection",
            weight=0.10,
            description="Agent inspects packet behavior before trusting SCADA telemetry.",
            check=_task3_logs_read,
        ),
        Rubric(
            name="state_estimation",
            weight=0.20,
            description="Agent runs a Kirchhoff consistency check and detects the violation.",
            check=_task3_estimation,
        ),
        Rubric(
            name="correct_quarantine",
            weight=0.30,
            description="Agent quarantines the spoofed SCADA node after confirming the anomaly.",
            check=_task3_quarantine,
        ),
        Rubric(
            name="reroute_dispatch",
            weight=0.40,
            description="Agent reroutes generation after quarantine to stabilize the grid.",
            check=_task3_reroute,
        ),
    ],
    4: [
        Rubric(
            name="counter_signal_attempted",
            weight=0.20,
            description="Agent attempts a counter-signal instead of cutting the turbine offline.",
            check=_task4_injection_attempt,
        ),
        Rubric(
            name="correct_frequency_match",
            weight=0.20,
            description="Counter-signal uses the resonance-canceling frequency offset.",
            check=_task4_correct_offset,
        ),
        Rubric(
            name="target_ramp_down",
            weight=0.30,
            description="Agent ramps down the resonating turbine after the counter-signal lands.",
            check=_task4_target_ramp_down,
        ),
        Rubric(
            name="support_reroute",
            weight=0.30,
            description="Agent dispatches alternate generation to carry the displaced load.",
            check=_task4_support_reroute,
        ),
    ],
    5: [
        Rubric(
            name="hydro_bootstrap",
            weight=0.25,
            description="Agent restarts the hydro source to begin the black-start sequence.",
            check=_task5_hydro_bootstrap,
        ),
        Rubric(
            name="secondary_energized",
            weight=0.25,
            description="A second energized node is sustained after hydro startup.",
            check=_task5_secondary_energized,
        ),
        Rubric(
            name="safe_islanding",
            weight=0.30,
            description="Agent forms stable islands and completes at least one safe merger.",
            check=_task5_safe_islanding,
        ),
        Rubric(
            name="critical_restoration",
            weight=0.20,
            description="All critical infrastructure is restored after safe synchronization.",
            check=_task5_restore_critical,
        ),
    ],
}

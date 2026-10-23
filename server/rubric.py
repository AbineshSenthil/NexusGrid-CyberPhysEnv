"""
Composable task rubrics for NexusGrid scoring and training feedback.

Each rubric is a named, weighted check ``(action_history, episode_state) -> [0, 1]``.
Rubrics marked ``final_only`` are only awarded once the episode is done, so
that intermediate scores (used for reward shaping) never hand out credit for
outcomes that have not happened yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

CheckFn = Callable[[List[Dict[str, Any]], Dict[str, Any]], float]

BAND_LOW, BAND_HIGH = 59.8, 60.2


@dataclass(frozen=True)
class Rubric:
    """A single named rubric component."""

    name: str
    weight: float
    description: str
    check: CheckFn
    final_only: bool = False

    def evaluate(self, action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.final_only and not episode_state.get("done", True):
            raw_score = 0.0
        else:
            raw_score = float(self.check(action_history, episode_state))
        raw_score = max(0.0, min(1.0, raw_score))
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "raw": raw_score,
            "weighted": raw_score * self.weight,
            "final_only": self.final_only,
        }


def evaluate_task_rubrics(task_id: int, action_history: List[Dict[str, Any]], episode_state: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all rubrics for a task and return a serializable breakdown."""
    results = [rubric.evaluate(action_history, episode_state) for rubric in TASK_RUBRICS.get(task_id, [])]
    return {
        "task_id": task_id,
        "rubrics": {item["name"]: item["raw"] for item in results},
        "weights": {item["name"]: item["weight"] for item in results},
        "descriptions": {item["name"]: item["description"] for item in results},
        "weighted_components": {item["name"]: item["weighted"] for item in results},
        "weighted_score": sum(item["weighted"] for item in results),
    }


def get_task_rubric_specs(task_id: int) -> List[Dict[str, Any]]:
    return [
        {"name": r.name, "weight": r.weight, "description": r.description, "final_only": r.final_only}
        for r in TASK_RUBRICS.get(task_id, [])
    ]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _in_band(freq: float, low: float = BAND_LOW, high: float = BAND_HIGH) -> bool:
    return low <= freq <= high


def _freqs_after_reset(state: Dict[str, Any]) -> List[float]:
    history = state.get("frequency_history", [])
    return list(history[1:]) if len(history) > 1 else []


def _band_fraction_of_horizon(state: Dict[str, Any], low: float = BAND_LOW, high: float = BAND_HIGH) -> float:
    freqs = _freqs_after_reset(state)
    horizon = max(1, int(state.get("max_ticks", len(freqs) or 1)))
    return sum(1 for f in freqs if _in_band(f, low, high)) / horizon


def _last_k_band(state: Dict[str, Any], k: int, low: float = BAND_LOW, high: float = BAND_HIGH) -> float:
    freqs = _freqs_after_reset(state)[-k:]
    if not freqs:
        return 0.0
    return sum(1 for f in freqs if _in_band(f, low, high)) / len(freqs)


def mean_served(state: Dict[str, Any]) -> float:
    """Average share of demand served after reset (1.0 when no history yet)."""
    served = state.get("served_history", [])[1:]
    return sum(served) / len(served) if served else 1.0


def _events(state: Dict[str, Any], kind: str) -> List[Dict[str, Any]]:
    return [e for e in state.get("events", []) if e.get("type") == kind]


def _successful(actions: List[Dict[str, Any]], action_type: str) -> List[Dict[str, Any]]:
    return [
        a for a in actions
        if a.get("action_type") == action_type and (a.get("result") or {}).get("success", a.get("result") is None)
    ]


def _positive_dispatches(actions: List[Dict[str, Any]], exclude: Optional[str] = None) -> List[Dict[str, Any]]:
    return [
        a for a in _successful(actions, "dispatch_generation")
        if (a.get("mw") or 0.0) > 0.0 and (exclude is None or a.get("node_id") != exclude)
    ]


# ---------------------------------------------------------------------------
# Task 0 — smoke test
# ---------------------------------------------------------------------------

def _t0_valid_dispatch(actions: List[Dict[str, Any]], _: Dict[str, Any]) -> float:
    return 1.0 if _positive_dispatches(actions) else 0.0


def _t0_stable(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    freqs = state.get("frequency_history", [60.0])
    return 1.0 if _positive_dispatches(actions) and _in_band(freqs[-1]) else 0.0


# ---------------------------------------------------------------------------
# Task 1 — duck curve
# ---------------------------------------------------------------------------

def _t1_band_time(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    return _band_fraction_of_horizon(state)


def _t1_no_emergency(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    freqs = _freqs_after_reset(state) or [60.0]
    if _events(state, "ufls") or _events(state, "island_collapse"):
        return 0.0
    return 1.0 if min(freqs) >= 59.5 else 0.0


def _t1_no_overshoot(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    freqs = _freqs_after_reset(state) or [60.0]
    return 1.0 if max(freqs) <= 60.3 and _band_fraction_of_horizon(state) >= 0.5 else 0.0


def _t1_proactive(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    dispatches = _positive_dispatches(actions)
    if not dispatches:
        return 0.0
    first_dispatch = min(a.get("tick", 999) for a in dispatches)
    freqs = state.get("frequency_history", [])
    drop_ticks = [t for t, f in enumerate(freqs) if f < 59.9]
    first_drop = drop_ticks[0] if drop_ticks else None
    return 1.0 if first_drop is None or first_dispatch < first_drop else 0.0


# ---------------------------------------------------------------------------
# Task 2 — cascade overload
# ---------------------------------------------------------------------------

def _t2_cleared_from(state: Dict[str, Any], tick: int) -> bool:
    history = state.get("max_loading_history", [])
    tail = history[tick:]
    return bool(tail) and all(x <= 1.0 + 1e-6 for x in tail)


def _t2_relieve(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    trips = len(_events(state, "protection_trip"))
    if trips == 0 and _t2_cleared_from(state, 3):
        return 1.0
    if trips <= 1 and _t2_cleared_from(state, 6):
        return 0.5
    return 0.0


def _t2_protect_critical(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    return 0.0 if state.get("critical_lost_ever", False) else 1.0


def _t2_serve_load(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    if not state.get("served_history", [])[1:]:
        return 0.0
    return max(0.0, min(1.0, (mean_served(state) - 0.85) / 0.15))


def _t2_frequency(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    return _last_k_band(state, 5)


# ---------------------------------------------------------------------------
# Task 3 — phantom injection (false data injection)
# ---------------------------------------------------------------------------

def _t3_evidence_tick(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> Optional[int]:
    target = state.get("attack_target")
    for a in actions:
        if a.get("action_type") != "run_state_estimation":
            continue
        result = a.get("result") or {}
        if not result.get("consistent", True) and result.get("violation_node") == target:
            return a.get("tick")
    return None


def _t3_quarantine_tick(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> Optional[int]:
    target = state.get("attack_target")
    for a in _successful(actions, "quarantine_scada_node"):
        if a.get("node_id") == target:
            return a.get("tick")
    return None


def _t3_evidence(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    evidence = _t3_evidence_tick(actions, state)
    quarantine = _t3_quarantine_tick(actions, state)
    if evidence is None:
        return 0.0
    return 1.0 if quarantine is None or evidence <= quarantine else 0.0


def _t3_quarantine(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    quarantine = _t3_quarantine_tick(actions, state)
    evidence = _t3_evidence_tick(actions, state)
    if quarantine is None or evidence is None or evidence > quarantine:
        return 0.0
    start = int(state.get("attack_start_tick", 0))
    return 1.0 if quarantine <= start + 5 else 0.6


def _t3_recovery(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    quarantine = _t3_quarantine_tick(actions, state)
    if quarantine is None:
        return 0.0
    freqs = state.get("frequency_history", [])
    after = freqs[quarantine + 2:]
    if len(after) < 3:
        return 0.0
    return sum(1 for f in after if _in_band(f)) / len(after)


def _t3_triage(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    evidence = _t3_evidence_tick(actions, state)
    if evidence is None:
        return 0.0
    calls = sum(1 for a in actions if a.get("action_type") == "run_state_estimation" and a.get("tick", 999) <= evidence)
    return 1.0 if calls <= 2 else (0.5 if calls <= 4 else 0.0)


# ---------------------------------------------------------------------------
# Task 4 — forced oscillation (Stuxnet-style resonance)
# ---------------------------------------------------------------------------

def _t4_mitigate(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    quality = state.get("best_counter_signal", "none")
    return {"correct": 1.0, "partial": 0.4}.get(quality, 0.0)


def _t4_turbine_saved(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    if state.get("turbine_destroyed") or state.get("target_disconnected"):
        return 0.0
    return 1.0


def _t4_detune(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    return 1.0 if state.get("attack_neutralized") else 0.0


def _t4_frequency(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    return _last_k_band(state, 4)


def _t4_support(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    neutralized_tick = state.get("neutralized_tick")
    if neutralized_tick is None:
        return 0.0
    target = state.get("attack_target")
    support = [a for a in _positive_dispatches(actions, exclude=target) if a.get("tick", -1) >= neutralized_tick - 1]
    freqs = state.get("frequency_history", [60.0])
    return 1.0 if support and _in_band(freqs[-1]) else 0.0


# ---------------------------------------------------------------------------
# Task 5 — black start
# ---------------------------------------------------------------------------

def _t5_bootstrap(actions: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    unit = state.get("black_start_unit")
    return 1.0 if any(a.get("node_id") == unit for a in _positive_dispatches(actions)) else 0.0


def _t5_critical(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    return float(state.get("critical_energized_fraction", 0.0))


def _t5_load(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    return float(state.get("load_restored_fraction", 0.0))


def _t5_stability(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    if state.get("load_restored_fraction", 0.0) <= 0.0:
        return 0.0
    bad = len(_events(state, "island_collapse")) + len(_events(state, "out_of_phase_close"))
    return 1.0 if bad == 0 else (0.5 if bad == 1 else 0.0)


def _t5_frequency(_: List[Dict[str, Any]], state: Dict[str, Any]) -> float:
    if state.get("load_restored_fraction", 0.0) <= 0.0:
        return 0.0
    return _last_k_band(state, 5, low=59.5, high=60.5)


TASK_RUBRICS: Dict[int, List[Rubric]] = {
    0: [
        Rubric("valid_dispatch", 0.70, "Issue a valid positive dispatch on an energised generator.", _t0_valid_dispatch),
        Rubric("stable_after_dispatch", 0.30, "Frequency is inside 59.8-60.2 Hz after the dispatch.", _t0_stable, final_only=True),
    ],
    1: [
        Rubric("band_time", 0.50, "Share of the horizon with frequency inside 59.8-60.2 Hz.", _t1_band_time),
        Rubric("no_emergency", 0.25, "Frequency never drops below 59.5 Hz and no UFLS or collapse occurs.", _t1_no_emergency, final_only=True),
        Rubric("proactive_response", 0.15, "Support is dispatched before frequency first falls below 59.9 Hz.", _t1_proactive),
        Rubric("no_overshoot", 0.10, "No over-dispatch above 60.3 Hz (requires at least half the horizon in band).", _t1_no_overshoot, final_only=True),
    ],
    2: [
        Rubric("relieve_overload", 0.35, "All lines back under 100% by tick 3 with no relay trips (half credit by tick 6 with at most one trip).", _t2_relieve, final_only=True),
        Rubric("protect_critical_load", 0.25, "No critical load is de-energised at any point.", _t2_protect_critical, final_only=True),
        Rubric("serve_load", 0.20, "Average share of demand served stays high (85% -> 0, 100% -> 1).", _t2_serve_load, final_only=True),
        Rubric("frequency_band", 0.20, "Frequency inside 59.8-60.2 Hz over the last five ticks.", _t2_frequency, final_only=True),
    ],
    3: [
        Rubric("evidence", 0.25, "State estimation proves the falsified bus before any quarantine.", _t3_evidence),
        Rubric("correct_quarantine", 0.30, "The compromised bus is quarantined after evidence (full credit within five ticks of the attack).", _t3_quarantine),
        Rubric("recovery", 0.30, "Frequency stays inside 59.8-60.2 Hz after the quarantine.", _t3_recovery, final_only=True),
        Rubric("triage", 0.15, "Packet-log triage finds the compromised bus within two estimation calls.", _t3_triage),
    ],
    4: [
        Rubric("mitigate_oscillation", 0.35, "Counter-signal with the matching offset from a battery within two hops, before the unit fails.", _t4_mitigate),
        Rubric("turbine_saved", 0.20, "The oscillating unit survives and stays connected.", _t4_turbine_saved, final_only=True),
        Rubric("detune_unit", 0.15, "The unit is ramped down at least 20% while damped, neutralising the mode.", _t4_detune),
        Rubric("frequency_band", 0.15, "Frequency inside 59.8-60.2 Hz over the last four ticks.", _t4_frequency, final_only=True),
        Rubric("support_dispatch", 0.15, "Other units pick up the lost output and frequency ends in band.", _t4_support, final_only=True),
    ],
    5: [
        Rubric("bootstrap", 0.10, "The black-start unit is brought online.", _t5_bootstrap),
        Rubric("critical_restored", 0.30, "Share of critical loads energised.", _t5_critical),
        Rubric("load_restored", 0.30, "Share of total demand served.", _t5_load),
        Rubric("stability", 0.20, "No island collapse or out-of-phase closure during restoration.", _t5_stability, final_only=True),
        Rubric("frequency_band", 0.10, "Main island inside 59.5-60.5 Hz over the last five ticks.", _t5_frequency, final_only=True),
    ],
}

"""
NexusGrid — Dense Reward Calculator.

Per-tick reward computation with positive signals, penalties,
graduated frequency deviation bands, and the stability bonus.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Reward signal values
# ---------------------------------------------------------------------------

# Positive signals
REWARD_FAULT_ISOLATION = 0.20   # Isolating a fault without dropping critical nodes
REWARD_CYBER_DETECTION = 0.15   # Correctly classifying + quarantining a spoofed sensor
REWARD_FREQUENCY_STABLE = 0.10  # Per-tick frequency in nominal band (59.7–60.3Hz)
REWARD_PROACTIVE_DISPATCH = 0.08  # Dispatch before frequency deviation
REWARD_REASONING_ORDER = 0.05   # Reading packet logs before state estimation
REWARD_STABILITY_BONUS = 0.03   # Frequency within ±0.1Hz of 60.0Hz (tightest band)

# Negative signals
PENALTY_OVERLOAD_ROUTING = -0.20  # Routing through ≥95% capacity line
PENALTY_QUARANTINE_NO_EST = -0.15  # Quarantine without preceding state estimation
PENALTY_REDUNDANT_ESTIMATION = -0.05  # State estimation twice on same subgraph without action

# Graduated frequency deviation penalties
PENALTY_FREQ_59_2_TO_59_5 = -0.05  # Per tick in 59.2–59.5Hz band
PENALTY_FREQ_59_0_TO_59_2 = -0.15  # Per tick in 59.0–59.2Hz band


class RewardCalculator:
    """
    Computes per-tick rewards for the NexusGrid environment.

    Tracks action history to determine reward eligibility.
    Returns a breakdown dict and total for each tick.
    """

    def __init__(self):
        self._total_reward = 0.0
        self._has_read_logs = False
        self._has_run_estimation = False
        self._estimation_subgraphs: List[set] = []
        self._last_action_was_estimation = False
        self._actions_since_estimation = 0

    def reset(self) -> None:
        """Reset reward state for a new episode."""
        self._total_reward = 0.0
        self._has_read_logs = False
        self._has_run_estimation = False
        self._estimation_subgraphs = []
        self._last_action_was_estimation = False
        self._actions_since_estimation = 0

    def compute_tick_reward(
        self,
        action_type: str,
        action_params: Dict[str, Any],
        frequency_hz: float,
        overloaded_edges: List[str],
        critical_nodes_shed: int,
        is_proactive: bool,
        spoof_detected: bool,
        fault_isolated: bool,
        has_read_logs_before_estimation: bool,
    ) -> Dict[str, float]:
        """
        Compute all reward signals for a single tick.

        Returns:
            Dict with signal names as keys and values as floats.
            Includes 'total' key with the sum.
        """
        breakdown: Dict[str, float] = {
            "fault_isolation": 0.0,
            "cyber_detection": 0.0,
            "frequency_stable": 0.0,
            "proactive_dispatch": 0.0,
            "reasoning_order": 0.0,
            "stability_bonus": 0.0,
            "penalties": 0.0,
        }

        # --- Positive signals ---

        # Fault isolation (toggle_circuit_breaker that isolates without dropping critical)
        if fault_isolated and critical_nodes_shed == 0:
            breakdown["fault_isolation"] = REWARD_FAULT_ISOLATION

        # Cyber detection (quarantine after state estimation found violation)
        if spoof_detected and action_type == "quarantine_scada_node":
            if self._has_run_estimation:
                breakdown["cyber_detection"] = REWARD_CYBER_DETECTION

        # Frequency stability — nominal band
        if 59.7 <= frequency_hz <= 60.3:
            breakdown["frequency_stable"] = REWARD_FREQUENCY_STABLE

        # Proactive dispatch
        if action_type == "dispatch_generation" and is_proactive:
            breakdown["proactive_dispatch"] = REWARD_PROACTIVE_DISPATCH

        # Reasoning order — read logs before estimation
        if action_type == "run_state_estimation" and has_read_logs_before_estimation:
            breakdown["reasoning_order"] = REWARD_REASONING_ORDER

        # Stability bonus — tight band
        if 59.9 <= frequency_hz <= 60.1:
            breakdown["stability_bonus"] = REWARD_STABILITY_BONUS

        # --- Negative signals (penalties) ---

        penalties = 0.0

        # Overload routing
        if overloaded_edges and action_type in ("dispatch_generation", "toggle_circuit_breaker"):
            penalties += PENALTY_OVERLOAD_ROUTING

        # Quarantine without state estimation
        if action_type == "quarantine_scada_node" and not self._has_run_estimation:
            penalties += PENALTY_QUARANTINE_NO_EST

        # Redundant state estimation
        if action_type == "run_state_estimation":
            subgraph_set = set(action_params.get("subgraph", []))
            if self._last_action_was_estimation and subgraph_set in self._estimation_subgraphs:
                if self._actions_since_estimation == 0:
                    penalties += PENALTY_REDUNDANT_ESTIMATION
            self._estimation_subgraphs.append(subgraph_set)
            self._last_action_was_estimation = True
            self._has_run_estimation = True
            self._actions_since_estimation = 0
        else:
            self._last_action_was_estimation = False
            if self._has_run_estimation:
                self._actions_since_estimation += 1

        # Graduated frequency deviation penalties
        if 59.2 <= frequency_hz < 59.5:
            penalties += PENALTY_FREQ_59_2_TO_59_5
        elif 59.0 <= frequency_hz < 59.2:
            penalties += PENALTY_FREQ_59_0_TO_59_2

        breakdown["penalties"] = penalties

        # Total
        total = sum(breakdown.values())
        breakdown["total"] = total

        self._total_reward += total

        return breakdown

    @property
    def total_reward(self) -> float:
        """Get accumulated total reward."""
        return self._total_reward

    def mark_logs_read(self) -> None:
        """Mark that packet logs have been read (for reasoning order check)."""
        self._has_read_logs = True

    @property
    def has_read_logs(self) -> bool:
        """Whether packet logs have been read."""
        return self._has_read_logs

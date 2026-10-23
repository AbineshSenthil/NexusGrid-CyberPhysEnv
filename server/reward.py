"""
NexusGrid — Dense Reward.

The per-step reward is a potential-based shaping of the task score plus a
few small physical terms:

    r_t = (S_t - S_{t-1})          progress on the task rubrics
        + 0.02 * [f in 59.8-60.2]  frequency regulation
        - 0.02 * overloaded lines  thermal stress
        - event penalties          relay trips, load shedding, collapses
        - 0.05 * invalid action    malformed or rejected commands

Because the progress term telescopes, the sum of progress rewards over an
episode equals the final rubric score, so maximising return and maximising
the graded score point in the same direction.

When an episode ends early because the task was completed (black start), the
in-band bonus for the remaining ticks is paid at once. Otherwise the per-tick
bonus would reward an agent for delaying completion.
"""

from __future__ import annotations

from typing import Any, Dict, List

REWARD_IN_BAND = 0.02
PENALTY_PER_OVERLOAD = -0.02
PENALTY_INVALID_ACTION = -0.05
EVENT_PENALTIES = {
    "protection_trip": -0.10,
    "ufls": -0.05,
    "island_collapse": -0.30,
    "out_of_phase_close": -0.20,
    "turbine_destroyed": -0.30,
    "battery_depleted": -0.02,
}


class RewardCalculator:
    """Computes per-step rewards and keeps the running total."""

    def __init__(self) -> None:
        self._total_reward = 0.0

    def reset(self) -> None:
        self._total_reward = 0.0

    def compute_tick_reward(
        self,
        score_now: float,
        score_prev: float,
        frequency_hz: float,
        overloaded_count: int,
        events: List[Dict[str, Any]],
        invalid_action: bool,
        completed_ticks_remaining: int = 0,
    ) -> Dict[str, float]:
        in_band = REWARD_IN_BAND if 59.8 <= frequency_hz <= 60.2 else 0.0
        breakdown = {
            "progress": round(score_now - score_prev, 6),
            "frequency_band": round(in_band + REWARD_IN_BAND * max(0, int(completed_ticks_remaining)), 6),
            "overload": PENALTY_PER_OVERLOAD * int(overloaded_count),
            "events": sum(EVENT_PENALTIES.get(e.get("type", ""), 0.0) for e in events),
            "invalid_action": PENALTY_INVALID_ACTION if invalid_action else 0.0,
        }
        breakdown["total"] = round(sum(breakdown.values()), 6)
        self._total_reward += breakdown["total"]
        return breakdown

    @property
    def total_reward(self) -> float:
        return self._total_reward

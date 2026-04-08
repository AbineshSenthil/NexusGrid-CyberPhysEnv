"""
Test Reproducibility — runs environment episodes twice with the same seed,
asserts scores are identical (byte-identical).

Verifies the seed-lock contract: all randomness derives from
numpy.random.Generator(PCG64(episode_seed)).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import GridAction, ActionType
from server.nexusgrid_environment import NexusgridEnvironment


def run_deterministic_episode(task_id: int, seed: int) -> dict:
    """
    Run a deterministic episode using fixed actions (no LLM).

    Returns dict with score, frequency_history, and final tick.
    """
    env = NexusgridEnvironment()
    obs = env.reset(seed=seed, task_id=task_id)

    # Execute a fixed sequence of actions based on task
    if task_id == 0:
        actions = [
            GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id="NODE_01", mw=100),
        ]
    elif task_id == 1:
        actions = [
            GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id="NODE_04", mw=200),
            GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id="NODE_08", mw=200),
            GridAction(action_type=ActionType.ADVANCE_TICK),
            GridAction(action_type=ActionType.ADVANCE_TICK),
        ]
    else:
        actions = [
            GridAction(action_type=ActionType.ADVANCE_TICK),
            GridAction(action_type=ActionType.ADVANCE_TICK),
        ]

    rewards = []
    frequencies = []
    for action in actions:
        obs = env.step(action)
        rewards.append(obs.reward)
        frequencies.append(obs.grid_frequency_hz)
        if obs.done:
            break

    score = env.get_score()

    return {
        "score": score,
        "rewards": rewards,
        "frequencies": frequencies,
        "tick": obs.tick,
    }


class TestReproducibility:
    """Verify seed-lock contract: same seed → identical results."""

    def test_task0_reproducible(self):
        """Task 0 with same seed should produce identical scores."""
        result1 = run_deterministic_episode(task_id=0, seed=42)
        result2 = run_deterministic_episode(task_id=0, seed=42)

        assert result1["score"] == result2["score"], (
            f"Score mismatch: {result1['score']} != {result2['score']}"
        )
        assert result1["rewards"] == result2["rewards"], "Reward sequence mismatch"
        assert result1["frequencies"] == result2["frequencies"], "Frequency sequence mismatch"

    def test_task1_reproducible(self):
        """Task 1 with same seed should produce identical scores."""
        result1 = run_deterministic_episode(task_id=1, seed=42)
        result2 = run_deterministic_episode(task_id=1, seed=42)

        assert result1["score"] == result2["score"]
        assert result1["rewards"] == result2["rewards"]

    def test_task1_different_seed_differs(self):
        """Task 1 with different seeds should produce different results."""
        result1 = run_deterministic_episode(task_id=1, seed=42)
        result2 = run_deterministic_episode(task_id=1, seed=99)

        # Frequencies should differ (different weather evolution)
        assert result1["frequencies"] != result2["frequencies"], (
            "Different seeds should produce different frequency trajectories"
        )

    def test_all_tasks_reproducible(self):
        """All tasks should be reproducible with the same seed."""
        for task_id in range(6):
            result1 = run_deterministic_episode(task_id=task_id, seed=42)
            result2 = run_deterministic_episode(task_id=task_id, seed=42)

            assert result1["score"] == result2["score"], (
                f"Task {task_id} score not reproducible: "
                f"{result1['score']} != {result2['score']}"
            )

    def test_reset_idempotent(self):
        """Calling reset(42) 10 times should produce identical observations."""
        env = NexusgridEnvironment()
        freq_values = []

        for _ in range(10):
            obs = env.reset(seed=42, task_id=0)
            freq_values.append(obs.grid_frequency_hz)

        assert len(set(freq_values)) == 1, (
            f"reset(42) produced different frequencies: {freq_values}"
        )

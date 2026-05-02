"""Curriculum scheduling for NexusGrid training runs."""

from __future__ import annotations

from typing import Dict, List


class CurriculumManager:
    """Track recent task scores and recommend the next training task."""

    def __init__(self, unlock_threshold: float = 0.7, window: int = 10, num_tasks: int = 6):
        if num_tasks <= 0:
            raise ValueError("num_tasks must be positive")
        if window <= 0:
            raise ValueError("window must be positive")

        self.threshold = float(unlock_threshold)
        self.window = int(window)
        self.num_tasks = int(num_tasks)
        self.scores: Dict[int, List[float]] = {task_id: [] for task_id in range(self.num_tasks)}

    def reset(self) -> None:
        """Clear curriculum history."""
        self.scores = {task_id: [] for task_id in range(self.num_tasks)}

    def record_score(self, task_id: int, score: float) -> None:
        """Record a task score and keep only the latest `window` entries."""
        self._validate_task_id(task_id)
        bounded_score = max(0.0, min(1.0, float(score)))
        history = self.scores[task_id]
        history.append(bounded_score)
        if len(history) > self.window:
            del history[0 : len(history) - self.window]

    def get_task_average(self, task_id: int) -> float:
        """Return the recent rolling average for a task."""
        self._validate_task_id(task_id)
        history = self.scores[task_id]
        if not history:
            return 0.0
        return sum(history) / len(history)

    def is_unlocked(self, task_id: int) -> bool:
        """Whether a task is unlocked by the previous task's recent average."""
        self._validate_task_id(task_id)
        if task_id == 0:
            return True
        return self.get_task_average(task_id - 1) >= self.threshold

    def get_unlocked_tasks(self) -> List[int]:
        """Return all currently unlocked tasks."""
        return [task_id for task_id in range(self.num_tasks) if self.is_unlocked(task_id)]

    def get_recommended_task(self) -> int:
        """
        Recommend the next task to train on.
        Starts at Task 0 and unlocks subsequent tasks once the previous task's
        rolling average meets the threshold.
        """
        for task_id in range(self.num_tasks):
            if not self.is_unlocked(task_id):
                return max(0, task_id - 1)
            if self.get_task_average(task_id) < self.threshold:
                return task_id
        return self.num_tasks - 1

    def to_dict(self) -> Dict[str, object]:
        """Serialize the curriculum state for logs or dashboards."""
        return {
            "threshold": self.threshold,
            "window": self.window,
            "recommended_task": self.get_recommended_task(),
            "unlocked_tasks": self.get_unlocked_tasks(),
            "averages": {
                str(task_id): self.get_task_average(task_id) for task_id in range(self.num_tasks)
            },
        }

    def _validate_task_id(self, task_id: int) -> None:
        if task_id not in self.scores:
            raise ValueError(f"Unknown task_id: {task_id}")

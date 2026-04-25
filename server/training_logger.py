"""Structured JSONL logging utilities for training and evaluation runs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, TextIO


class TrainingLogger:
    """Write one JSON document per line for episode-level training telemetry."""

    def __init__(self, stream: TextIO | None = None):
        self._stream = stream or sys.stdout

    def build_record(
        self,
        episode: int,
        task_id: int,
        seed: int,
        score: float,
        rubrics: Dict[str, float],
        actions_taken: Iterable[str],
        frequency_min: float,
        ticks_used: int,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "episode": episode,
            "task_id": task_id,
            "seed": seed,
            "score": score,
            "rubrics": dict(rubrics),
            "actions_taken": list(actions_taken),
            "frequency_min": frequency_min,
            "ticks_used": ticks_used,
        }
        if extra:
            record.update(extra)
        return record

    def write_episode(self, record: Dict[str, Any]) -> str:
        line = json.dumps(record, sort_keys=True)
        self._stream.write(f"{line}\n")
        self._stream.flush()
        return line

"""Shared fixtures for the NexusGrid test-suite."""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import ActionType, GridAction  # noqa: E402
from server.grid_engine import GridEngine  # noqa: E402
from server.nexusgrid_environment import NexusgridEnvironment  # noqa: E402
from server.scenarios import build_base_topology  # noqa: E402


@pytest.fixture
def env() -> NexusgridEnvironment:
    return NexusgridEnvironment()


@pytest.fixture
def engine() -> GridEngine:
    eng = GridEngine(seed=7)
    build_base_topology(eng, load_factor=0.85, solar_irradiance=0.5)
    return eng


def noop() -> GridAction:
    return GridAction(action_type=ActionType.ADVANCE_TICK)


def run_policy(env: NexusgridEnvironment, policy, task_id: int, seed: int) -> float:
    obs = env.reset(seed=seed, task_id=task_id)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    while not obs.done:
        obs = env.step(policy.act(obs))
    return env.get_score()

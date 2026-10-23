"""
Shared pieces for GRPO training of language-model operators.

Nothing here needs a GPU or a model, so it is unit-tested in CI.

Prompts
    States are collected from expert trajectories with epsilon-exploration
    (random catalogue actions), so the model also sees off-nominal situations.
    Each record stores how to rebuild the state exactly: task, seed and the
    action prefix (the simulator is deterministic). With ``min_gap`` > 0 only
    *decisive* states are kept: those where the expert's action earns at least
    ``min_gap`` more reward than waiting. The expert only selects which states
    become prompts; it never contributes to the reward.

Reward
    The model's own action is executed in the rebuilt environment, followed by
    ``horizon`` passive ticks (``advance_tick``). The reward is the sum of the
    environment's dense rewards over those steps, so it measures what *this*
    action did, including delayed consequences such as relay trips. Output that
    is not a valid action JSON gets ``INVALID_REWARD``. No scripted fallback is
    ever substituted for the model's action.
"""

from __future__ import annotations

import copy
import json
import random
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from gym_env import CATALOG
from inference import SYSTEM_PROMPT, parse_action, render_observation, to_grid_action
from models import ActionType, GridAction
from policies import ExpertPolicy, GridView
from server.nexusgrid_environment import NexusgridEnvironment

INVALID_REWARD = -1.0
TRAIN_SEED_RANGE = (0, 9000)


def action_to_dict(action: GridAction) -> Dict[str, Any]:
    data = {k: v for k, v in action.model_dump().items() if v is not None}
    data["action_type"] = action.action_type.value
    return data


def history_line(tick: int, action: GridAction) -> str:
    return f"t{tick} {json.dumps(action_to_dict(action), separators=(',', ':'))}"


def build_messages(obs: Any, history: Sequence[str]) -> List[Dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": render_observation(obs, list(history))}]


def rollout_reward(env: NexusgridEnvironment, action: GridAction, horizon: int = 2) -> float:
    """Reward of ``action`` plus ``horizon`` passive ticks, on a copy of ``env`` (left untouched)."""
    env = copy.deepcopy(env)
    obs = env.step(action)
    total = float(obs.reward)
    for _ in range(horizon):
        if obs.done:
            break
        obs = env.step(GridAction(action_type=ActionType.ADVANCE_TICK))
        total += float(obs.reward)
    return total


def collect_states(tasks: Sequence[int], episodes: int, epsilon: float = 0.25, max_states_per_episode: int = 6,
                   seed: int = 0, seed_range: Tuple[int, int] = TRAIN_SEED_RANGE, min_gap: float = 0.0,
                   horizon: int = 2) -> List[Dict[str, Any]]:
    """Roll out the expert with exploration and sample decision states from each episode.

    ``min_gap`` > 0 keeps only states where acting like the expert beats waiting by at least that much
    reward (plus one quiet state per episode), so most prompts have an answer that measurably matters.
    """
    rng = random.Random(seed)
    env = NexusgridEnvironment()
    records: List[Dict[str, Any]] = []
    for episode in range(episodes):
        task_id = tasks[episode % len(tasks)]
        ep_seed = rng.randrange(*seed_range)
        expert = ExpertPolicy()
        obs = env.reset(seed=ep_seed, task_id=task_id)
        prefix: List[Dict[str, Any]] = []
        history: List[str] = []
        candidates: List[Dict[str, Any]] = []
        while not obs.done:
            expert_action = expert.act(obs)
            gap = None
            if min_gap > 0 and expert_action.action_type != ActionType.ADVANCE_TICK:
                gap = (rollout_reward(env, expert_action, horizon)
                       - rollout_reward(env, GridAction(action_type=ActionType.ADVANCE_TICK), horizon))
            candidates.append({
                "task_id": task_id,
                "seed": ep_seed,
                "tick": obs.tick,
                "prefix": json.dumps(prefix),
                "messages": build_messages(obs, history),
                "expert_action": json.dumps(action_to_dict(expert_action)),
                "gap": None if gap is None else round(gap, 6),
            })
            if rng.random() < epsilon:
                view = GridView(obs)
                valid = [i for i, ok in enumerate(CATALOG.mask(view)) if ok]
                action = CATALOG.to_action(rng.choice(valid), view)
            else:
                action = expert_action
            history.append(history_line(obs.tick, action))
            prefix.append(action_to_dict(action))
            obs = env.step(action)
        # Prefer states where the expert actually acted; keep a few quiet ones too.
        acted = [c for c in candidates if json.loads(c["expert_action"])["action_type"] != "advance_tick"]
        quiet = [c for c in candidates if c not in acted]
        active = [c for c in acted if min_gap <= 0 or c["gap"] >= min_gap]
        rng.shuffle(active)
        rng.shuffle(quiet)
        keep = active[: max_states_per_episode - 1] + quiet[:1]
        records.extend(keep[:max_states_per_episode])
    return records


_SNAPSHOTS: "OrderedDict[str, Tuple[NexusgridEnvironment, Any]]" = OrderedDict()
_SNAPSHOT_LIMIT = 64


def rebuild(task_id: int, seed: int, prefix: Iterable[Dict[str, Any]]) -> Tuple[NexusgridEnvironment, Any]:
    """Recreate the exact state (the simulator is deterministic); snapshots are cached per prompt."""
    prefix = list(prefix)
    key = json.dumps([int(task_id), int(seed), prefix], sort_keys=True)
    if key not in _SNAPSHOTS:
        env = NexusgridEnvironment()
        obs = env.reset(seed=int(seed), task_id=int(task_id))
        for action in prefix:
            obs = env.step(GridAction(**action))
        _SNAPSHOTS[key] = (env, obs)
        if len(_SNAPSHOTS) > _SNAPSHOT_LIMIT:
            _SNAPSHOTS.popitem(last=False)
    else:
        _SNAPSHOTS.move_to_end(key)
    env, obs = _SNAPSHOTS[key]
    return copy.deepcopy(env), obs


def score_action(action: Optional[GridAction], task_id: int, seed: int, prefix: Sequence[Dict[str, Any]],
                 horizon: int = 2) -> Dict[str, Any]:
    """Dense reward of ``action`` plus ``horizon`` passive ticks, from the rebuilt state."""
    if action is None:
        return {"reward": INVALID_REWARD, "valid": False, "error": "unparseable"}
    env, obs = rebuild(task_id, seed, prefix)
    if obs.done:
        return {"reward": 0.0, "valid": True, "error": "episode already finished"}
    obs = env.step(action)
    total = float(obs.reward)
    error = obs.last_action_error
    for _ in range(horizon):
        if obs.done:
            break
        obs = env.step(GridAction(action_type=ActionType.ADVANCE_TICK))
        total += float(obs.reward)
    return {"reward": round(total, 6), "valid": True, "error": error}


def score_completion(text: str, task_id: int, seed: int, prefix: str | Sequence[Dict[str, Any]],
                     horizon: int = 2) -> Dict[str, Any]:
    prefix_list = json.loads(prefix) if isinstance(prefix, str) else list(prefix)
    return score_action(to_grid_action(parse_action(text)), task_id, seed, prefix_list, horizon)


def completion_text(completion: Any) -> str:
    """TRL passes plain strings or chat-style message lists depending on the dataset format."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[-1].get("content", ""))
    return str(completion)


def make_reward_fn(horizon: int = 2):
    """Reward function with the signature TRL's GRPOTrainer expects."""

    def nexusgrid_reward(prompts, completions, task_id, seed, prefix, **kwargs) -> List[float]:
        rewards = []
        for text, t, s, p in zip(completions, task_id, seed, prefix):
            rewards.append(score_completion(completion_text(text), int(t), int(s), p, horizon)["reward"])
        return rewards

    nexusgrid_reward.__name__ = "nexusgrid_reward"
    return nexusgrid_reward

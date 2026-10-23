#!/usr/bin/env python3
"""
Train a MaskablePPO agent on NexusGrid (CPU is enough).

Three regimes, selected with ``--bc-episodes``:

* ``0``   PPO from scratch.
* ``N``   behaviour cloning on N expert episodes first (the observation-only
          expert, restricted to the discrete action catalogue), then PPO
          fine-tuning from that initialisation.
* add ``--bc-only`` to stop after behaviour cloning.

Training episodes use seeds 0-8999. Seeds 9000-9099 are used for model
selection during training; seeds >= 10000 stay untouched for the final
held-out evaluation (``scripts/evaluate.py --policies ppo --ppo-model ...``).

Outputs in ``--out``: ``curve.csv`` (timesteps, validation score, episode
return), ``config.json``, ``best.zip`` (best validation score), ``final.zip``.

Example:
    python training/train_ppo.py --tasks 1 --timesteps 300000 --bc-episodes 200 --out runs/ppo/task1_bc
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from sb3_contrib import MaskablePPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: E402

from gym_env import CATALOG, DiscretizedExpert, NexusGridGymEnv, featurize  # noqa: E402
from policies import GridView  # noqa: E402
from server.nexusgrid_environment import NexusgridEnvironment  # noqa: E402

TRAIN_SEEDS = (0, 9000)
VALIDATION_SEEDS = list(range(9000, 9100))


def make_env(tasks: Sequence[int], rank: int):
    def _init():
        env = NexusGridGymEnv(tasks=tasks, seed_range=TRAIN_SEEDS)
        env.reset(seed=10_000 * (rank + 1))
        return env
    return _init


def evaluate_model(model: MaskablePPO, tasks: Sequence[int], seeds: Sequence[int]) -> Dict[str, float]:
    env = NexusgridEnvironment()
    scores: List[float] = []
    for task_id in tasks:
        for seed in seeds:
            obs = env.reset(seed=seed, task_id=task_id)
            while not obs.done:
                view = GridView(obs)
                action, _ = model.predict(featurize(view), action_masks=CATALOG.mask(view), deterministic=True)
                obs = env.step(CATALOG.to_action(int(action), view))
            scores.append(env.get_score())
    return {"mean": float(np.mean(scores)), "std": float(np.std(scores))}


def collect_demonstrations(tasks: Sequence[int], episodes: int, seed: int = 0):
    """Roll out the discretised expert; returns features, masks, actions and discounted returns."""
    rng = np.random.default_rng(seed)
    env = NexusgridEnvironment()
    feats, masks, actions, returns = [], [], [], []
    for _ in range(episodes):
        task_id = int(rng.choice(tasks))
        ep_seed = int(rng.integers(*TRAIN_SEEDS))
        expert = DiscretizedExpert()
        obs = env.reset(seed=ep_seed, task_id=task_id)
        rewards = []
        while not obs.done:
            view = GridView(obs)
            idx = expert.act_index(obs)
            feats.append(featurize(view))
            masks.append(CATALOG.mask(view))
            actions.append(idx)
            obs = env.step(CATALOG.to_action(idx, view))
            rewards.append(float(obs.reward))
        g, ep_returns = 0.0, []
        for r in reversed(rewards):
            g = r + 0.99 * g
            ep_returns.append(g)
        returns.extend(reversed(ep_returns))
    return (np.asarray(feats, dtype=np.float32), np.asarray(masks, dtype=bool),
            np.asarray(actions, dtype=np.int64), np.asarray(returns, dtype=np.float32))


def behaviour_cloning(model: MaskablePPO, data, epochs: int, batch_size: int = 256, lr: float = 1e-3) -> List[float]:
    """Supervised pre-training of the actor (cross-entropy) and critic (Monte-Carlo returns)."""
    feats, masks, actions, returns = data
    policy = model.policy
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    n = len(actions)
    losses = []
    for _ in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            obs_t = torch.as_tensor(feats[idx])
            act_t = torch.as_tensor(actions[idx])
            ret_t = torch.as_tensor(returns[idx])
            values, log_prob, entropy = policy.evaluate_actions(obs_t, act_t, action_masks=masks[idx])
            loss = -log_prob.mean() + 0.5 * torch.nn.functional.mse_loss(values.flatten(), ret_t) \
                - 0.001 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(loss.item())
        losses.append(statistics.mean(epoch_loss))
    return losses


class CurveCallback(BaseCallback):
    def __init__(self, tasks: Sequence[int], out: Path, every: int, val_seeds: Sequence[int], writer):
        super().__init__()
        self.tasks, self.out, self.every, self.val_seeds, self.writer = tasks, out, every, val_seeds, writer
        self.next_eval = every
        self.best = -1.0
        self.scores: List[float] = []
        self.started = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "score" in info:
                self.scores.append(info["score"])
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            self.record()
        return True

    def record(self) -> None:
        if getattr(self, "_last_recorded", None) == self.num_timesteps and self.num_timesteps > 0:
            return
        self._last_recorded = self.num_timesteps
        val = evaluate_model(self.model, self.tasks, self.val_seeds)
        train_score = float(np.mean(self.scores[-200:])) if self.scores else float("nan")
        row = {"timesteps": self.num_timesteps, "val_score": round(val["mean"], 4), "val_std": round(val["std"], 4),
               "train_score": round(train_score, 4), "minutes": round((time.time() - self.started) / 60, 2)}
        self.writer.writerow(row)
        print(json.dumps(row), flush=True)
        if val["mean"] > self.best:
            self.best = val["mean"]
            self.model.save(self.out / "best.zip")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="1")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--subproc", action="store_true", help="one process per environment")
    parser.add_argument("--bc-episodes", type=int, default=0)
    parser.add_argument("--bc-epochs", type=int, default=30)
    parser.add_argument("--bc-only", action="store_true")
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--val-seeds", type=int, default=10, help="validation seeds per task")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", default="runs/ppo/task1")
    args = parser.parse_args(argv)

    torch.set_num_threads(args.threads)
    tasks = [int(t) for t in args.tasks.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    val_seeds = VALIDATION_SEEDS[: args.val_seeds]

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    venv = vec_cls([make_env(tasks, i) for i in range(args.n_envs)])
    model = MaskablePPO(
        "MlpPolicy", venv, learning_rate=args.learning_rate, n_steps=512, batch_size=256, n_epochs=8,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])), seed=args.seed, verbose=0, device="cpu",
    )

    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["timesteps", "val_score", "val_std", "train_score", "minutes"])
        writer.writeheader()
        callback = CurveCallback(tasks, out, args.eval_every, val_seeds, writer)
        callback.model = model
        callback.num_timesteps = 0
        callback.record()  # untrained (or nothing yet)
        fh.flush()

        if args.bc_episodes > 0:
            started = time.time()
            data = collect_demonstrations(tasks, args.bc_episodes, seed=args.seed)
            losses = behaviour_cloning(model, data, args.bc_epochs)
            info = {"bc_samples": int(len(data[2])), "bc_loss_first": losses[0], "bc_loss_last": losses[-1],
                    "bc_minutes": round((time.time() - started) / 60, 2)}
            (out / "bc.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(json.dumps(info), flush=True)
            callback.record()
            fh.flush()
            model.save(out / "bc.zip")
        if not args.bc_only and args.timesteps > 0:
            model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=False)
            callback.record()
        fh.flush()
    model.save(out / "final.zip")
    venv.close()
    print(f"done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

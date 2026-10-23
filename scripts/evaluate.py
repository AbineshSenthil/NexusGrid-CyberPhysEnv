#!/usr/bin/env python3
"""
Evaluate policies on NexusGrid.

Runs every (policy, task, seed) episode in-process and reports the mean task
score with its standard deviation and a 95 % confidence interval. Training
seeds start at 0; held-out evaluation seeds start at 10 000 and draw from
parameter ranges (and, for Task 2, contingencies) never used in training.

Examples
--------
    python scripts/evaluate.py                       # baselines, 50 held-out seeds per task
    python scripts/evaluate.py --policies expert --tasks 2,3 --seeds 200
    python scripts/evaluate.py --policies llm --seeds 5 --model qwen3:1.7b \\
        --api-base http://localhost:11434/v1        # any OpenAI-compatible server
    python scripts/evaluate.py --policies ppo --ppo-model runs/ppo/task2/final.zip
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.nexusgrid_environment import NexusgridEnvironment  # noqa: E402
from server.scenarios import EVAL_SEED_START, TASK_NAMES  # noqa: E402


def build_policy(name: str, seed: int, options: Dict[str, Any]):
    if name in ("noop", "random", "expert"):
        from policies import make_policy
        return make_policy(name, seed)
    if name == "llm":
        from inference import LLMPolicy
        return LLMPolicy(
            model=options["model"],
            api_base=options["api_base"],
            api_key=options.get("api_key"),
            fallback=options.get("fallback", False),
            temperature=options.get("temperature", 0.0),
        )
    if name == "ppo":
        from gym_env import PPOPolicy
        return PPOPolicy(options["ppo_model"], deterministic=True)
    raise ValueError(f"Unknown policy: {name}")


def run_episode(policy_name: str, task_id: int, seed: int, options: Dict[str, Any]) -> Dict[str, Any]:
    env = NexusgridEnvironment()
    policy = build_policy(policy_name, seed, options)
    if hasattr(policy, "reset"):
        policy.reset(seed)
    started = time.time()
    obs = env.reset(seed=seed, task_id=task_id)
    while not obs.done:
        obs = env.step(policy.act(obs))
    summary = env.get_episode_summary()
    record = {
        "policy": policy_name,
        "task_id": task_id,
        "task_name": TASK_NAMES[task_id],
        "seed": seed,
        "split": summary["split"],
        "score": round(summary["score"], 6),
        "total_reward": summary["total_reward"],
        "rubrics": {k: round(v, 4) for k, v in summary["rubrics"].items()},
        "steps": summary["ticks_used"],
        "invalid_actions": summary["invalid_actions"],
        "events": summary["events"],
        "seconds": round(time.time() - started, 3),
    }
    stats = getattr(policy, "stats", None)
    if callable(stats):
        record["policy_stats"] = stats()
    return record


def summarise(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["policy"], r["task_id"], r["split"]), []).append(r)
    rows = []
    for (policy, task_id, split), items in sorted(groups.items()):
        scores = [r["score"] for r in items]
        mean = statistics.mean(scores)
        sd = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        ci = 1.96 * (statistics.stdev(scores) / math.sqrt(len(scores))) if len(scores) > 1 else 0.0
        rows.append({
            "policy": policy,
            "task_id": task_id,
            "task_name": TASK_NAMES[task_id],
            "split": split,
            "episodes": len(items),
            "mean_score": round(mean, 4),
            "std": round(sd, 4),
            "ci95": round(ci, 4),
            "success_rate": round(sum(1 for s in scores if s >= 0.8) / len(scores), 4),
            "mean_invalid_actions": round(statistics.mean(r["invalid_actions"] for r in items), 3),
            "mean_steps": round(statistics.mean(r["steps"] for r in items), 2),
        })
    return rows


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    policies = sorted({r["policy"] for r in rows}, key=lambda p: ["noop", "random", "expert"].index(p)
                      if p in ("noop", "random", "expert") else 10)
    tasks = sorted({(r["task_id"], r["task_name"]) for r in rows})
    splits = sorted({r["split"] for r in rows})
    lookup = {(r["policy"], r["task_id"], r["split"]): r for r in rows}
    lines = []
    for split in splits:
        lines.append(f"**{split} seeds**\n")
        lines.append("| Task | " + " | ".join(policies) + " |")
        lines.append("|---|" + "---|" * len(policies))
        for task_id, name in tasks:
            cells = []
            for p in policies:
                r = lookup.get((p, task_id, split))
                cells.append(f"{r['mean_score']:.3f} ± {r['std']:.3f}" if r else "–")
            lines.append(f"| {task_id} · {name} | " + " | ".join(cells) + " |")
        means = []
        for p in policies:
            vals = [lookup[(p, t, split)]["mean_score"] for t, _ in tasks if (p, t, split) in lookup]
            means.append(f"**{statistics.mean(vals):.3f}**" if vals else "–")
        lines.append("| **Mean** | " + " | ".join(means) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policies", default="noop,random,expert")
    parser.add_argument("--tasks", default="0,1,2,3,4,5")
    parser.add_argument("--seeds", type=int, default=50, help="episodes per task and split")
    parser.add_argument("--split", choices=["eval", "train", "both"], default="eval")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default="results/eval")
    parser.add_argument("--model", default="qwen3:1.7b", help="LLM model name (llm policy)")
    parser.add_argument("--api-base", default="http://localhost:11434/v1", help="OpenAI-compatible base URL")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--fallback", action="store_true", help="LLM: replace invalid outputs with the expert action")
    parser.add_argument("--ppo-model", default=None, help="path to a trained MaskablePPO .zip")
    args = parser.parse_args(argv)

    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    tasks = [int(t) for t in args.tasks.split(",")]
    seed_sets = []
    if args.split in ("train", "both"):
        seed_sets.append(list(range(args.seeds)))
    if args.split in ("eval", "both"):
        seed_sets.append(list(range(EVAL_SEED_START, EVAL_SEED_START + args.seeds)))
    options = {"model": args.model, "api_base": args.api_base, "api_key": args.api_key,
               "fallback": args.fallback, "ppo_model": args.ppo_model}

    jobs = [(p, t, s) for p in policies for t in tasks for seeds in seed_sets for s in seeds]
    started = time.time()
    records: List[Dict[str, Any]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_episode, p, t, s, options) for p, t, s in jobs]
            for f in futures:
                records.append(f.result())
    else:
        for i, (p, t, s) in enumerate(jobs, 1):
            records.append(run_episode(p, t, s, options))
            if p == "llm":
                print(f"[{i}/{len(jobs)}] {p} task {t} seed {s}: {records[-1]['score']:.3f}", flush=True)

    rows = summarise(records)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({
        "policies": policies, "tasks": tasks, "seeds_per_task": args.seeds, "split": args.split,
        "eval_seed_start": EVAL_SEED_START, "rows": rows,
    }, indent=2), encoding="utf-8")
    table = markdown_table(rows)
    (out / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"{len(records)} episodes in {time.time() - started:.1f}s -> {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

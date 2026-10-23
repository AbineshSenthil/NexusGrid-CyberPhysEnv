#!/usr/bin/env python3
"""
Build the README figures from saved results (static SVG, light surface).

SVG keeps the repository free of binary files, which Hugging Face rejects in plain git pushes.

    python scripts/make_figures.py

Inputs (missing ones are skipped):
    results/baselines/summary.json          scripts/evaluate.py --policies noop,random,expert
    results/ppo/<run>/summary.json          scripts/evaluate.py --policies ppo
    results/ppo/curves/<run>.csv            copies of runs/ppo/<run>/curve.csv
    results/llm/<run>/summary.json          training/eval_llm_local.py
    results/llm/grpo_train_log.json         copy of runs/grpo/<run>/train_log.json
Outputs: docs/figures/*.svg
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"

# Reference palette (light mode) and chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
FONT = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
TASKS_RL = [1, 2, 3, 4, 5]

POLICY_LABELS = {
    "noop": "No-op",
    "random": "Random",
    "expert": "Expert (physics heuristics)",
    "ppo_scratch": "PPO from scratch",
    "bc_only": "Behaviour cloning",
    "bc_ppo": "BC → PPO",
    "llm-base": "Qwen3-1.7B (no training)",
    "llm-sft": "Qwen3-1.7B, warm start",
    "llm-grpo": "Qwen3-1.7B, warm start → GRPO",
}


def _setup() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT,
        "font.size": 10,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "path",        # render identically without the fonts installed
        "svg.hashsalt": "nexusgrid",   # stable element ids, so unchanged charts give unchanged files
    })


def _style(ax, title: str, subtitle: str = "") -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0)


def _mean_over(rows: List[Dict], tasks: List[int]) -> Optional[float]:
    vals = [r["mean_score"] for r in rows if r["task_id"] in tasks and r.get("split", "eval") == "eval"]
    return statistics.mean(vals) if len(vals) == len(tasks) else None


def _load_rows(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def collect_means(tasks: List[int]) -> List[Tuple[str, float, int]]:
    """(label, mean score over tasks, seeds per task) for every available policy."""
    out: List[Tuple[str, float, int]] = []
    base = _load_rows(RESULTS / "baselines" / "summary.json")
    for policy in ("noop", "random", "expert"):
        rows = [r for r in base if r["policy"] == policy]
        mean = _mean_over(rows, tasks)
        if mean is not None:
            out.append((POLICY_LABELS[policy], mean, rows[0]["episodes"]))
    for run in ("ppo_scratch", "bc_only", "bc_ppo"):
        rows = _load_rows(RESULTS / "ppo" / run / "summary.json")
        mean = _mean_over(rows, tasks)
        if mean is not None:
            out.append((POLICY_LABELS[run], mean, rows[0]["episodes"]))
    for run, key in (("qwen3-1.7b-base", "llm-base"), ("qwen3-1.7b-sft", "llm-sft"),
                     ("qwen3-1.7b-grpo", "llm-grpo")):
        rows = _load_rows(RESULTS / "llm" / run / "summary.json")
        mean = _mean_over(rows, tasks)
        if mean is not None:
            out.append((POLICY_LABELS[key], mean, rows[0]["episodes"]))
    return out


def _span(counts: List[int]) -> str:
    counts = sorted(set(counts))
    return f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"


def _seed_note(means: List[Tuple[str, float, int]], llm_labels: set) -> str:
    """'50–100 seeds per task (LLMs: 5)' style note for the chart subtitle."""
    other = [n for label, _, n in means if label not in llm_labels]
    llm = [n for label, _, n in means if label in llm_labels]
    note = f"{_span(other or llm)} seeds per task"
    if other and llm:
        note += f" (LLMs: {_span(llm)})"
    return note


def _rounded_hbar(ax, y: float, value: float, height: float, color: str, radius_px: float = 4.0) -> None:
    """Horizontal bar, square at the baseline with a rounded data end."""
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform((0, 0)), inv.transform((radius_px, radius_px))
    rx, ry = abs(x1 - x0), abs(y1 - y0)
    if value <= 0:
        return
    if value <= 2 * rx:
        ax.add_patch(Rectangle((0, y - height / 2), value, height, color=color, lw=0))
        return
    ax.add_patch(Rectangle((0, y - height / 2), value - rx, height, color=color, lw=0))
    ax.add_patch(FancyBboxPatch((value - 2 * rx, y - height / 2), 2 * rx, height,
                                boxstyle=f"round,pad=0,rounding_size={rx}", mutation_aspect=ry / rx,
                                color=color, lw=0))


def plot_policy_comparison(out: Path) -> Optional[Path]:
    means = collect_means(TASKS_RL)
    if not means:
        return None
    means.sort(key=lambda item: item[1])
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(means) + 1.4), dpi=150)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.6, len(means) - 0.4)
    fig.canvas.draw()
    for i, (label, value, n) in enumerate(means):
        _rounded_hbar(ax, i, value, 0.56, SERIES[0])
        ax.text(value + 0.012, i, f"{value:.2f}", va="center", ha="left", color=INK, fontsize=9)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels([label for label, _, _ in means], color=INK_2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    llm_labels = {POLICY_LABELS[k] for k in ("llm-base", "llm-sft", "llm-grpo")}
    _style(ax, "Mean score on held-out scenarios (tasks 1-5)",
           "Seeds ≥ 10 000, never used in training; " + _seed_note(means, llm_labels) + ".")
    ax.set_xlabel("mean task score (1.0 = every rubric met)")
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.grid(False, axis="y")
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Date": None})
    plt.close(fig)
    return out


def _read_curve(path: Path) -> List[Tuple[float, float]]:
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("val_score") not in (None, "", "nan")]
    return [(float(r["timesteps"]), float(r["val_score"])) for r in rows]


def plot_ppo_curves(out: Path) -> Optional[Path]:
    curves_dir = RESULTS / "ppo" / "curves"
    runs = [("bc_ppo", "BC → PPO"), ("ppo_scratch", "PPO from scratch")]
    available = [(name, label, curves_dir / f"{name}.csv") for name, label in runs
                 if (curves_dir / f"{name}.csv").exists()]
    if not available:
        return None
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    ends = []
    for i, (name, label, path) in enumerate(available):
        points = _read_curve(path)
        if name == "bc_ppo" and len(points) > 1 and points[0][0] == points[1][0] == 0:
            points = points[1:]  # keep the post-cloning point at step 0
        xs = [p[0] / 1000 for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=SERIES[i], linewidth=1.8, solid_capstyle="round", solid_joinstyle="round",
                label=label, marker="o", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.0)
        ends.append((xs[-1], ys[-1], label))
    base = _load_rows(RESULTS / "baselines" / "summary.json")
    refs = []
    for policy, name in (("expert", "expert"), ("random", "random")):
        mean = _mean_over([r for r in base if r["policy"] == policy], TASKS_RL)
        if mean is not None:
            refs.append((name, mean))
    for name, value in refs:
        ax.axhline(value, color=MUTED, linewidth=0.9)
        ax.text(0.995, value - 0.02 if value > 0.9 else value + 0.015, f"{name} {value:.2f}",
                transform=ax.get_yaxis_transform(), color=INK_2, fontsize=8.5, ha="right",
                va="top" if value > 0.9 else "bottom")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("validation score")
    _style(ax, "Masked PPO on tasks 1-5: validation score during training",
           "50 validation episodes per point (seeds 9000-9009, excluded from training).")
    ax.legend(frameon=False, loc="center right", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Date": None})
    plt.close(fig)
    return out


def plot_grpo_reward(out: Path) -> Optional[Path]:
    path = RESULTS / "llm" / "grpo_train_log.json"
    if not path.exists():
        return None
    history = json.loads(path.read_text(encoding="utf-8"))
    points = [(h["step"], float(h["reward"])) for h in history if "reward" in h and "step" in h]
    if not points:
        return None
    steps, rewards = zip(*points)
    window = max(5, len(rewards) // 5)
    smooth = [statistics.mean(rewards[max(0, i - window + 1): i + 1]) for i in range(len(rewards))]
    fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
    ax.plot(steps, rewards, color="#b7d3f6", linewidth=1.0, label="mean reward per step")
    ax.plot(steps, smooth, color=SERIES[0], linewidth=2.0, label=f"moving average ({window} steps)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("simulator reward of sampled actions")
    _style(ax, "GRPO on Qwen3-1.7B after the warm start: reward of its own actions",
           "Mean over 16 sampled actions per step, each executed in the rebuilt scenario; invalid JSON scores −1.")
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Date": None})
    plt.close(fig)
    return out


def main() -> int:
    _setup()
    made = [plot_policy_comparison(FIGURES / "policy_comparison.svg"),
            plot_ppo_curves(FIGURES / "ppo_learning_curves.svg"),
            plot_grpo_reward(FIGURES / "grpo_reward.svg")]
    for path in made:
        if path:
            print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

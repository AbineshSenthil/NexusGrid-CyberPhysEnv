"""Plots for training curves and evaluation summaries (matplotlib, headless)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1f2937"
GRID = "#e5e7eb"
SERIES = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=INK, fontsize=12, loc="left")
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def read_curve(path: Path) -> List[Dict[str, float]]:
    with open(path, encoding="utf-8") as fh:
        return [{k: float(v) for k, v in row.items() if v not in ("", "nan")} for row in csv.DictReader(fh)]


def plot_ppo_curves(runs: Dict[str, Path], out: Path, title: str, references: Dict[str, float] | None = None) -> None:
    """Validation score vs. timesteps for several runs, with horizontal reference lines."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    for i, (label, path) in enumerate(runs.items()):
        rows = [r for r in read_curve(path) if "val_score" in r]
        ax.plot([r["timesteps"] for r in rows], [r["val_score"] for r in rows], marker="o", markersize=3,
                color=SERIES[i % len(SERIES)], label=label, linewidth=2)
    for j, (label, value) in enumerate((references or {}).items()):
        ax.axhline(value, linestyle="--", linewidth=1.2, color=["#6b7280", "#9ca3af", "#111827"][j % 3])
        ax.text(ax.get_xlim()[1], value, f" {label} {value:.2f}", va="center", fontsize=8, color=INK)
    ax.set_ylim(-0.02, 1.05)
    _style(ax, title, "environment steps", "validation score (30 seeds)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_grpo_log(log_path: Path, out: Path) -> None:
    history = json.loads(Path(log_path).read_text(encoding="utf-8"))
    points = [(h["step"], h["reward"]) for h in history if "reward" in h and "step" in h]
    if not points:
        return
    steps, rewards = zip(*points)
    window = max(1, len(rewards) // 20)
    smooth = [sum(rewards[max(0, i - window + 1): i + 1]) / len(rewards[max(0, i - window + 1): i + 1])
              for i in range(len(rewards))]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    ax.plot(steps, rewards, color="#93c5fd", linewidth=1, label="per step")
    ax.plot(steps, smooth, color="#1d4ed8", linewidth=2.2, label=f"moving average ({window})")
    _style(ax, "GRPO training reward (simulator-scored actions)", "optimizer step", "mean reward")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_policy_bars(rows: Sequence[Dict], out: Path, policies: Sequence[str], title: str) -> None:
    """Grouped bars: mean held-out score per task for each policy."""
    tasks = sorted({(r["task_id"], r["task_name"]) for r in rows})
    lookup = {(r["policy"], r["task_id"]): r for r in rows}
    width = 0.8 / max(1, len(policies))
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=150)
    for i, policy in enumerate(policies):
        xs = [t + (i - (len(policies) - 1) / 2) * width for t, _ in tasks]
        ys = [lookup.get((policy, t), {}).get("mean_score", 0.0) for t, _ in tasks]
        err = [lookup.get((policy, t), {}).get("ci95", 0.0) for t, _ in tasks]
        ax.bar(xs, ys, width=width * 0.95, color=SERIES[i % len(SERIES)], label=policy, yerr=err,
               error_kw={"elinewidth": 0.8, "capsize": 2, "ecolor": INK})
    ax.set_xticks([t for t, _ in tasks])
    ax.set_xticklabels([f"T{t}\n{name.replace('_', ' ')}" for t, name in tasks], fontsize=8)
    ax.set_ylim(0, 1.08)
    _style(ax, title, "", "mean score (held-out seeds, 95% CI)")
    ax.legend(frameon=False, fontsize=9, ncol=len(policies), loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)

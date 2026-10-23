"""
NexusGrid-CyberPhysEnv — LLM agent runner.

Runs a language model as the grid operator through any OpenAI-compatible
endpoint (Ollama, vLLM, LM Studio, Hugging Face router, OpenAI ...).

The model sees a compact text rendering of the observation and must answer
with one JSON action. By default the run is *pure*: unparseable answers
become ``advance_tick`` and are counted as invalid outputs. ``--fallback``
substitutes the expert policy's action instead and reports how often that
happened, so a fallback score is never mistaken for the model's own score.

Environment variables (also read from .env)
    API_BASE_URL   OpenAI-compatible base URL   (default http://localhost:11434/v1)
    MODEL_NAME     model name                   (default qwen3:1.7b)
    API_KEY / HF_TOKEN                           (default "ollama")
    EPISODE_SEED   seed for the default run     (default 42)
    ENV_URL + USE_REMOTE_ENV=1  drive a running server over WebSocket instead of in-process

Structured log lines: ``[START]``, ``[STEP]`` and ``[END]``.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import ActionType, GridAction  # noqa: E402

API_BASE_URL = os.getenv("API_BASE_URL") or "http://localhost:11434/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "qwen3:1.7b"
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "ollama"
EPISODE_SEED = int(os.getenv("EPISODE_SEED", "42"))
DEBUG_LLM_OUTPUT = os.getenv("DEBUG_LLM_OUTPUT", "0") == "1"
SCORE_EPSILON = 0.001
MAX_LLM_CALL_SECONDS = 60
TASK_BUDGETS = {0: 60, 1: 300, 2: 300, 3: 300, 4: 240, 5: 900}
TASK_NAMES = {0: "smoke_test", 1: "duck_curve", 2: "cascade_overload", 3: "phantom_inject",
              4: "stuxnet_resonance", 5: "black_start"}


def _env_flag_is_true(raw_value: Optional[str]) -> bool:
    return str(raw_value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_ollama_backend(api_base_url: Optional[str], compat_override: Optional[str] = None) -> bool:
    """Local Ollama is detected automatically; remote Ollama can opt in with OLLAMA_COMPAT=1."""
    if _env_flag_is_true(compat_override):
        return True
    base_url = (api_base_url or "").lower()
    return any(h in base_url for h in ("localhost:11434", "127.0.0.1:11434", "host.docker.internal:11434"))


# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_start(task_id: int, episode_seed: int, model_name: str) -> None:
    print(f"[START] task={task_id} seed={episode_seed} env=NexusGrid-CyberPhysEnv model={model_name}", flush=True)


def log_step(task_id: int, tick: int, action: str, params: Dict[str, Any], reward: float, done: bool,
             error: Optional[str] = None) -> None:
    action_str = json.dumps({"action_type": action, **params}, separators=(",", ":"))
    error_val = str(error) if error else "null"
    print(f"[STEP] step={tick + 1} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_val}",
          flush=True)


def log_end(task_id: int, score: float, ticks: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(score >= 0.5).lower()} steps={ticks} score={score:.3f} rewards={rewards_str}",
          flush=True)


def clamp_submission_score(score: float) -> float:
    """Clamp a score into the open interval (0, 1) for validators that reject 0.0 and 1.0."""
    return max(SCORE_EPSILON, min(1.0 - SCORE_EPSILON, float(score)))


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the control-room operator of a 20-bus transmission grid (DC power flow, one tick = 5 minutes).
Each turn you see the grid and send exactly ONE action as a JSON object.

ACTIONS
- {"action_type":"dispatch_generation","node_id":"NODE_01","mw":120}
  Change a generator or battery setpoint by mw (+ raise, - lower). Limited by its ramp per tick and capacity.
- {"action_type":"shed_load","node_id":"NODE_07","mw":100}
  Shed load at a non-critical load bus. Negative mw restores shed load (also load cut by UFLS relays).
- {"action_type":"toggle_circuit_breaker","edge_id":"LINE_12","status":"OPEN"}   (status OPEN or CLOSED)
- {"action_type":"run_state_estimation","subgraph":["NODE_09","NODE_10"]}
  Kirchhoff consistency check of up to 6 buses. A violation proves that bus's telemetry is falsified.
- {"action_type":"quarantine_scada_node","node_id":"NODE_09"}
  Isolate a compromised SCADA channel. Refused (and penalised) unless state estimation proved that bus first.
- {"action_type":"inject_counter_signal","node_id":"NODE_20","hz_offset":-0.45,"duration":4}
  Damping signal from a battery within 2 line hops of an oscillating unit; hz_offset must be the negative of the oscillation mode.
- {"action_type":"advance_tick"}   Do nothing this tick.

GRID FACTS
- Keep frequency inside 59.8-60.2 Hz. Island frequency ~ 60 + mismatch / (0.25 x online dispatchable MW), where
  mismatch = generation setpoints - demand. A mismatch of -300 MW is fixed by dispatching about +300 MW in total.
- Demand and solar change over time: the forecast shows what is coming, act before frequency moves.
- Below 59.2 Hz relays shed load automatically (UFLS). Below 59.0 Hz the episode ends.
- Lines above 100% loading trip after 3 ticks, instantly at 140%. Relieve them by raising generation near the
  receiving end and lowering it near the sending end, or by shedding non-critical load. Lines marked out of service cannot be closed.
- Critical loads (hospitals, water, data centre) must stay energised.
- Closing a breaker between two live islands needs matching frequency (<=0.2 Hz) and phase angle (<=20 deg).
- In a blackout, energise lines outward from the black-start unit, raise generation before picking up load,
  and restore critical loads first.

Answer with the JSON object only."""

ACTION_ALIASES = {
    "dispatch": "dispatch_generation",
    "dispatch_generation": "dispatch_generation",
    "shed": "shed_load",
    "shed_load": "shed_load",
    "load_shed": "shed_load",
    "toggle_breaker": "toggle_circuit_breaker",
    "toggle_circuit_breaker": "toggle_circuit_breaker",
    "run_state_estimation": "run_state_estimation",
    "state_estimation": "run_state_estimation",
    "quarantine_scada_node": "quarantine_scada_node",
    "quarantine_node": "quarantine_scada_node",
    "quarantine": "quarantine_scada_node",
    "inject_counter_signal": "inject_counter_signal",
    "counter_signal": "inject_counter_signal",
    "advance_tick": "advance_tick",
    "wait": "advance_tick",
    "noop": "advance_tick",
}


def _obs_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "observation"):
        obs = obs.observation
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    return dict(obs)


def render_observation(obs: Any, history: Optional[List[str]] = None) -> str:
    """Compact text rendering of an observation for language models."""
    d = _obs_dict(obs)
    topo = d.get("topology_graph") or {}
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])
    stream = d.get("telemetry_stream") or []
    tele = {r["node_id"]: r for r in (stream[-1] if stream else [])}
    task_id = int(d.get("task_id", 0))
    lines: List[str] = []
    lines.append(f"TASK {task_id} {TASK_NAMES.get(task_id, '')} | tick {d.get('tick', 0)}/{d.get('max_ticks', '?')} | "
                 f"system frequency {float(d.get('grid_frequency_hz', 60.0)):.3f} Hz (target 59.8-60.2)")
    if d.get("task_brief"):
        lines.append(f"BRIEF: {d['task_brief']}")

    islands = d.get("islands") or []
    if islands:
        lines.append("ISLANDS:")
        for isl in islands:
            lines.append(f"  #{isl['island']} {len(isl['nodes'])} buses f={isl['frequency_hz']:.3f} Hz "
                         f"gen {isl['generation_mw']:.0f} MW load {isl['load_mw']:.0f} MW "
                         f"mismatch {isl['setpoint_mismatch_mw']:+.0f} MW")
    else:
        lines.append("ISLANDS: none energised")
    fc = d.get("forecast") or {}
    if fc.get("system_demand_mw"):
        dem = ", ".join(f"{v:.0f}" for v in fc["system_demand_mw"][:4])
        sol = ", ".join(f"{v:.0f}" for v in (fc.get("solar_available_mw") or [])[:4])
        lines.append(f"FORECAST next ticks: demand {dem} MW | solar {sol} MW")

    lines.append("GENERATORS (id type region | capacity setpoint measured ramp):")
    for n in nodes:
        if n["node_type"] == "load":
            continue
        r = tele.get(n["id"], {})
        flags = []
        if not n.get("energized"):
            flags.append("DARK")
        if n.get("tripped"):
            flags.append("TRIPPED")
        if n.get("quarantined"):
            flags.append("QUARANTINED")
        if n.get("black_start_capable"):
            flags.append("black-start")
        if n["node_type"] == "battery":
            flags.append(f"soc {n.get('soc_pct', 0):.0f}%")
        lines.append(f"  {n['id']} {n['node_type']} {n['region']} | cap {n['capacity_mw']:.0f} "
                     f"set {n.get('setpoint_mw', 0):.0f} meas {r.get('generation_mw', 0):.0f} "
                     f"ramp {n.get('ramp_mw_per_tick', 0):.0f}" + (f" [{', '.join(flags)}]" if flags else ""))
    lines.append("LOADS (id region | served MW):")
    for n in nodes:
        if n["node_type"] != "load":
            continue
        r = tele.get(n["id"], {})
        flags = ["CRITICAL"] if n.get("critical") else []
        if not n.get("energized"):
            flags.append(f"DARK, peak {n.get('peak_load_mw', 0):.0f}")
        if n.get("shed_mw"):
            flags.append(f"shed {n['shed_mw']:.0f}")
        if n.get("ufls_shed_pct"):
            flags.append(f"UFLS {n['ufls_shed_pct']:.0f}%")
        lines.append(f"  {n['id']} {n['region']} | {r.get('consumption_mw', 0):.0f}" +
                     (f" [{', '.join(flags)}]" if flags else ""))

    adjacency: Dict[str, List[str]] = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["status"] == "LIVE":
            adjacency[e["source"]].append(e["target"])
            adjacency[e["target"]].append(e["source"])
    connected = [f"{nid}-{'/'.join(nbs)}" for nid, nbs in adjacency.items() if nbs]
    lines.append("LIVE CONNECTIONS: " + ("; ".join(connected) if connected else "none"))

    shown = []
    for e in edges:
        load = float(e.get("loading_pct", 0.0))
        if e["status"] == "LIVE" and load < 80.0:
            continue
        state = "LIVE" if e["status"] == "LIVE" else ("OUT OF SERVICE" if e.get("out_of_service") else "OPEN")
        detail = f" {load:.0f}% ({abs(e.get('flow_mw', 0)):.0f}/{e['capacity_mw']:.0f} MW)" if e["status"] == "LIVE" \
            else f" cap {e['capacity_mw']:.0f}"
        shown.append(f"  {e['id']} {e['source']}-{e['target']} {state}{detail}")
    if shown:
        lines.append("LINES (loaded >= 80% or not live):")
        lines.extend(shown)

    packets = d.get("network_packet_logs") or []
    if packets:
        last = max(int(p["timestamp"]) // 300 for p in packets)
        flagged = [p for p in packets if int(p["timestamp"]) // 300 == last and p.get("anomaly_flag")]
        if flagged:
            lines.append("SCADA PACKET ANOMALIES: " + ", ".join(
                f"{p['source_node']} {p['latency_ms']:.0f} ms" for p in flagged))
    se = d.get("last_state_estimation")
    if se:
        if se.get("consistent", True):
            lines.append(f"LAST STATE ESTIMATION: consistent for {', '.join(se.get('checked_nodes', []))}")
        else:
            lines.append(f"LAST STATE ESTIMATION: VIOLATION at {se.get('violation_node')} - reported "
                         f"{se.get('reported_mw', 0):.0f} MW, physics says {se.get('estimated_true_mw', 0):.0f} MW")
    osc = [(nid, r) for nid, r in tele.items() if r.get("oscillation_amplitude_hz", 0) > 0]
    if osc:
        osc.sort(key=lambda item: -item[1]["oscillation_amplitude_hz"])
        lines.append("PMU OSCILLATION: " + "; ".join(
            f"{nid} mode {r['oscillation_hz']:.2f} Hz amplitude {r['oscillation_amplitude_hz']:.3f} Hz" for nid, r in osc[:5]))
    signals = d.get("active_counter_signals") or []
    if signals:
        lines.append("COUNTER-SIGNALS: " + "; ".join(
            f"{s['node_id']} {s['hz_offset']:+.2f} Hz ({s['remaining_ticks']} ticks left)" for s in signals))
    events = d.get("events") or []
    if events:
        lines.append("EVENTS LAST TICK: " + "; ".join(
            f"{e['type']} {e.get('edge') or e.get('node') or ','.join(e.get('nodes', [])[:4])}" for e in events))
    if d.get("last_action_error"):
        lines.append(f"LAST ACTION ERROR: {d['last_action_error']}")
    if history:
        lines.append("YOUR RECENT ACTIONS: " + " | ".join(history[-4:]))
    lines.append("Reply with one JSON action.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing model output
# ---------------------------------------------------------------------------

def _strip_reasoning(text: str) -> str:
    cleaned = text.strip().replace("﻿", "")
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")


def _extract_balanced_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    stack: List[str] = []
    start_idx: Optional[int] = None
    for idx, ch in enumerate(text):
        if ch in "{[":
            if not stack:
                start_idx = idx
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
                if not stack and start_idx is not None:
                    candidates.append(text[start_idx: idx + 1])
                    start_idx = None
            else:
                stack = []
                start_idx = None
    return candidates


def _normalize_action_dict(action: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    normalized = dict(action)
    action_type = normalized.get("action_type") or normalized.get("action") or normalized.get("type")
    if not action_type:
        return None
    canonical = ACTION_ALIASES.get(str(action_type).strip().lower())
    if canonical is None:
        return None
    normalized["action_type"] = canonical
    aliases = {"node": "node_id", "nodeId": "node_id", "target_node": "node_id", "edge": "edge_id",
               "edgeId": "edge_id", "line": "edge_id", "line_id": "edge_id", "target_edge": "edge_id",
               "nodes": "subgraph", "buses": "subgraph", "hz": "hz_offset", "offset_hz": "hz_offset",
               "ticks": "duration", "amount": "mw", "delta_mw": "mw"}
    for source, target in aliases.items():
        if source in normalized and target not in normalized:
            normalized[target] = normalized[source]
    if isinstance(normalized.get("status"), str):
        normalized["status"] = normalized["status"].strip().upper()
    for key in ("node_id", "edge_id"):
        if isinstance(normalized.get(key), str):
            normalized[key] = normalized[key].strip().upper()
    if isinstance(normalized.get("subgraph"), str):
        normalized["subgraph"] = re.findall(r"NODE_\d{2}", normalized["subgraph"].upper())
    for key in ("mw", "hz_offset", "duration"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                normalized[key] = float(value) if key != "duration" else int(float(value))
            except ValueError:
                return None
    allowed = {"action_type", "node_id", "edge_id", "mw", "status", "subgraph", "hz_offset", "duration"}
    return {k: v for k, v in normalized.items() if k in allowed}


def _load_action_candidate(candidate: str) -> Optional[Dict[str, Any]]:
    candidate = candidate.strip()
    if not candidate:
        return None
    decoders = [
        lambda raw: json.loads(raw),
        lambda raw: json.loads(re.sub(r",(\s*[}\]])", r"\1", raw)),
        lambda raw: ast.literal_eval(raw),
    ]
    for decoder in decoders:
        try:
            parsed = decoder(candidate)
        except Exception:
            continue
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if isinstance(item, dict):
                normalized = _normalize_action_dict(item)
                if normalized:
                    return normalized
    return None


def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    """Extract the first valid action object from free-form model output."""
    text = _strip_reasoning(response_text or "")
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    for candidate in fenced + _extract_balanced_candidates(text) + [text]:
        action = _load_action_candidate(candidate)
        if action:
            return action
    return None


def to_grid_action(action_dict: Optional[Dict[str, Any]]) -> Optional[GridAction]:
    if not action_dict:
        return None
    try:
        return GridAction(**action_dict)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LLM policy
# ---------------------------------------------------------------------------

class LLMPolicy:
    """Language-model operator behind an OpenAI-compatible endpoint."""

    name = "llm"

    def __init__(self, model: str = MODEL_NAME, api_base: str = API_BASE_URL, api_key: Optional[str] = None,
                 temperature: float = 0.0, max_tokens: int = 256, fallback: bool = False,
                 timeout: float = MAX_LLM_CALL_SECONDS, client: Any = None):
        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback = fallback
        self.timeout = timeout
        self.is_ollama = _is_ollama_backend(api_base, os.getenv("OLLAMA_COMPAT"))
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=api_base, api_key=api_key or API_KEY or "none")
        self.client = client
        self._expert = None
        if fallback:
            from policies import ExpertPolicy
            self._expert = ExpertPolicy()
        self._history: List[str] = []
        self._stats = {"calls": 0, "invalid_outputs": 0, "fallbacks": 0, "api_errors": 0, "latency_s": 0.0}
        self.last_response = ""

    def reset(self, seed: Optional[int] = None) -> None:
        self._history = []
        if self._expert is not None:
            self._expert.reset()

    def stats(self) -> Dict[str, Any]:
        calls = max(1, self._stats["calls"])
        return {**self._stats, "latency_s": round(self._stats["latency_s"], 2),
                "mean_latency_s": round(self._stats["latency_s"] / calls, 3)}

    def complete(self, prompt: str) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.is_ollama:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs["reasoning_effort"] = "none"
        completion = self.client.chat.completions.create(**kwargs)
        message = completion.choices[0].message
        return ((message.content or "") or getattr(message, "reasoning", "") or "").strip()

    def act(self, obs: Any) -> GridAction:
        expert_action = self._expert.act(obs) if self._expert is not None else None
        prompt = render_observation(obs, self._history)
        started = time.time()
        text = ""
        try:
            text = self.complete(prompt)
        except Exception as exc:  # network errors, timeouts
            self._stats["api_errors"] += 1
            if DEBUG_LLM_OUTPUT:
                print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        self._stats["calls"] += 1
        self._stats["latency_s"] += time.time() - started
        self.last_response = text
        action = to_grid_action(parse_action(text))
        if action is None:
            self._stats["invalid_outputs"] += 1
            if DEBUG_LLM_OUTPUT:
                print(f"[DEBUG] Unparseable output: {text[:300]!r}", flush=True)
            if expert_action is not None:
                self._stats["fallbacks"] += 1
                action = expert_action
            else:
                action = GridAction(action_type=ActionType.ADVANCE_TICK)
        tick = _obs_dict(obs).get("tick", 0)
        summary = {k: v for k, v in action.model_dump().items() if v is not None}
        summary["action_type"] = action.action_type.value
        self._history.append(f"t{tick} {json.dumps(summary, separators=(',', ':'))}")
        return action


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

class RemoteEnvAdapter:
    """Wraps the WebSocket client so it behaves like the in-process environment."""

    def __init__(self, base_url: str):
        from client import NexusgridEnv
        self._env = NexusgridEnv(base_url=base_url).sync()
        self._env.__enter__()
        self._last = None

    def reset(self, seed: int, task_id: int):
        self._last = self._env.reset(seed=seed, task_id=task_id).observation
        return self._last

    def step(self, action: GridAction):
        result = self._env.step(action)
        obs = result.observation
        obs.done = result.done
        obs.reward = result.reward or 0.0
        self._last = obs
        return obs

    def get_score(self) -> float:
        return float(getattr(self._last, "score", 0.0) or 0.0)


def run_task(policy: Any, env: Any, task_id: int, seed: int, budget_s: Optional[float] = None,
             verbose: bool = True) -> Dict[str, Any]:
    """Run one episode and return the result record."""
    budget_s = budget_s or TASK_BUDGETS.get(task_id, 300)
    started = time.time()
    if verbose:
        log_start(task_id, seed, getattr(policy, "model", getattr(policy, "name", "policy")))
    if hasattr(policy, "reset"):
        policy.reset(seed)
    obs = env.reset(seed=seed, task_id=task_id)
    rewards: List[float] = []
    ticks = 0
    timed_out = False
    while not obs.done:
        if time.time() - started > budget_s:
            timed_out = True
            break
        action = policy.act(obs)
        obs = env.step(action)
        rewards.append(float(obs.reward or 0.0))
        if verbose:
            params = {k: v for k, v in action.model_dump().items() if k != "action_type" and v is not None}
            log_step(task_id, ticks, action.action_type.value, params, rewards[-1], obs.done, obs.last_action_error)
        ticks += 1
    score = float(env.get_score())
    if verbose:
        log_end(task_id, clamp_submission_score(score), ticks, rewards)
    record = {"task_id": task_id, "task_name": TASK_NAMES[task_id], "seed": seed, "score": round(score, 6),
              "steps": ticks, "total_reward": round(sum(rewards), 6), "timed_out": timed_out,
              "seconds": round(time.time() - started, 2)}
    if hasattr(policy, "stats"):
        record["policy_stats"] = policy.stats()
    return record


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run an LLM operator on NexusGrid")
    parser.add_argument("--tasks", default="0,1,2,3,4,5")
    parser.add_argument("--seeds", default=str(EPISODE_SEED), help="comma-separated seeds, e.g. 10000,10001")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--api-base", default=API_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fallback", action="store_true", help="use the expert action when the output is invalid")
    parser.add_argument("--out", default="scores.json")
    args = parser.parse_args(argv)

    if _env_flag_is_true(os.getenv("USE_REMOTE_ENV")):
        env = RemoteEnvAdapter(os.getenv("ENV_URL", "http://localhost:8000"))
    else:
        from server.nexusgrid_environment import NexusgridEnvironment
        env = NexusgridEnvironment()
    policy = LLMPolicy(model=args.model, api_base=args.api_base, temperature=args.temperature,
                       fallback=args.fallback)
    tasks = [int(t) for t in args.tasks.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    records = []
    started = time.time()
    for task_id in tasks:
        for seed in seeds:
            try:
                records.append(run_task(policy, env, task_id, seed))
            except Exception as exc:  # keep going: one failing task must not abort the run
                print(f"[DEBUG] task {task_id} seed {seed} failed: {exc}", flush=True)
                log_end(task_id, clamp_submission_score(0.0), 0, [])
                records.append({"task_id": task_id, "seed": seed, "score": 0.0, "error": str(exc)})
    by_task: Dict[int, List[float]] = {}
    for r in records:
        by_task.setdefault(r["task_id"], []).append(r["score"])
    summary = {
        "model": args.model,
        "api_base": args.api_base,
        "fallback": args.fallback,
        "seeds": seeds,
        "scores": {str(t): round(sum(v) / len(v), 4) for t, v in by_task.items()},
        "average": round(sum(r["score"] for r in records) / max(1, len(records)), 4),
        "policy_stats": policy.stats(),
        "episodes": records,
        "total_time_seconds": round(time.time() - started, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nAverage score {summary['average']:.3f} over {len(records)} episodes "
          f"(invalid outputs {policy.stats()['invalid_outputs']}, fallbacks {policy.stats()['fallbacks']}) "
          f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

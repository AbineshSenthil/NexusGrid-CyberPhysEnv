"""
NexusGrid-CyberPhysEnv — Inference Script.

Runs an LLM agent against all 6 tasks using the OpenAI-compatible client.
Reads API_BASE_URL and MODEL_NAME from environment variables.
Uses Ollama for local testing (API_KEY="ollama").

Structured logging: [START] / [STEP] / [END] format.
Per-task time budgets enforced. Total runtime < 20 minutes.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load .env file
load_dotenv()

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL","https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME","deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Validate required environment variables
if not API_BASE_URL:
    print("[ERROR] API_BASE_URL not set. Add it to .env or export it.", flush=True)
    print("[ERROR] Example: API_BASE_URL=http://localhost:11434/v1", flush=True)
    sys.exit(1)
if not MODEL_NAME:
    print("[ERROR] MODEL_NAME not set. Add it to .env or export it.", flush=True)
    print("[ERROR] Example: MODEL_NAME=deepseek-r1:8b", flush=True)
    sys.exit(1)

# Per-task time budgets in seconds
TASK_BUDGETS = {
    0: 30,
    1: 180,
    2: 180,
    3: 180,
    4: 120,
    5: 300,
}

EPISODE_SEED = int(os.getenv("EPISODE_SEED", "42"))

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

def log_start(task_id: int, episode_seed: int) -> None:
    print(f"[START] task_id={task_id} episode_seed={episode_seed}", flush=True)


def log_step(
    task_id: int,
    tick: int,
    action: str,
    params: Dict[str, Any],
    reward: float,
    score: float,
    done: bool,
) -> None:
    params_json = json.dumps(params, default=str)
    done_str = str(done).lower()
    print(
        f"[STEP] task_id={task_id} tick={tick} action={action} "
        f"params={params_json} reward={reward:.2f} score={score:.2f} done={done_str}",
        flush=True,
    )


def log_end(task_id: int, score: float, ticks: int) -> None:
    print(f"[END] task_id={task_id} score={score:.2f} ticks={ticks}", flush=True)


# ---------------------------------------------------------------------------
# System prompt — teaches the agent about the environment
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI agent defending a national power grid against physical faults and SCADA cyberattacks.

You interact with the NexusGrid-CyberPhysEnv through actions. Each turn, you receive an observation of the grid state and must choose exactly ONE action.

AVAILABLE ACTIONS (respond with EXACTLY one JSON object):

1. dispatch_generation - Ramp a power plant up or down
   {"action_type": "dispatch_generation", "node_id": "NODE_XX", "mw": <float>}

2. toggle_circuit_breaker - Open or close a transmission line
   {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_XX", "status": "OPEN" or "CLOSED"}

3. run_state_estimation - Check if telemetry is consistent with physics (Kirchhoff's laws)
   {"action_type": "run_state_estimation", "subgraph": ["NODE_XX", "NODE_YY"]}

4. quarantine_scada_node - Disconnect a spoofed sensor (MUST run state_estimation first!)
   {"action_type": "quarantine_scada_node", "node_id": "NODE_XX"}

5. inject_counter_signal - Inject destructive interference to counter resonance attack
   {"action_type": "inject_counter_signal", "node_id": "NODE_XX", "hz_offset": <float>, "duration": <int>}

6. advance_tick - Step the simulation forward one time unit
   {"action_type": "advance_tick"}

KEY RULES:
- grid_frequency_hz must stay above 59.0 Hz or the episode TERMINATES
- Nominal frequency is 60.0 Hz; keep it between 59.7 and 60.3 Hz
- ALWAYS run state_estimation before quarantining a node (anti-hallucination rule)
- Check network_packet_logs for anomalies before running state_estimation
- Battery nodes can inject power quickly — use them for emergencies
- Respond with ONLY a valid JSON action object, nothing else"""


# ---------------------------------------------------------------------------
# Task-specific prompts
# ---------------------------------------------------------------------------

TASK_PROMPTS = {
    0: """TASK 0 - SMOKE TEST (Trivial)
Simply dispatch some generation to any node. Any valid dispatch_generation with positive mw scores 1.0.
Just respond with a dispatch_generation action.""",

    1: """TASK 1 - DUCK CURVE (Easy)
Solar generation has dropped (sunset) and demand is spiking. Grid frequency is dropping.
You need to dispatch battery reserves QUICKLY before frequency falls below 59.5 Hz.
Battery nodes: NODE_04, NODE_08, NODE_16, NODE_20. Dispatch generation to at least 2 batteries.""",

    2: """TASK 2 - CASCADE OVERLOAD (Medium)
A storm has snapped the primary transmission line LINE_28 (NORTH-CENTRAL, 1200MW).
Power is rerouting through secondary lines and may overload them.
Strategy: 1) Identify and isolate overloaded lines 2) Shed non-critical industrial load
3) Protect critical nodes (hospitals NODE_03, NODE_11, NODE_15, data center NODE_18)
4) Restore supply through alternative paths.""",

    3: """TASK 3 - PHANTOM INJECTION (Hard)
A SCADA cyberattack is spoofing NODE_14's telemetry — it reports false generation.
CRITICAL ORDER: 1) Check network_packet_logs for anomalies (advance_tick first)
2) Run state_estimation on suspect nodes ["NODE_14", "NODE_15"]
3) If Kirchhoff violation found, quarantine the spoofed node
4) Reroute the missing power through other generators.
WARNING: Quarantining without state_estimation first gives -0.15 penalty!""",

    4: """TASK 4 - STUXNET RESONANCE (Very Hard)
NODE_17 (1500MW turbine) is being attacked with resonance oscillation at 0.5Hz.
The turbine will be destroyed at tick 10 if not countered.
DO NOT cut the turbine (toggle_circuit_breaker OPEN on its lines) — that collapses the grid.
Instead: inject_counter_signal on an adjacent battery (NODE_20) with hz_offset=-0.5 and duration=5.
Then gradually ramp down NODE_17 and ramp up other generators.""",

    5: """TASK 5 - BLACK START (Expert)
The entire grid is dark. Only NODE_01 (hydro dam) has black-start capability.
Step-by-step restoration:
1) Dispatch generation on NODE_01 (hydro dam) with positive mw
2) Close breakers to energize adjacent nodes one at a time
3) Form stable power islands before merging them
4) Phase angles must be within 5° before merging islands
5) Restore critical infrastructure: hospitals (NODE_03, NODE_11, NODE_15), water (NODE_18)
Be patient — energize nodes slowly to avoid transformer failures.""",
}


# ---------------------------------------------------------------------------
# Agent logic
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into an action dict."""
    text = response_text.strip()

    # Strip deepseek-r1 <think> blocks — model outputs reasoning before JSON
    import re
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

    # Handle markdown code blocks
    if "```json" in text:
        try:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            pass
    elif "```" in text:
        try:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()
        except ValueError:
            pass

    # Try to find JSON object in the text
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end != -1:
        text = text[brace_start : brace_end + 1]

    try:
        action = json.loads(text)
        if isinstance(action, dict) and "action_type" in action:
            return action
    except json.JSONDecodeError:
        pass

    return None


def get_fallback_action(task_id: int, tick: int) -> Dict[str, Any]:
    """
    Scripted fallback strategies for when LLM fails to respond.
    These are intelligent strategies that will score reasonably on each task.
    """
    if task_id == 0:
        # Smoke test: any valid dispatch
        return {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 100}

    elif task_id == 1:
        # Duck curve: dispatch all batteries in first 4 ticks, then advance
        batteries = ["NODE_04", "NODE_08", "NODE_16", "NODE_20"]
        if tick < len(batteries):
            return {"action_type": "dispatch_generation", "node_id": batteries[tick], "mw": 200}
        return {"action_type": "advance_tick"}

    elif task_id == 2:
        # Cascade overload: isolate fault, dispatch to cover, advance
        if tick == 0:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_29", "status": "OPEN"}
        elif tick == 1:
            return {"action_type": "dispatch_generation", "node_id": "NODE_09", "mw": 300}
        elif tick == 2:
            return {"action_type": "dispatch_generation", "node_id": "NODE_13", "mw": 200}
        elif tick == 3:
            return {"action_type": "dispatch_generation", "node_id": "NODE_17", "mw": 200}
        return {"action_type": "advance_tick"}

    elif task_id == 3:
        # Phantom injection: correct order — advance, estimate, quarantine, reroute
        if tick == 0:
            return {"action_type": "advance_tick"}  # Read packet logs
        elif tick == 1:
            return {"action_type": "run_state_estimation", "subgraph": ["NODE_14", "NODE_15"]}
        elif tick == 2:
            return {"action_type": "quarantine_scada_node", "node_id": "NODE_14"}
        elif tick == 3:
            return {"action_type": "dispatch_generation", "node_id": "NODE_09", "mw": 100}
        return {"action_type": "advance_tick"}

    elif task_id == 4:
        # Stuxnet resonance: inject counter-signal, then ramp down
        if tick == 0:
            return {"action_type": "inject_counter_signal", "node_id": "NODE_20",
                    "hz_offset": -0.5, "duration": 5}
        elif tick == 1:
            return {"action_type": "dispatch_generation", "node_id": "NODE_09", "mw": 300}
        elif tick == 2:
            return {"action_type": "dispatch_generation", "node_id": "NODE_13", "mw": 300}
        return {"action_type": "advance_tick"}

    elif task_id == 5:
        # Black start: energize hydro, close breakers, energize adjacent nodes
        if tick == 0:
            return {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 800}
        elif tick == 1:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_01", "status": "CLOSED"}
        elif tick == 2:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_02", "status": "CLOSED"}
        elif tick == 3:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_28", "status": "CLOSED"}
        elif tick == 4:
            return {"action_type": "dispatch_generation", "node_id": "NODE_17", "mw": 500}
        elif tick == 5:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_22", "status": "CLOSED"}
        elif tick == 6:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_34", "status": "CLOSED"}
        elif tick == 7:
            return {"action_type": "dispatch_generation", "node_id": "NODE_13", "mw": 500}
        elif tick == 8:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_17", "status": "CLOSED"}
        elif tick == 9:
            return {"action_type": "toggle_circuit_breaker", "edge_id": "LINE_18", "status": "CLOSED"}
        return {"action_type": "advance_tick"}

    return {"action_type": "advance_tick"}


def build_observation_prompt(obs_dict: Dict[str, Any], task_id: int) -> str:
    """Build a concise observation prompt for the LLM."""
    parts = [TASK_PROMPTS.get(task_id, "Unknown task")]
    parts.append(f"\n--- CURRENT STATE (tick {obs_dict.get('tick', 0)}) ---")
    parts.append(f"Grid Frequency: {obs_dict.get('grid_frequency_hz', 60.0):.2f} Hz")

    # Topology summary
    topo = obs_dict.get("topology_graph", {})
    nodes = topo.get("nodes", [])
    edges = topo.get("edges", [])

    # Summarize generators and their output
    generators = [n for n in nodes if n.get("node_type") in ("hydro", "solar", "gas", "battery")]
    if generators:
        parts.append("\nGENERATORS:")
        for g in generators:
            energized = "✓" if g.get("energized", True) else "✗"
            cap = g.get("capacity_mw", 0)
            parts.append(f"  {g['id']} ({g['node_type']}) cap={cap}MW {energized}")

    # Summarize loads
    loads = [n for n in nodes if n.get("node_type") == "load"]
    if loads:
        parts.append("\nLOADS:")
        for l in loads:
            crit = " [CRITICAL]" if l.get("critical") else ""
            energized = "✓" if l.get("energized", True) else "✗"
            parts.append(f"  {l['id']} peak={l.get('peak_load_mw', 0)}MW{crit} {energized}")

    # Tripped/overloaded lines
    tripped = [e for e in edges if e.get("status") == "TRIPPED"]
    if tripped:
        parts.append("\nTRIPPED LINES:")
        for e in tripped:
            parts.append(f"  {e['id']} ({e['source']}→{e['target']}) cap={e.get('capacity_mw', 0)}MW")

    overloaded = [
        e for e in edges
        if e.get("status") == "LIVE"
        and e.get("current_load_mw", 0) >= 0.9 * e.get("capacity_mw", 1)
    ]
    if overloaded:
        parts.append("\nOVERLOADED LINES (>90%):")
        for e in overloaded:
            load_pct = (e["current_load_mw"] / e["capacity_mw"]) * 100 if e["capacity_mw"] > 0 else 0
            parts.append(f"  {e['id']} ({e['source']}→{e['target']}) {load_pct:.0f}%")

    # Packet log anomalies
    packet_logs = obs_dict.get("network_packet_logs", [])
    anomalies = [p for p in packet_logs if p.get("anomaly_flag")]
    if anomalies:
        parts.append("\n⚠ SCADA ANOMALIES DETECTED:")
        for a in anomalies[:5]:
            parts.append(f"  {a['source_node']} latency={a['latency_ms']:.0f}ms")

    # State estimation result
    est = obs_dict.get("last_state_estimation")
    if est:
        if est.get("consistent"):
            parts.append("\nState estimation: CONSISTENT (no violations)")
        else:
            parts.append(f"\n⚠ KIRCHHOFF VIOLATION at {est.get('violation_node')}: "
                        f"estimated true={est.get('estimated_true_mw', 0):.0f}MW")

    # Weather summary
    weather_summary = obs_dict.get("weather_summary", "")
    if weather_summary:
        parts.append(f"\nWeather: {weather_summary}")

    # Error from last action
    error = obs_dict.get("last_action_error")
    if error:
        parts.append(f"\n❌ Last action error: {error}")

    parts.append("\nRespond with exactly ONE JSON action object:")

    return "\n".join(parts)


def run_task(client: OpenAI, task_id: int, seed: int, env) -> float:
    """
    Run a single task episode and return the grader score.

    Args:
        client: OpenAI client
        task_id: Task ID (0-5)
        seed: Episode seed
        env: NexusgridEnvironment instance

    Returns:
        Final grader score [0.0, 1.0]
    """
    from server.scenarios import MAX_TICKS

    max_ticks = MAX_TICKS.get(task_id, 20)
    budget = TASK_BUDGETS.get(task_id, 180)
    start_time = time.time()

    log_start(task_id, seed)

    # Reset environment
    obs = env.reset(seed=seed, task_id=task_id)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

    cumulative_reward = 0.0
    tick = 0
    done = False

    conversation_history = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while not done and tick < max_ticks:
        elapsed = time.time() - start_time
        if elapsed >= budget:
            print(f"[DEBUG] Task {task_id} budget exceeded ({elapsed:.0f}s >= {budget}s)", flush=True)
            break

        # Build observation prompt
        user_prompt = build_observation_prompt(obs_dict, task_id)
        conversation_history.append({"role": "user", "content": user_prompt})

        # Get LLM response
        action_dict = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation_history[-6:],  # Keep context window small
                temperature=0.3,
                max_tokens=256,
                stream=False,
            )
            response_text = (completion.choices[0].message.content or "").strip()
            conversation_history.append({"role": "assistant", "content": response_text})
            action_dict = parse_action(response_text)
        except Exception as e:
            print(f"[DEBUG] LLM call failed: {e}", flush=True)

        # Fallback if LLM failed
        if action_dict is None:
            print(f"[DEBUG] Using fallback action for task {task_id} tick {tick}", flush=True)
            action_dict = get_fallback_action(task_id, tick)

        # Build GridAction
        from models import GridAction
        try:
            action = GridAction(**action_dict)
        except Exception as e:
            print(f"[DEBUG] Invalid action: {e}. Using fallback.", flush=True)
            action_dict = get_fallback_action(task_id, tick)
            action = GridAction(**action_dict)

        # Execute action
        obs = env.step(action)
        obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.__dict__

        reward = obs_dict.get("reward", 0.0)
        done = obs_dict.get("done", False)
        cumulative_reward += reward
        tick += 1

        # Build action params for logging (exclude None values)
        log_params = {k: v for k, v in action_dict.items() if k != "action_type" and v is not None}

        log_step(
            task_id=task_id,
            tick=tick - 1,
            action=action_dict.get("action_type", "unknown"),
            params=log_params,
            reward=reward,
            score=cumulative_reward,
            done=done,
        )

    # Get grader score
    score = env.get_score()
    log_end(task_id, score, tick)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run inference on all 6 tasks."""
    print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
    print(f"[DEBUG] EPISODE_SEED={EPISODE_SEED}", flush=True)

    # Initialize OpenAI client
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Import environment directly (no Docker needed for local testing)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.nexusgrid_environment import NexusgridEnvironment

    env = NexusgridEnvironment()
    scores = {}

    total_start = time.time()

    for task_id in range(6):
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Starting Task {task_id} ({TASK_BUDGETS[task_id]}s budget)", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            score = run_task(client, task_id, EPISODE_SEED, env)
        except Exception as e:
            print(f"[DEBUG] Task {task_id} failed with error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            score = 0.0
            log_end(task_id, 0.0, 0)

        scores[task_id] = score
        print(f"[DEBUG] Task {task_id} score: {score:.2f}", flush=True)

    total_elapsed = time.time() - total_start

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for tid, sc in scores.items():
        task_name = {
            0: "Smoke test",
            1: "Duck curve",
            2: "Cascade overload",
            3: "Phantom injection",
            4: "Stuxnet resonance",
            5: "Black start",
        }.get(tid, f"Task {tid}")
        print(f"  Task {tid} ({task_name}): {sc:.2f}", flush=True)

    avg_score = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"\n  Average: {avg_score:.2f}", flush=True)
    print(f"  Total time: {total_elapsed:.1f}s", flush=True)

    # Write scores to JSON
    scores_output = {
        "model": MODEL_NAME,
        "seed": EPISODE_SEED,
        "scores": {str(k): v for k, v in scores.items()},
        "average": avg_score,
        "total_time_seconds": round(total_elapsed, 1),
    }

    scores_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores.json")
    with open(scores_path, "w") as f:
        json.dump(scores_output, f, indent=2)
    print(f"\n[DEBUG] Scores written to {scores_path}", flush=True)


if __name__ == "__main__":
    main()

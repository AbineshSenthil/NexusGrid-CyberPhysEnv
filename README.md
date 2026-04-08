---
title: NexusGrid-CyberPhysEnv
emoji: ⚡
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - critical-infrastructure
  - cybersecurity
  - energy
  - scada
---

# ⚡ NexusGrid-CyberPhysEnv

**National Power Grid & SCADA Cyber-Warfare Defense Benchmark**

An OpenEnv environment that simulates the defense of a national power grid under simultaneous physical faults and SCADA cyberattacks. Built for the OpenEnv Hackathon.

---

## 1. Environment Description & Motivation

NexusGrid-CyberPhysEnv models a **20-node transmission network** with real power flow physics (DC Kirchhoff formulation) running alongside a **deterministic sensor spoofing engine**. An AI agent must distinguish real grid failures from adversarially fabricated telemetry — a task that requires engineering reasoning, not just pattern matching.

**Motivation**: The 2015 Ukraine Sandworm attack demonstrated that SCADA cyberattacks on critical infrastructure are a present-day threat. No existing RL benchmark simulates the physics-versus-deception paradox this environment is built around. The environment is designed to be immediately useful for evaluating planning and reasoning capabilities in frontier models.

### Architecture

```
Layer 1  Physical grid simulation    numpy graph + Kirchhoff DC power flow solver
Layer 2  SCADA cyber layer           packet logs + seeded sensor spoof engine (3 attack vectors)
Layer 3  OpenEnv API                 step() / reset(seed) / state() + typed Pydantic models
Layer 4  Inference                   inference.py + OpenAI client + structured logging
```

---

## 2. Observation Space

| Field | Type | Range | Spoofable? | Description |
|-------|------|-------|------------|-------------|
| `topology_graph` | Dict (nodes + edges) | 20 nodes, 40 edges | ❌ Never | Immutable physical map. Node: id, region, type, capacity_mw, critical. Edge: id, source, target, capacity_mw, current_load_mw, status. |
| `telemetry_stream` | List[List[Dict]] | Last 10 ticks | ⚠️ Yes | Per-node time-series: voltage_kv [0–765], frequency_hz [58–62], generation_mw [0–capacity], consumption_mw [0–peak_load]. |
| `weather_forecast_matrix` | List[Dict] | 5 zones | ❌ Never | Per-zone: solar_irradiance [0–1], wind_speed_ms [0–30], cloud_cover [0–1]. |
| `network_packet_logs` | List[Dict] | Last 20 entries | ❌ Never | SCADA traffic: timestamp, source/dest node, latency_ms [0–500], anomaly_flag. Spikes precede spoofing by 1–2 ticks. |
| `grid_frequency_hz` | float | [58.0, 62.0] | ❌ Never | Truth engine output. Nominal 60.0Hz. Termination below 59.0Hz. |

---

## 3. Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `dispatch_generation` | node_id, mw | Ramp a plant/battery up or down. mw ∈ [-capacity, +capacity]. |
| `toggle_circuit_breaker` | edge_id, status | Open/close a transmission line. status: "OPEN" \| "CLOSED". |
| `run_state_estimation` | subgraph (list of node IDs) | Kirchhoff consistency check. Returns {consistent, violation_node, estimated_true_mw}. Costs 1 tick. |
| `quarantine_scada_node` | node_id | Disconnect spoofed sensor. **Must** be preceded by `run_state_estimation` or -0.15 penalty. |
| `inject_counter_signal` | node_id, hz_offset, duration | Counter resonance attack via battery. Tolerance ±0.05Hz. |
| `advance_tick` | (none) | Step simulation forward ~5 minutes. Weather/load/attacks evolve. |

---

## 4. Task Descriptions

| Task | Name | Difficulty | Max Ticks | Expected Score | Description |
|------|------|-----------|-----------|---------------|-------------|
| 0 | Smoke Test | Trivial | 3 | 1.0 | Any valid `dispatch_generation` → 1.0. Infrastructure validation. |
| 1 | Duck Curve | Easy | 15 | 0.7–1.0 | Solar drops at sunset, demand spikes. Dispatch batteries before frequency falls. |
| 2 | Cascade Overload | Medium | 20 | 0.5–0.8 | Storm snaps primary line. Isolate fault, protect critical nodes, restore supply. |
| 3 | Phantom Injection | Hard | 18 | 0.3–0.6 | SCADA spoofs NODE_14. Must: read logs → state estimation → quarantine → reroute. |
| 4 | Stuxnet Resonance | Very Hard | 12 | 0.1–0.4 | Turbine under resonance attack. Inject counter-signal at correct frequency. |
| 5 | Black Start | Expert | 50 | 0.0–0.3 | Grid is dark. Restart from hydro dam, form islands, sync phases, restore critical infra. |

---

## 5. Reward Function

### Positive Signals (per tick)
| Signal | Value | Condition |
|--------|-------|-----------|
| Fault isolation | +0.20 | Isolating a fault without dropping critical nodes |
| Cyber detection | +0.15 | Classifying + quarantining a spoofed sensor after state estimation |
| Frequency stable | +0.10 | Grid frequency in nominal band (59.7–60.3Hz) |
| Proactive dispatch | +0.08 | Dispatch before frequency deviation |
| Reasoning order | +0.05 | Reading packet logs before running state estimation |
| Stability bonus | +0.03 | Frequency within ±0.1Hz of 60.0Hz |

### Penalties
| Signal | Value | Condition |
|--------|-------|-----------|
| Overload routing | -0.20 | Routing through ≥95% capacity line |
| Quarantine w/o estimation | -0.15 | Anti-hallucination penalty |
| Redundant estimation | -0.05 | Same subgraph without action between |

### Graduated Frequency Bands
| Band | Effect |
|------|--------|
| 59.7–60.3 Hz | Nominal. No penalty. Stability bonus applies. |
| 59.5–59.7 Hz | Warning zone. No penalty. |
| 59.2–59.5 Hz | -0.05 per tick |
| 59.0–59.2 Hz | -0.15 per tick (critical) |
| Below 59.0 Hz | **Episode termination.** done=True. |

---

## 6. Environment Contract

### reset() Contract
`reset(seed)` must:
1. Reconstruct full topology from scenario definition — no mutations persist
2. Clear all telemetry history buffers
3. Re-initialize the spoof engine with the new seed
4. Reset tick counter to 0
5. Return a `GridObservation` with all fields freshly computed

**Calling `reset(42)` ten times in a row returns byte-identical `GridObservation` each time.**

### Reproducibility Guarantee (Seed-Lock Contract)
All stochastic elements derive exclusively from `numpy.random.Generator(numpy.random.PCG64(episode_seed))` where `episode_seed` is passed to `reset()`. Python's `random` module is never used. `datetime.now()` is never used for game-state computation. Running `inference.py` twice with the same `EPISODE_SEED` produces byte-identical scores.

---

## 7. Setup & Usage

### Prerequisites
```bash
ollama pull deepseek-r1:8b
```

### Environment Variables
```bash
# Required — set in .env or export before running
API_BASE_URL=http://localhost:11434/v1
MODEL_NAME=deepseek-r1:8b

# For HF Space deployment only (set as Space Secrets)
HF_TOKEN=your_hf_token
```

### Install Dependencies
```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Run Server Locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run Inference (Local with Ollama)
```bash
python inference.py
```
Runs all 6 tasks. Per-task time budgets enforced. Total under 20 minutes on vcpu=2, memory=8GB.

### Docker
```bash
docker build -t nexusgrid .
docker run -p 8000:8000 \
  -e API_BASE_URL=http://host.docker.internal:11434/v1 \
  -e MODEL_NAME=deepseek-r1:8b \
  nexusgrid
```

### Run Tests
```bash
pytest tests/ -v
```
48 tests across 5 test files: Kirchhoff physics, grader determinism, spoof engine reproducibility, API schema compliance, seed-lock verification.

### Health Check
```
GET /health  →  {"status": "healthy"}
```

### Validation
```bash
openenv validate
```

---

## 8. Baseline Scores

| Task | Name | Difficulty | Baseline Score |
|------|------|-----------|---------------|
| 0 | Smoke test | Trivial | 1.00 |
| 1 | Duck curve | Easy | 1.00 |
| 2 | Cascade overload | Medium | 0.80 |
| 3 | Phantom injection | Hard | 1.00 |
| 4 | Stuxnet resonance | Very hard | 1.00 |
| 5 | Black start | Expert | 0.025 |

Model: `deepseek-r1:8b` via Ollama · Seed: 42 · Average: **0.80** · Total time: 766s

> **Note:** Scores measured using deterministic fallback actions. When connected to Ollama, the inference script attempts LLM reasoning; fallback actions activate on LLM timeout/error.

---

## 9. Project Structure

```
nexusgrid/
├── .env                          # API_BASE_URL, MODEL_NAME (local testing)
├── __init__.py                   # Package exports
├── models.py                     # GridObservation, GridAction, GridReward (Pydantic)
├── client.py                     # NexusgridEnv WebSocket client
├── inference.py                  # Agent inference script (OpenAI client)
├── openenv.yaml                  # OpenEnv specification
├── pyproject.toml                # Dependencies & build config
├── Dockerfile                    # python:3.11-slim container
├── requirements.txt              # Pinned dependencies
├── README.md                     # This file
├── server/
│   ├── __init__.py
│   ├── __main__.py               # python -m server entry point
│   ├── app.py                    # FastAPI + /health + Gradio mount
│   ├── dashboard.py              # Gradio visual dashboard (7 panels)
│   ├── nexusgrid_environment.py  # Core OpenEnv environment
│   ├── grid_engine.py            # DC power flow physics engine
│   ├── spoof_engine.py           # SCADA attack simulation
│   ├── scenarios.py              # 6 task scenario definitions
│   ├── graders.py                # 6 task graders (pure functions)
│   └── reward.py                 # Dense reward calculator
└── tests/
    ├── test_kirchhoff.py         # Physics conservation tests (8)
    ├── test_graders.py           # Grader unit tests (18)
    ├── test_spoof_engine.py      # Spoof determinism tests (7)
    ├── test_api.py               # API interface tests (10)
    ├── test_reproducibility.py   # Seed-lock verification (5)
    └── verify_server.py          # Endpoint health checker
```

---

## 10. Structured Logging Format

```
[START] task_id=<int> episode_seed=<int>
[STEP] task_id=<int> tick=<int> action=<name> params=<json> reward=<float> score=<float> done=<bool>
[END] task_id=<int> score=<float> ticks=<int>
```

Example:
```
[START] task_id=0 episode_seed=42
[STEP] task_id=0 tick=0 action=dispatch_generation params={"node_id": "NODE_01", "mw": 100} reward=0.18 score=0.18 done=false
[END] task_id=0 score=1.00 ticks=3
```

---

## 11. LLM Client Pattern

All LLM calls use the OpenAI client with environment variables:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.environ["API_BASE_URL"],
    api_key=os.environ.get("HF_TOKEN") or "ollama",
)

response = client.chat.completions.create(
    model=os.environ["MODEL_NAME"],
    messages=[{"role": "user", "content": prompt}],
)
```

---

## License

This environment is built for the OpenEnv Hackathon.

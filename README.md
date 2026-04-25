---
title: NexusGrid-CyberPhysEnv
emoji: "⚡"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web/
tags:
  - openenv
  - critical-infrastructure
  - cybersecurity
  - scada
  - reinforcement-learning
---

# ⚡ NexusGrid-CyberPhysEnv
Training an LLM to defend a national power grid against cyber-physical attacks.

## The Problem
The 2015 Ukraine power-grid attack proved that SCADA compromises can spill directly into real-world infrastructure failures. Most LLM benchmarks still ignore the combination of physical constraints, deceptive telemetry, and multi-step operator workflows that make these incidents hard. NexusGrid turns that gap into an OpenEnv benchmark with live grid physics, seeded cyberattacks, and task-specific grading that rewards correct operational reasoning.

## What The Agent Learns
The agent does not just classify a screenshot or pick from a toy menu. It must dispatch power, isolate overloaded lines, verify suspicious telemetry with state estimation, quarantine spoofed SCADA nodes, and keep grid frequency inside safe limits while the scenario evolves tick by tick.

Round 2 Phase 2 adds the environment scaffolding needed for training:
- curriculum scheduling via `server/curriculum.py`
- composable task rubrics via `server/rubric.py`
- structured JSONL episode logging via `server/training_logger.py`
- richer `metadata` in observations, including `reward_breakdown` and `rubric_breakdown`

## Environment Architecture
```text
Layer 1  Physical grid simulation    numpy graph + Kirchhoff DC power flow solver
Layer 2  SCADA cyber layer           packet logs + deterministic spoof engine
Layer 3  OpenEnv API                 step() / reset(seed, task_id) / state() + Pydantic models
Layer 4  Training + inference        inference.py, curriculum, rubrics, structured logs
```

## Task Suite
| Task | Name | Difficulty | Max Ticks | Baseline | Goal |
|------|------|------------|-----------|----------|------|
| 0 | Smoke Test | Trivial | 3 | 0.999 | Validate the environment pipeline with any valid positive dispatch |
| 1 | Duck Curve | Easy | 15 | 0.999 | Dispatch early as solar falls and demand rises |
| 2 | Cascade Overload | Medium | 20 | 0.800 | Isolate the overload, protect critical loads, restore service |
| 3 | Phantom Injection | Hard | 18 | 0.600 | Read logs, estimate state, quarantine spoof, reroute power |
| 4 | Stuxnet Resonance | Very Hard | 12 | 0.400 | Cancel resonance with the right counter-signal and recovery actions |
| 5 | Black Start | Expert | 50 | 0.060 | Restart the grid safely from a dark state |

## Rubric System
Each task now exposes named sub-rubrics in [`openenv.yaml`](openenv.yaml) and through observation metadata. This gives training code denser, more interpretable reward signals than a single opaque terminal score.

Examples:
- Task 2: `fault_isolation`, `protect_critical_load`, `fast_restoration`
- Task 3: `log_inspection`, `state_estimation`, `correct_quarantine`, `reroute_dispatch`
- Task 5: `hydro_bootstrap`, `secondary_energized`, `safe_islanding`, `critical_restoration`

## API Contract
The public environment interface is unchanged:
- `reset(seed, task_id)` returns a typed `GridObservation`
- `step(action)` returns a typed `GridObservation`
- `state` remains available through the OpenEnv runtime
- `/health` stays available for HF Space health checks

Persistent multi-step control should use the WebSocket-backed OpenEnv client. The raw HTTP routes are present for schema/compliance and smoke checks, but the supported session flow for repeated `reset → step → state` interactions is `NexusgridEnv(...).sync()` or the async client.

Observation metadata now includes:
- `reward_breakdown`
- `rubric_breakdown`
- `episode_seed`
- `active_spoofs`
- `episode_summary` when the episode is done

## Deployment
### Local run
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Local inference
```bash
python inference.py
```

### Docker / HF Space
```bash
docker build -t nexusgrid-cyberphysenv .
docker run --rm -p 8000:8000 \
  -e API_BASE_URL=http://host.docker.internal:11434/v1 \
  -e MODEL_NAME=deepseek-r1:8b \
  -e OLLAMA_COMPAT=1 \
  nexusgrid-cyberphysenv
```

For Hugging Face Spaces:
- use Docker SDK
- expose port `8000`
- keep the README header in this file unchanged
- set `API_BASE_URL`, `MODEL_NAME`, and any `HF_TOKEN` as Space Secrets, not in `.env`

## Verified Phase 2 Status
The environment has been cleaned for deployment readiness:
- `.gitignore` added
- secret-like HF token comment removed from `.env`
- curriculum, rubric, and training logger modules added
- dashboard dependencies promoted into installable package dependencies
- rubric metadata added to [`openenv.yaml`](openenv.yaml)
- local caches / generated artifacts are excluded from deployment

Current verified local baseline:
- Task 0: `0.999`
- Task 1: `0.999`
- Task 2: `0.800`
- Task 3: `0.600`
- Task 4: `0.400`
- Task 5: `0.06030215480283671`
- Average: `0.6430503591338061`

## Try It
### Python client
```python
from client import NexusgridEnv
from models import GridAction, ActionType

with NexusgridEnv(base_url="http://localhost:8000").sync() as env:
    env.reset(seed=42, task_id=2)
    env.step(
        GridAction(
            action_type=ActionType.TOGGLE_CIRCUIT_BREAKER,
            edge_id="LINE_29",
            status="OPEN",
        )
    )
```

### Health check
```bash
curl http://localhost:8000/health
```

## Project Layout
```text
nexusgrid/
├── server/
│   ├── app.py
│   ├── curriculum.py
│   ├── graders.py
│   ├── nexusgrid_environment.py
│   ├── reward.py
│   ├── rubric.py
│   ├── scenarios.py
│   └── training_logger.py
├── tests/
├── client.py
├── Dockerfile
├── inference.py
├── models.py
├── openenv.yaml
└── README.md
```

## Next Phase
Phase 3 is the training stage: Colab or HF credits, GRPO, prompt dataset generation, before/after rollout evidence, and reward curve plots. This repository is now prepared so that work can happen against a clean, HF Space-ready environment instead of a local-only benchmark.

# Changelog

## 3.0.0 — 2026-10-23

A rebuild of the benchmark so that scores mean something.

### Environment
- **Seeded scenario families.** Every task draws its disturbance, timing and operating point from the
  episode seed. Seeds ≥ 10 000 are held out; Task 2 uses a catalog of 240 validated contingencies split
  192/48 between training and evaluation.
- **Physics.** DC power flow per electrical island, primary frequency response with inertia, ramp limits,
  battery state of charge, relay protection (three ticks above 100 %, instantly at 140 %), under-frequency
  load shedding and island collapse. Line flows obey Kirchhoff's laws (tested).
- **Attacks with real consequences.** Phantom injection hides an actual loss of generation behind
  falsified SCADA readings; the forced oscillation grows each tick until a matched counter-signal damps it.
- **No information leaks.** Attack targets and hidden quantities are no longer in observations; observers
  can read them from `debug_state()`.
- New `shed_load` action; damaged and maintenance lines are locked out; PMU phase angles and oscillation
  measurements in telemetry; forecasts, islands, events and the running score are top-level observation
  fields (OpenEnv drops `metadata` over HTTP/WebSocket).
- Dense reward = potential-based task progress + small frequency/safety terms; the progress terms sum to the
  final score.

### Grading
- Rubrics rewritten so that doing nothing scores 0.04 and random actions 0.08 on held-out seeds, while the
  reference expert scores 1.00.
- Served-load gate: frequency must not be "fixed" by cutting customers.
- Quarantine without state-estimation evidence is refused and penalised.

### Agents and training
- `policies.py`: no-op, random and an observation-only expert that checks every action with a DC power flow.
- `inference.py`: LLM evaluation is pure by default; the expert fallback is opt-in and always reported.
- `scripts/evaluate.py`: held-out evaluation with 95 % confidence intervals.
- `gym_env.py`, `training/train_ppo.py`: masked PPO with optional behaviour cloning.
- `training/train_grpo.py`: supervised warm start on expert actions, then GRPO whose reward is the
  simulator's verdict on the model's own action; prompts are decisive states; VRAM is capped so Windows
  cannot silently spill into shared memory.
- `training/run_local_gpu.ps1` / `.cmd`: one-click baseline evaluation → GRPO → evaluation on a Windows
  laptop GPU (Unsloth, 4-bit LoRA; tested on an 8 GB RTX 4060).
- Compact text observation for language models, so prompts fit small context windows and 8 GB of VRAM.
- `scripts/make_figures.py`: the README charts (SVG), rebuilt from the saved results.

### Engineering
- 140+ tests, GitHub Actions CI, MIT license, `.env.example`, per-session dashboard state.
- Removed a Hugging Face token from the git remote configuration and local history.

## 2.0.0 — 2026-05

Hackathon submission (archived write-up in `docs/blog-v2-hackathon.md`).

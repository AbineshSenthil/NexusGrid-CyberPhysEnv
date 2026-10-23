"""Regression test for server/dashboard.py."""
from __future__ import annotations

import sys
import traceback

import plotly.graph_objects as go

EXPECTED = 11
ACTIONS = [
    ("dispatch_generation", "dispatch_generation", "NODE_01", "LINE_01", 100, "CLOSED", -0.5, 5, '["NODE_14"]'),
    ("toggle_circuit_breaker", "toggle_circuit_breaker", "NODE_01", "LINE_01", 0, "OPEN", 0, 0, "[]"),
    ("run_state_estimation", "run_state_estimation", "NODE_01", "LINE_01", 0, "CLOSED", 0, 0, '["NODE_01","NODE_02"]'),
    ("quarantine_scada_node", "quarantine_scada_node", "NODE_01", "LINE_01", 0, "CLOSED", 0, 0, "[]"),
    ("inject_counter_signal", "inject_counter_signal", "NODE_01", "LINE_01", 0, "CLOSED", -0.3, 3, "[]"),
    ("advance_tick", "advance_tick", "NODE_01", "LINE_01", 0, "CLOSED", 0, 0, "[]"),
]


FIG_IDX = (0, 1, 3, 4, 5)
STR_IDX = (2, 6, 7, 8, 9, 10)


def check(name: str, out, failures: list[str]) -> None:
    if len(out) != EXPECTED:
        failures.append(f"{name}: got {len(out)} outputs, need {EXPECTED}")
        return
    for i in FIG_IDX:
        if not isinstance(out[i], go.Figure):
            failures.append(f"{name}: out[{i}] is {type(out[i]).__name__}, not Figure")
    for i in STR_IDX:
        if not isinstance(out[i], str):
            failures.append(f"{name}: out[{i}] is {type(out[i]).__name__}, not str")


def main() -> int:
    failures: list[str] = []
    from server.dashboard import (
        PLOTLY_AVAILABLE,
        TASK_META,
        create_dashboard,
        create_topology_graph,
        create_frequency_gauge,
        create_power_flow_sankey,
        create_task_radar,
        create_reward_breakdown,
        on_reset,
        on_step,
        on_auto_run,
        dashboard_state,
        _refresh,
    )

    if not PLOTLY_AVAILABLE:
        failures.append("plotly not installed")

    try:
        create_dashboard()
        print("create_dashboard: OK")
    except Exception as exc:
        failures.append(f"create_dashboard: {exc}")
        traceback.print_exc()

    for tid, meta in enumerate(TASK_META):
        name = meta["name"]
        try:
            out = on_reset(tid, 42)
            check(f"reset T{tid}", out, failures)
            obs = dashboard_state.current_obs
            if not obs.get("topology_graph", {}).get("nodes"):
                failures.append(f"T{tid}: no topology nodes")
            create_topology_graph(obs)
            create_frequency_gauge(dashboard_state.frequency_hz, dashboard_state.freq_history)
            pf = create_power_flow_sankey(obs)
            if not pf.data:
                failures.append(f"T{tid}: power panel has no traces")
            create_task_radar(dashboard_state.task_scores)
            create_reward_breakdown(dashboard_state.reward_history)
            got = pf.data[0].type
            want = "bar" if tid == 5 else "sankey"
            if got != want:
                failures.append(f"T{tid}: expected {want}, got {got}")
            print(f"  T{tid} {name}: OK ({got})")
        except Exception as exc:
            failures.append(f"T{tid} {name}: {exc}")
            traceback.print_exc()

    on_reset(0, 42)
    for row in ACTIONS:
        label, args = row[0], row[1:]
        try:
            out = on_step(*args)
            check(f"step {label}", out, failures)
            print(f"  step {label}: OK")
        except Exception as exc:
            failures.append(f"step {label}: {exc}")

    for tid in range(len(TASK_META)):
        try:
            out = on_auto_run(tid, 42, 5)
            check(f"auto T{tid}", out, failures)
            print(f"  auto T{tid}: OK")
        except Exception as exc:
            failures.append(f"auto T{tid}: {exc}")

    on_reset(3, 7)
    check("_refresh", _refresh(), failures)

    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nALL CHECKS PASSED — dashboard.py is healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

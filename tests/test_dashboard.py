"""Regression checks for the Gradio dashboard (skipped when gradio/plotly are missing)."""

from __future__ import annotations

import copy

import pytest

pytest.importorskip("gradio")
go = pytest.importorskip("plotly.graph_objects")

from server import dashboard as dash  # noqa: E402

FIG_IDX = (0, 1, 3, 4, 5)
STR_IDX = (2, 6, 7, 8, 9, 10)
ACTIONS = [
    ("dispatch_generation", "NODE_01", "LINE_01", 100, "CLOSED", -0.5, 5, '["NODE_14"]'),
    ("shed_load", "NODE_07", "LINE_01", 50, "CLOSED", 0, 0, "[]"),
    ("toggle_circuit_breaker", "NODE_01", "LINE_01", 0, "OPEN", 0, 0, "[]"),
    ("run_state_estimation", "NODE_01", "LINE_01", 0, "CLOSED", 0, 0, '["NODE_01","NODE_02"]'),
    ("quarantine_scada_node", "NODE_01", "LINE_01", 0, "CLOSED", 0, 0, "[]"),
    ("inject_counter_signal", "NODE_04", "LINE_01", 0, "CLOSED", -0.3, 3, "[]"),
    ("advance_tick", "NODE_01", "LINE_01", 0, "CLOSED", 0, 0, "[]"),
]


def _check(out):
    assert len(out) == 11
    for i in FIG_IDX:
        assert isinstance(out[i], go.Figure), i
    for i in STR_IDX:
        assert isinstance(out[i], str), i


def test_dashboard_builds():
    assert dash.create_dashboard() is not None


@pytest.mark.parametrize("task_id", range(6))
def test_reset_renders_every_panel(task_id):
    state = dash.DashboardState()
    out = dash.on_reset(task_id, 10_000, state)
    _check(out)
    assert state.current_obs["topology_graph"]["nodes"]
    panel = out[3].data[0].type
    assert panel == ("bar" if task_id == 5 else "sankey")


def test_manual_steps_render():
    state = dash.DashboardState()
    dash.on_reset(3, 10_000, state)
    for row in ACTIONS:
        _check(dash.on_step(*row, state))
    assert len(state.action_log) == len(ACTIONS)


@pytest.mark.parametrize("task_id", range(6))
def test_expert_autorun_scores(task_id):
    state = dash.DashboardState()
    _check(dash.on_auto_run(task_id, 10_000, 50, state, "expert"))
    assert state.done
    assert state.task_scores[task_id] >= 0.8


def test_sessions_are_isolated():
    template = dash.DashboardState()
    a, b = copy.deepcopy(template), copy.deepcopy(template)
    dash.on_reset(1, 10_000, a)
    dash.on_step("advance_tick", "", "", 0, "CLOSED", 0, 0, "[]", a)
    assert a.tick == 1 and b.tick == 0 and b.env is None

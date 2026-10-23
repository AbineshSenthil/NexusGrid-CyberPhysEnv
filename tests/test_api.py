"""HTTP and WebSocket API served by server.app."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server.app import app  # noqa: E402

client = TestClient(app)


def test_health_and_metadata():
    assert client.get("/health").status_code == 200
    meta = client.get("/metadata").json()
    assert meta["name"] == "NexusGrid-CyberPhysEnv" and meta["version"] == "3.0.0"


def test_tasks_endpoint_lists_rubrics():
    data = client.get("/tasks").json()
    assert data["eval_seed_start"] == 10_000
    assert [t["id"] for t in data["tasks"]] == list(range(6))
    for task in data["tasks"]:
        assert abs(sum(r["weight"] for r in task["rubrics"]) - 1.0) < 1e-9


def test_http_reset_and_step_are_stateless_but_valid():
    reset = client.post("/reset", json={"seed": 10_000, "task_id": 1})
    assert reset.status_code == 200
    body = reset.json()
    assert body["observation"]["task_id"] == 1
    assert "metadata" not in body["observation"]
    # Plain HTTP creates a fresh environment per request; multi-step episodes use the WebSocket.
    step = client.post("/step", json={"action": {"action_type": "advance_tick"}})
    assert step.status_code == 200
    assert "reset" in step.json()["observation"]["last_action_error"]


def test_websocket_session_runs_an_episode():
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reset", "data": {"seed": 10_000, "task_id": 0}})
        first = ws.receive_json()
        assert first["data"]["observation"]["tick"] == 0
        done = False
        steps = 0
        while not done and steps < 5:
            action = {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 40} if steps == 0 \
                else {"action_type": "advance_tick"}
            ws.send_json({"type": "step", "data": action})
            msg = ws.receive_json()
            done = msg["data"]["done"]
            steps += 1
        assert done
        assert msg["data"]["observation"]["score"] == pytest.approx(1.0)

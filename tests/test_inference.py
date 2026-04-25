"""
Test inference helpers.

These tests focus on the local Ollama/DeepSeek path where the model often
wraps JSON in reasoning text or code fences.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from inference import (
    _is_ollama_backend,
    _should_replace_with_fallback,
    clamp_submission_score,
    get_fallback_action,
    parse_action,
)


class TestParseAction:
    def test_parses_reasoning_plus_fenced_json(self):
        response = """
<think>
I should bring up the hydro dam first.
</think>
```json
{"action_type":"dispatch_generation","node_id":"NODE_01","mw":600}
```
"""
        action = parse_action(response)
        assert action == {
            "action_type": "dispatch_generation",
            "node_id": "NODE_01",
            "mw": 600,
        }

    def test_parses_single_quoted_payload(self):
        response = "{'action_type': 'toggle_circuit_breaker', 'edge_id': 'LINE_28', 'status': 'closed'}"
        action = parse_action(response)
        assert action == {
            "action_type": "toggle_circuit_breaker",
            "edge_id": "LINE_28",
            "status": "CLOSED",
        }

    def test_parses_action_wrapped_in_list(self):
        response = """
Here is the action:
[
  {"action_type":"advance_tick"}
]
"""
        action = parse_action(response)
        assert action == {"action_type": "advance_tick"}


class TestClampSubmissionScore:
    def test_clamp_keeps_scores_inside_open_interval(self):
        assert 0.0 < clamp_submission_score(0.0) < 1.0
        assert 0.0 < clamp_submission_score(1.0) < 1.0
        assert clamp_submission_score(0.42) == 0.42


class TestOllamaDetection:
    def test_local_ollama_url_detected_automatically(self):
        assert _is_ollama_backend("http://localhost:11434/v1") is True

    def test_remote_openai_compatible_url_requires_override(self):
        assert _is_ollama_backend("https://example.com/v1") is False
        assert _is_ollama_backend("https://example.com/v1", compat_override="1") is True


class TestFallbackPolicy:
    def test_cascade_overload_replaces_non_isolation_action_while_line_is_overloaded(self):
        obs = {
            "topology_graph": {
                "nodes": [
                    {"id": "NODE_01", "energized": True, "capacity_mw": 1500},
                ],
                "edges": [
                    {
                        "id": "LINE_29",
                        "status": "LIVE",
                        "current_load_mw": 950.0,
                        "capacity_mw": 1000.0,
                    },
                ],
            },
            "telemetry_stream": [[
                {"node_id": "NODE_01", "generation_mw": 600.0},
            ]],
        }

        action = {"action_type": "dispatch_generation", "node_id": "NODE_01", "mw": 100}
        assert _should_replace_with_fallback(task_id=2, tick=0, action_dict=action, obs_dict=obs)

    def test_task3_replaces_early_dispatch_before_logs(self):
        obs = {
            "network_packet_logs": [],
            "metadata": {"active_spoofs": ["NODE_14"]},
        }
        action = {"action_type": "dispatch_generation", "node_id": "NODE_09", "mw": 100}
        assert _should_replace_with_fallback(task_id=3, tick=0, action_dict=action, obs_dict=obs)

    def test_task3_replaces_early_quarantine_before_estimation(self):
        obs = {
            "network_packet_logs": [{"source_node": "NODE_14", "anomaly_flag": True}],
            "metadata": {"active_spoofs": ["NODE_14"]},
        }
        action = {"action_type": "quarantine_scada_node", "node_id": "NODE_14"}
        assert _should_replace_with_fallback(task_id=3, tick=1, action_dict=action, obs_dict=obs)

    def test_task3_replaces_dispatch_before_quarantine(self):
        obs = {
            "network_packet_logs": [{"source_node": "NODE_14", "anomaly_flag": True}],
            "last_state_estimation": {"consistent": False, "violation_node": "NODE_14"},
            "metadata": {"active_spoofs": ["NODE_14"]},
        }
        action = {"action_type": "dispatch_generation", "node_id": "NODE_09", "mw": 100}
        assert _should_replace_with_fallback(task_id=3, tick=2, action_dict=action, obs_dict=obs)

    def test_task3_allows_reroute_after_quarantine(self):
        obs = {
            "topology_graph": {
                "nodes": [
                    {"id": "NODE_09", "energized": True, "capacity_mw": 900},
                ],
                "edges": [],
            },
            "telemetry_stream": [[
                {"node_id": "NODE_09", "generation_mw": 400.0},
            ]],
            "network_packet_logs": [{"source_node": "NODE_14", "anomaly_flag": True}],
            "last_state_estimation": {"consistent": False, "violation_node": "NODE_14"},
            "metadata": {"active_spoofs": []},
        }
        action = {"action_type": "dispatch_generation", "node_id": "NODE_09", "mw": 100}
        assert _should_replace_with_fallback(task_id=3, tick=3, action_dict=action, obs_dict=obs) is False

    def test_task4_replaces_non_injection_opening_move(self):
        obs = {
            "topology_graph": {"nodes": [], "edges": []},
            "telemetry_stream": [],
        }
        action = {"action_type": "run_state_estimation", "subgraph": ["NODE_17", "NODE_20"]}
        assert _should_replace_with_fallback(task_id=4, tick=0, action_dict=action, obs_dict=obs)

    def test_task3_replaces_redundant_estimation_after_quarantine(self):
        obs = {
            "network_packet_logs": [{"source_node": "NODE_14", "anomaly_flag": True}],
            "last_state_estimation": {"consistent": False, "violation_node": "NODE_14"},
            "metadata": {"active_spoofs": []},
        }
        action = {"action_type": "run_state_estimation", "subgraph": ["NODE_14", "NODE_15"]}
        assert _should_replace_with_fallback(task_id=3, tick=4, action_dict=action, obs_dict=obs)

    def test_black_start_fallback_only_makes_minimal_progress(self):
        obs = {
            "topology_graph": {
                "nodes": [
                    {"id": "NODE_01", "energized": True},
                    {"id": "NODE_03", "energized": False},
                ],
                "edges": [
                    {"id": "LINE_02", "status": "TRIPPED"},
                ],
            },
            "telemetry_stream": [[
                {"node_id": "NODE_01", "generation_mw": 600.0},
                {"node_id": "NODE_03", "generation_mw": 0.0},
            ]],
        }

        action = get_fallback_action(task_id=5, tick=3, obs_dict=obs)
        assert action == {
            "action_type": "toggle_circuit_breaker",
            "edge_id": "LINE_02",
            "status": "CLOSED",
        }

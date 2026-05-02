"""Round 2 environment enhancements: curriculum, rubrics, and training logging."""

import io
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from client import NexusgridEnv
from models import ActionType, GridAction
from server.curriculum import CurriculumManager
from server.nexusgrid_environment import NexusgridEnvironment
from server.rubric import evaluate_task_rubrics, get_task_rubric_specs
from server.training_logger import TrainingLogger


class TestCurriculumManager:
    def test_recommends_task_zero_initially(self):
        curriculum = CurriculumManager(unlock_threshold=0.7, window=3)
        assert curriculum.get_recommended_task() == 0

    def test_unlocks_next_task_after_threshold(self):
        curriculum = CurriculumManager(unlock_threshold=0.7, window=3)
        curriculum.record_score(0, 0.8)
        curriculum.record_score(0, 0.75)

        assert curriculum.is_unlocked(1) is True
        assert curriculum.get_recommended_task() == 1


class TestRubrics:
    def test_task_three_rubrics_reflect_forensic_sequence(self):
        actions = [
            {"action_type": "advance_tick", "tick": 0},
            {"action_type": "run_state_estimation", "tick": 1, "result": {"consistent": False}},
            {"action_type": "quarantine_scada_node", "tick": 2, "node_id": "NODE_14"},
            {"action_type": "dispatch_generation", "tick": 3, "node_id": "NODE_09", "mw": 100},
        ]
        state = {"spoof_target": "NODE_14", "logs_read_before_estimation": True}

        result = evaluate_task_rubrics(3, actions, state)

        assert result["rubrics"]["log_inspection"] == 1.0
        assert result["rubrics"]["state_estimation"] == 1.0
        assert result["rubrics"]["correct_quarantine"] == 1.0
        assert result["rubrics"]["reroute_dispatch"] == 1.0
        assert result["weighted_score"] == 1.0

    def test_task_specs_available_for_manifest_sync(self):
        specs = get_task_rubric_specs(5)
        assert any(spec["name"] == "hydro_bootstrap" for spec in specs)


class TestTrainingLogger:
    def test_training_logger_writes_jsonl(self):
        buffer = io.StringIO()
        logger = TrainingLogger(stream=buffer)
        record = logger.build_record(
            episode=7,
            task_id=3,
            seed=42,
            score=0.6,
            rubrics={"state_estimation": 1.0},
            actions_taken=["advance_tick", "run_state_estimation"],
            frequency_min=59.8,
            ticks_used=2,
        )
        logger.write_episode(record)

        payload = json.loads(buffer.getvalue().strip())
        assert payload["episode"] == 7
        assert payload["task_id"] == 3
        assert payload["rubrics"]["state_estimation"] == 1.0


class TestRoundTwoMetadata:
    def test_environment_exposes_rubric_breakdown(self):
        env = NexusgridEnvironment()
        obs = env.reset(seed=42, task_id=3)

        assert "rubric_breakdown" in obs.metadata
        assert obs.metadata["rubric_breakdown"]["task_id"] == 3

    def test_emit_training_log_uses_episode_summary(self):
        env = NexusgridEnvironment()
        env.reset(seed=42, task_id=0)
        env.step(GridAction(action_type=ActionType.DISPATCH_GENERATION, node_id="NODE_01", mw=100))

        buffer = io.StringIO()
        line = env.emit_training_log(episode=1, logger=TrainingLogger(stream=buffer))
        payload = json.loads(line)

        assert payload["episode"] == 1
        assert payload["task_id"] == 0
        assert "valid_dispatch" in payload["rubrics"]


class TestClientMetadataParsing:
    def test_client_preserves_metadata(self):
        client = NexusgridEnv(base_url="http://localhost:8000")
        payload = {
            "observation": {
                "topology_graph": {},
                "telemetry_stream": [],
                "weather_forecast_matrix": [],
                "network_packet_logs": [],
                "grid_frequency_hz": 60.0,
                "tick": 1,
                "task_id": 2,
                "done": False,
                "reward": 0.2,
                "weather_summary": "",
                "metadata": {"rubric_breakdown": {"task_id": 2}},
            },
            "reward": 0.2,
            "done": False,
        }

        result = client._parse_result(payload)
        assert result.observation.metadata["rubric_breakdown"]["task_id"] == 2

    def test_client_parses_state_with_model_validation(self):
        client = NexusgridEnv(base_url="http://localhost:8000")
        state = client._parse_state({"episode_id": "ep-1", "step_count": 4, "extra": "ok"})

        assert state.episode_id == "ep-1"
        assert state.step_count == 4

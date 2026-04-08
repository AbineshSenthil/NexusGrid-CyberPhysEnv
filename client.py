"""NexusGrid-CyberPhysEnv Environment Client."""

from typing import Any, Dict, List

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import GridAction, GridObservation


class NexusgridEnv(
    EnvClient[GridAction, GridObservation, State]
):
    """
    Client for the NexusGrid-CyberPhysEnv.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with NexusgridEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     result = env.step(GridAction(
        ...         action_type="dispatch_generation",
        ...         node_id="NODE_01",
        ...         mw=100.0,
        ...     ))
    """

    def _step_payload(self, action: GridAction) -> Dict:
        """Convert GridAction to JSON payload."""
        payload = {
            "action_type": action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type),
        }
        if action.node_id is not None:
            payload["node_id"] = action.node_id
        if action.edge_id is not None:
            payload["edge_id"] = action.edge_id
        if action.mw is not None:
            payload["mw"] = action.mw
        if action.status is not None:
            payload["status"] = action.status
        if action.subgraph is not None:
            payload["subgraph"] = action.subgraph
        if action.hz_offset is not None:
            payload["hz_offset"] = action.hz_offset
        if action.duration is not None:
            payload["duration"] = action.duration
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[GridObservation]:
        """Parse server response into StepResult[GridObservation]."""
        obs_data = payload.get("observation", payload)

        observation = GridObservation(
            topology_graph=obs_data.get("topology_graph", {}),
            telemetry_stream=obs_data.get("telemetry_stream", []),
            weather_forecast_matrix=obs_data.get("weather_forecast_matrix", []),
            network_packet_logs=obs_data.get("network_packet_logs", []),
            grid_frequency_hz=obs_data.get("grid_frequency_hz", 60.0),
            tick=obs_data.get("tick", 0),
            task_id=obs_data.get("task_id", 0),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            last_action_error=obs_data.get("last_action_error"),
            last_state_estimation=obs_data.get("last_state_estimation"),
            weather_summary=obs_data.get("weather_summary", ""),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
            done=payload.get("done", obs_data.get("done", False)),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

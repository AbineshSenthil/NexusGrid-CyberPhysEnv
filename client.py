"""NexusGrid-CyberPhysEnv Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import GridAction, GridObservation
except ImportError:
    from models import GridAction, GridObservation


class NexusgridEnv(
    EnvClient[GridAction, GridObservation, State]
):
    """
    Client for the NexusGrid-CyberPhysEnv.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with NexusgridEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset(seed=10_000, task_id=1)
        ...     result = env.step(GridAction(
        ...         action_type="dispatch_generation",
        ...         node_id="NODE_01",
        ...         mw=100.0,
        ...     ))
        ...     print(result.observation.score, result.observation.grid_frequency_hz)
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
        """Parse a server response into StepResult[GridObservation]."""
        obs_data = dict(payload.get("observation", payload))
        done = payload.get("done", obs_data.get("done", False))
        reward = payload.get("reward", obs_data.get("reward", 0.0))
        obs_data["done"] = bool(done)
        obs_data["reward"] = float(reward or 0.0)
        observation = GridObservation.model_validate(obs_data)
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        if hasattr(State, "model_validate"):
            return State.model_validate(payload)
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )

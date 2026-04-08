"""
NexusGrid-CyberPhysEnv Data Models.

Typed Pydantic models for the national power grid defense environment.
All fields have explicit range contracts via Field(ge=, le=, description=).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    HYDRO = "hydro"
    SOLAR = "solar"
    GAS = "gas"
    BATTERY = "battery"
    LOAD = "load"


class EdgeStatus(str, Enum):
    LIVE = "LIVE"
    TRIPPED = "TRIPPED"


class ActionType(str, Enum):
    DISPATCH_GENERATION = "dispatch_generation"
    TOGGLE_CIRCUIT_BREAKER = "toggle_circuit_breaker"
    RUN_STATE_ESTIMATION = "run_state_estimation"
    QUARANTINE_SCADA_NODE = "quarantine_scada_node"
    INJECT_COUNTER_SIGNAL = "inject_counter_signal"
    ADVANCE_TICK = "advance_tick"


# ---------------------------------------------------------------------------
# Topology sub-models
# ---------------------------------------------------------------------------

class GridNode(BaseModel):
    """A substation node in the power grid."""
    id: str = Field(..., description="Unique node identifier, e.g. NODE_01")
    region: str = Field(..., description="Geographic region name")
    node_type: NodeType = Field(..., description="Type of substation")
    capacity_mw: float = Field(..., ge=0, le=5000, description="Maximum generation/load capacity in MW")
    peak_load_mw: float = Field(0.0, ge=0, le=5000, description="Peak demand for load nodes in MW")
    critical: bool = Field(False, description="True if this is critical infrastructure (hospital, water)")
    phase_angle_deg: float = Field(0.0, ge=-180.0, le=180.0, description="AC phase angle in degrees")


class GridEdge(BaseModel):
    """A transmission line edge in the power grid."""
    id: str = Field(..., description="Unique edge identifier, e.g. LINE_01")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    capacity_mw: float = Field(..., ge=50, le=5000, description="Line thermal capacity in MW")
    current_load_mw: float = Field(0.0, ge=0, le=5000, description="Current power flow on line in MW")
    status: EdgeStatus = Field(EdgeStatus.LIVE, description="Line status: LIVE or TRIPPED")


# ---------------------------------------------------------------------------
# Telemetry sub-models
# ---------------------------------------------------------------------------

class NodeTelemetry(BaseModel):
    """Per-node telemetry reading (may be spoofed)."""
    node_id: str = Field(..., description="Node this reading belongs to")
    voltage_kv: float = Field(345.0, ge=0.0, le=765.0, description="Voltage in kV (nominal 345)")
    frequency_hz: float = Field(60.0, ge=58.0, le=62.0, description="Local frequency in Hz (nominal 60)")
    generation_mw: float = Field(0.0, ge=0.0, le=5000.0, description="Generation output in MW")
    consumption_mw: float = Field(0.0, ge=0.0, le=5000.0, description="Consumption demand in MW")


class WeatherZone(BaseModel):
    """Weather data for a geographic zone."""
    zone: str = Field(..., description="Zone identifier")
    solar_irradiance: float = Field(0.5, ge=0.0, le=1.0, description="Solar irradiance [0-1]")
    wind_speed_ms: float = Field(5.0, ge=0.0, le=30.0, description="Wind speed in m/s")
    cloud_cover: float = Field(0.3, ge=0.0, le=1.0, description="Cloud cover fraction [0-1]")


class PacketLog(BaseModel):
    """A simulated SCADA network packet log entry."""
    timestamp: float = Field(..., ge=0.0, description="Simulated timestamp")
    source_node: str = Field(..., description="Source node ID")
    dest_node: str = Field(..., description="Destination node ID")
    latency_ms: float = Field(5.0, ge=0.0, le=500.0, description="Packet latency in ms")
    anomaly_flag: bool = Field(False, description="True if latency > 50ms for 2+ consecutive packets")


# ---------------------------------------------------------------------------
# State estimation result
# ---------------------------------------------------------------------------

class StateEstimationResult(BaseModel):
    """Result from run_state_estimation action."""
    consistent: bool = Field(..., description="True if Kirchhoff's laws hold for the subgraph")
    violation_node: Optional[str] = Field(None, description="Node ID where violation was detected")
    estimated_true_mw: float = Field(0.0, description="Estimated true power at violation node")


# ---------------------------------------------------------------------------
# Observation — what the agent sees
# ---------------------------------------------------------------------------

class GridObservation(BaseModel):
    """
    Complete observation from the NexusGrid environment.

    The agent sees topology (never spoofed), telemetry (may be spoofed),
    weather, SCADA packet logs, and the true grid frequency.
    """
    model_config = {"extra": "allow"}

    topology_graph: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dict with 'nodes' and 'edges' keys — the immutable physical map, never spoofed"
    )
    telemetry_stream: List[List[Dict[str, Any]]] = Field(
        default_factory=list,
        description="Per-tick telemetry history (last 10 ticks). Each tick is a list of node readings. May be spoofed."
    )
    weather_forecast_matrix: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="24-hour per-zone weather forecast"
    )
    network_packet_logs: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent SCADA packet logs with anomaly flags"
    )
    grid_frequency_hz: float = Field(
        60.0, ge=58.0, le=62.0,
        description="Instantaneous national grid frequency from truth engine. Cannot be spoofed."
    )
    tick: int = Field(0, ge=0, description="Current simulation tick")
    task_id: int = Field(0, ge=0, le=5, description="Current task ID")
    done: bool = Field(False, description="Whether episode has ended")
    reward: float = Field(0.0, description="Reward for this step")
    last_action_error: Optional[str] = Field(None, description="Error message from last action, if any")
    last_state_estimation: Optional[Dict[str, Any]] = Field(
        None,
        description="Result of most recent run_state_estimation call"
    )
    weather_summary: str = Field("", description="Natural language weather summary")


# ---------------------------------------------------------------------------
# Action — what the agent does
# ---------------------------------------------------------------------------

class GridAction(BaseModel):
    """
    Action for the NexusGrid environment.

    The agent selects an action_type and provides relevant parameters.
    Unused parameters for a given action_type should be left as None.
    """
    action_type: ActionType = Field(
        ...,
        description="Type of action to perform"
    )
    node_id: Optional[str] = Field(
        None,
        description="Target node ID (for dispatch_generation, quarantine_scada_node, inject_counter_signal)"
    )
    edge_id: Optional[str] = Field(
        None,
        description="Target edge ID (for toggle_circuit_breaker)"
    )
    mw: Optional[float] = Field(
        None, ge=-5000, le=5000,
        description="Megawatts to dispatch (for dispatch_generation)"
    )
    status: Optional[str] = Field(
        None,
        description="OPEN or CLOSED (for toggle_circuit_breaker)"
    )
    subgraph: Optional[List[str]] = Field(
        None,
        description="List of node IDs to check (for run_state_estimation)"
    )
    hz_offset: Optional[float] = Field(
        None, ge=-5.0, le=5.0,
        description="Frequency offset for counter-signal injection (for inject_counter_signal)"
    )
    duration: Optional[int] = Field(
        None, ge=1, le=20,
        description="Duration in ticks for counter-signal (for inject_counter_signal)"
    )


# ---------------------------------------------------------------------------
# Reward — breakdown of reward signals
# ---------------------------------------------------------------------------

class GridReward(BaseModel):
    """Breakdown of reward signals for transparency."""
    fault_isolation: float = Field(0.0, description="Reward for isolating transmission faults")
    cyber_detection: float = Field(0.0, description="Reward for detecting spoofed sensors")
    frequency_stable: float = Field(0.0, description="Reward for keeping frequency in nominal band")
    proactive_dispatch: float = Field(0.0, description="Reward for proactive (early) dispatch")
    reasoning_order: float = Field(0.0, description="Reward for correct investigation order")
    stability_bonus: float = Field(0.0, description="Bonus for tight frequency control (±0.1Hz)")
    penalties: float = Field(0.0, description="Sum of all penalties (negative value)")
    total: float = Field(0.0, description="Total reward for this tick")


# ---------------------------------------------------------------------------
# State — full internal state for debugging
# ---------------------------------------------------------------------------

class GridState(BaseModel):
    """Full internal environment state for debugging and reproducibility."""
    episode_id: str = Field("", description="Unique episode identifier")
    episode_seed: int = Field(42, description="Seed used for this episode")
    task_id: int = Field(0, ge=0, le=5, description="Current task")
    tick: int = Field(0, ge=0, description="Current tick")
    grid_frequency_hz: float = Field(60.0, ge=58.0, le=62.0, description="Current grid frequency")
    done: bool = Field(False, description="Whether episode is finished")
    total_reward: float = Field(0.0, description="Accumulated reward")
    active_spoofs: List[str] = Field(default_factory=list, description="Currently spoofed node IDs")
    quarantined_nodes: List[str] = Field(default_factory=list, description="Quarantined node IDs")
    state_estimation_run: bool = Field(False, description="Whether state estimation has been run this episode")
    action_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of all actions taken")

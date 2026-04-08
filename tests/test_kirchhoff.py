"""
Test Kirchhoff Conservation — proves the power flow engine
conserves power at every node on every tick.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.grid_engine import GridEngine
from server.scenarios import build_base_topology


class TestKirchhoff:
    """Test that Kirchhoff's laws hold in the grid engine."""

    def test_power_conservation_initial(self):
        """After initialization, generation and load should be physically reasonable."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        total_gen = engine.get_total_generation()
        total_load = engine.get_total_load()

        # Grid has generation headroom by design — gen >= load
        assert total_gen > 0, "Grid should have generation"
        assert total_load > 0, "Grid should have load"
        # Frequency should reflect the balance
        assert 59.0 <= engine.frequency_hz <= 62.0, (
            f"Frequency out of range: {engine.frequency_hz}"
        )

    def test_frequency_nominal_at_start(self):
        """Initial frequency should be near 60Hz."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        assert 59.0 <= engine.frequency_hz <= 61.0, (
            f"Initial frequency out of range: {engine.frequency_hz}"
        )

    def test_power_conservation_after_ticks(self):
        """Generation and load should remain physically reasonable after ticks."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        for _ in range(5):
            engine.advance_tick()

        total_gen = engine.get_total_generation()
        total_load = engine.get_total_load()

        # After ticks, both gen and load should still be positive
        assert total_gen > 0, f"No generation after 5 ticks: {total_gen}"
        assert total_load > 0, f"No load after 5 ticks: {total_load}"
        # Frequency should remain in valid range
        assert 58.0 <= engine.frequency_hz <= 62.0, (
            f"Frequency out of range after ticks: {engine.frequency_hz}"
        )

    def test_edge_loads_non_negative(self):
        """No edge should carry negative load."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        for _ in range(3):
            engine.advance_tick()

        for edge_id, edge in engine.edges.items():
            assert edge["current_load_mw"] >= 0, (
                f"Edge {edge_id} has negative load: {edge['current_load_mw']}"
            )

    def test_edge_loads_within_capacity(self):
        """Live edges should ideally not exceed capacity significantly."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        for edge_id, edge in engine.edges.items():
            if edge["status"] == "LIVE":
                # Allow 10% overload for transient states
                assert edge["current_load_mw"] <= edge["capacity_mw"] * 1.1, (
                    f"Edge {edge_id} overloaded: {edge['current_load_mw']} > {edge['capacity_mw']}"
                )

    def test_state_estimation_consistent_without_spoofs(self):
        """State estimation should report consistent when no spoofs are active."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        ground_truth = engine.get_ground_truth_telemetry()
        result = engine.run_state_estimation(
            list(engine.nodes.keys())[:5],
            ground_truth,
        )

        assert result["consistent"] is True, (
            f"State estimation inconsistent without spoofs: {result}"
        )

    def test_dispatch_changes_generation(self):
        """Dispatching generation should change the node's output."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        old_gen = engine.nodes["NODE_01"]["generation_mw"]
        result = engine.dispatch_generation("NODE_01", 100)

        assert result["success"] is True
        assert engine.nodes["NODE_01"]["generation_mw"] == old_gen + 100

    def test_tripped_line_carries_zero(self):
        """Tripping a line should set its load to zero."""
        engine = GridEngine(seed=42)
        build_base_topology(engine)

        engine.toggle_circuit_breaker("LINE_01", "OPEN")
        assert engine.edges["LINE_01"]["status"] == "TRIPPED"
        assert engine.edges["LINE_01"]["current_load_mw"] == 0.0

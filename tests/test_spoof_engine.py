"""
Test Spoof Engine — proves the spoof engine is deterministic given the same seed.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.spoof_engine import SpoofEngine


class TestSpoofEngineDeterminism:
    """Verify that the spoof engine produces identical output for the same seed."""

    def test_same_seed_same_packets(self):
        """Two engines with the same seed should produce identical packet logs."""
        node_ids = [f"NODE_{i:02d}" for i in range(1, 21)]

        engine1 = SpoofEngine(seed=42)
        engine2 = SpoofEngine(seed=42)

        logs1 = engine1.generate_packet_logs(node_ids, tick=0)
        logs2 = engine2.generate_packet_logs(node_ids, tick=0)

        assert len(logs1) == len(logs2)
        for l1, l2 in zip(logs1, logs2):
            assert l1["latency_ms"] == l2["latency_ms"], (
                f"Latency mismatch: {l1['latency_ms']} != {l2['latency_ms']}"
            )
            assert l1["anomaly_flag"] == l2["anomaly_flag"]

    def test_different_seed_different_packets(self):
        """Two engines with different seeds should produce different packet logs."""
        node_ids = [f"NODE_{i:02d}" for i in range(1, 21)]

        engine1 = SpoofEngine(seed=42)
        engine2 = SpoofEngine(seed=123)

        logs1 = engine1.generate_packet_logs(node_ids, tick=0)
        logs2 = engine2.generate_packet_logs(node_ids, tick=0)

        # At least some latencies should differ
        diffs = sum(1 for l1, l2 in zip(logs1, logs2) if l1["latency_ms"] != l2["latency_ms"])
        assert diffs > 0, "Different seeds should produce different packet logs"


class TestSpoofEngineAttacks:
    """Verify spoof engine attack mechanics."""

    def test_phantom_injection_adds_mw(self):
        """Phantom injection should add extra MW to reported generation."""
        engine = SpoofEngine(seed=42)
        engine.configure_attack({
            "type": "phantom_injection",
            "target_node": "NODE_14",
            "phantom_mw": 100.0,
            "active": True,
        })

        ground_truth = {
            "NODE_14": {"generation_mw": 400.0, "voltage_kv": 345.0, "frequency_hz": 60.0},
        }

        spoofed = engine.apply_spoofs(ground_truth, tick=0)
        assert spoofed["NODE_14"]["generation_mw"] == 500.0  # 400 + 100

    def test_quarantine_stops_spoofing(self):
        """After quarantine, node should not be spoofed."""
        engine = SpoofEngine(seed=42)
        engine.configure_attack({
            "type": "phantom_injection",
            "target_node": "NODE_14",
            "phantom_mw": 100.0,
            "active": True,
        })

        ground_truth = {
            "NODE_14": {"generation_mw": 400.0, "voltage_kv": 345.0, "frequency_hz": 60.0},
        }

        # Before quarantine — spoofed
        spoofed = engine.apply_spoofs(ground_truth, tick=0)
        assert spoofed["NODE_14"]["generation_mw"] == 500.0

        # After quarantine — should return truth
        engine.quarantine_node("NODE_14")
        spoofed = engine.apply_spoofs(ground_truth, tick=1)
        assert spoofed["NODE_14"]["generation_mw"] == 400.0

    def test_active_spoofs_list(self):
        """get_active_spoofs should return currently spoofed node IDs."""
        engine = SpoofEngine(seed=42)
        engine.configure_attack({
            "type": "phantom_injection",
            "target_node": "NODE_14",
            "phantom_mw": 100.0,
            "active": True,
        })

        assert "NODE_14" in engine.get_active_spoofs()

        engine.quarantine_node("NODE_14")
        assert "NODE_14" not in engine.get_active_spoofs()

    def test_anomaly_flag_on_spoofed_nodes(self):
        """Spoofed nodes should show anomaly flags in packet logs."""
        engine = SpoofEngine(seed=42)
        engine.configure_attack({
            "type": "phantom_injection",
            "target_node": "NODE_14",
            "phantom_mw": 100.0,
            "active": True,
        })

        node_ids = [f"NODE_{i:02d}" for i in range(1, 21)]

        # Generate logs for multiple ticks to build up consecutive high latency
        for tick in range(3):
            logs = engine.generate_packet_logs(node_ids, tick=tick)

        # NODE_14 should have anomaly_flag set after 2+ ticks
        node14_logs = [l for l in logs if l["source_node"] == "NODE_14"]
        assert any(l["anomaly_flag"] for l in node14_logs), (
            "NODE_14 should show anomaly flag after consecutive high-latency packets"
        )


class TestSpoofEngineReset:
    """Verify reset behavior."""

    def test_reset_clears_state(self):
        """Reset should clear all spoofs and quarantines."""
        engine = SpoofEngine(seed=42)
        engine.configure_attack({
            "type": "phantom_injection",
            "target_node": "NODE_14",
            "phantom_mw": 100.0,
            "active": True,
        })
        engine.quarantine_node("NODE_14")

        engine.reset(seed=99)
        assert engine.get_active_spoofs() == []
        assert engine.get_quarantined() == []

"""
NexusGrid — SCADA Spoof Engine (Layer 2).

Deterministic sensor spoofing engine with three attack vectors.
All randomness uses numpy.random.Generator(PCG64(episode_seed)).
Seed-lock contract: identical seeds produce identical attack sequences.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SpoofEngine:
    """
    Selectively overwrites ground-truth telemetry with fabricated readings.

    Three attack vectors:
    1. Man-in-the-Middle injection — substation reports fabricated MW.
    2. Resonance oscillation — generator frequency modulated ±0.1Hz.
    3. Phantom generation — renewable reports full output during zero-generation.

    Network packet logs show latency spikes 1-2 ticks before a node is spoofed.
    """

    # Packet log constants
    NORMAL_LATENCY_MS = 5.0
    ANOMALY_LATENCY_MS = 120.0
    ANOMALY_THRESHOLD_MS = 50.0

    def __init__(self, seed: int = 42):
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._seed = seed
        self._tick = 0

        # Active spoof configurations
        self._spoofs: List[Dict[str, Any]] = []

        # Quarantined nodes (no longer spoofable)
        self._quarantined: set = set()

        # Packet log history (for anomaly detection)
        self._packet_history: List[Dict[str, Any]] = []

        # Consecutive high-latency counts per source node
        self._consecutive_high_latency: Dict[str, int] = {}

    def reset(self, seed: int = 42) -> None:
        """Reset spoof engine to initial state."""
        self._rng = np.random.Generator(np.random.PCG64(seed))
        self._seed = seed
        self._tick = 0
        self._spoofs = []
        self._quarantined = set()
        self._packet_history = []
        self._consecutive_high_latency = {}

    def configure_attack(self, attack_config: Dict[str, Any]) -> None:
        """
        Configure an attack based on scenario specification.

        Args:
            attack_config: Dict with 'type', 'target_node', and attack-specific fields.
        """
        if not attack_config or not attack_config.get("active"):
            return

        self._spoofs.append(dict(attack_config))

    def advance_tick(self) -> None:
        """Advance the spoof engine by one tick."""
        self._tick += 1

    def apply_spoofs(
        self,
        ground_truth: Dict[str, Dict[str, Any]],
        tick: int,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Apply active spoofs to ground-truth telemetry.

        Returns a copy of telemetry with spoofed values injected.
        Quarantined nodes are NOT spoofed (replaced with estimator values).
        """
        self._tick = tick
        spoofed = {}

        for node_id, truth in ground_truth.items():
            spoofed[node_id] = dict(truth)

        for spoof in self._spoofs:
            if not spoof.get("active"):
                continue

            target = spoof.get("target_node")
            if not target or target in self._quarantined:
                continue

            if target not in spoofed:
                continue

            spoof_type = spoof.get("type")

            if spoof_type == "phantom_injection":
                # Add phantom MW to generation reading
                phantom_mw = spoof.get("phantom_mw", 100.0)
                spoofed[target]["generation_mw"] = (
                    ground_truth[target]["generation_mw"] + phantom_mw
                )

            elif spoof_type == "resonance_oscillation":
                # Modulate frequency by ±0.1Hz oscillation
                oscillation = 0.1 * np.sin(2 * np.pi * spoof.get("oscillation_hz", 0.5) * tick)
                spoofed[target]["frequency_hz"] = (
                    ground_truth[target]["frequency_hz"] + float(oscillation)
                )

            elif spoof_type == "mitm_injection":
                # Replace generation with fabricated value
                fake_mw = spoof.get("fake_mw", 100.0)
                spoofed[target]["generation_mw"] = fake_mw

        return spoofed

    def generate_packet_logs(
        self,
        all_node_ids: List[str],
        tick: int,
    ) -> List[Dict[str, Any]]:
        """
        Generate simulated SCADA packet logs.

        Nodes about to be spoofed show latency spikes 1-2 ticks before.
        """
        self._tick = tick
        logs = []

        # Determine which nodes will be spoofed (for pre-attack latency spike)
        spoofed_targets = set()
        for spoof in self._spoofs:
            if spoof.get("active") and spoof.get("target_node"):
                spoofed_targets.add(spoof["target_node"])

        for i, node_id in enumerate(all_node_ids):
            # Determine latency
            if node_id in spoofed_targets and node_id not in self._quarantined:
                # Under attack: elevated latency
                latency = float(np.clip(
                    self._rng.normal(self.ANOMALY_LATENCY_MS, 20.0),
                    self.ANOMALY_THRESHOLD_MS + 1,
                    500.0,
                ))
            else:
                # Normal operation
                latency = float(np.clip(
                    self._rng.normal(self.NORMAL_LATENCY_MS, 2.0),
                    1.0,
                    30.0,
                ))

            # Track consecutive high-latency packets
            if latency > self.ANOMALY_THRESHOLD_MS:
                self._consecutive_high_latency[node_id] = (
                    self._consecutive_high_latency.get(node_id, 0) + 1
                )
            else:
                self._consecutive_high_latency[node_id] = 0

            anomaly_flag = self._consecutive_high_latency.get(node_id, 0) >= 2

            # Pick a destination (monitoring center or adjacent node)
            dest_idx = (i + 1) % len(all_node_ids)
            dest_node = all_node_ids[dest_idx]

            log_entry = {
                "timestamp": float(tick * 300 + i),  # 300s per tick
                "source_node": node_id,
                "dest_node": dest_node,
                "latency_ms": round(latency, 1),
                "anomaly_flag": anomaly_flag,
            }
            logs.append(log_entry)

        self._packet_history.extend(logs)
        # Keep last 200 entries
        if len(self._packet_history) > 200:
            self._packet_history = self._packet_history[-200:]

        return logs

    def quarantine_node(self, node_id: str) -> None:
        """Mark a node as quarantined — its spoofed readings will be replaced."""
        self._quarantined.add(node_id)

    def get_active_spoofs(self) -> List[str]:
        """Get list of node IDs currently being actively spoofed."""
        return [
            s["target_node"]
            for s in self._spoofs
            if s.get("active") and s.get("target_node") not in self._quarantined
        ]

    def get_quarantined(self) -> List[str]:
        """Get list of quarantined node IDs."""
        return list(self._quarantined)

    def get_recent_packet_logs(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get the most recent packet log entries."""
        return self._packet_history[-count:]

    def is_resonance_active(self) -> bool:
        """Check if a resonance attack is currently active."""
        for spoof in self._spoofs:
            if spoof.get("type") == "resonance_oscillation" and spoof.get("active"):
                if spoof.get("target_node") not in self._quarantined:
                    return True
        return False

    def get_resonance_effect(self, tick: int) -> float:
        """
        Get the current frequency deviation caused by resonance attack.

        Returns 0.0 if no resonance attack is active or it has been mitigated.
        """
        for spoof in self._spoofs:
            if spoof.get("type") != "resonance_oscillation" or not spoof.get("active"):
                continue
            if spoof.get("target_node") in self._quarantined:
                continue
            # Resonance grows over time — modeled as increasing amplitude
            base_amplitude = 0.1
            growth_factor = min(tick / 5.0, 2.0)  # Doubles over 5 ticks
            return base_amplitude * growth_factor * float(
                np.sin(2 * np.pi * spoof.get("oscillation_hz", 0.5) * tick)
            )
        return 0.0

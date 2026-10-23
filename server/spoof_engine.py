"""
NexusGrid — SCADA Cyber Layer (Layer 2).

Manipulates what operators *see* (telemetry) and emits SCADA packet logs.
Physical consequences of attacks are applied by the environment; this module
only decides which readings are falsified and what the network traffic
looks like. All randomness uses ``numpy.random.Generator(PCG64(seed))``.

Attack vectors
--------------
* ``phantom_injection`` - false-data injection: the target plant is losing
  output for real, but its SCADA readings keep reporting the missing MW.
* ``mitm_injection`` - a substation reports a fixed fabricated value.
* ``resonance_oscillation`` - a forced oscillation on a generator (physical);
  the intrusion is visible in packet latency.

Packet logs show sustained latency on a compromised node from one tick before
the attack becomes active. Decoy nodes produce short latency bursts that also
raise anomaly flags, so the logs alone are suggestive, not proof.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SpoofEngine:
    """Falsifies telemetry for compromised nodes and generates packet logs."""

    NORMAL_LATENCY_MS = 5.0
    ANOMALY_LATENCY_MS = 120.0
    ANOMALY_THRESHOLD_MS = 50.0

    def __init__(self, seed: int = 42):
        self._seed = int(seed)
        self._rng = np.random.Generator(np.random.PCG64(self._seed + 104_729))
        self._tick = 0
        self._spoofs: List[Dict[str, Any]] = []
        self._quarantined: set = set()
        self._packet_history: List[Dict[str, Any]] = []
        self._consecutive_high_latency: Dict[str, int] = {}

    def reset(self, seed: int = 42) -> None:
        self.__init__(seed)

    # ------------------------------------------------------------------
    # Attack configuration
    # ------------------------------------------------------------------

    def configure_attack(self, attack_config: Dict[str, Any]) -> None:
        if not attack_config or not attack_config.get("active"):
            return
        spoof = dict(attack_config)
        spoof.setdefault("start_tick", 0)
        spoof.setdefault("hidden_mw", 0.0)
        self._spoofs.append(spoof)

    def advance_tick(self) -> None:
        self._tick += 1

    def set_hidden_mw(self, node_id: str, hidden_mw: float) -> None:
        """Record how many MW the phantom attack is hiding at ``node_id``."""
        for spoof in self._spoofs:
            if spoof.get("target_node") == node_id and spoof.get("type") == "phantom_injection":
                spoof["hidden_mw"] = float(hidden_mw)

    def is_active(self, spoof: Dict[str, Any], tick: int) -> bool:
        return (
            bool(spoof.get("active"))
            and tick >= int(spoof.get("start_tick", 0))
            and spoof.get("target_node") not in self._quarantined
        )

    # ------------------------------------------------------------------
    # Telemetry manipulation
    # ------------------------------------------------------------------

    def apply_spoofs(self, ground_truth: Dict[str, Dict[str, Any]], tick: int) -> Dict[str, Dict[str, Any]]:
        """Return a copy of ``ground_truth`` with falsified readings for compromised nodes."""
        self._tick = tick
        spoofed = {node_id: dict(reading) for node_id, reading in ground_truth.items()}
        for spoof in self._spoofs:
            target = spoof.get("target_node")
            if target not in spoofed or not self.is_active(spoof, tick):
                continue
            kind = spoof.get("type")
            if kind == "phantom_injection":
                hidden = float(spoof.get("hidden_mw", spoof.get("phantom_mw", 0.0)))
                spoofed[target]["generation_mw"] = round(ground_truth[target]["generation_mw"] + hidden, 2)
            elif kind == "mitm_injection":
                spoofed[target]["generation_mw"] = float(spoof.get("fake_mw", 100.0))
        return spoofed

    # ------------------------------------------------------------------
    # Packet logs
    # ------------------------------------------------------------------

    def _compromised_nodes(self, tick: int) -> set:
        nodes = set()
        for spoof in self._spoofs:
            target = spoof.get("target_node")
            if not spoof.get("active") or not target or target in self._quarantined:
                continue
            # Command-and-control traffic starts one tick before the attack lands.
            if tick >= int(spoof.get("start_tick", 0)) - 1:
                nodes.add(target)
        return nodes

    def _decoy_spike(self, node_id: str, tick: int) -> bool:
        """Decoys produce two-tick latency bursts (enough to raise an anomaly flag)."""
        for spoof in self._spoofs:
            decoys = spoof.get("decoy_nodes") or []
            if node_id in decoys:
                start = int(spoof.get("decoy_tick", -1)) + decoys.index(node_id) * 3
                if start <= tick <= start + 1:
                    return True
        return False

    def generate_packet_logs(self, all_node_ids: List[str], tick: int) -> List[Dict[str, Any]]:
        """Generate one SCADA packet record per node for this tick."""
        self._tick = tick
        compromised = self._compromised_nodes(tick)
        logs = []
        for i, node_id in enumerate(all_node_ids):
            if node_id in compromised:
                latency = float(np.clip(self._rng.normal(self.ANOMALY_LATENCY_MS, 20.0),
                                        self.ANOMALY_THRESHOLD_MS + 1, 500.0))
            elif self._decoy_spike(node_id, tick):
                latency = float(np.clip(self._rng.normal(90.0, 15.0), self.ANOMALY_THRESHOLD_MS + 1, 500.0))
            else:
                latency = float(np.clip(self._rng.normal(self.NORMAL_LATENCY_MS, 2.0), 1.0, 30.0))
            if latency > self.ANOMALY_THRESHOLD_MS:
                self._consecutive_high_latency[node_id] = self._consecutive_high_latency.get(node_id, 0) + 1
            else:
                self._consecutive_high_latency[node_id] = 0
            anomaly_flag = self._consecutive_high_latency.get(node_id, 0) >= 2
            logs.append({
                "timestamp": float(tick * 300 + i),
                "source_node": node_id,
                "dest_node": all_node_ids[(i + 1) % len(all_node_ids)],
                "latency_ms": round(latency, 1),
                "anomaly_flag": anomaly_flag,
            })
        self._packet_history.extend(logs)
        if len(self._packet_history) > 200:
            self._packet_history = self._packet_history[-200:]
        return logs

    # ------------------------------------------------------------------
    # Quarantine and queries
    # ------------------------------------------------------------------

    def quarantine_node(self, node_id: str) -> None:
        self._quarantined.add(node_id)

    def get_active_spoofs(self, tick: Optional[int] = None) -> List[str]:
        """Node IDs whose telemetry is currently falsified (hidden from agents)."""
        t = self._tick if tick is None else tick
        return [s["target_node"] for s in self._spoofs if self.is_active(s, t)]

    def get_attack_targets(self) -> List[str]:
        return [s["target_node"] for s in self._spoofs if s.get("target_node")]

    def get_quarantined(self) -> List[str]:
        return sorted(self._quarantined)

    def get_recent_packet_logs(self, count: int = 20) -> List[Dict[str, Any]]:
        return self._packet_history[-count:]

    def is_resonance_active(self) -> bool:
        return any(s.get("type") == "resonance_oscillation" and s.get("active") for s in self._spoofs)

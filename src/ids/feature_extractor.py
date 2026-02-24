"""
Feature Extractor for the Baseline IDS.

Transforms raw simulated traffic flows into a fixed-size numerical feature
vector suitable for anomaly detection and RL state representation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..simulation.traffic_generator import TrafficFlow


class FeatureExtractor:
    """
    Extracts and normalises network traffic features from flow data.

    Maintains a sliding window of historical stats for temporal context
    and produces a fixed-size feature vector each time step.
    """

    FEATURE_NAMES = [
        "traffic_volume",
        "packet_rate",
        "avg_packet_size",
        "tcp_ratio",
        "udp_ratio",
        "icmp_ratio",
        "unique_src_ips",
        "unique_dst_ips",
        "unique_dst_ports",
        "failed_connections",
        "avg_flow_duration",
        "entropy_src_ip",
        "entropy_dst_port",
        "dns_query_rate",
        "outbound_data_ratio",
    ]

    NUM_FEATURES = len(FEATURE_NAMES)

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self._history: List[np.ndarray] = []
        self._running_mean: Optional[np.ndarray] = None
        self._running_var: Optional[np.ndarray] = None
        self._count = 0

    # ── Public API ──────────────────────────────────────────────────

    def extract(self, flows: List[TrafficFlow]) -> np.ndarray:
        """
        Extract a feature vector from a list of flows for the current time step.

        Returns:
            1-D numpy array of shape (NUM_FEATURES,) with normalised values.
        """
        raw = self._raw_features(flows)
        self._update_stats(raw)
        normalised = self._normalise(raw)

        self._history.append(normalised.copy())
        if len(self._history) > self.window_size:
            self._history.pop(0)

        return normalised

    def get_temporal_features(self) -> np.ndarray:
        """
        Get aggregated temporal features from the sliding window.

        Returns mean and standard deviation of the window, concatenated.
        Shape: (NUM_FEATURES * 2,) if history exists, else zeros.
        """
        if not self._history:
            return np.zeros(self.NUM_FEATURES * 2, dtype=np.float32)

        window = np.array(self._history)
        mean = window.mean(axis=0)
        std = window.std(axis=0)
        return np.concatenate([mean, std]).astype(np.float32)

    def get_full_state(self, flows: List[TrafficFlow]) -> np.ndarray:
        """
        Get the full state vector: current features + temporal context.

        Shape: (NUM_FEATURES * 3,)
        """
        current = self.extract(flows)
        temporal = self.get_temporal_features()
        return np.concatenate([current, temporal]).astype(np.float32)

    def reset(self) -> None:
        self._history.clear()
        self._running_mean = None
        self._running_var = None
        self._count = 0

    # ── Internal ────────────────────────────────────────────────────

    def _raw_features(self, flows: List[TrafficFlow]) -> np.ndarray:
        if not flows:
            return np.zeros(self.NUM_FEATURES, dtype=np.float32)

        n = len(flows)
        total_bytes = sum(f.bytes_sent + f.bytes_received for f in flows)
        total_packets = sum(f.packets_sent + f.packets_received for f in flows)
        protocols = [f.protocol for f in flows]

        features = np.array([
            float(total_bytes),
            float(total_packets),
            total_bytes / max(total_packets, 1),
            protocols.count("tcp") / n,
            protocols.count("udp") / n,
            protocols.count("icmp") / n,
            float(len(set(f.src_ip for f in flows))),
            float(len(set(f.dst_ip for f in flows))),
            float(len(set(f.dst_port for f in flows))),
            float(sum(1 for f in flows if f.metadata.get("failed", False))),
            float(np.mean([f.duration for f in flows])),
            self._entropy([f.src_ip for f in flows]),
            self._entropy([str(f.dst_port) for f in flows]),
            float(sum(1 for f in flows if f.dst_port == 53)),
            sum(f.bytes_sent for f in flows) / max(total_bytes, 1),
        ], dtype=np.float32)

        return features

    def _update_stats(self, raw: np.ndarray) -> None:
        """Welford's online algorithm for running mean/variance."""
        self._count += 1
        if self._running_mean is None:
            self._running_mean = raw.copy()
            self._running_var = np.zeros_like(raw)
        else:
            delta = raw - self._running_mean
            self._running_mean += delta / self._count
            delta2 = raw - self._running_mean
            self._running_var += delta * delta2

    def _normalise(self, raw: np.ndarray) -> np.ndarray:
        if self._count < 2 or self._running_var is None or self._running_mean is None:
            return raw

        std = np.sqrt(self._running_var / max(self._count - 1, 1)) + 1e-8
        return ((raw - self._running_mean) / std).astype(np.float32)

    @staticmethod
    def _entropy(values: List[str]) -> float:
        if not values:
            return 0.0
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        n = len(values)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

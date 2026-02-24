"""
Benign Traffic Generator for the RL-Enhanced IDS Simulation.

Generates statistically realistic network traffic using Poisson arrivals,
Pareto flow sizes, diurnal patterns and concept-drift modelling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .network_topology import DeviceType, NetworkTopology


@dataclass
class TrafficFlow:
    """A single simulated network flow."""

    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str          # tcp, udp, icmp
    bytes_sent: int
    bytes_received: int
    packets_sent: int
    packets_received: int
    duration: float        # seconds
    timestamp: float       # simulation time
    is_malicious: bool = False
    attack_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrafficGenerator:
    """
    Model-based benign traffic generator.

    Produces flows that follow configurable statistical distributions and
    realistic protocol mixes with optional diurnal variation and concept drift.
    """

    COMMON_PORTS = {
        "http": 80,
        "https": 443,
        "dns": 53,
        "smtp": 25,
        "imap": 143,
        "ssh": 22,
        "ftp": 21,
        "smb": 445,
        "mysql": 3306,
        "postgresql": 5432,
        "mqtt": 1883,
        "ntp": 123,
    }

    def __init__(
        self,
        topology: NetworkTopology,
        benign_rate: float = 100.0,
        protocol_distribution: Optional[Dict[str, float]] = None,
        diurnal_variation: bool = True,
        concept_drift_rate: float = 0.001,
        seed: Optional[int] = None,
    ) -> None:
        self.topology = topology
        self.benign_rate = benign_rate
        self.protocol_dist = protocol_distribution or {
            "tcp": 0.65, "udp": 0.25, "icmp": 0.05, "other": 0.05,
        }
        self.diurnal = diurnal_variation
        self.drift_rate = concept_drift_rate
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._flow_counter = 0
        self._drift_offset = 0.0

    # ── Public API ──────────────────────────────────────────────────

    def generate_flows(
        self, sim_time: float, time_step: float = 60.0
    ) -> List[TrafficFlow]:
        """
        Generate a batch of benign flows for one simulation time step.

        Args:
            sim_time: Current simulation time in seconds.
            time_step: Duration of this step in seconds.

        Returns:
            List of TrafficFlow objects.
        """
        rate = self._effective_rate(sim_time)
        n_flows = self._np_rng.poisson(rate * (time_step / 60.0))
        flows: List[TrafficFlow] = []

        for _ in range(n_flows):
            flows.append(self._create_benign_flow(sim_time))

        return flows

    def apply_concept_drift(self) -> None:
        """Advance concept drift by one step, slowly shifting traffic profile."""
        self._drift_offset += self.drift_rate
        # Slowly shift protocol distribution
        drift_delta = self._rng.gauss(0, self.drift_rate * 0.1)
        self.protocol_dist["tcp"] = max(0.3, min(0.85, self.protocol_dist["tcp"] + drift_delta))
        remaining = 1.0 - self.protocol_dist["tcp"]
        udp_frac = 0.7
        self.protocol_dist["udp"] = remaining * udp_frac
        self.protocol_dist["icmp"] = remaining * 0.15
        self.protocol_dist["other"] = remaining * 0.15

    def get_traffic_stats(self, flows: List[TrafficFlow]) -> Dict[str, float]:
        """Compute aggregate statistics from a list of flows."""
        if not flows:
            return self._empty_stats()

        total_bytes = sum(f.bytes_sent + f.bytes_received for f in flows)
        total_packets = sum(f.packets_sent + f.packets_received for f in flows)
        protocols = [f.protocol for f in flows]
        src_ips = set(f.src_ip for f in flows)
        dst_ips = set(f.dst_ip for f in flows)
        dst_ports = set(f.dst_port for f in flows)

        tcp_count = protocols.count("tcp")
        udp_count = protocols.count("udp")
        icmp_count = protocols.count("icmp")
        n = len(flows)

        return {
            "traffic_volume": float(total_bytes),
            "packet_rate": float(total_packets),
            "avg_packet_size": total_bytes / max(total_packets, 1),
            "tcp_ratio": tcp_count / n,
            "udp_ratio": udp_count / n,
            "icmp_ratio": icmp_count / n,
            "unique_src_ips": float(len(src_ips)),
            "unique_dst_ips": float(len(dst_ips)),
            "unique_dst_ports": float(len(dst_ports)),
            "failed_connections": float(sum(1 for f in flows if f.metadata.get("failed", False))),
            "avg_flow_duration": float(np.mean([f.duration for f in flows])),
            "entropy_src_ip": self._entropy([f.src_ip for f in flows]),
            "entropy_dst_port": self._entropy([str(f.dst_port) for f in flows]),
            "dns_query_rate": float(sum(1 for f in flows if f.dst_port == 53)),
            "outbound_data_ratio": self._outbound_ratio(flows),
            "num_flows": float(n),
        }

    # ── Internal ────────────────────────────────────────────────────

    def _effective_rate(self, sim_time: float) -> float:
        rate = self.benign_rate + self._drift_offset * 10
        if self.diurnal:
            hour_of_day = (sim_time / 3600.0) % 24
            # Peak at 10–14h, trough at 2–5h
            diurnal_factor = 0.5 + 0.5 * math.sin(
                math.pi * (hour_of_day - 6) / 12
            )
            rate *= max(0.2, diurnal_factor)
        return max(1.0, rate)

    def _create_benign_flow(self, sim_time: float) -> TrafficFlow:
        self._flow_counter += 1
        protocol = self._pick_protocol()

        workstations = self.topology.get_workstations()
        servers = self.topology.get_servers()

        if workstations and servers:
            src = self._rng.choice(workstations)
            dst = self._rng.choice(servers)
        else:
            all_devs = list(self.topology.devices.values())
            src, dst = self._rng.sample(all_devs, min(2, len(all_devs)))

        dst_port = self._pick_dst_port(protocol, dst)
        src_port = self._rng.randint(1024, 65535)

        # Pareto-distributed flow size (heavy tail)
        bytes_sent = int(self._np_rng.pareto(1.5) * 500 + 64)
        bytes_received = int(self._np_rng.pareto(1.2) * 1000 + 64)
        duration = max(0.01, self._np_rng.exponential(5.0))
        packets_sent = max(1, bytes_sent // 1400 + 1)
        packets_received = max(1, bytes_received // 1400 + 1)

        failed = self._rng.random() < 0.02  # 2% connection failures

        return TrafficFlow(
            flow_id=f"F{self._flow_counter:08d}",
            src_ip=src.ip_address,
            dst_ip=dst.ip_address,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            bytes_sent=bytes_sent,
            bytes_received=bytes_received if not failed else 0,
            packets_sent=packets_sent,
            packets_received=packets_received if not failed else 0,
            duration=duration if not failed else 0.01,
            timestamp=sim_time,
            is_malicious=False,
            metadata={"failed": failed, "service": self._port_to_service(dst_port)},
        )

    def _pick_protocol(self) -> str:
        r = self._rng.random()
        cumulative = 0.0
        for proto, prob in self.protocol_dist.items():
            cumulative += prob
            if r <= cumulative:
                return proto
        return "tcp"

    def _pick_dst_port(self, protocol: str, dst_device: Any) -> int:
        if protocol == "icmp":
            return 0
        if dst_device.services:
            service = self._rng.choice(dst_device.services)
            return self.COMMON_PORTS.get(service, self._rng.randint(1024, 65535))
        return self._rng.choice([80, 443, 53, 22, 3306])

    def _port_to_service(self, port: int) -> str:
        for svc, p in self.COMMON_PORTS.items():
            if p == port:
                return svc
        return "unknown"

    @staticmethod
    def _entropy(values: List[str]) -> float:
        if not values:
            return 0.0
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        n = len(values)
        return -sum((c / n) * math.log2(c / n) for c in counts.values())

    def _outbound_ratio(self, flows: List[TrafficFlow]) -> float:
        total_out = sum(f.bytes_sent for f in flows)
        total_in = sum(f.bytes_received for f in flows)
        return total_out / max(total_out + total_in, 1)

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
        return {k: 0.0 for k in [
            "traffic_volume", "packet_rate", "avg_packet_size",
            "tcp_ratio", "udp_ratio", "icmp_ratio",
            "unique_src_ips", "unique_dst_ips", "unique_dst_ports",
            "failed_connections", "avg_flow_duration",
            "entropy_src_ip", "entropy_dst_port",
            "dns_query_rate", "outbound_data_ratio", "num_flows",
        ]}

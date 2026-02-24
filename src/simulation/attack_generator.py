"""
Attack Traffic Generator for the RL-Enhanced IDS Simulation.

Produces parameterized malicious traffic across 12 attack categories:
port scans, DDoS (SYN/UDP), C2 beaconing, encrypted C2, data exfiltration,
slow exfiltration, lateral movement, DNS tunneling, ICMP tunneling,
polymorphic attacks, and multi-stage coordinated campaigns.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np

from .network_topology import DeviceType, NetworkTopology
from .traffic_generator import TrafficFlow


class AttackGenerator:
    """
    Generates diverse, parameterized attack traffic for RL training.

    Each attack type is configurable in intensity, timing, and targets,
    ensuring the RL agent encounters sufficient variability.
    """

    ATTACK_TYPES = [
        "port_scan",
        "ddos_syn_flood",
        "ddos_udp_flood",
        "c2_beaconing",
        "data_exfiltration",
        "lateral_movement",
        "dns_tunneling",
        # ── v2.0 advanced attacks ────────────────────────────────
        "encrypted_c2",
        "polymorphic_attack",
        "slow_exfiltration",
        "icmp_tunneling",
        "multi_stage_attack",
    ]

    def __init__(
        self,
        topology: NetworkTopology,
        min_intensity: float = 0.1,
        max_intensity: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        self.topology = topology
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self._flow_counter = 0

    # ── Public API ──────────────────────────────────────────────────

    def generate_attack(
        self,
        attack_type: str,
        sim_time: float,
        intensity: Optional[float] = None,
        target_device_id: Optional[str] = None,
        source_device_id: Optional[str] = None,
    ) -> List[TrafficFlow]:
        """
        Generate attack flows of the given type.

        Args:
            attack_type: One of ATTACK_TYPES.
            sim_time: Current simulation time.
            intensity: Attack intensity 0‒1 (randomized if None).
            target_device_id: Specific target (randomized if None).
            source_device_id: Specific source (randomized if None).

        Returns:
            List of malicious TrafficFlow objects.
        """
        if attack_type not in self.ATTACK_TYPES:
            raise ValueError(f"Unknown attack type: {attack_type}")

        intensity = intensity or self._rng.uniform(self.min_intensity, self.max_intensity)

        dispatch = {
            "port_scan": self._port_scan,
            "ddos_syn_flood": self._ddos_syn_flood,
            "ddos_udp_flood": self._ddos_udp_flood,
            "c2_beaconing": self._c2_beaconing,
            "data_exfiltration": self._data_exfiltration,
            "lateral_movement": self._lateral_movement,
            "dns_tunneling": self._dns_tunneling,
            "encrypted_c2": self._encrypted_c2,
            "polymorphic_attack": self._polymorphic_attack,
            "slow_exfiltration": self._slow_exfiltration,
            "icmp_tunneling": self._icmp_tunneling,
            "multi_stage_attack": self._multi_stage_attack,
        }

        return dispatch[attack_type](sim_time, intensity, target_device_id, source_device_id)

    def random_attack(self, sim_time: float) -> List[TrafficFlow]:
        """Generate a random attack type with random parameters."""
        attack_type = self._rng.choice(self.ATTACK_TYPES)
        return self.generate_attack(attack_type, sim_time)

    # ── Attack implementations ──────────────────────────────────────

    def _port_scan(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """TCP SYN scan across ports on the target host."""
        src = self._pick_source(source_id)
        tgt = self._pick_target(target_id, DeviceType.SERVER)
        num_ports = int(20 + intensity * 980)  # 20 to 1000 ports
        ports = self._rng.sample(range(1, 65536), min(num_ports, 65535))

        flows = []
        for port in ports:
            self._flow_counter += 1
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=tgt.ip_address,
                src_port=self._rng.randint(1024, 65535),
                dst_port=port,
                protocol="tcp",
                bytes_sent=44,       # SYN packet
                bytes_received=0 if self._rng.random() > 0.3 else 44,
                packets_sent=1,
                packets_received=0 if self._rng.random() > 0.3 else 1,
                duration=0.001 + self._rng.random() * 0.1,
                timestamp=sim_time + self._rng.random() * 5,
                is_malicious=True,
                attack_type="port_scan",
                metadata={"scan_type": "syn", "intensity": intensity},
            ))
        return flows

    def _ddos_syn_flood(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """SYN flood DDoS with spoofed source IPs."""
        tgt = self._pick_target(target_id, DeviceType.SERVER)
        num_flows = int(50 + intensity * 950)
        dst_port = self._rng.choice([80, 443, 8080])
        flows = []
        for _ in range(num_flows):
            self._flow_counter += 1
            spoofed_ip = f"{self._rng.randint(1,223)}.{self._rng.randint(0,255)}.{self._rng.randint(0,255)}.{self._rng.randint(1,254)}"
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=spoofed_ip,
                dst_ip=tgt.ip_address,
                src_port=self._rng.randint(1024, 65535),
                dst_port=dst_port,
                protocol="tcp",
                bytes_sent=44,
                bytes_received=0,
                packets_sent=1,
                packets_received=0,
                duration=0.0,
                timestamp=sim_time + self._rng.random() * 2,
                is_malicious=True,
                attack_type="ddos_syn_flood",
                metadata={"spoofed": True, "intensity": intensity},
            ))
        return flows

    def _ddos_udp_flood(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """UDP flood with large payloads."""
        tgt = self._pick_target(target_id, DeviceType.SERVER)
        num_flows = int(30 + intensity * 500)
        flows = []
        for _ in range(num_flows):
            self._flow_counter += 1
            spoofed_ip = f"{self._rng.randint(1,223)}.{self._rng.randint(0,255)}.{self._rng.randint(0,255)}.{self._rng.randint(1,254)}"
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=spoofed_ip,
                dst_ip=tgt.ip_address,
                src_port=self._rng.randint(1024, 65535),
                dst_port=self._rng.randint(1, 65535),
                protocol="udp",
                bytes_sent=self._rng.randint(512, 1400),
                bytes_received=0,
                packets_sent=1,
                packets_received=0,
                duration=0.0,
                timestamp=sim_time + self._rng.random() * 2,
                is_malicious=True,
                attack_type="ddos_udp_flood",
                metadata={"intensity": intensity},
            ))
        return flows

    def _c2_beaconing(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """Command-and-Control beaconing at regular intervals."""
        src = self._pick_source(source_id, DeviceType.WORKSTATION)
        c2_ip = f"198.51.100.{self._rng.randint(1, 254)}"
        # Low-and-slow: fewer beacons at lower intensity
        num_beacons = int(3 + intensity * 15)
        interval = max(5, 60 * (1 - intensity))
        flows = []
        for i in range(num_beacons):
            self._flow_counter += 1
            jitter = self._rng.gauss(0, interval * 0.05)
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=c2_ip,
                src_port=self._rng.randint(1024, 65535),
                dst_port=self._rng.choice([443, 8443, 4444]),
                protocol="tcp",
                bytes_sent=self._rng.randint(64, 256),
                bytes_received=self._rng.randint(64, 512),
                packets_sent=self._rng.randint(1, 5),
                packets_received=self._rng.randint(1, 5),
                duration=self._rng.uniform(0.5, 3.0),
                timestamp=sim_time + i * interval + jitter,
                is_malicious=True,
                attack_type="c2_beaconing",
                metadata={"beacon_interval": interval, "c2_server": c2_ip, "intensity": intensity},
            ))
        return flows

    def _data_exfiltration(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """Large outbound data transfers to external IPs."""
        src = self._pick_source(source_id, DeviceType.WORKSTATION)
        ext_ip = f"203.0.113.{self._rng.randint(1, 254)}"
        num_flows = int(2 + intensity * 10)
        flows = []
        for _ in range(num_flows):
            self._flow_counter += 1
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=ext_ip,
                src_port=self._rng.randint(1024, 65535),
                dst_port=self._rng.choice([443, 80, 21, 8080]),
                protocol="tcp",
                bytes_sent=int(self._np_rng.pareto(1.0) * 50000 + 10000),
                bytes_received=self._rng.randint(64, 512),
                packets_sent=self._rng.randint(50, 500),
                packets_received=self._rng.randint(5, 30),
                duration=self._rng.uniform(10, 120),
                timestamp=sim_time + self._rng.random() * 30,
                is_malicious=True,
                attack_type="data_exfiltration",
                metadata={"exfil_target": ext_ip, "intensity": intensity},
            ))
        return flows

    def _lateral_movement(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """Internal host-to-host scanning and exploitation attempts."""
        ws_list = self.topology.get_workstations()
        srv_list = self.topology.get_servers()
        if not ws_list:
            return []
        src = self._pick_source(source_id, DeviceType.WORKSTATION)
        targets = self._rng.sample(ws_list + srv_list, min(int(2 + intensity * 8), len(ws_list + srv_list)))
        flows = []
        for tgt in targets:
            if tgt.device_id == src.device_id:
                continue
            for port in [445, 3389, 22, 135]:
                self._flow_counter += 1
                flows.append(TrafficFlow(
                    flow_id=f"ATK{self._flow_counter:08d}",
                    src_ip=src.ip_address,
                    dst_ip=tgt.ip_address,
                    src_port=self._rng.randint(1024, 65535),
                    dst_port=port,
                    protocol="tcp",
                    bytes_sent=self._rng.randint(64, 2048),
                    bytes_received=self._rng.randint(0, 1024),
                    packets_sent=self._rng.randint(1, 10),
                    packets_received=self._rng.randint(0, 8),
                    duration=self._rng.uniform(0.1, 5.0),
                    timestamp=sim_time + self._rng.random() * 20,
                    is_malicious=True,
                    attack_type="lateral_movement",
                    metadata={"target_host": tgt.device_id, "intensity": intensity},
                ))
        return flows

    def _dns_tunneling(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """DNS queries encoding exfiltrated data in subdomains."""
        src = self._pick_source(source_id, DeviceType.WORKSTATION)
        dns_servers = [d for d in self.topology.get_servers() if d.role and d.role.value == "dns"]
        if not dns_servers:
            dns_ip = "8.8.8.8"
        else:
            dns_ip = self._rng.choice(dns_servers).ip_address

        num_queries = int(10 + intensity * 200)
        flows = []
        for i in range(num_queries):
            self._flow_counter += 1
            # DNS tunneling: anomalously large queries
            query_size = self._rng.randint(100, 255)  # much larger than normal
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=dns_ip,
                src_port=self._rng.randint(1024, 65535),
                dst_port=53,
                protocol="udp",
                bytes_sent=query_size,
                bytes_received=self._rng.randint(100, 300),
                packets_sent=1,
                packets_received=1,
                duration=self._rng.uniform(0.01, 0.5),
                timestamp=sim_time + i * (60 / num_queries) * self._rng.uniform(0.5, 1.5),
                is_malicious=True,
                attack_type="dns_tunneling",
                metadata={"encoded_data_size": query_size, "intensity": intensity},
            ))
        return flows

    # ── Advanced attack implementations (v2.0) ─────────────────────

    def _encrypted_c2(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """
        Encrypted C2 beaconing — simulates HTTPS-based C2 channels.

        Hard to detect: uses port 443, realistic TLS payload sizes,
        and jittered intervals that mimic legitimate browsing.
        """
        src = self._pick_source(source_id, DeviceType.WORKSTATION)
        c2_ip = f"104.{self._rng.randint(16, 31)}.{self._rng.randint(0, 255)}.{self._rng.randint(1, 254)}"
        num_beacons = int(5 + intensity * 20)
        base_interval = max(10, 120 * (1 - intensity))
        flows = []
        for i in range(num_beacons):
            self._flow_counter += 1
            # Jittered interval mimicking legitimate browsing patterns
            jitter = self._np_rng.exponential(base_interval * 0.3)
            # TLS-like handshake + encrypted payload
            tls_overhead = self._rng.choice([517, 573, 609, 660])  # realistic TLS record sizes
            payload = self._rng.randint(200, 2000)
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=c2_ip,
                src_port=self._rng.randint(49152, 65535),
                dst_port=443,
                protocol="tcp",
                bytes_sent=tls_overhead + payload,
                bytes_received=tls_overhead + self._rng.randint(100, 5000),
                packets_sent=self._rng.randint(3, 12),
                packets_received=self._rng.randint(3, 15),
                duration=self._rng.uniform(0.5, 8.0),
                timestamp=sim_time + i * base_interval + jitter,
                is_malicious=True,
                attack_type="encrypted_c2",
                metadata={
                    "encryption": "TLS1.3",
                    "beacon_interval": base_interval,
                    "c2_server": c2_ip,
                    "intensity": intensity,
                },
            ))
        return flows

    def _polymorphic_attack(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """
        Polymorphic attack — mutates protocol, ports, and payload sizes
        on every attempt to evade signature-based detection.
        """
        src = self._pick_source(source_id)
        tgt = self._pick_target(target_id, DeviceType.SERVER)
        num_flows = int(10 + intensity * 50)
        protocols = ["tcp", "udp"]
        flows = []
        for i in range(num_flows):
            self._flow_counter += 1
            # Mutate characteristics each attempt
            proto = self._rng.choice(protocols)
            port = self._rng.choice([80, 443, 8080, 8443, 3306, 5432, 27017, self._rng.randint(1024, 65535)])
            # Vary payload sizes significantly (polymorphic mutation)
            base_size = int(self._np_rng.lognormal(6, 1.5))  # ~400 median, high variance
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=tgt.ip_address,
                src_port=self._rng.randint(1024, 65535),
                dst_port=port,
                protocol=proto,
                bytes_sent=max(40, base_size),
                bytes_received=max(0, base_size // self._rng.randint(2, 10)),
                packets_sent=self._rng.randint(1, 20),
                packets_received=self._rng.randint(0, 15),
                duration=self._rng.uniform(0.01, 30.0),
                timestamp=sim_time + self._rng.random() * 30,
                is_malicious=True,
                attack_type="polymorphic_attack",
                metadata={
                    "mutation_id": i,
                    "variant_hash": f"{self._rng.getrandbits(64):016x}",
                    "intensity": intensity,
                },
            ))
        return flows

    def _slow_exfiltration(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """
        Slow-and-low data exfiltration — sub-threshold data leakage
        spread over many tiny transfers that individually look benign.
        """
        src = self._pick_source(source_id, DeviceType.WORKSTATION)
        ext_ip = f"172.{self._rng.randint(64, 127)}.{self._rng.randint(0, 255)}.{self._rng.randint(1, 254)}"
        # Many small transfers over extended time
        num_flows = int(20 + intensity * 100)
        flows = []
        for i in range(num_flows):
            self._flow_counter += 1
            # Each flow is tiny — below typical alerting thresholds
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=ext_ip,
                src_port=self._rng.randint(49152, 65535),
                dst_port=self._rng.choice([443, 80, 53]),
                protocol="tcp",
                bytes_sent=self._rng.randint(50, 500),   # deliberately small
                bytes_received=self._rng.randint(40, 200),
                packets_sent=self._rng.randint(1, 3),
                packets_received=self._rng.randint(1, 3),
                duration=self._rng.uniform(0.1, 2.0),
                timestamp=sim_time + i * (60.0 / max(num_flows, 1)) * self._rng.uniform(0.8, 1.2),
                is_malicious=True,
                attack_type="slow_exfiltration",
                metadata={
                    "total_leaked": num_flows * 250,  # estimated total bytes
                    "stealth_rating": 1.0 - intensity,
                    "intensity": intensity,
                },
            ))
        return flows

    def _icmp_tunneling(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """
        ICMP tunneling — data hidden in echo request/reply payloads.

        Uses ICMP protocol with abnormally large payloads (normal ping
        is ~64 bytes; tunneling uses 200–1400 bytes).
        """
        src = self._pick_source(source_id, DeviceType.WORKSTATION)
        tunnel_ip = f"198.18.{self._rng.randint(0, 255)}.{self._rng.randint(1, 254)}"
        num_pings = int(10 + intensity * 100)
        flows = []
        for i in range(num_pings):
            self._flow_counter += 1
            # Abnormal ICMP payload size (normal is ~64 bytes)
            payload = self._rng.randint(200, 1400)
            flows.append(TrafficFlow(
                flow_id=f"ATK{self._flow_counter:08d}",
                src_ip=src.ip_address,
                dst_ip=tunnel_ip,
                src_port=0,   # ICMP has no ports
                dst_port=0,
                protocol="icmp",
                bytes_sent=payload,
                bytes_received=payload + self._rng.randint(-50, 50),
                packets_sent=1,
                packets_received=1,
                duration=self._rng.uniform(0.001, 0.5),
                timestamp=sim_time + i * (60.0 / num_pings),
                is_malicious=True,
                attack_type="icmp_tunneling",
                metadata={
                    "payload_size": payload,
                    "tunnel_endpoint": tunnel_ip,
                    "intensity": intensity,
                },
            ))
        return flows

    def _multi_stage_attack(
        self, sim_time: float, intensity: float,
        target_id: Optional[str], source_id: Optional[str],
    ) -> List[TrafficFlow]:
        """
        Multi-stage coordinated attack campaign:
          Stage 1: Reconnaissance (port scan)
          Stage 2: Exploitation (lateral movement)
          Stage 3: Exfiltration (data exfiltration)

        This tests the IDS's ability to correlate events across time.
        """
        flows = []

        # Stage 1: Recon (first 20 seconds)
        recon_flows = self._port_scan(
            sim_time, intensity * 0.3, target_id, source_id
        )
        for f in recon_flows:
            f.attack_type = "multi_stage_attack"
            f.metadata["stage"] = "recon"
        flows.extend(recon_flows[:int(5 + intensity * 20)])  # limit scope

        # Stage 2: Exploitation (20–40 seconds in)
        exploit_flows = self._lateral_movement(
            sim_time + 20, intensity * 0.5, target_id, source_id
        )
        for f in exploit_flows:
            f.attack_type = "multi_stage_attack"
            f.metadata["stage"] = "exploit"
        flows.extend(exploit_flows[:int(3 + intensity * 10)])

        # Stage 3: Exfiltration (40+ seconds in)
        exfil_flows = self._data_exfiltration(
            sim_time + 40, intensity * 0.7, target_id, source_id
        )
        for f in exfil_flows:
            f.attack_type = "multi_stage_attack"
            f.metadata["stage"] = "exfiltrate"
        flows.extend(exfil_flows)

        return flows

    # ── Helpers ──────────────────────────────────────────────────────

    def _pick_source(
        self, device_id: Optional[str], preferred: DeviceType = DeviceType.WORKSTATION
    ) -> Any:
        if device_id:
            return self.topology.get_device(device_id)
        candidates = self.topology.get_devices_by_type(preferred)
        if candidates:
            return self._rng.choice(candidates)
        return self._rng.choice(list(self.topology.devices.values()))

    def _pick_target(
        self, device_id: Optional[str], preferred: DeviceType = DeviceType.SERVER
    ) -> Any:
        if device_id:
            return self.topology.get_device(device_id)
        candidates = self.topology.get_devices_by_type(preferred)
        if candidates:
            return self._rng.choice(candidates)
        return self._rng.choice(list(self.topology.devices.values()))

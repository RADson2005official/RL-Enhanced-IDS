"""
Network Topology Model for the RL-Enhanced IDS Simulation.

Builds a configurable graph-based network topology with routers, switches,
servers, workstations and IoT devices using NetworkX.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import networkx as nx


class DeviceType(str, Enum):
    ROUTER = "router"
    SWITCH = "switch"
    FIREWALL = "firewall"
    SERVER = "server"
    WORKSTATION = "workstation"
    IOT = "iot"


class ServerRole(str, Enum):
    WEB = "web"
    DATABASE = "database"
    MAIL = "mail"
    DNS = "dns"
    FILE = "file"
    APP = "application"


@dataclass
class NetworkDevice:
    """Represents a single device on the network."""

    device_id: str
    device_type: DeviceType
    ip_address: str
    mac_address: str
    subnet: str
    services: List[str] = field(default_factory=list)
    is_compromised: bool = False
    role: Optional[ServerRole] = None

    def __hash__(self) -> int:
        return hash(self.device_id)


class NetworkTopology:
    """
    Graph-based model of the simulated network.

    Nodes represent devices; edges represent network links with bandwidth
    and latency attributes.
    """

    def __init__(
        self,
        num_routers: int = 3,
        num_switches: int = 6,
        num_servers: int = 8,
        num_workstations: int = 20,
        num_iot_devices: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self._rng = random.Random(seed)
        self.graph = nx.Graph()
        self.devices: Dict[str, NetworkDevice] = {}
        self._subnet_counter = 0

        self._build_topology(
            num_routers, num_switches, num_servers, num_workstations, num_iot_devices
        )

    # ── Public API ──────────────────────────────────────────────────

    def get_device(self, device_id: str) -> NetworkDevice:
        return self.devices[device_id]

    def get_devices_by_type(self, dtype: DeviceType) -> List[NetworkDevice]:
        return [d for d in self.devices.values() if d.device_type == dtype]

    def get_servers(self) -> List[NetworkDevice]:
        return self.get_devices_by_type(DeviceType.SERVER)

    def get_workstations(self) -> List[NetworkDevice]:
        return self.get_devices_by_type(DeviceType.WORKSTATION)

    def get_neighbors(self, device_id: str) -> List[str]:
        return list(self.graph.neighbors(device_id))

    def get_all_device_ids(self) -> List[str]:
        return list(self.devices.keys())

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    def compromise_device(self, device_id: str) -> None:
        self.devices[device_id].is_compromised = True

    def reset_compromises(self) -> None:
        for d in self.devices.values():
            d.is_compromised = False

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for d in self.devices.values():
            counts[d.device_type.value] = counts.get(d.device_type.value, 0) + 1
        counts["total_links"] = self.graph.number_of_edges()
        return counts

    # ── Topology construction ───────────────────────────────────────

    def _next_subnet(self) -> str:
        self._subnet_counter += 1
        return f"10.{self._subnet_counter}.0"

    def _random_mac(self) -> str:
        return ":".join(f"{self._rng.randint(0, 255):02x}" for _ in range(6))

    def _create_device(
        self,
        prefix: str,
        idx: int,
        dtype: DeviceType,
        subnet: str,
        host: int,
        services: Optional[List[str]] = None,
        role: Optional[ServerRole] = None,
    ) -> NetworkDevice:
        dev_id = f"{prefix}{idx:03d}"
        ip = f"{subnet}.{host}"
        dev = NetworkDevice(
            device_id=dev_id,
            device_type=dtype,
            ip_address=ip,
            mac_address=self._random_mac(),
            subnet=subnet,
            services=services or [],
            role=role,
        )
        self.devices[dev_id] = dev
        self.graph.add_node(dev_id, device_type=dtype.value, ip=ip)
        return dev

    def _add_link(
        self, a: str, b: str, bandwidth_mbps: int = 1000, latency_ms: float = 0.5
    ) -> None:
        self.graph.add_edge(a, b, bandwidth=bandwidth_mbps, latency=latency_ms)

    def _build_topology(
        self,
        nr: int,
        ns: int,
        nserv: int,
        nws: int,
        niot: int,
    ) -> None:
        # ── Core routers (full mesh) ────────────────────────────────
        core_subnet = self._next_subnet()
        routers = [
            self._create_device("RTR", i, DeviceType.ROUTER, core_subnet, i + 1)
            for i in range(nr)
        ]
        for i in range(nr):
            for j in range(i + 1, nr):
                self._add_link(routers[i].device_id, routers[j].device_id, 10000, 0.1)

        # ── Firewall between core and edge ──────────────────────────
        fw = self._create_device(
            "FW", 0, DeviceType.FIREWALL, core_subnet, 254,
            services=["stateful_inspection", "nat"],
        )
        self._add_link(fw.device_id, routers[0].device_id, 10000, 0.05)

        # ── Distribution switches ───────────────────────────────────
        switches: List[NetworkDevice] = []
        for i in range(ns):
            sw_subnet = self._next_subnet()
            sw = self._create_device("SW", i, DeviceType.SWITCH, sw_subnet, 1)
            parent_router = routers[i % nr]
            self._add_link(sw.device_id, parent_router.device_id, 10000, 0.2)
            switches.append(sw)

        # ── Servers ────────────────────────────────────────────────
        roles = list(ServerRole)
        for i in range(nserv):
            role = roles[i % len(roles)]
            sw = switches[i % len(switches)]
            services_map = {
                ServerRole.WEB: ["http", "https"],
                ServerRole.DATABASE: ["postgresql", "mysql"],
                ServerRole.MAIL: ["smtp", "imap"],
                ServerRole.DNS: ["dns"],
                ServerRole.FILE: ["smb", "ftp"],
                ServerRole.APP: ["https", "grpc"],
            }
            srv = self._create_device(
                "SRV", i, DeviceType.SERVER, sw.subnet, 10 + i,
                services=services_map.get(role, []),
                role=role,
            )
            self._add_link(srv.device_id, sw.device_id, 10000, 0.1)

        # ── Workstations ────────────────────────────────────────────
        for i in range(nws):
            sw = switches[i % len(switches)]
            ws = self._create_device(
                "WS", i, DeviceType.WORKSTATION, sw.subnet, 100 + i,
                services=["browser", "email_client", "office"],
            )
            self._add_link(ws.device_id, sw.device_id, 1000, 0.5)

        # ── IoT devices ────────────────────────────────────────────
        for i in range(niot):
            sw = switches[i % len(switches)]
            iot = self._create_device(
                "IOT", i, DeviceType.IOT, sw.subnet, 200 + i,
                services=["telemetry", "mqtt"],
            )
            self._add_link(iot.device_id, sw.device_id, 100, 2.0)

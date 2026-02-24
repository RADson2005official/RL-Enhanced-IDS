"""Tests for the simulation layer: network topology, traffic, attacks, orchestrator."""

import pytest
from src.simulation.network_topology import NetworkTopology, DeviceType
from src.simulation.traffic_generator import TrafficGenerator, TrafficFlow
from src.simulation.attack_generator import AttackGenerator
from src.simulation.scenario_orchestrator import ScenarioOrchestrator


# ── Network Topology ────────────────────────────────────────────────

class TestNetworkTopology:
    def test_default_topology(self):
        topo = NetworkTopology(seed=42)
        assert topo.num_devices > 0
        summary = topo.summary()
        assert summary["router"] == 3
        assert summary["switch"] == 6
        assert summary["server"] == 8
        assert summary["workstation"] == 20
        assert summary["iot"] == 5
        assert summary["total_links"] > 0

    def test_custom_topology(self):
        topo = NetworkTopology(num_routers=2, num_switches=3, num_servers=4,
                               num_workstations=10, num_iot_devices=2, seed=1)
        assert topo.summary()["router"] == 2
        assert topo.summary()["workstation"] == 10

    def test_get_device(self):
        topo = NetworkTopology(seed=42)
        dev = topo.get_device("RTR000")
        assert dev.device_type == DeviceType.ROUTER

    def test_get_devices_by_type(self):
        topo = NetworkTopology(seed=42)
        servers = topo.get_devices_by_type(DeviceType.SERVER)
        assert len(servers) == 8
        for s in servers:
            assert s.device_type == DeviceType.SERVER

    def test_compromise_and_reset(self):
        topo = NetworkTopology(seed=42)
        topo.compromise_device("WS000")
        assert topo.get_device("WS000").is_compromised
        topo.reset_compromises()
        assert not topo.get_device("WS000").is_compromised

    def test_neighbors(self):
        topo = NetworkTopology(seed=42)
        neighbors = topo.get_neighbors("RTR000")
        assert len(neighbors) > 0


# ── Traffic Generator ───────────────────────────────────────────────

class TestTrafficGenerator:
    @pytest.fixture
    def topo(self):
        return NetworkTopology(seed=42)

    def test_generate_flows(self, topo):
        gen = TrafficGenerator(topo, benign_rate=50, seed=42)
        flows = gen.generate_flows(sim_time=3600.0, time_step=60.0)
        assert len(flows) > 0
        for f in flows:
            assert isinstance(f, TrafficFlow)
            assert not f.is_malicious
            assert f.bytes_sent >= 0
            assert f.protocol in ("tcp", "udp", "icmp", "other")

    def test_traffic_stats(self, topo):
        gen = TrafficGenerator(topo, seed=42)
        flows = gen.generate_flows(3600.0)
        stats = gen.get_traffic_stats(flows)
        assert stats["traffic_volume"] > 0
        assert stats["num_flows"] > 0
        assert 0 <= stats["tcp_ratio"] <= 1

    def test_concept_drift(self, topo):
        gen = TrafficGenerator(topo, seed=42)
        old_tcp = gen.protocol_dist["tcp"]
        for _ in range(100):
            gen.apply_concept_drift()
        assert gen.protocol_dist["tcp"] != old_tcp

    def test_empty_stats(self, topo):
        gen = TrafficGenerator(topo, seed=42)
        stats = gen.get_traffic_stats([])
        assert stats["num_flows"] == 0.0


# ── Attack Generator ────────────────────────────────────────────────

class TestAttackGenerator:
    @pytest.fixture
    def topo(self):
        return NetworkTopology(seed=42)

    def test_all_attack_types(self, topo):
        gen = AttackGenerator(topo, seed=42)
        for atype in AttackGenerator.ATTACK_TYPES:
            flows = gen.generate_attack(atype, sim_time=100.0, intensity=0.5)
            assert len(flows) > 0
            for f in flows:
                assert f.is_malicious
                assert f.attack_type == atype

    def test_random_attack(self, topo):
        gen = AttackGenerator(topo, seed=42)
        flows = gen.random_attack(sim_time=100.0)
        assert len(flows) > 0
        assert all(f.is_malicious for f in flows)

    def test_invalid_attack_type(self, topo):
        gen = AttackGenerator(topo, seed=42)
        with pytest.raises(ValueError, match="Unknown attack type"):
            gen.generate_attack("nonexistent", sim_time=100.0)


# ── Scenario Orchestrator ──────────────────────────────────────────

class TestScenarioOrchestrator:
    @pytest.fixture
    def orchestrator(self):
        topo = NetworkTopology(seed=42)
        tgen = TrafficGenerator(topo, seed=42)
        agen = AttackGenerator(topo, seed=42)
        return ScenarioOrchestrator(topo, tgen, agen, max_steps=10, seed=42)

    def test_step_returns_flows_and_ground_truth(self, orchestrator):
        flows, gt = orchestrator.step()
        assert isinstance(flows, list)
        assert isinstance(gt, dict)
        assert "has_attack" in gt
        assert "step" in gt
        assert gt["step"] == 1

    def test_episode_runs_to_completion(self, orchestrator):
        while not orchestrator.is_done():
            orchestrator.step()
        assert orchestrator.current_step == 10

    def test_reset(self, orchestrator):
        orchestrator.step()
        orchestrator.reset()
        assert orchestrator.current_step == 0

    def test_curriculum_scenarios(self):
        events = ScenarioOrchestrator.create_curriculum_scenarios(100)
        assert len(events) == 3

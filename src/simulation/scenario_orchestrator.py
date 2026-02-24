"""
Scenario Orchestrator for the RL-Enhanced IDS Simulation.

Manages event timelines, schedules attacks, introduces concept drift,
and creates reproducible experimental episodes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .attack_generator import AttackGenerator
from .network_topology import NetworkTopology
from .traffic_generator import TrafficFlow, TrafficGenerator


class EventType(str, Enum):
    ATTACK = "attack"
    CONCEPT_DRIFT = "concept_drift"
    TOPOLOGY_CHANGE = "topology_change"
    TRAFFIC_SPIKE = "traffic_spike"


@dataclass
class ScheduledEvent:
    """A single scheduled event in the simulation timeline."""

    step: int
    event_type: EventType
    params: Dict[str, Any] = field(default_factory=dict)
    executed: bool = False


class ScenarioOrchestrator:
    """
    Controls the simulation episode lifecycle.

    Produces mixed benign + attack traffic each step, tracks ground truth,
    and manages scheduled events for reproducible experiments.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        traffic_gen: TrafficGenerator,
        attack_gen: AttackGenerator,
        max_steps: int = 200,
        attack_probability: float = 0.3,
        time_step_seconds: float = 60.0,
        seed: Optional[int] = None,
    ) -> None:
        self.topology = topology
        self.traffic_gen = traffic_gen
        self.attack_gen = attack_gen
        self.max_steps = max_steps
        self.attack_probability = attack_probability
        self.time_step = time_step_seconds
        self._rng = random.Random(seed)

        self.current_step: int = 0
        self.sim_time: float = 0.0
        self.scheduled_events: List[ScheduledEvent] = []
        self._episode_log: List[Dict[str, Any]] = []

    # ── Public API ──────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset for a new episode."""
        self.current_step = 0
        self.sim_time = 0.0
        self._episode_log.clear()
        self.topology.reset_compromises()
        for ev in self.scheduled_events:
            ev.executed = False

    def schedule_event(self, event: ScheduledEvent) -> None:
        self.scheduled_events.append(event)

    def step(self) -> Tuple[List[TrafficFlow], Dict[str, Any]]:
        """
        Advance one simulation step.

        Returns:
            Tuple of (all_flows, ground_truth) where ground_truth contains
            which flows are malicious and which attack types are active.
        """
        self.current_step += 1
        self.sim_time += self.time_step

        # ── Process scheduled events ────────────────────────────────
        self._process_events()

        # ── Generate benign traffic ─────────────────────────────────
        benign_flows = self.traffic_gen.generate_flows(self.sim_time, self.time_step)

        # ── Possibly generate attack traffic ────────────────────────
        attack_flows: List[TrafficFlow] = []
        active_attacks: List[str] = []

        if self._rng.random() < self.attack_probability:
            attack_type = self._rng.choice(AttackGenerator.ATTACK_TYPES)
            attack_flows = self.attack_gen.generate_attack(attack_type, self.sim_time)
            active_attacks.append(attack_type)

        all_flows = benign_flows + attack_flows
        self._rng.shuffle(all_flows)

        ground_truth = {
            "step": self.current_step,
            "sim_time": self.sim_time,
            "num_benign_flows": len(benign_flows),
            "num_attack_flows": len(attack_flows),
            "active_attacks": active_attacks,
            "has_attack": len(attack_flows) > 0,
            "malicious_flow_ids": [f.flow_id for f in attack_flows],
        }

        self._episode_log.append(ground_truth)
        return all_flows, ground_truth

    def is_done(self) -> bool:
        return self.current_step >= self.max_steps

    @property
    def episode_log(self) -> List[Dict[str, Any]]:
        return self._episode_log.copy()

    # ── Event processing ────────────────────────────────────────────

    def _process_events(self) -> None:
        for event in self.scheduled_events:
            if event.executed or event.step != self.current_step:
                continue
            event.executed = True
            if event.event_type == EventType.CONCEPT_DRIFT:
                self.traffic_gen.apply_concept_drift()
            elif event.event_type == EventType.TRAFFIC_SPIKE:
                multiplier = event.params.get("multiplier", 2.0)
                self.traffic_gen.benign_rate *= multiplier

    @staticmethod
    def create_curriculum_scenarios(max_steps: int) -> List[ScheduledEvent]:
        """
        Create a curriculum of events for progressive difficulty.

        Returns pre-built scheduled events introducing concept drift
        halfway through and a traffic spike near the end.
        """
        events = [
            ScheduledEvent(
                step=max_steps // 3,
                event_type=EventType.CONCEPT_DRIFT,
                params={"description": "Moderate protocol distribution shift"},
            ),
            ScheduledEvent(
                step=2 * max_steps // 3,
                event_type=EventType.CONCEPT_DRIFT,
                params={"description": "New application traffic pattern"},
            ),
            ScheduledEvent(
                step=int(max_steps * 0.8),
                event_type=EventType.TRAFFIC_SPIKE,
                params={"multiplier": 1.5, "description": "Legitimate traffic surge"},
            ),
        ]
        return events

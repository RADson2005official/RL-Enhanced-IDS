"""
Gymnasium-compatible Environment for RL-based IDS tuning.

The RL agent observes traffic features and adjusts the IDS parameters
(anomaly threshold and sensitivity) to maximise detection performance.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..ids.alert_manager import AlertManager
from ..ids.baseline_ids import BaselineIDS, IDSConfig
from ..ids.feature_extractor import FeatureExtractor
from ..simulation.attack_generator import AttackGenerator
from ..simulation.network_topology import NetworkTopology
from ..simulation.scenario_orchestrator import ScenarioOrchestrator
from ..simulation.traffic_generator import TrafficGenerator
from .reward_calculator import RewardCalculator


class IDSEnvironment(gym.Env):
    """
    Custom Gymnasium environment for training an RL agent to tune IDS parameters.

    Observation space (Box):
        Feature vector of shape (n_features * 3 + n_ids_params,) — current
        traffic features, temporal features, and IDS parameter state.

    Action space (Discrete):
        0 = no change
        1 = increase threshold
        2 = decrease threshold
        3 = increase sensitivity
        4 = decrease sensitivity
        5 = increase both
        6 = decrease both
    """

    metadata = {"render_modes": ["human"]}

    N_ACTIONS = 7
    THRESHOLD_DELTA = 0.05
    SENSITIVITY_DELTA = 0.1

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        cfg = config or {}

        sim_cfg = cfg.get("simulation", {})
        ids_cfg = cfg.get("ids", {})
        rl_cfg = cfg.get("rl_agent", {})
        reward_cfg = cfg.get("reward", {})

        # ── Build simulation ────────────────────────────────────────
        net_cfg = sim_cfg.get("network", {})
        self.topology = NetworkTopology(
            num_routers=net_cfg.get("num_routers", 3),
            num_switches=net_cfg.get("num_switches", 6),
            num_servers=net_cfg.get("num_servers", 8),
            num_workstations=net_cfg.get("num_workstations", 20),
            num_iot_devices=net_cfg.get("num_iot_devices", 5),
            seed=seed,
        )

        traffic_cfg = sim_cfg.get("traffic", {})
        self.traffic_gen = TrafficGenerator(
            topology=self.topology,
            benign_rate=traffic_cfg.get("benign_rate", 100),
            seed=seed,
        )

        self.attack_gen = AttackGenerator(
            topology=self.topology,
            seed=seed,
        )

        episode_cfg = sim_cfg.get("episodes", {})
        self.orchestrator = ScenarioOrchestrator(
            topology=self.topology,
            traffic_gen=self.traffic_gen,
            attack_gen=self.attack_gen,
            max_steps=episode_cfg.get("max_steps", 200),
            attack_probability=episode_cfg.get("attack_probability", 0.3),
            seed=seed,
        )

        # Add curriculum events
        for event in ScenarioOrchestrator.create_curriculum_scenarios(
            episode_cfg.get("max_steps", 200)
        ):
            self.orchestrator.schedule_event(event)

        # ── Build IDS ───────────────────────────────────────────────
        self.feature_extractor = FeatureExtractor(
            window_size=ids_cfg.get("window_size", 10)
        )
        ids_config = IDSConfig(
            anomaly_threshold=ids_cfg.get("anomaly_threshold", 0.5),
            contamination=ids_cfg.get("contamination", 0.05),
        )
        self.ids = BaselineIDS(config=ids_config)
        self.alert_manager = AlertManager()

        # ── Reward ──────────────────────────────────────────────────
        self.reward_calculator = RewardCalculator(
            tp_weight=reward_cfg.get("true_positive_weight", 1.0),
            fp_weight=reward_cfg.get("false_positive_weight", -0.5),
            fn_weight=reward_cfg.get("false_negative_weight", -1.0),
            tn_weight=reward_cfg.get("true_negative_weight", 0.2),
            stability_penalty=reward_cfg.get("stability_penalty", 0.1),
            adaptation_bonus=reward_cfg.get("adaptation_bonus", 0.3),
        )

        # ── Spaces ──────────────────────────────────────────────────
        n_features = FeatureExtractor.NUM_FEATURES
        # current + window_mean + window_std + IDS params (threshold, sensitivity)
        obs_dim = n_features * 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

        # ── Tracking ────────────────────────────────────────────────
        self._warmup_steps = 20
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_metrics: List[Dict[str, Any]] = []

    # ── Gymnasium API ───────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.orchestrator.reset()
        self.feature_extractor.reset()
        self.ids.reset()
        self.alert_manager.reset()
        self.reward_calculator.reset()
        self._step_count = 0
        self._episode_reward = 0.0
        self._episode_metrics.clear()

        # ── Warmup: collect benign data and train IDS ───────────────
        warmup_features: List[np.ndarray] = []
        for _ in range(self._warmup_steps):
            benign_flows = self.traffic_gen.generate_flows(
                self.orchestrator.sim_time, 60.0
            )
            features = self.feature_extractor.extract(benign_flows)
            warmup_features.append(features)
            self.orchestrator.sim_time += 60.0

        if len(warmup_features) >= 20:
            self.ids.fit_initial(warmup_features)

        obs = self._get_observation([])
        return obs, {"warmup_steps": self._warmup_steps}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # ── Apply RL action ─────────────────────────────────────────
        self._apply_action(action)

        # ── Advance simulation ──────────────────────────────────────
        all_flows, ground_truth = self.orchestrator.step()
        self._step_count += 1

        # ── Extract features and score ──────────────────────────────
        features = self.feature_extractor.extract(all_flows)
        result = self.ids.score(features)

        # ── Classify ────────────────────────────────────────────────
        has_attack = ground_truth["has_attack"]
        active_attacks = ground_truth["active_attacks"]
        attack_type = active_attacks[0] if active_attacks else None

        alert = self.alert_manager.record(
            step=self._step_count,
            is_anomalous=result.is_anomalous,
            anomaly_score=result.anomaly_score,
            ground_truth_malicious=has_attack,
            attack_type=attack_type,
        )

        # ── Compute reward ──────────────────────────────────────────
        counts = self.alert_manager.get_recent_counts(last_n_steps=1)
        reward = self.reward_calculator.compute(
            tp=counts["TP"],
            fp=counts["FP"],
            tn=counts["TN"],
            fn=counts["FN"],
            current_threshold=self.ids.current_threshold,
        )
        self._episode_reward += reward

        # ── Incremental model update ────────────────────────────────
        if not has_attack:
            self.ids.incremental_update(features, is_benign=True)

        # ── Check termination ───────────────────────────────────────
        terminated = self.orchestrator.is_done()
        truncated = False

        # ── Build info ──────────────────────────────────────────────
        info = {
            "step": self._step_count,
            "ground_truth": ground_truth,
            "anomaly_score": result.anomaly_score,
            "is_anomalous": result.is_anomalous,
            "classification": alert.classification,
            "threshold": self.ids.current_threshold,
            "sensitivity": self.ids.current_sensitivity,
            "episode_reward": self._episode_reward,
            "alert_counts": self.alert_manager.get_step_counts(),
        }
        self._episode_metrics.append(info)

        obs = self._get_observation(all_flows)
        return obs, reward, terminated, truncated, info

    # ── Helpers ─────────────────────────────────────────────────────

    def _apply_action(self, action: int) -> None:
        """
        Discrete action → IDS parameter adjustment.

        0 = no-op | 1 = ↑ threshold | 2 = ↓ threshold
        3 = ↑ sensitivity | 4 = ↓ sensitivity
        5 = ↑ both | 6 = ↓ both
        """
        d_t = self.THRESHOLD_DELTA
        d_s = self.SENSITIVITY_DELTA
        if action == 1:
            self.ids.adjust_threshold(d_t)
        elif action == 2:
            self.ids.adjust_threshold(-d_t)
        elif action == 3:
            self.ids.adjust_sensitivity(d_s)
        elif action == 4:
            self.ids.adjust_sensitivity(-d_s)
        elif action == 5:
            self.ids.adjust_threshold(d_t)
            self.ids.adjust_sensitivity(d_s)
        elif action == 6:
            self.ids.adjust_threshold(-d_t)
            self.ids.adjust_sensitivity(-d_s)
        # action == 0: no-op

    def _get_observation(self, flows) -> np.ndarray:
        current = self.feature_extractor.extract(flows) if flows else np.zeros(
            FeatureExtractor.NUM_FEATURES, dtype=np.float32
        )
        temporal = self.feature_extractor.get_temporal_features()
        ids_params = np.array(
            [self.ids.current_threshold, self.ids.current_sensitivity],
            dtype=np.float32,
        )
        return np.concatenate([current, temporal, ids_params])

    @property
    def episode_metrics(self) -> List[Dict[str, Any]]:
        return self._episode_metrics.copy()

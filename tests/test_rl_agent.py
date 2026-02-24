"""Tests for the RL agent layer: reward calculator, environment, agent."""

import numpy as np
import pytest

from src.rl_agent.reward_calculator import RewardCalculator
from src.rl_agent.ids_environment import IDSEnvironment
from src.evaluation.metrics import SecurityMetrics


# ── Reward Calculator ───────────────────────────────────────────────

class TestRewardCalculator:
    def test_perfect_detection(self):
        rc = RewardCalculator()
        reward = rc.compute(tp=5, fp=0, tn=10, fn=0, current_threshold=0.5)
        assert reward > 0

    def test_all_false_negatives(self):
        rc = RewardCalculator()
        reward = rc.compute(tp=0, fp=0, tn=10, fn=5, current_threshold=0.5)
        assert reward < 0

    def test_stability_penalty(self):
        rc = RewardCalculator(stability_penalty=1.0)
        rc.compute(tp=1, fp=0, tn=1, fn=0, current_threshold=0.5)
        r1 = rc.compute(tp=1, fp=0, tn=1, fn=0, current_threshold=0.9)
        rc2 = RewardCalculator(stability_penalty=1.0)
        rc2.compute(tp=1, fp=0, tn=1, fn=0, current_threshold=0.5)
        r2 = rc2.compute(tp=1, fp=0, tn=1, fn=0, current_threshold=0.5)
        # Large threshold change should yield lower reward
        assert r1 < r2

    def test_reset(self):
        rc = RewardCalculator()
        rc.compute(tp=1, fp=0, tn=1, fn=0, current_threshold=0.5)
        rc.reset()
        assert rc._prev_threshold is None


# ── Security Metrics ────────────────────────────────────────────────

class TestSecurityMetrics:
    def test_all_correct(self):
        m = SecurityMetrics.compute(tp=50, fp=0, tn=50, fn=0)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1_score"] == 1.0
        assert m["accuracy"] == 1.0
        assert m["false_positive_rate"] == 0.0

    def test_all_wrong(self):
        m = SecurityMetrics.compute(tp=0, fp=50, tn=0, fn=50)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["accuracy"] == 0.0

    def test_zeros(self):
        m = SecurityMetrics.compute(tp=0, fp=0, tn=0, fn=0)
        assert m["total_samples"] == 0

    def test_time_to_detect_empty(self):
        ttd = SecurityMetrics.compute_time_to_detect([])
        assert ttd["detections"] == 0


# ── IDS Environment (lightweight) ──────────────────────────────────

class TestIDSEnvironment:
    @pytest.fixture
    def env(self):
        """Create a small environment for quick tests."""
        config = {
            "simulation": {
                "network": {"num_routers": 1, "num_switches": 2,
                            "num_servers": 2, "num_workstations": 5,
                            "num_iot_devices": 1},
                "episodes": {"max_steps": 10, "attack_probability": 0.3},
            }
        }
        return IDSEnvironment(config=config, seed=42)

    def test_reset(self, env):
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert "warmup_steps" in info

    def test_step(self, env):
        obs, _ = env.reset()
        obs2, reward, term, trunc, info = env.step(0)
        assert obs2.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert "classification" in info

    def test_full_episode(self, env):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        assert steps == 10

    def test_all_actions_valid(self, env):
        env.reset()
        for action in range(env.N_ACTIONS):
            obs, reward, term, trunc, info = env.step(action)
            assert obs.shape == env.observation_space.shape

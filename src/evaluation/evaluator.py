"""
Evaluator: runs RL-tuned vs static-baseline comparison experiments.

Produces a comprehensive report comparing the RL agent's IDS performance
against a fixed-threshold baseline over multiple episodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..ids.alert_manager import AlertManager
from ..ids.baseline_ids import BaselineIDS, IDSConfig
from ..ids.feature_extractor import FeatureExtractor
from ..rl_agent.ids_environment import IDSEnvironment
from .metrics import SecurityMetrics


class Evaluator:
    """
    Runs comparative evaluation between RL-tuned and static IDS.
    """

    def __init__(
        self,
        env: IDSEnvironment,
        agent: Any = None,  # RLAgent type to avoid circular imports
        n_eval_episodes: int = 10,
    ) -> None:
        self.env = env
        self.agent = agent
        self.n_eval_episodes = n_eval_episodes

    # ── Public API ──────────────────────────────────────────────────

    def evaluate_rl_agent(self) -> Dict[str, Any]:
        """Run evaluation episodes using the RL agent."""
        if self.agent is None:
            raise ValueError("No RL agent provided for evaluation.")

        all_counts: List[Dict[str, int]] = []
        episode_rewards: List[float] = []
        episode_metrics_list: List[List[Dict[str, Any]]] = []

        for ep in range(self.n_eval_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated

            all_counts.append(info.get("alert_counts", {}))
            episode_rewards.append(ep_reward)
            episode_metrics_list.append(self.env.episode_metrics)

        return self._aggregate_results(
            "rl_agent", all_counts, episode_rewards, episode_metrics_list
        )

    def evaluate_static_baseline(
        self, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Run evaluation episodes with a fixed-threshold IDS (no RL)."""
        all_counts: List[Dict[str, int]] = []
        episode_rewards: List[float] = []
        episode_metrics_list: List[List[Dict[str, Any]]] = []

        for ep in range(self.n_eval_episodes):
            obs, info = self.env.reset()
            # Force the threshold and never change it
            self.env.ids.config.anomaly_threshold = threshold
            self.env.ids.config.sensitivity = 0.5
            done = False
            ep_reward = 0.0

            while not done:
                action = 0  # no-op action
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated

            all_counts.append(info.get("alert_counts", {}))
            episode_rewards.append(ep_reward)
            episode_metrics_list.append(self.env.episode_metrics)

        return self._aggregate_results(
            "static_baseline", all_counts, episode_rewards, episode_metrics_list
        )

    def run_comparison(self) -> Dict[str, Any]:
        """
        Compare RL agent against static baseline.

        Returns:
            Dict with both results and a comparison summary.
        """
        rl_results = self.evaluate_rl_agent()
        static_results = self.evaluate_static_baseline()

        comparison = {
            "reward_improvement": rl_results["mean_reward"] - static_results["mean_reward"],
            "f1_improvement": rl_results["metrics"]["f1_score"] - static_results["metrics"]["f1_score"],
            "fpr_reduction": static_results["metrics"]["false_positive_rate"] - rl_results["metrics"]["false_positive_rate"],
            "precision_improvement": rl_results["metrics"]["precision"] - static_results["metrics"]["precision"],
        }

        return {
            "rl_agent": rl_results,
            "static_baseline": static_results,
            "comparison": comparison,
        }

    def save_report(self, results: Dict[str, Any], path: str = "reports/evaluation_report.json") -> str:
        """Save evaluation report to JSON file."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        return str(save_path)

    # ── Internal ────────────────────────────────────────────────────

    def _aggregate_results(
        self,
        name: str,
        all_counts: List[Dict[str, int]],
        episode_rewards: List[float],
        episode_metrics_list: List[List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        total_tp = sum(c.get("TP", 0) for c in all_counts)
        total_fp = sum(c.get("FP", 0) for c in all_counts)
        total_tn = sum(c.get("TN", 0) for c in all_counts)
        total_fn = sum(c.get("FN", 0) for c in all_counts)

        metrics = SecurityMetrics.compute(total_tp, total_fp, total_tn, total_fn)

        return {
            "name": name,
            "n_episodes": self.n_eval_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "metrics": metrics,
            "confusion_matrix": {
                "TP": total_tp,
                "FP": total_fp,
                "TN": total_tn,
                "FN": total_fn,
            },
        }

"""
Evaluator: runs RL-tuned vs static-baseline comparison experiments.

Produces a comprehensive report comparing the RL agent's IDS performance
against a fixed-threshold baseline over multiple episodes, including
per-attack-type metrics, ROC/AUC, MCC, and adversarial robustness testing.
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
    Runs comparative evaluation between RL-tuned and static IDS,
    with adversarial robustness testing and per-attack-type analysis.
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
        all_scores: List[float] = []
        all_labels: List[bool] = []
        all_alert_details: List[Dict[str, Any]] = []

        for ep in range(self.n_eval_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated

                # Collect per-step data for ROC and per-attack analysis
                all_scores.append(info.get("anomaly_score", 0.0))
                gt = info.get("ground_truth", {})
                all_labels.append(gt.get("has_attack", False))
                all_alert_details.append({
                    "attack_type": gt.get("active_attacks", [None])[0] if gt.get("active_attacks") else None,
                    "classification": info.get("classification", ""),
                })

            all_counts.append(info.get("alert_counts", {}))
            episode_rewards.append(ep_reward)
            episode_metrics_list.append(self.env.episode_metrics)

        return self._aggregate_results(
            "rl_agent", all_counts, episode_rewards,
            episode_metrics_list, all_scores, all_labels, all_alert_details,
        )

    def evaluate_static_baseline(
        self, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Run evaluation episodes with a fixed-threshold IDS (no RL)."""
        all_counts: List[Dict[str, int]] = []
        episode_rewards: List[float] = []
        episode_metrics_list: List[List[Dict[str, Any]]] = []
        all_scores: List[float] = []
        all_labels: List[bool] = []
        all_alert_details: List[Dict[str, Any]] = []

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

                all_scores.append(info.get("anomaly_score", 0.0))
                gt = info.get("ground_truth", {})
                all_labels.append(gt.get("has_attack", False))
                all_alert_details.append({
                    "attack_type": gt.get("active_attacks", [None])[0] if gt.get("active_attacks") else None,
                    "classification": info.get("classification", ""),
                })

            all_counts.append(info.get("alert_counts", {}))
            episode_rewards.append(ep_reward)
            episode_metrics_list.append(self.env.episode_metrics)

        return self._aggregate_results(
            "static_baseline", all_counts, episode_rewards,
            episode_metrics_list, all_scores, all_labels, all_alert_details,
        )

    def evaluate_adversarial_robustness(
        self, n_episodes: int = 5
    ) -> Dict[str, Any]:
        """
        Test IDS robustness against adversarial evasion attempts.

        Simulates an attacker who crafts traffic to sit just below
        the detection threshold (mimicry attack).
        """
        if self.agent is None:
            raise ValueError("No RL agent provided for adversarial testing.")

        evasion_successes = 0
        total_attack_steps = 0
        adversarial_rewards: List[float] = []

        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.agent.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated

                gt = info.get("ground_truth", {})
                if gt.get("has_attack"):
                    total_attack_steps += 1
                    if not info.get("is_anomalous"):
                        evasion_successes += 1

            adversarial_rewards.append(ep_reward)

        evasion_rate = evasion_successes / max(total_attack_steps, 1)
        return {
            "evasion_rate": evasion_rate,
            "robustness_score": 1.0 - evasion_rate,
            "total_attack_steps": total_attack_steps,
            "evasion_successes": evasion_successes,
            "mean_reward_under_attack": float(np.mean(adversarial_rewards)),
            "n_episodes": n_episodes,
        }

    def run_comparison(self) -> Dict[str, Any]:
        """
        Compare RL agent against static baseline with full analysis.

        Returns:
            Dict with both results, comparison summary, and adversarial testing.
        """
        rl_results = self.evaluate_rl_agent()
        static_results = self.evaluate_static_baseline()
        adversarial = self.evaluate_adversarial_robustness()

        comparison = {
            "reward_improvement": rl_results["mean_reward"] - static_results["mean_reward"],
            "f1_improvement": rl_results["metrics"]["f1_score"] - static_results["metrics"]["f1_score"],
            "fpr_reduction": static_results["metrics"]["false_positive_rate"] - rl_results["metrics"]["false_positive_rate"],
            "precision_improvement": rl_results["metrics"]["precision"] - static_results["metrics"]["precision"],
            "mcc_improvement": rl_results["metrics"]["mcc"] - static_results["metrics"]["mcc"],
            "auc_improvement": rl_results["roc"]["auc"] - static_results["roc"]["auc"],
        }

        return {
            "rl_agent": rl_results,
            "static_baseline": static_results,
            "adversarial_robustness": adversarial,
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
        all_scores: List[float],
        all_labels: List[bool],
        all_alert_details: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        total_tp = sum(c.get("TP", 0) for c in all_counts)
        total_fp = sum(c.get("FP", 0) for c in all_counts)
        total_tn = sum(c.get("TN", 0) for c in all_counts)
        total_fn = sum(c.get("FN", 0) for c in all_counts)

        metrics = SecurityMetrics.compute(total_tp, total_fp, total_tn, total_fn)
        roc = SecurityMetrics.compute_roc_curve(all_scores, all_labels)
        per_attack = SecurityMetrics.compute_per_attack_type(all_alert_details)

        return {
            "name": name,
            "n_episodes": self.n_eval_episodes,
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "metrics": metrics,
            "roc": roc,
            "per_attack_type": per_attack,
            "confusion_matrix": {
                "TP": total_tp,
                "FP": total_fp,
                "TN": total_tn,
                "FN": total_fn,
            },
        }

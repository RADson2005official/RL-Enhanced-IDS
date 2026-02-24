"""
Security Metrics Calculator for the RL-Enhanced IDS.

Provides standard intrusion detection metrics: Precision, Recall, F1,
Accuracy, FPR, and Time-to-Detect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class SecurityMetrics:
    """
    Compute standard IDS evaluation metrics from TP/FP/TN/FN counts.
    """

    @staticmethod
    def compute(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """
        Compute all metrics from confusion-matrix counts.

        Returns:
            Dict with precision, recall, f1, accuracy, fpr, specificity.
        """
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)  # TPR / sensitivity
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
        fpr = fp / max(fp + tn, 1)
        specificity = tn / max(tn + fp, 1)

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": fpr,
            "specificity": specificity,
            "true_positive_rate": recall,
            "total_samples": tp + fp + tn + fn,
        }

    @staticmethod
    def compute_time_to_detect(
        episode_log: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Compute average time-to-detect from episode log entries.

        The episode log should contain 'has_attack' and 'active_attacks' fields
        from the scenario orchestrator, plus 'is_anomalous' from the IDS.

        Returns:
            Dict with mean and median time-to-detect in steps.
        """
        detection_delays: List[int] = []
        attack_start: Optional[int] = None

        for entry in episode_log:
            if entry.get("has_attack") and attack_start is None:
                attack_start = entry["step"]
            elif entry.get("has_attack") and entry.get("is_anomalous"):
                if attack_start is not None:
                    delay = entry["step"] - attack_start
                    detection_delays.append(delay)
                    attack_start = None
            elif not entry.get("has_attack"):
                attack_start = None  # Attack ended without detection

        if not detection_delays:
            return {"mean_ttd": float("inf"), "median_ttd": float("inf"), "detections": 0}

        import numpy as np
        return {
            "mean_ttd": float(np.mean(detection_delays)),
            "median_ttd": float(np.median(detection_delays)),
            "detections": len(detection_delays),
        }

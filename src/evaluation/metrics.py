"""
Security Metrics Calculator for the RL-Enhanced IDS.

Provides comprehensive intrusion detection metrics: Precision, Recall, F1,
Accuracy, FPR, MCC, ROC/AUC, per-attack-type breakdown, and Time-to-Detect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SecurityMetrics:
    """
    Compute standard and advanced IDS evaluation metrics.
    """

    @staticmethod
    def compute(tp: int, fp: int, tn: int, fn: int) -> Dict[str, float]:
        """
        Compute all metrics from confusion-matrix counts.

        Returns:
            Dict with precision, recall, f1, accuracy, fpr, specificity,
            mcc (Matthews Correlation Coefficient), and balanced accuracy.
        """
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)  # TPR / sensitivity
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
        fpr = fp / max(fp + tn, 1)
        specificity = tn / max(tn + fp, 1)

        # Matthews Correlation Coefficient — balanced measure for imbalanced data
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        mcc = mcc_num / max(mcc_den, 1e-8)

        # Balanced accuracy (average of TPR and TNR)
        balanced_accuracy = (recall + specificity) / 2.0

        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "false_positive_rate": fpr,
            "specificity": specificity,
            "true_positive_rate": recall,
            "mcc": float(mcc),
            "balanced_accuracy": float(balanced_accuracy),
            "total_samples": tp + fp + tn + fn,
        }

    @staticmethod
    def compute_per_attack_type(
        alerts: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute precision/recall per attack type.

        Args:
            alerts: List of alert dicts with 'attack_type', 'classification'
                    (TP/FP/TN/FN) fields.

        Returns:
            Dict mapping attack_type -> {precision, recall, f1, count}.
        """
        type_counts: Dict[str, Dict[str, int]] = {}
        for alert in alerts:
            atype = alert.get("attack_type")
            if not atype:
                continue
            if atype not in type_counts:
                type_counts[atype] = {"TP": 0, "FP": 0, "FN": 0}
            cls = alert.get("classification", "")
            if cls in type_counts[atype]:
                type_counts[atype][cls] += 1

        results = {}
        for atype, counts in type_counts.items():
            tp = counts["TP"]
            fp = counts["FP"]
            fn = counts["FN"]
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            results[atype] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "count": tp + fp + fn,
            }
        return results

    @staticmethod
    def compute_roc_curve(
        scores: List[float],
        labels: List[bool],
        n_thresholds: int = 50,
    ) -> Dict[str, Any]:
        """
        Compute ROC curve data points and AUC.

        Args:
            scores: Anomaly scores (higher = more anomalous).
            labels: Ground-truth labels (True = malicious).
            n_thresholds: Number of threshold points.

        Returns:
            Dict with 'fpr', 'tpr', 'thresholds', and 'auc'.
        """
        if not scores or not labels:
            return {"fpr": [], "tpr": [], "thresholds": [], "auc": 0.0}

        scores_arr = np.array(scores)
        labels_arr = np.array(labels, dtype=bool)

        thresholds = np.linspace(0.0, 1.0, n_thresholds)
        fprs, tprs = [], []

        for thresh in thresholds:
            predicted = scores_arr >= thresh
            tp = int(np.sum(predicted & labels_arr))
            fp = int(np.sum(predicted & ~labels_arr))
            fn = int(np.sum(~predicted & labels_arr))
            tn = int(np.sum(~predicted & ~labels_arr))
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            tprs.append(tpr)
            fprs.append(fpr)

        # Sort by FPR for AUC computation
        sorted_pairs = sorted(zip(fprs, tprs))
        sorted_fprs = [p[0] for p in sorted_pairs]
        sorted_tprs = [p[1] for p in sorted_pairs]

        # Trapezoidal AUC
        auc = float(np.trapz(sorted_tprs, sorted_fprs))

        return {
            "fpr": sorted_fprs,
            "tpr": sorted_tprs,
            "thresholds": thresholds.tolist(),
            "auc": max(0.0, min(1.0, auc)),
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

        return {
            "mean_ttd": float(np.mean(detection_delays)),
            "median_ttd": float(np.median(detection_delays)),
            "min_ttd": float(np.min(detection_delays)),
            "max_ttd": float(np.max(detection_delays)),
            "detections": len(detection_delays),
        }

"""
Alert Manager for the Baseline IDS.

Tracks alerts, manages ground-truth classification, and records
True Positive / False Positive / True Negative / False Negative counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Alert:
    """A single IDS alert."""

    alert_id: str
    step: int
    anomaly_score: float
    is_anomalous: bool
    ground_truth_malicious: bool
    classification: str  # TP, FP, TN, FN
    attack_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """
    Records IDS decisions against ground truth and maintains running metrics.
    """

    def __init__(self) -> None:
        self._alerts: List[Alert] = []
        self._tp = 0
        self._fp = 0
        self._tn = 0
        self._fn = 0
        self._alert_counter = 0

    # ── Public API ──────────────────────────────────────────────────

    def record(
        self,
        step: int,
        is_anomalous: bool,
        anomaly_score: float,
        ground_truth_malicious: bool,
        attack_type: Optional[str] = None,
    ) -> Alert:
        """
        Record a detection decision and classify it.

        Args:
            step: Current simulation step.
            is_anomalous: Whether the IDS flagged this as anomalous.
            anomaly_score: The anomaly score from the IDS.
            ground_truth_malicious: Whether this traffic was actually malicious.
            attack_type: Type of attack (if malicious).

        Returns:
            The classified Alert object.
        """
        self._alert_counter += 1

        if is_anomalous and ground_truth_malicious:
            classification = "TP"
            self._tp += 1
        elif is_anomalous and not ground_truth_malicious:
            classification = "FP"
            self._fp += 1
        elif not is_anomalous and ground_truth_malicious:
            classification = "FN"
            self._fn += 1
        else:
            classification = "TN"
            self._tn += 1

        alert = Alert(
            alert_id=f"ALERT{self._alert_counter:06d}",
            step=step,
            anomaly_score=anomaly_score,
            is_anomalous=is_anomalous,
            ground_truth_malicious=ground_truth_malicious,
            classification=classification,
            attack_type=attack_type,
        )
        self._alerts.append(alert)
        return alert

    @property
    def tp(self) -> int:
        return self._tp

    @property
    def fp(self) -> int:
        return self._fp

    @property
    def tn(self) -> int:
        return self._tn

    @property
    def fn(self) -> int:
        return self._fn

    def get_step_counts(self) -> Dict[str, int]:
        """Get current TP/FP/TN/FN counts."""
        return {"TP": self._tp, "FP": self._fp, "TN": self._tn, "FN": self._fn}

    def get_recent_alerts(self, n: int = 10) -> List[Alert]:
        return self._alerts[-n:]

    def get_recent_counts(self, last_n_steps: int = 5) -> Dict[str, int]:
        """Get TP/FP/TN/FN counts from the last N alerts."""
        recent = self._alerts[-last_n_steps:] if self._alerts else []
        counts = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        for a in recent:
            counts[a.classification] += 1
        return counts

    def reset(self) -> None:
        self._alerts.clear()
        self._tp = self._fp = self._tn = self._fn = 0
        self._alert_counter = 0

    @property
    def total_alerts(self) -> int:
        return len(self._alerts)

    @property
    def all_alerts(self) -> List[Alert]:
        return self._alerts.copy()

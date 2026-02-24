"""
Reward Calculator for the RL Agent.

Computes multi-component reward signals aligned with IDS objectives:
detection accuracy, false-positive minimisation, and adaptation speed.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class RewardCalculator:
    """
    Produces a scalar reward from IDS performance metrics.

    Reward components
    -----------------
    * **Detection reward**: +1 for true positives, −1 for false negatives
    * **False-positive penalty**: −0.5 per false positive
    * **True-negative bonus**: +0.2 for correctly passing benign traffic
    * **Threshold stability**: small penalty for excessive threshold changes
    * **Adaptation reward**: bonus when detection improves over time
    """

    def __init__(
        self,
        tp_weight: float = 1.0,
        fp_weight: float = -0.5,
        fn_weight: float = -1.0,
        tn_weight: float = 0.2,
        stability_penalty: float = 0.1,
        adaptation_bonus: float = 0.3,
    ) -> None:
        self.tp_weight = tp_weight
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.tn_weight = tn_weight
        self.stability_penalty = stability_penalty
        self.adaptation_bonus = adaptation_bonus

        self._prev_threshold: Optional[float] = None
        self._prev_detection_rate: Optional[float] = None

    def compute(
        self,
        tp: int,
        fp: int,
        tn: int,
        fn: int,
        current_threshold: float,
        info: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute the composite reward.

        Args:
            tp: True positives in this step.
            fp: False positives in this step.
            tn: True negatives in this step.
            fn: False negatives in this step.
            current_threshold: Current anomaly threshold used by the IDS.
            info: Optional extra info dict.

        Returns:
            Scalar reward value.
        """
        # ── Detection performance ───────────────────────────────────
        reward = (
            tp * self.tp_weight
            + fp * self.fp_weight
            + fn * self.fn_weight
            + tn * self.tn_weight
        )

        # ── Threshold stability ─────────────────────────────────────
        if self._prev_threshold is not None:
            delta = abs(current_threshold - self._prev_threshold)
            reward -= delta * self.stability_penalty
        self._prev_threshold = current_threshold

        # ── Adaptation bonus ────────────────────────────────────────
        positives = tp + fn
        det_rate = tp / max(positives, 1) if positives > 0 else 0.0
        if self._prev_detection_rate is not None and det_rate > self._prev_detection_rate:
            improvement = det_rate - self._prev_detection_rate
            reward += improvement * self.adaptation_bonus
        self._prev_detection_rate = det_rate

        return float(reward)

    def reset(self) -> None:
        """Reset internal tracking state for new episode."""
        self._prev_threshold = None
        self._prev_detection_rate = None

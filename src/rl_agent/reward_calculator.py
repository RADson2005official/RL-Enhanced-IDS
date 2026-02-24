"""
Reward Calculator for the RL Agent — Multi-Objective Shaping.

Computes composite reward signals aligned with IDS objectives:
detection accuracy (F1-driven), false-positive minimisation,
adaptation speed, attack diversity coverage, and time-to-detect.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RewardCalculator:
    """
    Produces a multi-objective scalar reward from IDS performance metrics.

    Reward components
    -----------------
    * **Detection reward**: weighted TP / FP / FN / TN
    * **F1 component**: holistic detection quality bonus
    * **Threshold stability**: penalty for excessive parameter oscillation
    * **Adaptation bonus**: reward for improving detection rate over time
    * **Attack diversity**: penalises specialisation (high on some, low on others)
    * **Time-to-detect shaping**: bonus for fast detection after attack onset
    * **Consecutive-miss penalty**: exponentially increasing FN penalty
    """

    def __init__(
        self,
        tp_weight: float = 1.0,
        fp_weight: float = -0.5,
        fn_weight: float = -1.0,
        tn_weight: float = 0.2,
        stability_penalty: float = 0.1,
        adaptation_bonus: float = 0.3,
        f1_bonus_scale: float = 0.5,
        diversity_penalty: float = 0.2,
        ttd_bonus: float = 0.3,
        consecutive_miss_scale: float = 0.5,
    ) -> None:
        self.tp_weight = tp_weight
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.tn_weight = tn_weight
        self.stability_penalty = stability_penalty
        self.adaptation_bonus = adaptation_bonus
        self.f1_bonus_scale = f1_bonus_scale
        self.diversity_penalty = diversity_penalty
        self.ttd_bonus = ttd_bonus
        self.consecutive_miss_scale = consecutive_miss_scale

        self._prev_threshold: Optional[float] = None
        self._prev_detection_rate: Optional[float] = None
        self._consecutive_fn: int = 0
        self._attack_type_counts: Dict[str, Dict[str, int]] = {}  # type -> {tp, fn}
        self._cumulative_tp: int = 0
        self._cumulative_fp: int = 0
        self._cumulative_fn: int = 0
        self._steps_since_attack_start: Optional[int] = None

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
            info: Optional extra info dict (can include 'attack_type',
                  'has_attack', 'detection_delay').

        Returns:
            Scalar reward value.
        """
        info = info or {}
        reward = 0.0

        # ── 1. Base detection performance ────────────────────────────
        reward += (
            tp * self.tp_weight
            + fp * self.fp_weight
            + fn * self.fn_weight
            + tn * self.tn_weight
        )

        # ── 2. F1-score bonus (holistic quality) ────────────────────
        self._cumulative_tp += tp
        self._cumulative_fp += fp
        self._cumulative_fn += fn
        cum_precision = self._cumulative_tp / max(self._cumulative_tp + self._cumulative_fp, 1)
        cum_recall = self._cumulative_tp / max(self._cumulative_tp + self._cumulative_fn, 1)
        f1 = 2 * cum_precision * cum_recall / max(cum_precision + cum_recall, 1e-8)
        reward += f1 * self.f1_bonus_scale

        # ── 3. Threshold stability ──────────────────────────────────
        if self._prev_threshold is not None:
            delta = abs(current_threshold - self._prev_threshold)
            reward -= delta * self.stability_penalty
        self._prev_threshold = current_threshold

        # ── 4. Adaptation bonus ─────────────────────────────────────
        positives = tp + fn
        det_rate = tp / max(positives, 1) if positives > 0 else 0.0
        if self._prev_detection_rate is not None and det_rate > self._prev_detection_rate:
            improvement = det_rate - self._prev_detection_rate
            reward += improvement * self.adaptation_bonus
        self._prev_detection_rate = det_rate

        # ── 5. Consecutive-miss exponential penalty ─────────────────
        if fn > 0:
            self._consecutive_fn += fn
            # Exponential penalty: 1st miss = 0.5x, 2nd = 1.0x, 3rd = 2.0x, ...
            exp_penalty = self.consecutive_miss_scale * (2 ** min(self._consecutive_fn - 1, 5))
            reward -= exp_penalty
        else:
            self._consecutive_fn = 0

        # ── 6. Attack diversity penalty ─────────────────────────────
        attack_type = info.get("attack_type")
        if attack_type:
            if attack_type not in self._attack_type_counts:
                self._attack_type_counts[attack_type] = {"tp": 0, "fn": 0}
            self._attack_type_counts[attack_type]["tp"] += tp
            self._attack_type_counts[attack_type]["fn"] += fn

            # Penalise if detection rates vary widely across attack types
            if len(self._attack_type_counts) >= 3:
                rates = []
                for counts in self._attack_type_counts.values():
                    total = counts["tp"] + counts["fn"]
                    if total >= 2:
                        rates.append(counts["tp"] / total)
                if len(rates) >= 2:
                    import numpy as np
                    rate_std = float(np.std(rates))
                    reward -= rate_std * self.diversity_penalty

        # ── 7. Time-to-detect shaping ───────────────────────────────
        has_attack = info.get("has_attack", (tp + fn) > 0)
        if has_attack:
            if self._steps_since_attack_start is None:
                self._steps_since_attack_start = 0
            self._steps_since_attack_start += 1
            if tp > 0:
                # Bonus inversely proportional to detection delay
                delay = self._steps_since_attack_start
                ttd_reward = self.ttd_bonus / max(delay, 1)
                reward += ttd_reward
                self._steps_since_attack_start = None
        else:
            self._steps_since_attack_start = None

        return float(reward)

    def reset(self) -> None:
        """Reset internal tracking state for new episode."""
        self._prev_threshold = None
        self._prev_detection_rate = None
        self._consecutive_fn = 0
        self._attack_type_counts.clear()
        self._cumulative_tp = 0
        self._cumulative_fp = 0
        self._cumulative_fn = 0
        self._steps_since_attack_start = None

    @property
    def attack_diversity_stats(self) -> Dict[str, float]:
        """Return per-attack-type detection rates."""
        stats = {}
        for atype, counts in self._attack_type_counts.items():
            total = counts["tp"] + counts["fn"]
            stats[atype] = counts["tp"] / max(total, 1)
        return stats

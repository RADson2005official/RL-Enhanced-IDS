"""
Baseline Anomaly Detection System (IDS) for the RL-Enhanced IDS.

Uses Isolation Forest with dynamically adjustable parameters that
the RL agent can tune at each time step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest


@dataclass
class IDSConfig:
    """Configuration for the baseline IDS, with RL-tunable parameters."""

    anomaly_threshold: float = 0.5
    min_threshold: float = 0.1
    max_threshold: float = 0.95
    contamination: float = 0.05
    n_estimators: int = 100
    sensitivity: float = 0.5      # 0‒1 overall sensitivity
    alert_cooldown: int = 0       # steps to suppress duplicate alerts


@dataclass
class AnomalyResult:
    """Result of anomaly scoring for one time step."""

    anomaly_score: float            # 0‒1 higher = more anomalous
    is_anomalous: bool
    raw_score: float                # raw Isolation Forest score
    threshold_used: float
    confidence: float               # model confidence estimate


class BaselineIDS:
    """
    Isolation Forest-based anomaly detection with RL-controllable parameters.

    The RL agent manipulates threshold, sensitivity and model parameters
    through discrete actions to improve detection performance.
    """

    def __init__(self, config: Optional[IDSConfig] = None) -> None:
        self.config = config or IDSConfig()
        self._model: Optional[IsolationForest] = None
        self._is_fitted = False
        self._training_data: List[np.ndarray] = []
        self._min_training_samples = 20

    # ── Public API ──────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def current_threshold(self) -> float:
        return self.config.anomaly_threshold

    @property
    def current_sensitivity(self) -> float:
        return self.config.sensitivity

    def fit_initial(self, benign_features: List[np.ndarray]) -> None:
        """
        Train the Isolation Forest on initial benign traffic data.

        Args:
            benign_features: List of feature vectors representing normal traffic.
        """
        if len(benign_features) < self._min_training_samples:
            raise ValueError(
                f"Need at least {self._min_training_samples} samples, got {len(benign_features)}"
            )

        X = np.array(benign_features)
        self._model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X)
        self._is_fitted = True
        self._training_data = list(benign_features)

    def score(self, features: np.ndarray) -> AnomalyResult:
        """
        Score a feature vector for anomaly.

        If model not yet fitted, uses a simpler statistical threshold approach.

        Args:
            features: 1-D feature vector.

        Returns:
            AnomalyResult with anomaly score and decision.
        """
        if not self._is_fitted:
            return self._fallback_score(features)

        # Isolation Forest decision_function returns negative for anomalies
        raw_score = float(self._model.decision_function(features.reshape(1, -1))[0])

        # Normalise to 0‒1 where 1 = most anomalous
        # decision_function: negative = anomaly, positive = normal
        anomaly_score = 1.0 / (1.0 + np.exp(raw_score * self.config.sensitivity * 5))

        effective_threshold = self.config.anomaly_threshold
        is_anomalous = anomaly_score >= effective_threshold

        confidence = abs(anomaly_score - effective_threshold) / max(effective_threshold, 1e-8)
        confidence = min(1.0, confidence)

        return AnomalyResult(
            anomaly_score=float(anomaly_score),
            is_anomalous=is_anomalous,
            raw_score=raw_score,
            threshold_used=effective_threshold,
            confidence=confidence,
        )

    def incremental_update(self, features: np.ndarray, is_benign: bool = True) -> None:
        """
        Optionally accumulate benign samples for periodic refit.
        Only adds confirmed benign samples.
        """
        if is_benign:
            self._training_data.append(features.copy())
            # Cap stored data to avoid memory issues
            if len(self._training_data) > 5000:
                self._training_data = self._training_data[-3000:]

    def refit(self) -> None:
        """Re-train the model on accumulated training data."""
        if len(self._training_data) >= self._min_training_samples:
            self.fit_initial(self._training_data)

    # ── RL action interface ─────────────────────────────────────────

    def adjust_threshold(self, delta: float) -> float:
        """
        Adjust the anomaly threshold by delta.

        Args:
            delta: Change amount (positive = more strict, negative = more lenient).

        Returns:
            New threshold value.
        """
        new = self.config.anomaly_threshold + delta
        self.config.anomaly_threshold = max(
            self.config.min_threshold, min(self.config.max_threshold, new)
        )
        return self.config.anomaly_threshold

    def adjust_sensitivity(self, delta: float) -> float:
        """
        Adjust model sensitivity by delta.

        Returns:
            New sensitivity value.
        """
        new = self.config.sensitivity + delta
        self.config.sensitivity = max(0.1, min(1.0, new))
        return self.config.sensitivity

    def get_params(self) -> Dict[str, float]:
        """Return current tunable parameters as a dict."""
        return {
            "anomaly_threshold": self.config.anomaly_threshold,
            "sensitivity": self.config.sensitivity,
            "contamination": self.config.contamination,
        }

    def reset(self) -> None:
        """Reset to default config while keeping the fitted model."""
        self.config.anomaly_threshold = 0.5
        self.config.sensitivity = 0.5

    # ── Internal ────────────────────────────────────────────────────

    def _fallback_score(self, features: np.ndarray) -> AnomalyResult:
        """Simple z-score-based fallback when model is not fitted."""
        magnitude = float(np.linalg.norm(features))
        score = min(1.0, magnitude / 10.0)
        return AnomalyResult(
            anomaly_score=score,
            is_anomalous=score >= self.config.anomaly_threshold,
            raw_score=magnitude,
            threshold_used=self.config.anomaly_threshold,
            confidence=0.3,
        )

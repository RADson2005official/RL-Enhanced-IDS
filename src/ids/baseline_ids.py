"""
Baseline Anomaly Detection System (IDS) for the RL-Enhanced IDS.

Uses an **ensemble** of Isolation Forests with randomised seeds for
robust, adversary-resistant anomaly detection.  The RL agent can
dynamically tune threshold, sensitivity, and contamination parameters.
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
    sensitivity: float = 0.5      # 0–1 overall sensitivity
    alert_cooldown: int = 0       # steps to suppress duplicate alerts
    ensemble_size: int = 3        # number of Isolation Forest models


@dataclass
class AnomalyResult:
    """Result of anomaly scoring for one time step."""

    anomaly_score: float            # 0–1 higher = more anomalous
    is_anomalous: bool
    raw_score: float                # raw Isolation Forest score
    threshold_used: float
    confidence: float               # model confidence estimate
    ensemble_agreement: float = 1.0  # fraction of models that agree


class BaselineIDS:
    """
    Ensemble Isolation Forest anomaly detection with RL-controllable parameters.

    Uses multiple Isolation Forest models with different random seeds to:
    - Reduce adversarial predictability (no single fixed seed)
    - Improve detection robustness via ensemble voting
    - Provide calibrated confidence via ensemble agreement

    The RL agent manipulates threshold, sensitivity and model parameters
    through discrete actions to improve detection performance.
    """

    def __init__(self, config: Optional[IDSConfig] = None) -> None:
        self.config = config or IDSConfig()
        self._models: List[IsolationForest] = []
        self._is_fitted = False
        self._training_data: List[np.ndarray] = []
        self._min_training_samples = 20
        self._refit_count = 0
        self._feature_importances: Optional[np.ndarray] = None
        self._rng = np.random.default_rng()

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

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Per-feature importance scores from the ensemble (if computed)."""
        return self._feature_importances

    def fit_initial(self, benign_features: List[np.ndarray]) -> None:
        """
        Train the Isolation Forest ensemble on initial benign traffic data.

        Creates `ensemble_size` models with different random seeds for
        robust, adversary-resistant detection.

        Args:
            benign_features: List of feature vectors representing normal traffic.
        """
        if len(benign_features) < self._min_training_samples:
            raise ValueError(
                f"Need at least {self._min_training_samples} samples, got {len(benign_features)}"
            )

        X = np.array(benign_features)
        self._models.clear()

        for i in range(self.config.ensemble_size):
            # Randomise seed each time for adversary resistance
            seed = int(self._rng.integers(0, 2**31))
            model = IsolationForest(
                n_estimators=self.config.n_estimators,
                contamination=self.config.contamination,
                random_state=seed,
                n_jobs=-1,
            )
            model.fit(X)
            self._models.append(model)

        self._is_fitted = True
        self._training_data = list(benign_features)
        self._refit_count += 1
        self._compute_feature_importances(X)

    def score(self, features: np.ndarray) -> AnomalyResult:
        """
        Score a feature vector for anomaly using ensemble voting.

        Each model votes independently; the final score is the mean.
        Ensemble agreement provides calibrated confidence.

        Args:
            features: 1-D feature vector.

        Returns:
            AnomalyResult with anomaly score, decision, and ensemble agreement.
        """
        if not self._is_fitted:
            return self._fallback_score(features)

        X = features.reshape(1, -1)
        raw_scores = []
        individual_anomaly_scores = []

        for model in self._models:
            raw = float(model.decision_function(X)[0])
            raw_scores.append(raw)
            # Normalise to 0–1 where 1 = most anomalous
            score = 1.0 / (1.0 + np.exp(raw * self.config.sensitivity * 5))
            individual_anomaly_scores.append(score)

        # Ensemble aggregation
        mean_raw = float(np.mean(raw_scores))
        mean_anomaly_score = float(np.mean(individual_anomaly_scores))

        effective_threshold = self.config.anomaly_threshold

        # Per-model decisions
        individual_decisions = [s >= effective_threshold for s in individual_anomaly_scores]
        ensemble_agreement = sum(individual_decisions) / len(individual_decisions)

        # Majority vote
        is_anomalous = ensemble_agreement >= 0.5

        # Confidence: combines distance from threshold + ensemble agreement
        distance_confidence = abs(mean_anomaly_score - effective_threshold) / max(effective_threshold, 1e-8)
        distance_confidence = min(1.0, distance_confidence)
        confidence = 0.6 * distance_confidence + 0.4 * ensemble_agreement

        return AnomalyResult(
            anomaly_score=float(mean_anomaly_score),
            is_anomalous=is_anomalous,
            raw_score=mean_raw,
            threshold_used=effective_threshold,
            confidence=float(confidence),
            ensemble_agreement=float(ensemble_agreement),
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
        """Re-train the ensemble on accumulated training data with fresh seeds."""
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
            "ensemble_size": float(self.config.ensemble_size),
            "refit_count": float(self._refit_count),
        }

    def reset(self) -> None:
        """Reset to default config while keeping the fitted models."""
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
            ensemble_agreement=1.0,
        )

    def _compute_feature_importances(self, X: np.ndarray) -> None:
        """
        Estimate feature importances via mean absolute anomaly score
        contribution across ensemble members (lightweight proxy for
        permutation importance).
        """
        if not self._models or X.shape[0] < 5:
            return

        n_features = X.shape[1]
        importances = np.zeros(n_features)

        # Use a sample of training data for efficiency
        sample_idx = self._rng.choice(len(X), min(100, len(X)), replace=False)
        X_sample = X[sample_idx]

        baseline_scores = np.mean(
            [model.decision_function(X_sample) for model in self._models], axis=0
        )

        for feat_idx in range(n_features):
            X_permuted = X_sample.copy()
            X_permuted[:, feat_idx] = self._rng.permutation(X_permuted[:, feat_idx])
            permuted_scores = np.mean(
                [model.decision_function(X_permuted) for model in self._models], axis=0
            )
            importances[feat_idx] = float(np.mean(np.abs(baseline_scores - permuted_scores)))

        # Normalise
        total = importances.sum()
        if total > 0:
            importances /= total
        self._feature_importances = importances

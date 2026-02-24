"""Tests for the IDS layer: feature extractor, baseline IDS, alert manager."""

import numpy as np
import pytest

from src.ids.feature_extractor import FeatureExtractor
from src.ids.baseline_ids import BaselineIDS, IDSConfig, AnomalyResult
from src.ids.alert_manager import AlertManager
from src.simulation.traffic_generator import TrafficFlow


# ── helpers ─────────────────────────────────────────────────────────

def _make_flow(malicious: bool = False, **kw) -> TrafficFlow:
    defaults = dict(
        flow_id="F00000001", src_ip="10.1.0.100", dst_ip="10.2.0.10",
        src_port=50000, dst_port=80, protocol="tcp",
        bytes_sent=1000, bytes_received=2000, packets_sent=5, packets_received=10,
        duration=1.0, timestamp=0.0, is_malicious=malicious,
    )
    defaults.update(kw)
    return TrafficFlow(**defaults)


def _make_flows(n: int = 50, malicious: bool = False) -> list:
    return [_make_flow(malicious=malicious, flow_id=f"F{i:08d}", src_port=1024 + i)
            for i in range(n)]


# ── Feature Extractor ──────────────────────────────────────────────

class TestFeatureExtractor:
    def test_extract_shape(self):
        fe = FeatureExtractor()
        features = fe.extract(_make_flows(10))
        assert features.shape == (FeatureExtractor.NUM_FEATURES,)

    def test_empty_extract(self):
        fe = FeatureExtractor()
        features = fe.extract([])
        assert np.all(features == 0)

    def test_temporal_features(self):
        fe = FeatureExtractor(window_size=5)
        for _ in range(5):
            fe.extract(_make_flows(10))
        temporal = fe.get_temporal_features()
        assert temporal.shape == (FeatureExtractor.NUM_FEATURES * 2,)

    def test_full_state(self):
        fe = FeatureExtractor()
        state = fe.get_full_state(_make_flows(10))
        assert state.shape == (FeatureExtractor.NUM_FEATURES * 3,)

    def test_reset(self):
        fe = FeatureExtractor()
        fe.extract(_make_flows(10))
        fe.reset()
        temporal = fe.get_temporal_features()
        assert np.all(temporal == 0)


# ── Baseline IDS ────────────────────────────────────────────────────

class TestBaselineIDS:
    def test_unfitted_fallback(self):
        ids = BaselineIDS()
        result = ids.score(np.ones(15, dtype=np.float32))
        assert isinstance(result, AnomalyResult)
        assert result.confidence == 0.3

    def test_fit_and_score(self):
        ids = BaselineIDS()
        features = [np.random.randn(15).astype(np.float32) for _ in range(30)]
        ids.fit_initial(features)
        assert ids.is_fitted
        result = ids.score(np.random.randn(15).astype(np.float32))
        assert 0 <= result.anomaly_score <= 1

    def test_fit_insufficient_samples(self):
        ids = BaselineIDS()
        with pytest.raises(ValueError, match="at least"):
            ids.fit_initial([np.zeros(15) for _ in range(5)])

    def test_adjust_threshold(self):
        ids = BaselineIDS()
        new = ids.adjust_threshold(0.1)
        assert new == 0.6

    def test_threshold_clamp(self):
        ids = BaselineIDS()
        ids.adjust_threshold(10)
        assert ids.current_threshold <= ids.config.max_threshold

    def test_adjust_sensitivity(self):
        ids = BaselineIDS()
        ids.adjust_sensitivity(0.3)
        assert ids.current_sensitivity == 0.8

    def test_reset(self):
        ids = BaselineIDS()
        ids.adjust_threshold(0.2)
        ids.reset()
        assert ids.current_threshold == 0.5


# ── Alert Manager ──────────────────────────────────────────────────

class TestAlertManager:
    def test_record_tp(self):
        am = AlertManager()
        alert = am.record(step=1, is_anomalous=True, anomaly_score=0.9,
                          ground_truth_malicious=True, attack_type="port_scan")
        assert alert.classification == "TP"
        assert am.tp == 1

    def test_record_fp(self):
        am = AlertManager()
        am.record(step=1, is_anomalous=True, anomaly_score=0.7, ground_truth_malicious=False)
        assert am.fp == 1

    def test_record_fn(self):
        am = AlertManager()
        am.record(step=1, is_anomalous=False, anomaly_score=0.2, ground_truth_malicious=True)
        assert am.fn == 1

    def test_record_tn(self):
        am = AlertManager()
        am.record(step=1, is_anomalous=False, anomaly_score=0.1, ground_truth_malicious=False)
        assert am.tn == 1

    def test_counts(self):
        am = AlertManager()
        am.record(1, True, 0.9, True)
        am.record(2, True, 0.7, False)
        am.record(3, False, 0.1, False)
        counts = am.get_step_counts()
        assert counts == {"TP": 1, "FP": 1, "TN": 1, "FN": 0}

    def test_reset(self):
        am = AlertManager()
        am.record(1, True, 0.9, True)
        am.reset()
        assert am.total_alerts == 0
        assert am.tp == 0

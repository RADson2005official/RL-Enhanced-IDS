"""Tests for utility modules: config and logger."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import load_config
from src.utils.logger import setup_logger


class TestConfig:
    def test_load_default(self):
        config = load_config()
        assert "simulation" in config
        assert "ids" in config
        assert "rl_agent" in config
        assert "dashboard" in config

    def test_load_custom_path(self, tmp_path):
        custom = {"simulation": {"network": {"num_routers": 5}}, "ids": {}}
        cfg_file = tmp_path / "test_config.yaml"
        cfg_file.write_text(yaml.dump(custom))
        config = load_config(str(cfg_file))
        assert config["simulation"]["network"]["num_routers"] == 5

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_env_override(self, monkeypatch, tmp_path):
        base = {"dashboard": {"port": 8050}, "logging": {"level": "INFO"}}
        cfg_file = tmp_path / "cfg.yaml"
        cfg_file.write_text(yaml.dump(base))
        monkeypatch.setenv("RL_IDS_DASHBOARD_PORT", "9999")
        config = load_config(str(cfg_file))
        assert config["dashboard"]["port"] == 9999


class TestLogger:
    def test_basic_logger(self):
        logger = setup_logger(name="test_basic", json_format=False)
        assert logger.name == "test_basic"
        assert len(logger.handlers) > 0

    def test_file_logger(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = setup_logger(name="test_file", log_file=str(log_file), json_format=False)
        logger.info("test message")
        assert log_file.exists()

    def test_idempotent(self):
        name = "test_idempotent"
        l1 = setup_logger(name=name, json_format=False)
        l2 = setup_logger(name=name, json_format=False)
        assert l1 is l2

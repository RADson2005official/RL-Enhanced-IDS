"""
Configuration management for the RL-Enhanced IDS.
Loads YAML config with environment variable overrides and validation.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "config.yaml"


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with optional environment variable overrides.

    Priority: ENV vars > YAML config > defaults

    Args:
        config_path: Path to YAML config file. Falls back to config/config.yaml.

    Returns:
        Merged configuration dictionary.
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # ── Environment variable overrides ──────────────────────────────
    env_overrides = {
        "RL_IDS_DASHBOARD_HOST": ("dashboard", "host"),
        "RL_IDS_DASHBOARD_PORT": ("dashboard", "port"),
        "RL_IDS_LOG_LEVEL": ("logging", "level"),
        "RL_IDS_RL_ALGORITHM": ("rl_agent", "algorithm"),
        "RL_IDS_TRAINING_EPISODES": ("rl_agent", "training_episodes"),
        "RL_IDS_MODEL_SAVE_PATH": ("rl_agent", "model_save_path"),
    }

    for env_key, config_path_tuple in env_overrides.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            section, key = config_path_tuple
            # Attempt numeric conversion
            try:
                env_val = int(env_val)
            except ValueError:
                try:
                    env_val = float(env_val)
                except ValueError:
                    pass
            config.setdefault(section, {})[key] = env_val

    return config


def get_nested(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely retrieve a nested config value."""
    current = config
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current

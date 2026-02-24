"""
RL Agent wrapper around Stable-Baselines3 DQN / PPO.

Provides a unified interface for training, evaluation, and model
persistence regardless of the underlying algorithm.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Type

import numpy as np

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from .ids_environment import IDSEnvironment


_ALGORITHMS: Dict[str, Type[BaseAlgorithm]] = {
    "DQN": DQN,
    "PPO": PPO,
}


class TrainingMetricsCallback(BaseCallback):
    """Records per-episode metrics for dashboard consumption."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_metrics: list[Dict[str, Any]] = []

    def _on_step(self) -> bool:
        # Check for episode ends via the monitor wrapper info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
            if info.get("alert_counts"):
                self.episode_metrics.append({
                    "step": self.num_timesteps,
                    "reward": info.get("episode_reward", 0),
                    "alert_counts": info["alert_counts"],
                    "threshold": info.get("threshold", 0.5),
                    "sensitivity": info.get("sensitivity", 0.5),
                })
        return True


class RLAgent:
    """
    Unified RL agent supporting DQN and PPO via Stable-Baselines3.

    Abstracts algorithm selection, hyperparameter configuration, training,
    evaluation, and model save/load.
    """

    def __init__(
        self,
        env: IDSEnvironment,
        algorithm: str = "DQN",
        hyperparams: Optional[Dict[str, Any]] = None,
        model_save_path: str = "models/rl_ids",
        seed: Optional[int] = None,
    ) -> None:
        self.env = env
        self.algorithm_name = algorithm.upper()
        self.hyperparams = hyperparams or {}
        self.model_save_path = Path(model_save_path)
        self.seed = seed

        if self.algorithm_name not in _ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. Use one of: {list(_ALGORITHMS)}"
            )

        self._model: Optional[BaseAlgorithm] = None
        self.metrics_callback = TrainingMetricsCallback()
        self._build_model()

    # ── Public API ──────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 50_000,
        callbacks: Optional[list[BaseCallback]] = None,
        progress_bar: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the RL agent.

        Args:
            total_timesteps: Total environment steps to train for.
            callbacks: Additional SB3 callbacks.
            progress_bar: Whether to display a progress bar.

        Returns:
            Dict with training summary metrics.
        """
        cb_list = [self.metrics_callback]
        if callbacks:
            cb_list.extend(callbacks)

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(cb_list),
            progress_bar=progress_bar,
        )

        return self._training_summary()

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> int:
        """
        Select an action given an observation.

        Returns:
            Integer action.
        """
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return int(action)

    def evaluate(self, n_episodes: int = 10) -> Dict[str, Any]:
        """
        Run evaluation episodes and collect metrics.

        Returns:
            Aggregated evaluation results.
        """
        episode_rewards = []
        episode_alert_counts = []

        for _ in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += reward
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            episode_alert_counts.append(info.get("alert_counts", {}))

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "n_episodes": n_episodes,
            "final_alert_counts": episode_alert_counts[-1] if episode_alert_counts else {},
        }

    def save(self, path: Optional[str] = None) -> str:
        """Save the current model to disk."""
        save_path = Path(path) if path else self.model_save_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(save_path))
        return str(save_path)

    def load(self, path: Optional[str] = None) -> None:
        """Load a model from disk."""
        load_path = Path(path) if path else self.model_save_path
        algo_cls = _ALGORITHMS[self.algorithm_name]
        self._model = algo_cls.load(str(load_path), env=self.env)

    @property
    def model(self) -> BaseAlgorithm:
        return self._model

    @property
    def training_metrics(self) -> Dict[str, Any]:
        return {
            "episode_rewards": self.metrics_callback.episode_rewards,
            "episode_lengths": self.metrics_callback.episode_lengths,
            "detailed_metrics": self.metrics_callback.episode_metrics,
        }

    # ── Internal ────────────────────────────────────────────────────

    def _build_model(self) -> None:
        algo_cls = _ALGORITHMS[self.algorithm_name]

        common_params: Dict[str, Any] = {
            "env": self.env,
            "verbose": 0,
            "seed": self.seed,
        }

        if self.algorithm_name == "DQN":
            defaults = {
                "learning_rate": 1e-4,
                "buffer_size": 50_000,
                "learning_starts": 1000,
                "batch_size": 64,
                "gamma": 0.99,
                "target_update_interval": 500,
                "exploration_fraction": 0.3,
                "exploration_final_eps": 0.05,
                "policy_kwargs": {"net_arch": [256, 256]},
            }
        else:  # PPO
            defaults = {
                "learning_rate": 3e-4,
                "n_steps": 2048,
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "policy_kwargs": {"net_arch": [dict(pi=[256, 256], vf=[256, 256])]},
            }

        merged = {**defaults, **self.hyperparams}
        self._model = algo_cls("MlpPolicy", **{**common_params, **merged})

    def _training_summary(self) -> Dict[str, Any]:
        rewards = self.metrics_callback.episode_rewards
        return {
            "algorithm": self.algorithm_name,
            "total_episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "final_rewards_mean": float(np.mean(rewards[-10:])) if len(rewards) >= 10 else 0.0,
        }

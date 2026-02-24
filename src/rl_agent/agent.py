"""
RL Agent wrapper for Stable-Baselines3 (DQN / PPO).

Adds SHA-256 model integrity verification, auto-checkpointing,
early stopping, and TensorBoard-compatible logging.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from ..rl_agent.ids_environment import IDSEnvironment


class TrainingMetricsCallback(BaseCallback):
    """Captures episode-level metrics for monitoring and dashboard."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_metrics: List[Dict[str, Any]] = []
        self._current_ep_reward = 0.0
        self._current_ep_length = 0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0.0])
        if hasattr(reward, "__len__"):
            self._current_ep_reward += float(reward[0])
        else:
            self._current_ep_reward += float(reward)
        self._current_ep_length += 1

        dones = self.locals.get("dones", [False])
        if hasattr(dones, "__len__"):
            done = bool(dones[0])
        else:
            done = bool(dones)

        if done:
            self.episode_rewards.append(self._current_ep_reward)
            self.episode_lengths.append(self._current_ep_length)
            self.episode_metrics.append({
                "episode": len(self.episode_rewards),
                "reward": self._current_ep_reward,
                "length": self._current_ep_length,
                "timestamp": time.time(),
            })
            self._current_ep_reward = 0.0
            self._current_ep_length = 0
        return True


class AutoCheckpointCallback(BaseCallback):
    """Periodically saves model checkpoints with SHA-256 integrity hashes."""

    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_freq: int = 5000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.save_freq = save_freq
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = self.save_dir / f"checkpoint_{self.num_timesteps}"
            self.model.save(str(path))
            # Compute SHA-256 of saved model
            model_file = str(path) + ".zip"
            sha = _sha256_file(model_file)
            meta = {
                "timesteps": self.num_timesteps,
                "sha256": sha,
                "saved_at": time.time(),
            }
            meta_path = str(path) + "_meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            if self.verbose:
                print(f"[Checkpoint] Saved at step {self.num_timesteps}, SHA-256: {sha[:16]}...")
        return True


class EarlyStoppingCallback(BaseCallback):
    """Stops training if the rolling reward plateaus for N episodes."""

    def __init__(
        self,
        patience: int = 20,
        min_improvement: float = 0.01,
        window: int = 10,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.patience = patience
        self.min_improvement = min_improvement
        self.window = window
        self._episode_rewards: List[float] = []
        self._best_mean: float = float("-inf")
        self._no_improve_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [False])
        if hasattr(dones, "__len__"):
            done = bool(dones[0])
        else:
            done = bool(dones)

        if done:
            infos = self.locals.get("infos", [{}])
            ep_r = infos[0].get("episode", {}).get("r", 0) if infos else 0
            self._episode_rewards.append(ep_r)

            if len(self._episode_rewards) >= self.window:
                mean_r = float(np.mean(self._episode_rewards[-self.window:]))
                if mean_r > self._best_mean + self.min_improvement:
                    self._best_mean = mean_r
                    self._no_improve_count = 0
                else:
                    self._no_improve_count += 1
                if self._no_improve_count >= self.patience:
                    if self.verbose:
                        print(f"[EarlyStopping] No improvement for {self.patience} episodes. Stopping.")
                    return False
        return True


class RLAgent:
    """
    Stable-Baselines3 RL Agent with integrity verification and checkpointing.

    Supports DQN and PPO algorithms, with SHA-256 model verification on
    save/load, auto-checkpointing, and early stopping.
    """

    SUPPORTED_ALGORITHMS = {"DQN": DQN, "PPO": PPO}

    def __init__(
        self,
        env: IDSEnvironment,
        algorithm: str = "DQN",
        hyperparams: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        if algorithm not in self.SUPPORTED_ALGORITHMS:
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use: {list(self.SUPPORTED_ALGORITHMS)}")

        self.algorithm_name = algorithm
        self.env = env
        self.metrics_callback = TrainingMetricsCallback()
        self._model_hash: Optional[str] = None

        algo_cls = self.SUPPORTED_ALGORITHMS[algorithm]
        default_hp: Dict[str, Any] = {
            "policy": "MlpPolicy",
            "learning_rate": 1e-3 if algorithm == "DQN" else 3e-4,
            "verbose": 0,
        }
        if hyperparams:
            default_hp.update(hyperparams)

        tb_log = tensorboard_log or "runs/"
        self._model = algo_cls(
            env=env,
            tensorboard_log=tb_log,
            seed=seed,
            **default_hp,
        )

    def train(
        self,
        total_timesteps: int = 50_000,
        callbacks: Optional[List[BaseCallback]] = None,
        progress_bar: bool = True,
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 5000,
        early_stopping: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the agent with optional checkpointing, early stopping, and TensorBoard.

        Returns:
            Training summary dict.
        """
        cb_list: List[BaseCallback] = [self.metrics_callback]
        cb_list.append(AutoCheckpointCallback(
            save_dir=checkpoint_dir,
            save_freq=checkpoint_freq,
        ))
        if early_stopping:
            cb_list.append(EarlyStoppingCallback(verbose=1))
        if callbacks:
            cb_list.extend(callbacks)

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(cb_list),
            progress_bar=progress_bar,
        )

        return self._training_summary()

    def predict(self, observation: Any, deterministic: bool = True) -> int:
        """Get action from the trained model."""
        action, _ = self._model.predict(observation, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> Dict[str, str]:
        """
        Save the model and compute SHA-256 checksum.

        Returns:
            Dict with model path and sha256 hash.
        """
        self._model.save(path)
        model_file = path if path.endswith(".zip") else path + ".zip"
        sha = _sha256_file(model_file)
        self._model_hash = sha

        # Save integrity metadata alongside model
        meta_path = path + "_integrity.json"
        meta = {
            "algorithm": self.algorithm_name,
            "sha256": sha,
            "saved_at": time.time(),
            "total_episodes": len(self.metrics_callback.episode_rewards),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return {"path": model_file, "sha256": sha}

    @classmethod
    def load(
        cls,
        path: str,
        env: IDSEnvironment,
        verify_integrity: bool = True,
    ) -> "RLAgent":
        """
        Load a saved model with optional SHA-256 verification.

        Args:
            path: Path to saved model (.zip).
            env: Environment instance.
            verify_integrity: If True, verify SHA-256 against stored hash.

        Returns:
            Loaded RLAgent instance.
        """
        model_file = path if path.endswith(".zip") else path + ".zip"

        if verify_integrity:
            meta_path = path.replace(".zip", "") + "_integrity.json"
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                expected_sha = meta.get("sha256")
                actual_sha = _sha256_file(model_file)
                if expected_sha and actual_sha != expected_sha:
                    raise RuntimeError(
                        f"Model integrity check FAILED!\n"
                        f"  Expected SHA-256: {expected_sha}\n"
                        f"  Actual SHA-256:   {actual_sha}\n"
                        f"Model file may have been tampered with."
                    )

        # Detect algorithm from file or metadata
        algorithm = "DQN"
        meta_path = path.replace(".zip", "") + "_integrity.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            algorithm = meta.get("algorithm", "DQN")

        agent = cls.__new__(cls)
        agent.algorithm_name = algorithm
        agent.env = env
        agent.metrics_callback = TrainingMetricsCallback()
        agent._model_hash = None

        algo_cls = cls.SUPPORTED_ALGORITHMS[algorithm]
        agent._model = algo_cls.load(path, env=env)

        return agent

    @property
    def training_metrics(self) -> Dict[str, Any]:
        """Return current training metrics snapshot."""
        rewards = self.metrics_callback.episode_rewards
        return {
            "total_episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0,
            "std_reward": float(np.std(rewards)) if rewards else 0,
            "max_reward": float(np.max(rewards)) if rewards else 0,
            "last_10_mean": float(np.mean(rewards[-10:])) if len(rewards) >= 10 else 0,
        }

    def _training_summary(self) -> Dict[str, Any]:
        """Produce a summary of the training run."""
        rewards = self.metrics_callback.episode_rewards
        lengths = self.metrics_callback.episode_lengths
        return {
            "algorithm": self.algorithm_name,
            "total_episodes": len(rewards),
            "total_timesteps": sum(lengths),
            "mean_reward": float(np.mean(rewards)) if rewards else 0,
            "std_reward": float(np.std(rewards)) if rewards else 0,
            "best_reward": float(np.max(rewards)) if rewards else 0,
            "last_10_mean": float(np.mean(rewards[-10:])) if len(rewards) >= 10 else 0,
            "mean_length": float(np.mean(lengths)) if lengths else 0,
        }


def _sha256_file(path: str) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

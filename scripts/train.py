#!/usr/bin/env python3
"""
Training script for the RL-Enhanced IDS.

Usage:
    python scripts/train.py [--config CONFIG] [--algorithm DQN|PPO]
                            [--timesteps N] [--seed SEED]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from src.dashboard.app import create_app
from src.rl_agent.agent import RLAgent
from src.rl_agent.ids_environment import IDSEnvironment
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RL-IDS agent")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--algorithm", type=str, default=None, help="DQN or PPO")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable dashboard")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        name="train",
        level=config.get("logging", {}).get("level", "INFO"),
        log_file=config.get("logging", {}).get("file", "logs/training.log"),
    )

    algorithm = args.algorithm or config.get("rl_agent", {}).get("algorithm", "DQN")
    timesteps = args.timesteps or config.get("rl_agent", {}).get("total_timesteps", 50_000)
    rl_cfg = config.get("rl_agent", {})

    logger.info("Starting RL-IDS training", extra={
        "algorithm": algorithm,
        "timesteps": timesteps,
        "seed": args.seed,
    })

    # ── Create environment ──────────────────────────────────────────
    env = IDSEnvironment(config=config, seed=args.seed)
    logger.info("Environment created", extra={
        "observation_space": str(env.observation_space.shape),
        "action_space": str(env.action_space.n),
        "network_devices": env.topology.num_devices,
    })

    # ── Shared metrics store ────────────────────────────────────────
    metrics_store = {
        "training": {"episode_rewards": [], "episode_lengths": [], "detailed": []},
        "evaluation": {},
        "status": "training",
    }

    # ── Start dashboard in background thread ────────────────────────
    if not args.no_dashboard:
        dash_cfg = config.get("dashboard", {})
        app = create_app(metrics_store=metrics_store)
        host = dash_cfg.get("host", "0.0.0.0")
        port = dash_cfg.get("port", 8050)

        def run_dashboard():
            uvicorn.run(app, host=host, port=port, log_level="warning")

        dash_thread = threading.Thread(target=run_dashboard, daemon=True)
        dash_thread.start()
        logger.info(f"Dashboard started at http://{host}:{port}")

    # ── Create and train agent ──────────────────────────────────────
    agent = RLAgent(
        env=env,
        algorithm=algorithm,
        hyperparams=rl_cfg.get("hyperparams", {}),
        model_save_path=rl_cfg.get("model_save_path", "models/rl_ids"),
        seed=args.seed,
    )

    summary = agent.train(total_timesteps=timesteps, progress_bar=True)

    # Update shared metrics
    metrics_store["training"] = agent.training_metrics
    metrics_store["status"] = "trained"

    logger.info("Training complete", extra=summary)

    # ── Save model ──────────────────────────────────────────────────
    save_path = agent.save()
    logger.info(f"Model saved to {save_path}")

    # ── Quick evaluation ────────────────────────────────────────────
    logger.info("Running quick evaluation...")
    eval_results = agent.evaluate(n_episodes=5)
    metrics_store["evaluation"] = eval_results
    metrics_store["status"] = "done"

    logger.info("Evaluation results", extra=eval_results)

    # Save training report
    report_path = Path("reports")
    report_path.mkdir(exist_ok=True)
    with open(report_path / "training_report.json", "w") as f:
        json.dump({"training": summary, "evaluation": eval_results}, f, indent=2, default=str)

    logger.info("Training pipeline complete!")
    print(f"\n{'='*60}")
    print(f"  Training Complete — {algorithm}")
    print(f"  Episodes: {summary['total_episodes']}")
    print(f"  Mean Reward: {summary['mean_reward']:.2f}")
    print(f"  Eval Mean Reward: {eval_results['mean_reward']:.2f}")
    print(f"  Model saved: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

"""
Training script for the RL-Enhanced IDS.

Usage:
    python -m scripts.train [--algorithm DQN|PPO] [--timesteps 100000]
                            [--config config.yaml] [--no-dashboard]
                            [--tensorboard-log runs/] [--seed 42]

Features:
    - Graceful Ctrl+C handling with model auto-save
    - Periodic IDS model refitting during training
    - TensorBoard logging integration
    - Auto-checkpointing every N steps
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.logging_config import setup_logger
from src.rl_agent.agent import RLAgent
from src.rl_agent.ids_environment import IDSEnvironment


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL-Enhanced IDS")
    parser.add_argument("--algorithm", choices=["DQN", "PPO"], default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-dashboard", action="store_true")
    parser.add_argument("--tensorboard-log", default="runs/")
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--refit-interval", type=int, default=10,
                        help="Refit IDS model every N episodes")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-early-stop", action="store_true")
    return parser.parse_args()


class GracefulShutdown:
    """Handles Ctrl+C by saving the model before exiting."""

    def __init__(self, agent: Optional[RLAgent] = None, save_path: str = "models/emergency_save") -> None:
        self.agent = agent
        self.save_path = save_path
        self._shutdown = False
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, signum: int, frame: Any) -> None:
        if self._shutdown:
            print("\n[!] Force quit. Exiting immediately.")
            sys.exit(1)
        self._shutdown = True
        print("\n[!] Shutdown requested. Saving model...")
        if self.agent:
            try:
                Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
                result = self.agent.save(self.save_path)
                print(f"[✓] Emergency save complete: {result['path']} (SHA-256: {result['sha256'][:16]}...)")
            except Exception as e:
                print(f"[✗] Emergency save failed: {e}")
        print("[!] Exiting gracefully.")
        sys.exit(0)

    @property
    def should_stop(self) -> bool:
        return self._shutdown


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        "rl_ids.train",
        log_level=config.get("logging", {}).get("level", "INFO"),
    )

    algorithm = args.algorithm or config.get("rl_agent", {}).get("algorithm", "DQN")
    timesteps = args.timesteps or config.get("rl_agent", {}).get("total_timesteps", 50_000)
    rl_cfg = config.get("rl_agent", {})

    logger.info("Starting RL-IDS training", extra={
        "algorithm": algorithm,
        "timesteps": timesteps,
        "tensorboard": args.tensorboard_log,
    })

    # ── Create environment ───────────────────────────────────────────
    env = IDSEnvironment(config=config, seed=args.seed)
    logger.info("Environment created", extra={
        "obs_space": str(env.observation_space.shape),
        "action_space": str(env.action_space.n),
    })

    # ── Shared metrics store for dashboard ───────────────────────────
    metrics_store: Dict[str, Any] = {
        "training_active": True,
        "episodes": [],
        "latest_metrics": {},
    }

    # ── Start dashboard ──────────────────────────────────────────────
    if not args.no_dashboard:
        try:
            from src.dashboard.app import create_app
            dash_cfg = config.get("dashboard", {})
            app = create_app(metrics_store, dash_cfg)
            dash_host = dash_cfg.get("host", "127.0.0.1")
            dash_port = dash_cfg.get("port", 8050)
            dash_thread = threading.Thread(
                target=app.run,
                kwargs={"host": dash_host, "port": dash_port},
                daemon=True,
            )
            dash_thread.start()
            logger.info(f"Dashboard running at http://{dash_host}:{dash_port}")
        except Exception as e:
            logger.warning(f"Dashboard failed to start: {e}")

    # ── Create and train agent ───────────────────────────────────────
    agent = RLAgent(
        env=env,
        algorithm=algorithm,
        hyperparams=rl_cfg.get("hyperparams", {}),
        tensorboard_log=args.tensorboard_log,
        seed=args.seed,
    )

    # Register graceful shutdown AFTER agent is created
    shutdown = GracefulShutdown(agent, save_path="models/emergency_save")

    logger.info("Training started...")
    start_time = time.time()

    summary = agent.train(
        total_timesteps=timesteps,
        progress_bar=True,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_freq=args.checkpoint_freq,
        early_stopping=not args.no_early_stop,
    )

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed:.1f}s", extra=summary)

    # ── Update shared metrics ────────────────────────────────────────
    metrics_store["training_active"] = False
    metrics_store["latest_metrics"] = summary

    # ── Save final model ─────────────────────────────────────────────
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    save_result = agent.save(str(model_dir / "rl_ids_final"))
    logger.info(f"Model saved: {save_result['path']} (SHA-256: {save_result['sha256'][:16]}...)")

    # ── Quick evaluation ─────────────────────────────────────────────
    from src.evaluation.evaluator import Evaluator
    evaluator = Evaluator(env=env, agent=agent, n_eval_episodes=5)
    results = evaluator.run_comparison()
    report_path = evaluator.save_report(results)
    logger.info(f"Evaluation report saved: {report_path}")

    # ── Print summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RL-IDS Training Complete!")
    print("=" * 60)
    print(f"  Algorithm:      {summary['algorithm']}")
    print(f"  Episodes:       {summary['total_episodes']}")
    print(f"  Mean Reward:    {summary['mean_reward']:.4f}")
    print(f"  Best Reward:    {summary['best_reward']:.4f}")
    print(f"  Last 10 Mean:   {summary['last_10_mean']:.4f}")
    print(f"  Model SHA-256:  {save_result['sha256'][:32]}...")
    if "comparison" in results:
        comp = results["comparison"]
        print(f"\n  RL vs Baseline:")
        print(f"    F1 improvement:  {comp['f1_improvement']:+.4f}")
        print(f"    FPR reduction:   {comp['fpr_reduction']:+.4f}")
        print(f"    MCC improvement: {comp['mcc_improvement']:+.4f}")
        print(f"    AUC improvement: {comp['auc_improvement']:+.4f}")
    if "adversarial_robustness" in results:
        adv = results["adversarial_robustness"]
        print(f"\n  Adversarial Robustness:")
        print(f"    Robustness score: {adv['robustness_score']:.4f}")
        print(f"    Evasion rate:     {adv['evasion_rate']:.4f}")
    print("=" * 60)

    logger.info("Training pipeline complete!")


if __name__ == "__main__":
    main()

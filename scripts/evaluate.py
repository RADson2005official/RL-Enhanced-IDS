#!/usr/bin/env python3
"""
Evaluation script for the RL-Enhanced IDS.

Compares the RL-tuned IDS against a static-threshold baseline and
produces a detailed report.

Usage:
    python scripts/evaluate.py [--config CONFIG] [--model-path PATH]
                               [--episodes N] [--seed SEED]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.evaluator import Evaluator
from src.rl_agent.agent import RLAgent
from src.rl_agent.ids_environment import IDSEnvironment
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RL-IDS agent")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config")
    parser.add_argument("--model-path", type=str, default=None, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(
        name="evaluate",
        level=config.get("logging", {}).get("level", "INFO"),
        log_file="logs/evaluation.log",
    )

    model_path = args.model_path or config.get("rl_agent", {}).get("model_save_path", "models/rl_ids")
    algorithm = config.get("rl_agent", {}).get("algorithm", "DQN")

    logger.info("Starting evaluation", extra={
        "model_path": model_path,
        "episodes": args.episodes,
    })

    # ── Create environment and agent ────────────────────────────────
    env = IDSEnvironment(config=config, seed=args.seed)
    agent = RLAgent(env=env, algorithm=algorithm, seed=args.seed)

    # Load trained model
    try:
        agent.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logger.warning(f"No model found at {model_path}, evaluating untrained agent")

    # ── Run comparison ──────────────────────────────────────────────
    evaluator = Evaluator(env=env, agent=agent, n_eval_episodes=args.episodes)
    results = evaluator.run_comparison()

    # ── Display results ─────────────────────────────────────────────
    rl = results["rl_agent"]
    static = results["static_baseline"]
    comp = results["comparison"]

    print(f"\n{'='*65}")
    print(f"  Evaluation Report — {args.episodes} episodes")
    print(f"{'='*65}")
    print(f"\n{'RL Agent':>20s} | {'Static Baseline':>16s} | {'Improvement':>12s}")
    print(f"{'─'*20} | {'─'*16} | {'─'*12}")

    rl_m = rl["metrics"]
    st_m = static["metrics"]

    for key, label in [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("false_positive_rate", "FPR"),
        ("accuracy", "Accuracy"),
    ]:
        rl_v = rl_m.get(key, 0)
        st_v = st_m.get(key, 0)
        diff = rl_v - st_v
        sign = "+" if diff >= 0 else ""
        print(f"  {label:>18s}: {rl_v:>7.1%} | {st_v:>15.1%} | {sign}{diff:>10.1%}")

    print(f"\n  {'Mean Reward':>18s}: {rl['mean_reward']:>7.2f} | {static['mean_reward']:>15.2f} | {comp['reward_improvement']:>+10.2f}")
    print(f"{'='*65}\n")

    # ── Save report ─────────────────────────────────────────────────
    report_path = evaluator.save_report(results)
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()

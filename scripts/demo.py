#!/usr/bin/env python3
"""
Demo script for the RL-Enhanced IDS.

Runs a single episode with verbose output showing each simulation step,
IDS decisions, and RL agent actions in real time.

Usage:
    python scripts/demo.py [--config CONFIG] [--model-path PATH] [--seed SEED]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rl_agent.agent import RLAgent
from src.rl_agent.ids_environment import IDSEnvironment
from src.utils.config import load_config
from src.utils.logger import setup_logger


ACTION_NAMES = {
    0: "No-op",
    1: "↑ Threshold",
    2: "↓ Threshold",
    3: "↑ Sensitivity",
    4: "↓ Sensitivity",
    5: "↑ Both",
    6: "↓ Both",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL-IDS Demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between steps (s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(name="demo", level="INFO")

    algorithm = config.get("rl_agent", {}).get("algorithm", "DQN")
    model_path = args.model_path or config.get("rl_agent", {}).get("model_save_path", "models/rl_ids")

    env = IDSEnvironment(config=config, seed=args.seed)
    agent = RLAgent(env=env, algorithm=algorithm, seed=args.seed)

    try:
        agent.load(model_path)
        print(f"✅ Loaded trained model from {model_path}")
    except FileNotFoundError:
        print("⚠️  No trained model found — using untrained agent for demo")

    print(f"\n{'═'*70}")
    print(f"  RL-Enhanced IDS — Live Demo")
    print(f"  Network: {env.topology.num_devices} devices")
    print(f"  Algorithm: {algorithm}")
    print(f"{'═'*70}\n")

    obs, info = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    while not done:
        step += 1
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        gt = info["ground_truth"]
        attack_str = ", ".join(gt["active_attacks"]) if gt["active_attacks"] else "None"
        clf = info["classification"]

        clf_emoji = {"TP": "🟢", "FP": "🟡", "TN": "🔵", "FN": "🔴"}.get(clf, "⚪")

        print(
            f"  Step {step:>3d} │ "
            f"Action: {ACTION_NAMES[action]:<15s} │ "
            f"Attack: {attack_str:<20s} │ "
            f"Score: {info['anomaly_score']:.3f} │ "
            f"{clf_emoji} {clf} │ "
            f"Reward: {reward:>+6.2f} │ "
            f"Thr: {info['threshold']:.2f}"
        )

        if args.delay > 0:
            time.sleep(args.delay)

    counts = info["alert_counts"]
    print(f"\n{'═'*70}")
    print(f"  Episode Complete — {step} steps")
    print(f"  Total Reward: {total_reward:.2f}")
    print(f"  TP: {counts['TP']}  FP: {counts['FP']}  TN: {counts['TN']}  FN: {counts['FN']}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()

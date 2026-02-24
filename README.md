# RL-Enhanced Anomaly-Based Intrusion Detection System

A production-ready **Reinforcement Learning-enhanced IDS** that dynamically adapts its detection baseline using DQN/PPO agents trained in a high-fidelity network simulation environment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL-Enhanced IDS                          │
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌────────────────┐  │
│  │  Simulation  │──▶│  Baseline    │──▶│   RL Agent     │  │
│  │  Environment │   │  IDS         │   │  (DQN / PPO)   │  │
│  │              │   │  (Isolation  │   │                │  │
│  │  • Topology  │   │   Forest)    │   │  • Gymnasium   │  │
│  │  • Traffic   │   │              │   │  • SB3         │  │
│  │  • Attacks   │   │  • Features  │   │  • Reward Fn   │  │
│  │  • Scenarios │   │  • Alerts    │   │                │  │
│  └─────────────┘   └──────────────┘   └────────────────┘  │
│                                                             │
│  ┌─────────────┐   ┌──────────────────────────────────┐    │
│  │  Evaluation  │   │  Dashboard (FastAPI)             │    │
│  │  Pipeline    │   │  Real-time metrics & monitoring  │    │
│  └─────────────┘   └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the agent
python scripts/train.py --algorithm DQN --timesteps 50000

# 4. Evaluate
python scripts/evaluate.py --episodes 10

# 5. Run interactive demo
python scripts/demo.py
```

### Docker

```bash
# Train (with live dashboard on port 8050)
docker-compose up rl-ids-train

# Run tests
docker-compose --profile test up rl-ids-test

# Evaluate (after training)
docker-compose --profile eval up rl-ids-eval

# Dashboard only
docker-compose --profile dashboard up rl-ids-dashboard
```

## Project Structure

```
Development/
├── config/
│   └── config.yaml          # Central YAML configuration
├── src/
│   ├── simulation/           # Network simulation environment
│   │   ├── network_topology.py
│   │   ├── traffic_generator.py
│   │   ├── attack_generator.py
│   │   └── scenario_orchestrator.py
│   ├── ids/                  # Baseline Intrusion Detection System
│   │   ├── feature_extractor.py
│   │   ├── baseline_ids.py
│   │   └── alert_manager.py
│   ├── rl_agent/             # Reinforcement Learning agent
│   │   ├── ids_environment.py
│   │   ├── reward_calculator.py
│   │   └── agent.py
│   ├── evaluation/           # Metrics and comparison evaluator
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── dashboard/            # FastAPI real-time dashboard
│   │   └── app.py
│   └── utils/                # Config loader and logging
│       ├── config.py
│       └── logger.py
├── scripts/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # RL vs static-baseline comparison
│   └── demo.py               # Interactive step-by-step demo
├── tests/                    # Pytest test suite
├── Dockerfile                # Multi-stage production image
├── docker-compose.yml        # 4-service compose setup
└── requirements.txt          # Pinned dependencies
```

## Key Features

| Feature | Description |
|---------|-------------|
| **7 Attack Types** | Port scan, SYN flood, UDP flood, C2 beaconing, exfiltration, lateral movement, DNS tunneling |
| **Adaptive IDS** | RL agent tunes threshold & sensitivity via 7 discrete actions |
| **Concept Drift** | Traffic patterns shift over time, testing agent adaptability |
| **Curriculum Learning** | Scheduled events progressively increase difficulty |
| **Full Metrics** | Precision, Recall, F1, FPR, Accuracy, Time-to-Detect |
| **Live Dashboard** | Dark-themed FastAPI UI with real-time polling |
| **Production Docker** | Multi-stage build, non-root user, healthchecks |

## Configuration

All settings are in `config/config.yaml` with environment variable overrides:

| Environment Variable | Override |
|---------------------|----------|
| `RL_IDS_DASHBOARD_HOST` | Dashboard bind host |
| `RL_IDS_DASHBOARD_PORT` | Dashboard port |
| `RL_IDS_LOG_LEVEL` | Logging level |
| `RL_IDS_ALGORITHM` | RL algorithm (DQN/PPO) |

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_simulation.py -v
python -m pytest tests/test_ids.py -v
python -m pytest tests/test_rl_agent.py -v
```

## Tech Stack

- **Python 3.11+**, **PyTorch**, **NumPy**, **Pandas**
- **scikit-learn** (Isolation Forest)
- **Gymnasium** + **Stable-Baselines3** (DQN, PPO)
- **NetworkX** (graph topology)
- **FastAPI** + **Uvicorn** (dashboard)
- **Docker** (containerisation)

---

*RL-Enhanced Anomaly-Based IDS — Sanjivani College of Engineering*

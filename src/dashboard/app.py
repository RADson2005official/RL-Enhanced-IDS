"""
Dashboard application for the RL-Enhanced IDS.

FastAPI app with a modern, dark-themed HTML UI for viewing training
progress, evaluation metrics, and live simulation state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


def create_app(
    title: str = "RL-IDS Dashboard",
    metrics_store: Optional[Dict[str, Any]] = None,
) -> FastAPI:
    """
    Create and return a configured FastAPI dashboard application.

    Args:
        title: Application title.
        metrics_store: Shared dict that training/evaluation routines update.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(title=title, version="1.0.0")

    # Mutable default for shared state
    if metrics_store is None:
        metrics_store = {
            "training": {"episode_rewards": [], "episode_lengths": [], "detailed": []},
            "evaluation": {},
            "status": "idle",
        }
    app.state.metrics = metrics_store

    # ── Routes ──────────────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        """Serve the main dashboard page."""
        return HTMLResponse(content=_DASHBOARD_HTML, status_code=200)

    @app.get("/api/metrics")
    async def get_metrics() -> JSONResponse:
        """Return current metrics as JSON."""
        return JSONResponse(content=app.state.metrics)

    @app.post("/api/metrics")
    async def update_metrics(request: Request) -> JSONResponse:
        """Update metrics store (used by training scripts)."""
        data = await request.json()
        app.state.metrics.update(data)
        return JSONResponse(content={"status": "ok"})

    @app.get("/api/status")
    async def get_status() -> JSONResponse:
        return JSONResponse(content={"status": app.state.metrics.get("status", "idle")})

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(content={"status": "healthy"})

    return app


# ── Embedded HTML dashboard ─────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RL-IDS Dashboard</title>
  <style>
    *, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }

    :root {
      --bg-primary: #0a0e17;
      --bg-secondary: #111827;
      --bg-card: #1a2234;
      --border: #2a3a52;
      --text-primary: #e2e8f0;
      --text-secondary: #94a3b8;
      --accent-blue: #3b82f6;
      --accent-green: #10b981;
      --accent-red: #ef4444;
      --accent-amber: #f59e0b;
      --accent-purple: #8b5cf6;
      --glow: rgba(59, 130, 246, 0.15);
    }

    body {
      font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: var(--bg-primary);
      color: var(--text-primary);
      min-height: 100vh;
    }

    .app-header {
      background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
      border-bottom: 1px solid var(--border);
      padding: 1.25rem 2rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .app-header h1 {
      font-size: 1.5rem;
      font-weight: 700;
      background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .status-badge {
      padding: 0.35rem 1rem;
      border-radius: 9999px;
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .status-badge.idle { background: rgba(148,163,184,0.15); color: var(--text-secondary); }
    .status-badge.training { background: rgba(59,130,246,0.2); color: var(--accent-blue); animation: pulse 2s infinite; }
    .status-badge.done { background: rgba(16,185,129,0.2); color: var(--accent-green); }

    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }

    .dashboard {
      max-width: 1400px;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 1.5rem;
    }

    .card {
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.5rem;
      transition: box-shadow 0.3s ease;
    }
    .card:hover { box-shadow: 0 0 30px var(--glow); }
    .card h2 {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 1rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1rem;
    }

    .metric-item {
      text-align: center;
      padding: 0.75rem;
      background: rgba(255,255,255,0.02);
      border-radius: 8px;
    }
    .metric-item .value {
      font-size: 1.75rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }
    .metric-item .label {
      font-size: 0.75rem;
      color: var(--text-secondary);
      margin-top: 0.25rem;
    }

    .bar-chart {
      display: flex;
      align-items: flex-end;
      gap: 3px;
      height: 120px;
      padding-top: 0.5rem;
    }
    .bar {
      flex: 1;
      min-width: 4px;
      background: var(--accent-blue);
      border-radius: 2px 2px 0 0;
      transition: height 0.3s ease;
      opacity: 0.8;
    }
    .bar:hover { opacity: 1; }

    .confusion-matrix {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
    }
    .cm-cell {
      text-align: center;
      padding: 1rem;
      border-radius: 8px;
      font-weight: 700;
      font-size: 1.25rem;
    }
    .cm-tp { background: rgba(16,185,129,0.15); color: var(--accent-green); }
    .cm-fp { background: rgba(245,158,11,0.15); color: var(--accent-amber); }
    .cm-fn { background: rgba(239,68,68,0.15); color: var(--accent-red); }
    .cm-tn { background: rgba(59,130,246,0.15); color: var(--accent-blue); }
    .cm-cell .cm-label { font-size: 0.7rem; font-weight: 400; opacity: 0.7; display: block; margin-top: 0.25rem; }

    .wide { grid-column: 1 / -1; }

    .log-list { max-height: 200px; overflow-y: auto; font-size: 0.8rem; }
    .log-list p { padding: 0.3rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); color: var(--text-secondary); }
  </style>
</head>
<body>
  <header class="app-header">
    <h1>⚡ RL-Enhanced IDS Dashboard</h1>
    <span id="statusBadge" class="status-badge idle">Idle</span>
  </header>

  <main class="dashboard">
    <div class="card">
      <h2>Training Progress</h2>
      <div class="metric-grid">
        <div class="metric-item">
          <div class="value" id="totalEpisodes">0</div>
          <div class="label">Episodes</div>
        </div>
        <div class="metric-item">
          <div class="value" id="meanReward">0.00</div>
          <div class="label">Mean Reward</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Detection Metrics</h2>
      <div class="metric-grid">
        <div class="metric-item">
          <div class="value" style="color:var(--accent-green)" id="precision">–</div>
          <div class="label">Precision</div>
        </div>
        <div class="metric-item">
          <div class="value" style="color:var(--accent-blue)" id="recall">–</div>
          <div class="label">Recall</div>
        </div>
        <div class="metric-item">
          <div class="value" style="color:var(--accent-purple)" id="f1Score">–</div>
          <div class="label">F1 Score</div>
        </div>
        <div class="metric-item">
          <div class="value" style="color:var(--accent-amber)" id="fpr">–</div>
          <div class="label">FPR</div>
        </div>
      </div>
    </div>

    <div class="card">
      <h2>Confusion Matrix</h2>
      <div class="confusion-matrix">
        <div class="cm-cell cm-tp"><span id="cmTP">0</span><span class="cm-label">True Positive</span></div>
        <div class="cm-cell cm-fp"><span id="cmFP">0</span><span class="cm-label">False Positive</span></div>
        <div class="cm-cell cm-fn"><span id="cmFN">0</span><span class="cm-label">False Negative</span></div>
        <div class="cm-cell cm-tn"><span id="cmTN">0</span><span class="cm-label">True Negative</span></div>
      </div>
    </div>

    <div class="card">
      <h2>IDS Parameters</h2>
      <div class="metric-grid">
        <div class="metric-item">
          <div class="value" style="color:var(--accent-blue)" id="threshold">0.50</div>
          <div class="label">Threshold</div>
        </div>
        <div class="metric-item">
          <div class="value" style="color:var(--accent-purple)" id="sensitivity">0.50</div>
          <div class="label">Sensitivity</div>
        </div>
      </div>
    </div>

    <div class="card wide">
      <h2>Episode Reward History</h2>
      <div class="bar-chart" id="rewardChart"></div>
    </div>
  </main>

  <script>
    async function fetchMetrics() {
      try {
        const res = await fetch('/api/metrics');
        const data = await res.json();
        updateUI(data);
      } catch (e) { /* retry on next poll */ }
    }

    function updateUI(data) {
      // Status
      const badge = document.getElementById('statusBadge');
      const status = data.status || 'idle';
      badge.textContent = status.charAt(0).toUpperCase() + status.slice(1);
      badge.className = 'status-badge ' + status;

      // Training
      const training = data.training || {};
      const rewards = training.episode_rewards || [];
      document.getElementById('totalEpisodes').textContent = rewards.length;
      if (rewards.length > 0) {
        const mean = rewards.reduce((a, b) => a + b, 0) / rewards.length;
        document.getElementById('meanReward').textContent = mean.toFixed(2);
      }

      // Reward chart
      const chart = document.getElementById('rewardChart');
      chart.innerHTML = '';
      const recent = rewards.slice(-80);
      if (recent.length > 0) {
        const max = Math.max(...recent.map(Math.abs), 1);
        recent.forEach(r => {
          const bar = document.createElement('div');
          bar.className = 'bar';
          const h = Math.max(2, (Math.abs(r) / max) * 100);
          bar.style.height = h + '%';
          bar.style.background = r >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
          chart.appendChild(bar);
        });
      }

      // Evaluation metrics
      const evalData = data.evaluation || {};
      const metrics = evalData.metrics || {};
      if (metrics.precision !== undefined) {
        document.getElementById('precision').textContent = (metrics.precision * 100).toFixed(1) + '%';
        document.getElementById('recall').textContent = (metrics.recall * 100).toFixed(1) + '%';
        document.getElementById('f1Score').textContent = (metrics.f1_score * 100).toFixed(1) + '%';
        document.getElementById('fpr').textContent = (metrics.false_positive_rate * 100).toFixed(1) + '%';
      }

      // Confusion matrix
      const cm = evalData.confusion_matrix || {};
      document.getElementById('cmTP').textContent = cm.TP || 0;
      document.getElementById('cmFP').textContent = cm.FP || 0;
      document.getElementById('cmFN').textContent = cm.FN || 0;
      document.getElementById('cmTN').textContent = cm.TN || 0;

      // IDS params
      const detailed = (training.detailed || []);
      if (detailed.length > 0) {
        const last = detailed[detailed.length - 1];
        document.getElementById('threshold').textContent = (last.threshold || 0.5).toFixed(2);
        document.getElementById('sensitivity').textContent = (last.sensitivity || 0.5).toFixed(2);
      }
    }

    // Poll every 3 seconds
    fetchMetrics();
    setInterval(fetchMetrics, 3000);
  </script>
</body>
</html>"""

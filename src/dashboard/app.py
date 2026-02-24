"""
Dashboard application for the RL-Enhanced IDS.

FastAPI app with CORS middleware, input validation, and a modern dark-themed
HTML UI for real-time training progress, per-attack-type detection,
ROC curve visualization, and training convergence charts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


# ── Input validation schemas ────────────────────────────────────────

class MetricsUpdate(BaseModel):
    """Schema for POST /api/metrics input validation."""
    training: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(idle|training|evaluating|done|error)$")
    latest_metrics: Optional[Dict[str, Any]] = None
    per_attack_type: Optional[Dict[str, Any]] = None
    roc_data: Optional[Dict[str, Any]] = None
    convergence: Optional[List[Dict[str, Any]]] = None


def create_app(
    metrics_store: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    title: str = "RL-IDS Dashboard",
) -> FastAPI:
    """
    Create and return a configured FastAPI dashboard application.

    Args:
        metrics_store: Shared dict that training/evaluation routines update.
        config: Dashboard-specific configuration.
        title: Application title.

    Returns:
        Configured FastAPI instance.
    """
    config = config or {}
    app = FastAPI(title=title, version="2.0.0")

    # ── CORS middleware ──────────────────────────────────────────────
    allowed_origins = config.get("cors_origins", ["http://127.0.0.1:*", "http://localhost:*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Shared state ─────────────────────────────────────────────────
    if metrics_store is None:
        metrics_store = {
            "training": {"episode_rewards": [], "episode_lengths": [], "detailed": []},
            "evaluation": {},
            "status": "idle",
            "per_attack_type": {},
            "roc_data": {},
            "convergence": [],
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
    async def update_metrics(update: MetricsUpdate) -> JSONResponse:
        """Update metrics store with validated data."""
        data = update.model_dump(exclude_none=True)
        for key, value in data.items():
            if key in app.state.metrics and isinstance(app.state.metrics[key], dict) and isinstance(value, dict):
                app.state.metrics[key].update(value)
            else:
                app.state.metrics[key] = value
        return JSONResponse(content={"status": "ok"})

    @app.get("/api/per-attack")
    async def get_per_attack() -> JSONResponse:
        """Per-attack-type detection rates."""
        return JSONResponse(content=app.state.metrics.get("per_attack_type", {}))

    @app.get("/api/roc")
    async def get_roc() -> JSONResponse:
        """ROC curve data points."""
        return JSONResponse(content=app.state.metrics.get("roc_data", {}))

    @app.get("/api/convergence")
    async def get_convergence() -> JSONResponse:
        """Training convergence data."""
        return JSONResponse(content=app.state.metrics.get("convergence", []))

    @app.get("/api/status")
    async def get_status() -> JSONResponse:
        return JSONResponse(content={"status": app.state.metrics.get("status", "idle")})

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse(content={"status": "healthy", "version": "2.0.0"})

    return app


# ── Embedded HTML dashboard ─────────────────────────────────────────

_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>RL-IDS Dashboard v2</title>
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
      --accent-cyan: #06b6d4;
      --accent-pink: #ec4899;
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

    .header-right { display: flex; align-items: center; gap: 1rem; }
    .version-tag { font-size: 0.7rem; color: var(--text-secondary); padding: 0.2rem 0.5rem; border: 1px solid var(--border); border-radius: 4px; }

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
    .status-badge.evaluating { background: rgba(139,92,246,0.2); color: var(--accent-purple); animation: pulse 2s infinite; }
    .status-badge.done { background: rgba(16,185,129,0.2); color: var(--accent-green); }
    .status-badge.error { background: rgba(239,68,68,0.2); color: var(--accent-red); }

    @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.6; } }

    .dashboard {
      max-width: 1500px;
      margin: 0 auto;
      padding: 2rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
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
      font-size: 0.85rem;
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
      font-size: 0.7rem;
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
      min-width: 3px;
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
    .cm-cell .cm-label { font-size: 0.65rem; font-weight: 400; opacity: 0.7; display: block; margin-top: 0.25rem; }

    .wide { grid-column: 1 / -1; }

    /* Per-attack chart */
    .attack-chart { display: flex; flex-direction: column; gap: 0.4rem; }
    .attack-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; }
    .attack-name { width: 120px; color: var(--text-secondary); text-align: right; font-size: 0.7rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .attack-bar-bg { flex: 1; height: 18px; background: rgba(255,255,255,0.05); border-radius: 4px; overflow: hidden; }
    .attack-bar-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 6px; font-size: 0.65rem; font-weight: 600; }
    .attack-bar-fill.good { background: linear-gradient(90deg, var(--accent-green), #34d399); }
    .attack-bar-fill.mid { background: linear-gradient(90deg, var(--accent-amber), #fbbf24); }
    .attack-bar-fill.bad { background: linear-gradient(90deg, var(--accent-red), #f87171); }

    /* ROC canvas */
    .roc-container { position: relative; width: 100%; padding-top: 100%; }
    .roc-container canvas { position: absolute; top: 0; left: 0; width: 100% !important; height: 100% !important; }

    /* Convergence chart */
    .convergence-chart { height: 120px; display: flex; align-items: flex-end; gap: 2px; }
    .conv-bar { flex: 1; min-width: 2px; border-radius: 1px 1px 0 0; }
  </style>
</head>
<body>
  <header class="app-header">
    <h1>⚡ RL-Enhanced IDS Dashboard</h1>
    <div class="header-right">
      <span class="version-tag">v2.0</span>
      <span id="statusBadge" class="status-badge idle">Idle</span>
    </div>
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
        <div class="metric-item">
          <div class="value" style="color:var(--accent-cyan)" id="mcc">–</div>
          <div class="label">MCC</div>
        </div>
        <div class="metric-item">
          <div class="value" style="color:var(--accent-pink)" id="auc">–</div>
          <div class="label">AUC</div>
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
        <div class="metric-item">
          <div class="value" style="color:var(--accent-green)" id="ensembleAgr">–</div>
          <div class="label">Ensemble Agreement</div>
        </div>
        <div class="metric-item">
          <div class="value" style="color:var(--accent-cyan)" id="robustness">–</div>
          <div class="label">Robustness Score</div>
        </div>
      </div>
    </div>

    <div class="card wide">
      <h2>Episode Reward History</h2>
      <div class="bar-chart" id="rewardChart"></div>
    </div>

    <div class="card">
      <h2>Per-Attack Detection Rate</h2>
      <div class="attack-chart" id="attackChart">
        <div style="color:var(--text-secondary);font-size:0.8rem;">Waiting for data…</div>
      </div>
    </div>

    <div class="card">
      <h2>ROC Curve</h2>
      <div class="roc-container">
        <canvas id="rocCanvas"></canvas>
      </div>
    </div>

    <div class="card wide">
      <h2>Training Convergence</h2>
      <div class="convergence-chart" id="convergenceChart"></div>
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
      const recent = rewards.slice(-100);
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
      if (metrics.mcc !== undefined) {
        document.getElementById('mcc').textContent = metrics.mcc.toFixed(3);
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
        if (last.ensemble_agreement !== undefined) {
          document.getElementById('ensembleAgr').textContent = (last.ensemble_agreement * 100).toFixed(0) + '%';
        }
      }

      // ROC data
      const roc = data.roc_data || evalData.roc || {};
      if (roc.auc !== undefined) {
        document.getElementById('auc').textContent = roc.auc.toFixed(3);
        drawROC(roc);
      }

      // Per-attack data
      const perAttack = data.per_attack_type || evalData.per_attack_type || {};
      if (Object.keys(perAttack).length > 0) {
        drawAttackChart(perAttack);
      }

      // Adversarial robustness
      const adv = data.adversarial_robustness || evalData.adversarial_robustness || {};
      if (adv.robustness_score !== undefined) {
        document.getElementById('robustness').textContent = (adv.robustness_score * 100).toFixed(0) + '%';
      }

      // Convergence
      const convergence = data.convergence || [];
      if (convergence.length > 0) {
        drawConvergence(convergence);
      }
    }

    function drawAttackChart(perAttack) {
      const container = document.getElementById('attackChart');
      container.innerHTML = '';
      const entries = Object.entries(perAttack).sort((a, b) => (b[1].f1_score || 0) - (a[1].f1_score || 0));
      entries.forEach(([name, stats]) => {
        const rate = (stats.recall || stats.detection_rate || 0) * 100;
        const cls = rate >= 70 ? 'good' : rate >= 40 ? 'mid' : 'bad';
        const row = document.createElement('div');
        row.className = 'attack-row';
        row.innerHTML = `
          <span class="attack-name">${name.replace(/_/g, ' ')}</span>
          <div class="attack-bar-bg">
            <div class="attack-bar-fill ${cls}" style="width:${Math.max(2, rate)}%">${rate.toFixed(0)}%</div>
          </div>`;
        container.appendChild(row);
      });
    }

    function drawROC(roc) {
      const canvas = document.getElementById('rocCanvas');
      const ctx = canvas.getContext('2d');
      const w = canvas.width = canvas.offsetWidth * 2;
      const h = canvas.height = canvas.offsetHeight * 2;
      ctx.clearRect(0, 0, w, h);
      ctx.scale(1, 1);

      // Background
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.fillRect(0, 0, w, h);

      // Diagonal reference
      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 4]);
      ctx.beginPath();
      ctx.moveTo(0, h);
      ctx.lineTo(w, 0);
      ctx.stroke();
      ctx.setLineDash([]);

      // ROC curve
      const fprs = roc.fpr || [];
      const tprs = roc.tpr || [];
      if (fprs.length > 1) {
        ctx.strokeStyle = 'var(--accent-blue)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(fprs[0] * w, h - tprs[0] * h);
        for (let i = 1; i < fprs.length; i++) {
          ctx.lineTo(fprs[i] * w, h - tprs[i] * h);
        }
        ctx.stroke();

        // Fill under curve
        ctx.globalAlpha = 0.15;
        ctx.fillStyle = 'var(--accent-blue)';
        ctx.beginPath();
        ctx.moveTo(fprs[0] * w, h);
        fprs.forEach((fpr, i) => ctx.lineTo(fpr * w, h - tprs[i] * h));
        ctx.lineTo(fprs[fprs.length - 1] * w, h);
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      // AUC label
      ctx.fillStyle = '#e2e8f0';
      ctx.font = `bold ${w * 0.06}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(`AUC: ${(roc.auc || 0).toFixed(3)}`, w / 2, h * 0.15);
    }

    function drawConvergence(data) {
      const container = document.getElementById('convergenceChart');
      container.innerHTML = '';
      const recent = data.slice(-120);
      const values = recent.map(d => d.reward || d.value || 0);
      const max = Math.max(...values.map(Math.abs), 1);
      values.forEach(v => {
        const bar = document.createElement('div');
        bar.className = 'conv-bar';
        const h = Math.max(1, (Math.abs(v) / max) * 100);
        bar.style.height = h + '%';
        bar.style.background = v >= 0
          ? `hsl(${140 + v / max * 30}, 70%, 50%)`
          : `hsl(0, 70%, ${50 + Math.abs(v) / max * 20}%)`;
        container.appendChild(bar);
      });
    }

    // Poll every 3 seconds
    fetchMetrics();
    setInterval(fetchMetrics, 3000);
  </script>
</body>
</html>"""

"""
app.py
======
SentinelAI — Intelligent Log Anomaly Detection System
Premium Streamlit Dashboard with interactive charts and animated UI.

Run with:
    streamlit run app.py
"""

import io
import logging

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data_processing import ingest_logs, extract_features
from model import fit_and_score, evaluate_with_synthetic_ground_truth
from explainer import explain_anomaly

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelAI — Log Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# PREMIUM CSS — animated cybersecurity dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* ── BASE ── */
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp {
    background: radial-gradient(ellipse at top left, #0a0e1a 0%, #060912 40%, #00050f 100%);
    color: #c9d1d9;
  }

  /* ── ANIMATED GRID BACKGROUND ── */
  .stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
      linear-gradient(rgba(56, 189, 248, 0.03) 1px, transparent 1px),
      linear-gradient(90deg, rgba(56, 189, 248, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  /* ── HERO TITLE BANNER ── */
  .hero-banner {
    position: relative;
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 50%, #0d1b2a 100%);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    overflow: hidden;
    box-shadow: 0 0 60px rgba(56, 189, 248, 0.1), inset 0 0 60px rgba(0,0,0,0.3);
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: conic-gradient(transparent, rgba(56,189,248,0.05), transparent 30%);
    animation: rotate 8s linear infinite;
  }
  @keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

  .hero-shield {
    font-size: 3.5rem;
    display: inline-block;
    animation: pulse-glow 2.5s ease-in-out infinite;
    filter: drop-shadow(0 0 20px rgba(56,189,248,0.8));
  }
  @keyframes pulse-glow {
    0%, 100% { filter: drop-shadow(0 0 10px rgba(56,189,248,0.6)); transform: scale(1); }
    50%       { filter: drop-shadow(0 0 30px rgba(56,189,248,1));   transform: scale(1.08); }
  }

  .hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 2.6rem;
    font-weight: 900;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399, #38bdf8);
    background-size: 300% 100%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin: 0;
    line-height: 1.2;
  }
  @keyframes shimmer { 0% { background-position: 0% 50%; } 100% { background-position: 300% 50%; } }

  .hero-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.5rem;
    letter-spacing: 0.12em;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.4);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace;
    color: #38bdf8;
    letter-spacing: 0.1em;
    margin-right: 0.5rem;
    margin-top: 0.8rem;
  }
  .badge-red   { border-color: rgba(248,113,113,0.4); color: #f87171; background: rgba(248,113,113,0.08); }
  .badge-green { border-color: rgba(52,211,153,0.4);  color: #34d399; background: rgba(52,211,153,0.08); }

  /* ── STATUS BAR ── */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-top: 1rem;
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #34d399;
    box-shadow: 0 0 8px #34d399;
    animation: blink 1.5s ease-in-out infinite;
    display: inline-block;
  }
  @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* ── METRIC CARDS ── */
  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
  }
  .kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: default;
  }
  .kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56,189,248,0.4);
    box-shadow: 0 12px 40px rgba(56,189,248,0.15), 0 0 0 1px rgba(56,189,248,0.1);
  }
  .kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 0 0 16px 16px;
  }
  .kpi-blue::after  { background: linear-gradient(90deg, #38bdf8, #818cf8); }
  .kpi-red::after   { background: linear-gradient(90deg, #f87171, #fb923c); }
  .kpi-green::after { background: linear-gradient(90deg, #34d399, #38bdf8); }
  .kpi-amber::after { background: linear-gradient(90deg, #fbbf24, #f87171); }

  .kpi-icon { font-size: 1.6rem; margin-bottom: 0.5rem; display: block; }
  .kpi-label {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.12em; color: #64748b; margin-bottom: 0.3rem;
  }
  .kpi-value {
    font-family: 'Orbitron', monospace;
    font-size: 2.1rem; font-weight: 700; line-height: 1;
  }
  .kpi-blue  .kpi-value { color: #38bdf8; text-shadow: 0 0 20px rgba(56,189,248,0.5); }
  .kpi-red   .kpi-value { color: #f87171; text-shadow: 0 0 20px rgba(248,113,113,0.5); }
  .kpi-green .kpi-value { color: #34d399; text-shadow: 0 0 20px rgba(52,211,153,0.5); }
  .kpi-amber .kpi-value { color: #fbbf24; text-shadow: 0 0 20px rgba(251,191,36,0.5); }
  .kpi-sub { font-size: 0.72rem; color: #475569; margin-top: 0.4rem; }

  /* ── SECTION HEADERS ── */
  .section-hdr {
    display: flex; align-items: center; gap: 0.6rem;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid rgba(56,189,248,0.15);
  }
  .section-hdr-icon { font-size: 1.1rem; }
  .section-hdr-text {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem; font-weight: 600;
    color: #94a3b8; letter-spacing: 0.1em; text-transform: uppercase;
  }
  .section-hdr-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3), transparent);
  }

  /* ── CHART CONTAINERS ── */
  .chart-container {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1rem;
    transition: border-color 0.3s;
  }
  .chart-container:hover { border-color: rgba(56,189,248,0.25); }

  /* ── DATA TABLE ── */
  .stDataFrame { border-radius: 12px; overflow: hidden; }
  .anomaly-marker {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #f87171;
    box-shadow: 0 0 8px #f87171;
    margin-right: 6px;
    animation: blink 1.5s ease-in-out infinite;
  }

  /* ── SIDEBAR ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060912 0%, #0a0e1a 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.12) !important;
  }
  [data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Orbitron', monospace;
    font-size: 0.9rem; color: #38bdf8; letter-spacing: 0.1em;
  }

  /* ── FILE UPLOADER ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed rgba(56,189,248,0.35) !important;
    border-radius: 16px !important;
    background: rgba(56,189,248,0.03) !important;
    transition: all 0.3s !important;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: rgba(56,189,248,0.7) !important;
    background: rgba(56,189,248,0.07) !important;
    box-shadow: 0 0 30px rgba(56,189,248,0.1) !important;
  }

  /* ── TABS ── */
  .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02);
    border-radius: 10px;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: #64748b;
    border-radius: 8px;
    padding: 0.4rem 1rem;
  }
  .stTabs [aria-selected="true"] {
    background: rgba(56,189,248,0.15) !important;
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
  }

  /* ── SCROLLBAR ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: #060912; }
  ::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: #334155; }

  /* ── ALERTS ── */
  .stAlert { border-radius: 12px; }

  /* ── DOWNLOAD BUTTON ── */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #0f2027, #203a43) !important;
    border: 1px solid rgba(56,189,248,0.4) !important;
    color: #38bdf8 !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    transition: all 0.3s !important;
  }
  .stDownloadButton > button:hover {
    background: rgba(56,189,248,0.15) !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.2) !important;
    transform: translateY(-1px);
  }

  /* ── AI SECURITY ANALYST PANEL ── */
  .ai-analyst-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #0f2236 50%, #0a1628 100%);
    border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 40px rgba(129,140,248,0.08), inset 0 0 40px rgba(0,0,0,0.25);
  }
  .ai-analyst-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #818cf8, #38bdf8, transparent);
    animation: scanline 3s ease-in-out infinite;
  }
  @keyframes scanline {
    0%, 100% { opacity: 0.4; } 50% { opacity: 1; }
  }
  .ai-analyst-title {
    font-family: 'Orbitron', monospace;
    font-size: 1rem; font-weight: 700;
    color: #818cf8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    display: flex; align-items: center; gap: 0.5rem;
  }
  .ai-analyst-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; color: #475569;
    margin-top: 0.3rem; letter-spacing: 0.08em;
  }

  /* AI insight card shown per anomaly */
  .ai-insight-card {
    background: rgba(129,140,248,0.05);
    border: 1px solid rgba(129,140,248,0.2);
    border-left: 3px solid #818cf8;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    font-family: 'Inter', sans-serif;
    font-size: 0.87rem;
    line-height: 1.65;
    color: #cbd5e1;
    position: relative;
    transition: border-color 0.3s;
  }
  .ai-insight-card:hover {
    border-color: rgba(129,140,248,0.5);
    border-left-color: #a5b4fc;
    box-shadow: 0 4px 20px rgba(129,140,248,0.08);
  }
  .ai-insight-log-ref {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #818cf8;
    background: rgba(129,140,248,0.1);
    border: 1px solid rgba(129,140,248,0.2);
    border-radius: 6px;
    padding: 0.15rem 0.5rem;
    display: inline-block;
    margin-bottom: 0.5rem;
  }
  .ai-insight-score-badge {
    float: right;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    color: #f87171;
    background: rgba(248,113,113,0.1);
    border: 1px solid rgba(248,113,113,0.25);
    border-radius: 8px;
    padding: 0.15rem 0.5rem;
  }
  .ai-powered-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    background: rgba(129,140,248,0.12);
    border: 1px solid rgba(129,140,248,0.3);
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: #818cf8;
    letter-spacing: 0.08em;
    margin-left: 0.8rem;
    vertical-align: middle;
  }
  .gemini-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: linear-gradient(135deg, #818cf8, #38bdf8);
    box-shadow: 0 0 6px #818cf8;
    display: inline-block;
    animation: blink 2s ease-in-out infinite;
  }
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

# ─── PLOTLY SHARED LAYOUT DEFAULTS ────────────────────────────────────────────
PLOT_LAYOUT = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter, sans-serif', color='#94a3b8', size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    hoverlabel=dict(
        bgcolor='#0f172a',
        bordercolor='rgba(56,189,248,0.5)',
        font=dict(family='JetBrains Mono', size=12, color='#e2e8f0'),
    ),
    xaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        linecolor='rgba(255,255,255,0.08)',
        zeroline=False,
    ),
    yaxis=dict(
        gridcolor='rgba(255,255,255,0.04)',
        linecolor='rgba(255,255,255,0.08)',
        zeroline=False,
    ),
)

TICK_STYLE = dict(family='JetBrains Mono', size=10)

LEGEND_STYLE = dict(
    bgcolor='rgba(15,23,42,0.8)',
    bordercolor='rgba(56,189,248,0.2)',
    borderwidth=1,
    font=dict(size=11),
)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Model Controls")
    st.markdown("---")

    fpr_target = st.slider(
        "🎯 FPR Target (%)",
        min_value=1, max_value=20, value=5, step=1,
        help="Maximum fraction of logs flagged as anomalies.",
    ) / 100.0

    contamination = st.slider(
        "🧪 Contamination Prior (%)",
        min_value=1, max_value=20, value=5, step=1,
        help="Prior estimate of anomaly fraction for Isolation Forest & One-Class SVM.",
    ) / 100.0

    show_score_dist   = st.checkbox("Show Score Distribution",    value=True)
    show_radar        = st.checkbox("Show Feature Radar Chart",   value=True)
    show_heatmap      = st.checkbox("Show Severity Heatmap",      value=True)
    show_raw_table    = st.checkbox("Show Raw Feature Table",     value=False)

    st.markdown("---")
    st.markdown("### 📖 Detection Signals")
    st.markdown("""
    - 🔴 **Log severity** (ERROR / FATAL)
    - 🌐 **HTTP 5xx errors**
    - ⏱️ **Response time spikes**
    - 📝 **Rare message vocabulary**
    - 🕐 **Off-hours activity**
    - 💥 **Error keyword density**
    """)
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:0.5rem;'>
      <span style='font-family:JetBrains Mono; font-size:0.65rem; color:#334155;'>
        SENTINEL AI  v2.0 · ENSEMBLE ML
      </span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# HERO HEADER
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
  <div style="display:flex; align-items:center; gap:1.5rem; position:relative; z-index:1;">
    <div>
      <span class="hero-shield">🛡️</span>
    </div>
    <div>
      <h1 class="hero-title">SentinelAI</h1>
      <p class="hero-subtitle">▸ INTELLIGENT LOG ANOMALY DETECTION SYSTEM</p>
      <div>
        <span class="hero-badge">ENSEMBLE ML</span>
        <span class="hero-badge badge-red">ANOMALY DETECTION</span>
        <span class="hero-badge badge-green">REAL-TIME ANALYSIS</span>
      </div>
    </div>
    <div style="margin-left:auto; text-align:right;">
      <div class="status-bar" style="justify-content:flex-end;">
        <span class="status-dot"></span>
        <span style="font-family:JetBrains Mono; font-size:0.75rem; color:#34d399;">SYSTEM ONLINE</span>
      </div>
      <div style="font-family:JetBrains Mono; font-size:0.7rem; color:#334155; margin-top:0.3rem;">
        Isolation Forest + One-Class SVM
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------------------------------
col_up, col_cb = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        label="📂 Drop your log file here or click to browse",
        type=['log', 'txt', 'csv', 'json', 'jsonl'],
        help="Supports Apache/Nginx .log, syslog .txt, CSV, JSON-lines",
    )
with col_cb:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    use_sample = st.checkbox("🧪 Use sample data", value=not bool(uploaded_file))

# ---------------------------------------------------------------------------
# SAMPLE LOG DATA
# ---------------------------------------------------------------------------
SAMPLE_LOG = """\
2024-01-15 08:01:02 [INFO] auth_service: User admin logged in successfully
2024-01-15 08:02:14 [INFO] api_gateway: GET /api/v1/users 200 45ms 1.2KB
2024-01-15 08:03:55 [INFO] api_gateway: GET /api/v1/products 200 38ms 4.5KB
2024-01-15 08:04:10 [WARNING] db_service: Slow query detected (312ms) on users table
2024-01-15 08:05:00 [INFO] api_gateway: POST /api/v1/orders 201 52ms 0.8KB
2024-01-15 08:06:33 [ERROR] payment_service: Connection refused to payment-gateway.prod:443 after 3 retries
2024-01-15 08:07:45 [CRITICAL] db_service: Connection pool exhausted — all 100 connections in use
2024-01-15 08:08:12 [ERROR] api_gateway: GET /api/v1/checkout 500 4823ms 0.1KB — upstream timeout
2024-01-15 08:09:01 [INFO] auth_service: User guest_7821 logged in successfully
2024-01-15 08:10:18 [INFO] api_gateway: GET /api/v1/products 200 41ms 4.5KB
2024-01-15 08:11:00 [INFO] api_gateway: GET /api/v1/categories 200 29ms 2.1KB
2024-01-15 08:12:05 [WARNING] cache_service: Cache miss rate elevated: 78% (threshold: 30%)
2024-01-15 08:13:44 [INFO] api_gateway: GET /api/v1/users/profile 200 35ms 1.0KB
2024-01-15 08:14:22 [ERROR] auth_service: Failed login attempt for user root from IP 203.0.113.42
2024-01-15 08:14:23 [ERROR] auth_service: Failed login attempt for user root from IP 203.0.113.42
2024-01-15 08:14:24 [ERROR] auth_service: Failed login attempt for user root from IP 203.0.113.42
2024-01-15 08:14:25 [ERROR] auth_service: Failed login attempt for user root from IP 203.0.113.42
2024-01-15 08:14:26 [CRITICAL] auth_service: Brute-force attack detected from IP 203.0.113.42 — account locked
2024-01-15 08:15:00 [INFO] api_gateway: GET /api/v1/health 200 5ms 0.1KB
2024-01-15 08:16:30 [INFO] api_gateway: POST /api/v1/users 201 61ms 0.9KB
2024-01-15 08:17:55 [ERROR] storage_service: Disk usage at 97% on /dev/sda1 — write operations failing
2024-01-15 08:18:10 [FATAL] storage_service: Filesystem /dev/sda1 is read-only due to I/O errors
2024-01-15 08:19:00 [INFO] api_gateway: GET /api/v1/orders 200 44ms 3.2KB
2024-01-15 08:20:11 [INFO] api_gateway: GET /api/v1/products/42 200 31ms 0.7KB
2024-01-15 08:21:08 [WARNING] api_gateway: Rate limit approaching for client 10.0.0.7 (950/1000 rpm)
2024-01-15 08:22:00 [INFO] scheduler: Nightly backup job started
2024-01-15 08:23:30 [ERROR] scheduler: Backup job FAILED — insufficient disk space
2024-01-15 08:24:45 [INFO] api_gateway: GET /api/v1/users 200 40ms 1.2KB
2024-01-15 08:25:10 [INFO] auth_service: User alice logged out
2024-01-15 08:26:00 [CRITICAL] memory_monitor: OOM killer invoked — process api-worker-3 (PID 9821) terminated
2024-01-15 08:27:15 [INFO] api_gateway: GET /api/v1/products 200 43ms 4.5KB
2024-01-15 08:28:02 [INFO] api_gateway: POST /api/v1/sessions 201 55ms 0.5KB
2024-01-15 08:29:44 [ERROR] network_service: Packet loss 34% on interface eth0 — possible network degradation
2024-01-15 08:30:00 [INFO] api_gateway: GET /api/v1/health 200 5ms 0.1KB
2024-01-15 08:31:30 [INFO] api_gateway: GET /api/v1/orders/history 200 68ms 8.1KB
2024-01-15 08:32:11 [WARNING] db_service: Replication lag 8.3 seconds — replica falling behind
2024-01-15 08:33:00 [INFO] api_gateway: GET /api/v1/categories 200 28ms 2.1KB
2024-01-15 08:34:05 [ERROR] api_gateway: POST /api/v1/payments 503 12045ms — service unavailable
2024-01-15 08:35:00 [INFO] api_gateway: GET /api/v1/health 200 5ms 0.1KB
2024-01-15 08:36:42 [CRITICAL] api_gateway: Circuit breaker OPEN for payment-service after 10 consecutive failures
"""

@st.cache_data(show_spinner=False)
def run_pipeline(content: str, filename: str, fpr: float, contamination: float):
    df      = ingest_logs(content, filename)
    X, feat = extract_features(df)
    labels, scores, threshold, _ = fit_and_score(X, fpr_target=fpr, contamination=contamination)
    metrics = evaluate_with_synthetic_ground_truth(df, labels, scores)
    df['anomaly_score'] = np.round(scores, 4)
    df['is_anomaly']    = labels
    return df, feat, scores, threshold, metrics

# ---- resolve content ----
content, filename = None, "sample.log"
if uploaded_file is not None:
    content  = uploaded_file.read().decode('utf-8', errors='replace')
    filename = uploaded_file.name
elif use_sample:
    content  = SAMPLE_LOG
    filename = "sample.log"

# ---------------------------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------------------------
if content:
    with st.spinner("🤖 Running SentinelAI detection pipeline…"):
        try:
            df, feat_df, scores, threshold, metrics = run_pipeline(
                content, filename, fpr_target, contamination
            )
        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.stop()

    anomaly_df = df[df['is_anomaly'] == 1]
    normal_df  = df[df['is_anomaly'] == 0]
    df['ts_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # ── SECTION: KPI Cards ─────────────────────────────────────────────────
    st.markdown("""
    <div class="section-hdr">
      <span class="section-hdr-icon">📊</span>
      <span class="section-hdr-text">Detection Summary</span>
      <div class="section-hdr-line"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    pct = round(100 * len(anomaly_df) / len(df), 1) if len(df) else 0
    acc = metrics.get('estimated_accuracy', 0)
    fpr_pct = metrics.get('fpr', 0)

    with c1:
        st.markdown(f"""
        <div class="kpi-card kpi-blue">
          <span class="kpi-icon">📋</span>
          <div class="kpi-label">Total Log Lines</div>
          <div class="kpi-value">{len(df):,}</div>
          <div class="kpi-sub">from {filename}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card kpi-red">
          <span class="kpi-icon">⚠️</span>
          <div class="kpi-label">Anomalies Detected</div>
          <div class="kpi-value">{len(anomaly_df):,}</div>
          <div class="kpi-sub">{pct}% of total logs flagged</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card kpi-green">
          <span class="kpi-icon">🎯</span>
          <div class="kpi-label">Est. Detection Accuracy</div>
          <div class="kpi-value">{acc}%</div>
          <div class="kpi-sub">vs. heuristic ground truth</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card kpi-amber">
          <span class="kpi-icon">🔎</span>
          <div class="kpi-label">Est. False Positive Rate</div>
          <div class="kpi-value">{fpr_pct}%</div>
          <div class="kpi-sub">Target ≤ {int(fpr_target*100)}%</div>
        </div>""", unsafe_allow_html=True)

    # ── Detailed Metrics Expander ──────────────────────────────────────────
    with st.expander("📐 Detailed Evaluation Metrics (vs Heuristic Ground Truth)"):
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Precision",       f"{metrics.get('precision',0)}%")
        m2.metric("Recall",          f"{metrics.get('recall',0)}%")
        m3.metric("F1-Score",        f"{metrics.get('f1',0)}%")
        m4.metric("True Positives",  metrics.get('TP', '-'))
        m5.metric("False Positives", metrics.get('FP', '-'))
        st.caption(
            "ℹ️ Ground truth is generated heuristically from rule-based labels "
            "(error level / HTTP 5xx / error keywords / p95 response time). "
            "Metrics are conservative lower bounds on real-world performance."
        )

    # ══════════════════════════════════════════════════════════════════════
    # CHART ROW 1 — Time Series + Score Gauge
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-hdr">
      <span class="section-hdr-icon">📈</span>
      <span class="section-hdr-text">Log Volume &amp; Anomaly Timeline</span>
      <div class="section-hdr-line"></div>
    </div>
    """, unsafe_allow_html=True)

    ts_available = df['ts_parsed'].notna().sum() > 0
    if ts_available:
        df['minute'] = df['ts_parsed'].dt.floor('1min')
        vol      = df.groupby('minute').size().reset_index(name='count')
        anom_vol = df[df['is_anomaly'] == 1].groupby('minute').size().reset_index(name='anom_count')

        fig_ts = go.Figure()

        # Gradient area — total log volume
        fig_ts.add_trace(go.Scatter(
            x=vol['minute'], y=vol['count'],
            fill='tozeroy',
            fillcolor='rgba(56,189,248,0.08)',
            line=dict(color='#38bdf8', width=2.5, shape='spline'),
            name='Log Volume',
            mode='lines',
            hovertemplate='<b>%{x}</b><br>Volume: <b>%{y}</b> lines<extra>Normal</extra>',
        ))

        # Anomaly bars
        if not anom_vol.empty:
            fig_ts.add_trace(go.Bar(
                x=anom_vol['minute'], y=anom_vol['anom_count'],
                marker=dict(
                    color='rgba(248,113,113,0.6)',
                    line=dict(color='#f87171', width=1.5),
                    cornerradius=4,
                ),
                name='Anomalies',
                hovertemplate='<b>%{x}</b><br>Anomalies: <b>%{y}</b><extra>⚠️ Anomaly</extra>',
                width=30_000,
            ))

        # Anomaly scatter markers
        anom_ts = df[df['is_anomaly'] == 1].dropna(subset=['ts_parsed'])
        if not anom_ts.empty:
            minute_cnt = vol.set_index('minute')['count']
            anom_ts = anom_ts.copy()
            anom_ts['minute'] = anom_ts['ts_parsed'].dt.floor('1min')
            anom_ts_y = anom_ts['minute'].map(minute_cnt).fillna(1)
            fig_ts.add_trace(go.Scatter(
                x=anom_ts['ts_parsed'], y=anom_ts_y,
                mode='markers',
                marker=dict(
                    color='#f87171', size=12, symbol='x-thin',
                    line=dict(width=3, color='#ffffff'),
                ),
                name='⚠️ Anomaly Events',
                hovertemplate=(
                    '<b>⚠️ ANOMALY DETECTED</b><br>'
                    'Time: %{x}<br>'
                    'Score: %{customdata:.4f}<br>'
                    '<extra></extra>'
                ),
                customdata=anom_ts['anomaly_score'],
            ))

        fig_ts.update_layout(
            **{k: v for k, v in PLOT_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
            height=340,
            hovermode='x unified',
            xaxis=dict(**PLOT_LAYOUT['xaxis'], title='Time', tickfont=TICK_STYLE),
            yaxis=dict(**PLOT_LAYOUT['yaxis'], title='Log Count', tickfont=TICK_STYLE),
            legend=dict(orientation='h', y=1.08, x=0, **LEGEND_STYLE),
        )
        st.plotly_chart(fig_ts, use_container_width=True, config={'displayModeBar': True})
    else:
        st.info("⚠️ No parseable timestamps — time-series unavailable.")

    # ══════════════════════════════════════════════════════════════════════
    # CHART ROW 2 — Score Distribution + Severity Pie
    # ══════════════════════════════════════════════════════════════════════
    if show_score_dist:
        st.markdown("""
        <div class="section-hdr">
          <span class="section-hdr-icon">🎯</span>
          <span class="section-hdr-text">Anomaly Score Distribution &amp; Level Breakdown</span>
          <div class="section-hdr-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_dist, col_pie = st.columns([3, 2])

        with col_dist:
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=scores[df['is_anomaly'] == 0],
                nbinsx=40, name='✅ Normal',
                marker=dict(color='rgba(56,189,248,0.55)', line=dict(color='#38bdf8', width=0.5)),
                hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra>Normal</extra>',
            ))
            fig_dist.add_trace(go.Histogram(
                x=scores[df['is_anomaly'] == 1],
                nbinsx=20, name='⚠️ Anomaly',
                marker=dict(color='rgba(248,113,113,0.65)', line=dict(color='#f87171', width=0.5)),
                hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra>⚠️ Anomaly</extra>',
            ))
            fig_dist.add_vline(
                x=threshold, line_dash='dash', line_color='#fbbf24', line_width=2.5,
                annotation=dict(
                    text=f"⚡ Threshold ({threshold:.3f})",
                    font=dict(color='#fbbf24', size=11, family='JetBrains Mono'),
                    bgcolor='rgba(15,23,42,0.8)',
                    bordercolor='rgba(251,191,36,0.4)',
                    borderwidth=1,
                    borderpad=4,
                ),
            )
            fig_dist.update_layout(
                **{k: v for k, v in PLOT_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
                barmode='overlay',
                height=300,
                xaxis=dict(**PLOT_LAYOUT['xaxis'], title='Ensemble Anomaly Score', tickfont=TICK_STYLE),
                yaxis=dict(**PLOT_LAYOUT['yaxis'], title='Count', tickfont=TICK_STYLE),
                legend=dict(orientation='h', y=1.08, **LEGEND_STYLE),
                title=dict(text='Score Distribution', font=dict(family='Orbitron', size=13, color='#64748b'), x=0),
            )
            st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': True})

        with col_pie:
            level_counts = df['level'].value_counts()
            LEVEL_COLORS = {
                'INFO': '#38bdf8', 'DEBUG': '#475569', 'NOTICE': '#818cf8',
                'WARNING': '#fbbf24', 'WARN': '#fbbf24',
                'ERROR': '#f87171', 'CRITICAL': '#ef4444', 'FATAL': '#dc2626',
                'EMERG': '#dc2626', 'ALERT': '#f97316',
            }
            pie_colors = [LEVEL_COLORS.get(l, '#94a3b8') for l in level_counts.index]
            fig_pie = go.Figure(go.Pie(
                labels=level_counts.index,
                values=level_counts.values,
                hole=0.55,
                marker=dict(colors=pie_colors, line=dict(color='#060912', width=2)),
                textfont=dict(family='JetBrains Mono', size=11),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
                pull=[0.05 if l in ('ERROR', 'CRITICAL', 'FATAL') else 0 for l in level_counts.index],
            ))
            fig_pie.update_layout(
                **{k: v for k, v in PLOT_LAYOUT.items() if k not in ('xaxis', 'yaxis', 'margin')},
                height=300,
                title=dict(text='Log Level Breakdown', font=dict(family='Orbitron', size=13, color='#64748b'), x=0),
                annotations=[dict(
                    text=f"<b style='font-size:14px'>{len(df)}</b><br>logs",
                    x=0.5, y=0.5, font=dict(size=14, color='#94a3b8', family='Orbitron'),
                    showarrow=False,
                )],
                showlegend=True,
                legend=dict(orientation='v', x=1.0, y=0.5, **LEGEND_STYLE),
                margin=dict(l=0, r=80, t=36, b=8),
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': True})

    # ══════════════════════════════════════════════════════════════════════
    # CHART ROW 3 — Radar Chart + Top Anomalies Horizontal Bar
    # ══════════════════════════════════════════════════════════════════════
    if show_radar and not feat_df.empty:
        st.markdown("""
        <div class="section-hdr">
          <span class="section-hdr-icon">🕸️</span>
          <span class="section-hdr-text">Feature Profile Radar — Anomaly vs Normal</span>
          <div class="section-hdr-line"></div>
        </div>
        """, unsafe_allow_html=True)

        col_radar, col_bar = st.columns([2, 3])

        with col_radar:
            radar_features = ['severity_score', 'http_status_sev', 'response_size',
                              'response_time', 'hour_of_day', 'error_keyword']
            radar_labels = ['Severity', 'HTTP Sev', 'Resp Size', 'Resp Time', 'Hour', 'Error KW']

            feat_cols = [c for c in radar_features if c in feat_df.columns]
            radar_labels_used = [radar_labels[radar_features.index(c)] for c in feat_cols]

            # Normalise to 0-1 for each feature
            norm_feat = feat_df[feat_cols].copy()
            for col in feat_cols:
                lo, hi = norm_feat[col].min(), norm_feat[col].max()
                if hi > lo:
                    norm_feat[col] = (norm_feat[col] - lo) / (hi - lo)
                else:
                    norm_feat[col] = 0.0

            anomaly_mask = df['is_anomaly'].values == 1
            anom_mean   = norm_feat[anomaly_mask].mean().tolist() if anomaly_mask.any() else [0]*len(feat_cols)
            normal_mean = norm_feat[~anomaly_mask].mean().tolist()
            # Close the radar loop
            anom_r   = anom_mean   + [anom_mean[0]]
            normal_r = normal_mean + [normal_mean[0]]
            theta  = radar_labels_used + [radar_labels_used[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=anom_r, theta=theta, name='⚠️ Anomaly',
                fill='toself',
                fillcolor='rgba(248,113,113,0.15)',
                line=dict(color='#f87171', width=2),
                hovertemplate='<b>%{theta}</b><br>Level: %{r:.3f}<extra>Anomaly</extra>',
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=normal_r, theta=theta, name='✅ Normal',
                fill='toself',
                fillcolor='rgba(56,189,248,0.12)',
                line=dict(color='#38bdf8', width=2),
                hovertemplate='<b>%{theta}</b><br>Level: %{r:.3f}<extra>Normal</extra>',
            ))
            fig_radar.update_layout(
                **{k: v for k, v in PLOT_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
                height=320,
                legend=dict(orientation='h', y=-0.1, **LEGEND_STYLE),
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(
                        visible=True, range=[0, 1],
                        tickfont=dict(family='JetBrains Mono', size=9, color='#475569'),
                        gridcolor='rgba(255,255,255,0.06)',
                        linecolor='rgba(255,255,255,0.06)',
                    ),
                    angularaxis=dict(
                        tickfont=dict(family='JetBrains Mono', size=10, color='#94a3b8'),
                        gridcolor='rgba(255,255,255,0.06)',
                        linecolor='rgba(255,255,255,0.06)',
                    ),
                ),
                title=dict(text='Mean Feature Profiles', font=dict(family='Orbitron', size=12, color='#64748b'), x=0.0),
            )
            st.plotly_chart(fig_radar, use_container_width=True, config={'displayModeBar': True})

        with col_bar:
            # Top anomalous entries — horizontal bar chart
            top_anom = anomaly_df.nlargest(min(12, len(anomaly_df)), 'anomaly_score').copy()
            if not top_anom.empty:
                top_anom['label'] = (
                    'L' + top_anom['line_no'].astype(str) + ' · ' +
                    top_anom['level'].astype(str) + ' · ' +
                    top_anom['message'].astype(str).str[:40]
                )
                score_colors = [
                    f'rgba(248,113,113,{0.4 + 0.6 * (s - top_anom["anomaly_score"].min()) / max(top_anom["anomaly_score"].max() - top_anom["anomaly_score"].min(), 1e-6)})'
                    for s in top_anom['anomaly_score']
                ]
                fig_bar = go.Figure(go.Bar(
                    x=top_anom['anomaly_score'],
                    y=top_anom['label'],
                    orientation='h',
                    marker=dict(color=score_colors, line=dict(color='rgba(248,113,113,0.5)', width=0.5)),
                    hovertemplate=(
                        '<b>Line %{customdata[0]}</b><br>'
                        'Level: %{customdata[1]}<br>'
                        'Score: %{x:.4f}<br>'
                        'Msg: %{customdata[2]}<extra>⚠️ Anomaly</extra>'
                    ),
                    customdata=top_anom[['line_no', 'level', 'message']].values,
                    text=[f'{s:.3f}' for s in top_anom['anomaly_score']],
                    textposition='outside',
                    textfont=dict(family='JetBrains Mono', size=10, color='#f87171'),
                ))
                fig_bar.update_layout(
                    **{k: v for k, v in PLOT_LAYOUT.items() if k not in ('xaxis', 'yaxis', 'margin')},
                    height=320,
                    xaxis=dict(**PLOT_LAYOUT['xaxis'], title='Anomaly Score', range=[0, 1.05], tickfont=TICK_STYLE),
                    yaxis=dict(
                        **PLOT_LAYOUT['yaxis'],
                        automargin=True,
                        tickfont=dict(family='JetBrains Mono', size=9),
                    ),
                    title=dict(text='Top Anomalous Log Lines (by Score)', font=dict(family='Orbitron', size=12, color='#64748b'), x=0.0),
                    margin=dict(l=8, r=60, t=36, b=8),
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': True})

    # ══════════════════════════════════════════════════════════════════════
    # CHART ROW 4 — Severity Heatmap over time
    # ══════════════════════════════════════════════════════════════════════
    if show_heatmap and ts_available:
        st.markdown("""
        <div class="section-hdr">
          <span class="section-hdr-icon">🌡️</span>
          <span class="section-hdr-text">Log Severity Heatmap Over Time</span>
          <div class="section-hdr-line"></div>
        </div>
        """, unsafe_allow_html=True)

        df_heat = df.dropna(subset=['ts_parsed']).copy()
        df_heat['minute_str'] = df_heat['ts_parsed'].dt.strftime('%H:%M')
        from data_processing import LEVEL_MAP
        df_heat['sev_num'] = df_heat['level'].map(lambda l: LEVEL_MAP.get(str(l).upper(), 1))

        hmap_pivot = df_heat.groupby(['minute_str', 'level'])['sev_num'].count().unstack(fill_value=0)
        level_order = [l for l in ['DEBUG','INFO','NOTICE','WARNING','WARN','ERROR','CRITICAL','FATAL'] if l in hmap_pivot.columns]
        hmap_pivot = hmap_pivot.reindex(columns=level_order, fill_value=0)

        color_scales = {
            'DEBUG': [[0,'#060912'],[1,'#475569']],
            'INFO':  [[0,'#060912'],[1,'#38bdf8']],
            'NOTICE':[[0,'#060912'],[1,'#818cf8']],
            'WARNING':[[0,'#060912'],[1,'#fbbf24']],
            'WARN':  [[0,'#060912'],[1,'#fbbf24']],
            'ERROR': [[0,'#060912'],[1,'#f87171']],
            'CRITICAL':[[0,'#060912'],[1,'#ef4444']],
            'FATAL': [[0,'#060912'],[1,'#dc2626']],
        }
        default_cs = [[0,'#060912'],[1,'#94a3b8']]

        fig_heat = make_subplots(
            rows=1, cols=len(level_order),
            shared_yaxes=True,
            subplot_titles=level_order,
            horizontal_spacing=0.01,
        )
        for i, lvl in enumerate(level_order, 1):
            z_vals = hmap_pivot[lvl].values.reshape(-1, 1)
            fig_heat.add_trace(
                go.Heatmap(
                    z=z_vals,
                    y=hmap_pivot.index.tolist(),
                    x=[lvl],
                    name=lvl,
                    colorscale=color_scales.get(lvl, default_cs),
                    showscale=(i == len(level_order)),
                    hovertemplate=f'Time: %{{y}}<br>{lvl}: %{{z}} logs<extra></extra>',
                    xgap=2, ygap=1,
                ),
                row=1, col=i,
            )
        fig_heat.update_layout(
            **{k: v for k, v in PLOT_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
            height=350,
            title=dict(text='Log Level Distribution by Minute', font=dict(family='Orbitron', size=12, color='#64748b'), x=0),
        )
        fig_heat.update_annotations(font=dict(family='JetBrains Mono', size=10, color='#64748b'))
        st.plotly_chart(fig_heat, use_container_width=True, config={'displayModeBar': True})

    # ══════════════════════════════════════════════════════════════════════
    # ANNOTATED LOG TABLE
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-hdr">
      <span class="section-hdr-icon">📋</span>
      <span class="section-hdr-text">Annotated Log Table</span>
      <div class="section-hdr-line"></div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚠️ Anomalies Only", "✅ Normal Logs", "📄 All Logs"])

    display_cols = ['line_no', 'timestamp', 'level', 'http_status', 'anomaly_score', 'message']
    display_cols = [c for c in display_cols if c in df.columns]

    def _style_table(sub_df: pd.DataFrame, highlight_red: bool = False):
        rename_map = {
            'line_no': 'Line', 'timestamp': 'Timestamp', 'level': 'Level',
            'http_status': 'Status', 'anomaly_score': 'Score', 'message': 'Message',
        }
        sub = sub_df[display_cols].rename(columns=rename_map).copy()

        def colour_level(val):
            c = {
                'ERROR': 'color:#f87171;font-weight:600',
                'CRITICAL': 'color:#ef4444;font-weight:700',
                'FATAL': 'color:#dc2626;font-weight:700',
                'WARNING': 'color:#fbbf24;font-weight:600',
                'WARN': 'color:#fbbf24;font-weight:600',
                'INFO': 'color:#38bdf8',
                'DEBUG': 'color:#64748b',
            }
            return c.get(str(val).upper(), '')

        def colour_score(val):
            try:
                v = float(val)
                if v >= 0.75: return 'color:#f87171;font-weight:700'
                if v >= 0.5:  return 'color:#fbbf24'
                return 'color:#34d399'
            except Exception: return ''

        def row_bg(row):
            base = 'background-color:rgba(248,113,113,0.08);' if highlight_red else ''
            return [base] * len(sub.columns)

        return (
            sub.style
            .map(colour_level, subset=['Level'] if 'Level' in sub.columns else [])
            .map(colour_score, subset=['Score']  if 'Score'  in sub.columns else [])
            .apply(row_bg, axis=1)
        )

    with tab1:
        if anomaly_df.empty:
            st.success("✅ No anomalies detected at current threshold. Try increasing the FPR Target.")
        else:
            st.markdown(f"**{len(anomaly_df)} anomalous log lines** sorted by score ↓")
            st.dataframe(
                _style_table(anomaly_df.sort_values('anomaly_score', ascending=False), highlight_red=True),
                use_container_width=True, height=400, hide_index=True,
            )
    with tab2:
        st.markdown(f"**{len(normal_df)} normal log lines**")
        st.dataframe(
            _style_table(normal_df, highlight_red=False),
            use_container_width=True, height=400, hide_index=True,
        )
    with tab3:
        st.markdown(f"**All {len(df)} log lines** — ⚠️ anomalies · ✅ normal")
        st.dataframe(
            _style_table(df.sort_values('is_anomaly', ascending=False), highlight_red=False),
            use_container_width=True, height=450, hide_index=True,
        )

    # ── Raw Feature Table ─────────────────────────────────────────────────
    if show_raw_table:
        st.markdown("""
        <div class="section-hdr">
          <span class="section-hdr-icon">🔬</span>
          <span class="section-hdr-text">Raw Feature Matrix (Debug View)</span>
          <div class="section-hdr-line"></div>
        </div>
        """, unsafe_allow_html=True)
        st.dataframe(feat_df.round(4), use_container_width=True, height=300)

    # ══════════════════════════════════════════════════════════════════════
    # AI SECURITY ANALYST — Gemini 1.5 Flash per-anomaly explanations
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div class="section-hdr">
      <span class="section-hdr-icon">🤖</span>
      <span class="section-hdr-text">AI Security Analyst</span>
      <span class="ai-powered-badge"><span class="gemini-dot"></span>GEMINI 1.5 FLASH</span>
      <div class="section-hdr-line"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Analyst intro banner ────────────────────────────────────────────
    st.markdown("""
    <div class="ai-analyst-banner">
      <div class="ai-analyst-title">🧠 AI Security Analyst</div>
      <div class="ai-analyst-subtitle">
        Powered by Gemini 1.5 Flash · Senior SOC Analyst persona ·
        Results cached per log line to preserve API quota
      </div>
    </div>
    """, unsafe_allow_html=True)

    if anomaly_df.empty:
        st.info("✅ No anomalies to analyse — AI Analyst is on standby.")
    else:
        # Sidebar control injected here — how many anomalies to explain
        ai_col1, ai_col2 = st.columns([3, 1])
        with ai_col1:
            st.markdown(
                f"Select anomalies below and click **▶ Get AI Insight** to receive a "
                f"Gemini-powered threat analysis. Results are cached for 1 hour."
            )
        with ai_col2:
            max_explain = st.number_input(
                "Max anomalies to explain",
                min_value=1, max_value=min(20, len(anomaly_df)),
                value=min(5, len(anomaly_df)),
                step=1,
                key="ai_max_explain",
            )

        # Work on the top-N anomalies by score
        top_anomalies = anomaly_df.sort_values('anomaly_score', ascending=False).head(int(max_explain))

        # ── Expandable per-row AI analysis ────────────────────────────────
        for _, row in top_anomalies.iterrows():
            line_no   = int(row.get('line_no', 0))
            level     = str(row.get('level', ''))
            score     = float(row.get('anomaly_score', 0))
            message   = str(row.get('message', ''))
            timestamp = str(row.get('timestamp', ''))
            raw_line  = str(row.get('raw', message))  # use raw if present

            expander_label = (
                f"⚠️  Line {line_no:>4}  |  [{level}]  |  Score {score:.4f}  "
                f"|  {message[:70]}{'…' if len(message) > 70 else ''}"
            )

            with st.expander(expander_label, expanded=False):
                # Show the raw log line
                st.markdown(
                    f"<div class='ai-insight-log-ref'>📄 {timestamp} · Line {line_no} · {level}</div>",
                    unsafe_allow_html=True,
                )
                st.code(raw_line, language="", line_numbers=False)

                # Trigger analysis with spinner
                btn_key = f"ai_btn_{line_no}_{score}"
                if st.button("▶ Get AI Insight", key=btn_key, type="secondary"):
                    with st.spinner("🤖 Consulting Gemini 1.5 Flash…"):
                        ai_result = explain_anomaly(raw_line)

                    st.markdown(
                        f"""
                        <div class="ai-insight-card">
                          <span class="ai-insight-score-badge">SCORE {score:.4f}</span>
                          {ai_result}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # ── Batch Explain All Button ───────────────────────────────────────
        st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
        if st.button(
            f"⚡ Batch Analyse All {int(max_explain)} Anomalies",
            key="ai_batch_btn",
            type="primary",
        ):
            batch_results = []
            progress = st.progress(0, text="🤖 Analysing anomalies with Gemini…")
            total = len(top_anomalies)

            for idx, (_, row) in enumerate(top_anomalies.iterrows()):
                line_no = int(row.get('line_no', 0))
                score   = float(row.get('anomaly_score', 0))
                message = str(row.get('message', ''))
                raw_line = str(row.get('raw', message))

                with st.spinner(f"Analysing line {line_no}…"):
                    ai_result = explain_anomaly(raw_line)

                batch_results.append((line_no, score, raw_line[:80], ai_result))
                progress.progress((idx + 1) / total, text=f"🤖 Analysed {idx+1}/{total} anomalies…")

            progress.empty()

            st.success(f"✅ Batch analysis complete — {total} anomalies explained.")
            for line_no, score, preview, result in batch_results:
                st.markdown(
                    f"""
                    <div class="ai-insight-card">
                      <span class="ai-insight-score-badge">SCORE {score:.4f}</span>
                      <div class="ai-insight-log-ref">Line {line_no} · {preview}…</div>
                      {result}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ── Download ───────────────────────────────────────────────────────────
    st.markdown("---")
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download Full Results as CSV",
        data=csv_buf.getvalue().encode('utf-8'),
        file_name="sentinelai_anomaly_results.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# IDLE STATE
# ---------------------------------------------------------------------------
else:
    st.markdown("""
    <div style="
      text-align:center; padding:5rem 2rem;
      background:rgba(255,255,255,0.01);
      border:1px dashed rgba(56,189,248,0.2);
      border-radius:20px; margin-top:2rem;
    ">
      <div style="font-size:5rem; animation: pulse-glow 2.5s ease-in-out infinite;">📂</div>
      <h3 style="
        font-family:'Orbitron',monospace; font-size:1.3rem;
        color:#38bdf8; margin:1.5rem 0 0.5rem 0; letter-spacing:0.05em;
      ">AWAITING LOG DATA</h3>
      <p style="color:#475569; font-family:'JetBrains Mono',monospace; font-size:0.85rem;">
        Upload a log file above or enable sample data to run the detection pipeline
      </p>
      <p style="color:#334155; font-size:0.8rem; margin-top:1rem;">
        Supported formats: .log · .txt · .csv · .json · .jsonl
      </p>
    </div>
    """, unsafe_allow_html=True)

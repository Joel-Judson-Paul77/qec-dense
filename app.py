"""
app.py
Streamlit web UI for QEC Dense — 5 tabs:
  Tab 1: Send Message     — run superdense coding with noise slider
  Tab 2: Noise Analysis   — accuracy vs noise for all 4 messages
  Tab 3: Circuit Viewer   — show quantum circuit diagrams
  Tab 4: Network Demo     — Alice/Bob TCP communication log
  Tab 5: Secure Channel   — E91 QKD key generation and CHSH test

Run:
    streamlit run app.py
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import socket
import sys
import subprocess
import threading
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# Dark quantum chart theme applied globally to all figures
plt.style.use("dark_background")
mpl.rcParams.update({
    "figure.facecolor":  "#0b0f20",
    "axes.facecolor":    "#0b0f20",
    "axes.edgecolor":    "#1a2d4a",
    "axes.labelcolor":   "#7a9ab8",
    "axes.titlecolor":   "#d1e8f5",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.grid":         True,
    "grid.color":        "#1a2d4a",
    "grid.linewidth":    0.7,
    "xtick.color":       "#4a6080",
    "ytick.color":       "#4a6080",
    "text.color":        "#7a9ab8",
    "legend.facecolor":  "#0e1e30",
    "legend.edgecolor":  "#1a2d4a",
    "legend.labelcolor": "#94a3b8",
    "figure.dpi":        110,
})
import numpy as np
import streamlit as st

# Add project root to Python path
sys.path.insert(0, os.path.dirname(__file__))
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from quantum_protocols import (
    full_superdense_circuit,
    full_superdense_with_qec,
    decode_result,
)
from noise_channel import (
    create_noise_model,
    run_noisy_simulation,
    run_accuracy_sweep,
    run_channel_noise_sweep,
    create_channel_noise_model,
)
from visualization import (
    plot_accuracy_comparison,
    plot_histogram_comparison,
    save_circuit_diagram,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Network helpers (used by Tab 4)
# ---------------------------------------------------------------------------

def get_local_ip() -> str:
    """Return this machine's LAN IP using a UDP trick (no data sent)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _check_host(ip: str, port: int = 5000, timeout: float = 0.5):
    """Return (ip, True) if Bob's port open, (ip, False) if host alive, else None."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        if sock.connect_ex((ip, port)) == 0:
            sock.close()
            return (ip, True)   # Bob's port is open
        sock.close()
    except Exception:
        pass
    # Fall back to SMB/RPC ports — these are usually not firewalled between LAN devices
    for probe_port in (445, 139, 135):
        try:
            s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s2.settimeout(0.3)
            if s2.connect_ex((ip, probe_port)) == 0:
                s2.close()
                return (ip, False)
            s2.close()
        except Exception:
            pass
    return None


def _ping_host(ip: str) -> bool:
    """Ping a host using system ping. Works through Windows Firewall (ICMP)."""
    try:
        # Windows: -n 1 (1 packet), -w 200 (200ms timeout)
        result = subprocess.run(
            ["ping", "-n", "1", "-w", "200", ip],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1,
        )
        return result.returncode == 0
    except Exception:
        return False


def _scan_single(ip: str, port: int) -> tuple | None:
    """Check one IP: ping first, then check Bob's port if alive."""
    # First check Bob's port directly (fast path if firewall is open)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        if s.connect_ex((ip, port)) == 0:
            s.close()
            return (ip, True)
        s.close()
    except Exception:
        pass
    # Fall back to ping to detect host even if port is firewalled
    if _ping_host(ip):
        return (ip, False)
    return None


def scan_network(local_ip: str, port: int = 5000) -> list:
    """Scan the local /24 subnet for live devices. Returns [(ip, bob_running), ...]."""
    parts = local_ip.split(".")
    if len(parts) != 4:
        return []
    subnet = ".".join(parts[:3])
    targets = [f"{subnet}.{i}" for i in range(1, 255) if f"{subnet}.{i}" != local_ip]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=80) as pool:
        for res in pool.map(lambda ip: _scan_single(ip, port), targets):
            if res is not None:
                results.append(res)
    results.sort(key=lambda x: (not x[1], x[0]))
    return results


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QEC Dense — Quantum Superdense Communication",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Quantum dark theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
.stApp {
    background-color: #030008;
    background-image: 
        radial-gradient(ellipse at 20% 30%, rgba(0, 242, 254, 0.2) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 70%, rgba(255, 8, 68, 0.2) 0%, transparent 50%),
        linear-gradient(135deg, rgba(10,5,30,0.8) 0%, transparent 100%),
        repeating-linear-gradient(45deg, rgba(0,242,254,0.02) 0px, rgba(0,242,254,0.02) 1px, transparent 1px, transparent 20px),
        repeating-linear-gradient(-45deg, rgba(255,8,68,0.02) 0px, rgba(255,8,68,0.02) 1px, transparent 1px, transparent 20px);
    background-attachment: fixed;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}
/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* ── 3D Rectangular Card Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: none;
    gap: 20px;
    padding: 10px 5px 35px 5px;
    justify-content: center;
}
.stTabs [data-baseweb="tab"] {
    background: linear-gradient(145deg, rgba(20, 26, 45, 0.85) 0%, rgba(6, 9, 20, 0.95) 100%);
    color: #7a9ab8;
    border-radius: 24px;
    padding: 20px;
    height: 160px;
    font-weight: 800;
    font-size: 1.15rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    line-height: 1.4;
    white-space: normal;
    text-align: center;
    border: 1px solid rgba(0, 242, 254, 0.15);
    border-bottom: 8px solid rgba(0, 242, 254, 0.3);
    box-shadow: 0 12px 24px rgba(0,0,0,0.6), inset 0 2px 2px rgba(255,255,255,0.05);
    transition: all 0.35s cubic-bezier(0.25, 0.8, 0.25, 1);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    flex: 1;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(145deg, rgba(0, 242, 254, 0.15) 0%, rgba(255, 8, 68, 0.15) 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(0, 242, 254, 0.6) !important;
    border-bottom: 8px solid #ff0844 !important;
    box-shadow: 0 25px 45px rgba(0,242,254,0.35), inset 0 0 25px rgba(255,8,68,0.2) !important;
    transform: translateY(-12px);
    text-shadow: 0 0 12px rgba(255,255,255,0.6);
}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) { 
    color: #00f2fe; 
    border-color: rgba(0, 242, 254, 0.5);
    transform: translateY(-6px);
    box-shadow: 0 18px 30px rgba(0,242,254,0.2), inset 0 2px 2px rgba(255,255,255,0.1);
}
.stTabs [data-baseweb="tab-panel"] { 
    background: rgba(11, 15, 32, 0.6); 
    border-radius: 20px;
    padding: 2.5rem;
    margin-top: 1rem;
    border: 1px solid rgba(255, 255, 255, 0.05);
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    backdrop-filter: blur(12px);
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #8b0000 0%, #ff003c 60%, #ff4d6d 100%);
    color: #fff;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    font-size: 0.83rem;
    box-shadow: 0 0 22px rgba(0,180,216,0.32), 0 4px 14px rgba(0,0,0,0.45);
    transition: box-shadow 0.25s, transform 0.2s;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 36px rgba(0,180,216,0.62), 0 6px 22px rgba(0,0,0,0.5);
    transform: translateY(-2px);
}
.stButton > button:not([kind="primary"]) {
    background: #0e1e30;
    color: #7a9ab8;
    border: 1px solid #1a2d4a;
    border-radius: 9px;
    font-weight: 600;
    transition: background 0.2s, border-color 0.2s, color 0.2s;
}
.stButton > button:not([kind="primary"]):hover {
    background: #132338;
    color: #e2e8f0;
    border-color: #ff2e63;
    box-shadow: 0 0 12px rgba(255,46,99,0.18);
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #0b0f20 0%, #0e1e30 100%);
    border: 1px solid #1a2d4a;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 0 18px rgba(0,180,216,0.07), 0 4px 22px rgba(0,0,0,0.35);
    transition: box-shadow 0.25s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 0 28px rgba(0,180,216,0.18), 0 6px 28px rgba(0,0,0,0.4);
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    color: #4a6080 !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ff2e63 !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    text-shadow: 0 0 18px rgba(255,46,99,0.45);
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-weight: 700 !important;
    font-size: 0.82rem !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stAlert"][data-baseweb="notification"][kind="positive"],
.stSuccess > div {
    background: rgba(0,255,136,0.06) !important;
    border-left: 4px solid #00ff88 !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="negative"],
.stError > div {
    background: rgba(255,51,102,0.06) !important;
    border-left: 4px solid #ff3366 !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
.stWarning > div {
    background: rgba(251,191,36,0.06) !important;
    border-left: 4px solid #fbbf24 !important;
}
[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
.stInfo > div {
    background: rgba(255,46,99,0.05) !important;
    border-left: 4px solid #ff2e63 !important;
}

/* ── Inputs ── */
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div {
    background: #0e1e30 !important;
    border-color: #1a2d4a !important;
    border-radius: 9px !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
textarea {
    background: #0e1e30 !important;
    border-color: #1a2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 9px !important;
}
/* Slider track and thumb */
[data-testid="stSlider"] [role="slider"] {
    background: #ff2e63 !important;
    box-shadow: 0 0 10px rgba(255,46,99,0.5) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #4a6080 !important; }

/* ── Code blocks ── */
[data-testid="stCode"], .stCode > div, pre {
    background: #080c18 !important;
    border: 1px solid #1a2d4a !important;
    border-radius: 9px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #0b0f20;
    border: 1px solid #1a2d4a !important;
    border-radius: 11px;
}
[data-testid="stExpander"] summary p { color: #7a9ab8 !important; font-weight: 600; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] iframe,
[data-testid="stDataFrame"] { border: 1px solid #1a2d4a; border-radius: 10px; overflow: hidden; }

/* ── Typography ── */
h1, h2, h3 { color: #d1e8f5 !important; }
h2 { border-bottom: 1px solid #1a2d4a; padding-bottom: 0.45rem; margin-bottom: 1rem; }
.stMarkdown p { color: #7a9ab8; }
.stCaption p, [data-testid="stCaptionContainer"] { color: #384f66 !important; }

/* ── Radio ── */
[data-testid="stRadio"] label span { color: #94a3b8 !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #ff003c, #ff2e63) !important;
    border-radius: 4px;
}

/* ── Divider ── */
hr { border-color: #1a2d4a !important; margin: 1.2rem 0; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #06091a; }
::-webkit-scrollbar-thumb { background: #1a2d4a; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #ff2e63; }

/* ── Spinner ── */
[data-testid="stSpinner"] svg { color: #ff2e63 !important; }

/* ── Toggle ── */
[data-testid="stToggle"] { color: #94a3b8; }
/* ── Animations & Enhanced Header ── */
@keyframes gradientFlow {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes floatGlow {
    0%, 100% { transform: translateY(0); text-shadow: 0 0 25px rgba(255,46,99,0.6); }
    50% { transform: translateY(-8px); text-shadow: 0 0 50px rgba(0,242,254,0.8), 0 0 70px rgba(255,8,68,0.6); }
}
@keyframes pulseOrb {
    0%, 100% { transform: scale(1); opacity: 0.3; }
    50% { transform: scale(1.3); opacity: 0.7; }
}
@keyframes spinSlow {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
@keyframes iconFloat {
    0%, 100% { transform: translateY(0) scale(1); }
    50% { transform: translateY(-5px) scale(1.05); }
}

.qec-title {
  margin: 0;
  font-size: 7.5rem;
  font-weight: 900;
  line-height: 1.1;
  letter-spacing: -2.5px;
  position: relative;
  display: inline-block;

  background: linear-gradient(270deg, #00f2fe, #4facfe, #ff0844, #ffb199);
  background-size: 400% 400%;

  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;

  animation: gradientFlow 8s ease infinite, floatGlow 5s ease-in-out infinite;
}

.innovative-icon-container {
    width: 100px;
    height: 100px;
    border-radius: 24px;
    background: rgba(11, 15, 32, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 4rem;
    box-shadow: 0 0 40px rgba(0, 242, 254, 0.3), inset 0 0 25px rgba(255, 8, 68, 0.3);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(12px);
    border: 2px solid rgba(255, 255, 255, 0.05);
}

.innovative-icon-container::before {
    content: '';
    position: absolute;
    width: 150%;
    height: 150%;
    background: conic-gradient(transparent, transparent, transparent, #00f2fe, #ff0844);
    animation: spinSlow 4s linear infinite;
    z-index: 0;
}

.innovative-icon-inner {
    position: absolute;
    inset: 3px;
    background: #0b0f20;
    border-radius: 21px;
    z-index: 1;
}

.innovative-icon {
    position: relative;
    z-index: 2;
    background: linear-gradient(135deg, #00f2fe, #ff0844);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 15px rgba(0,242,254,0.6));
    animation: iconFloat 4s ease-in-out infinite;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="padding: 3rem 0 2.5rem; margin-bottom: 2.5rem; text-align:center; position: relative; overflow: hidden; border-radius: 30px; background: radial-gradient(ellipse at center, rgba(14,30,58,0.6) 0%, rgba(6,9,26,0) 70%); box-shadow: 0 10px 40px rgba(0,0,0,0.5); border: 1px solid rgba(255,255,255,0.03);">
<!-- glow orbs -->
<div style="position:absolute; top: -10%; left: 15%; width: 250px; height: 250px; background: #ff0844; border-radius: 50%; filter: blur(120px); animation: pulseOrb 6s infinite alternate; z-index: 0; opacity: 0.4;"></div>
<div style="position:absolute; bottom: -20%; right: 15%; width: 300px; height: 300px; background: #00f2fe; border-radius: 50%; filter: blur(140px); animation: pulseOrb 7s infinite alternate-reverse; z-index: 0; opacity: 0.3;"></div>
<!-- content -->
<div style="position:relative; z-index:1;">
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; gap:1.5rem; margin-bottom:1rem;">
<div class="innovative-icon-container">
<div class="innovative-icon-inner"></div>
<div style="position:relative; z-index:2; display:flex; align-items:center; justify-content:center; animation:iconFloat 4s ease-in-out infinite;">
<svg viewBox="0 0 100 100" width="75" height="75" style="overflow:visible; filter: drop-shadow(0 0 10px rgba(0,242,254,0.6));">
<defs>
<linearGradient id="atomGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#00f2fe" />
<stop offset="100%" stop-color="#ff0844" />
</linearGradient>
<radialGradient id="nucGrad" cx="30%" cy="30%" r="70%">
<stop offset="0%" stop-color="#a8e6cf" />
<stop offset="100%" stop-color="#10cf8b" />
</radialGradient>
</defs>
<g fill="none" stroke="url(#atomGrad)" stroke-width="5" stroke-linecap="round">
<g style="transform-origin: 50px 50px; animation: spinSlow 15s linear infinite;">
<ellipse cx="50" cy="50" rx="42" ry="14" transform="rotate(0 50 50)" />
<ellipse cx="50" cy="50" rx="42" ry="14" transform="rotate(60 50 50)" />
<ellipse cx="50" cy="50" rx="42" ry="14" transform="rotate(120 50 50)" />
</g>
<path d="M 32,3 Q 50,-10 68,3" />
<path d="M 32,97 Q 50,110 68,97" />
</g>
<circle cx="50" cy="50" r="14" fill="url(#nucGrad)" />
</svg>
</div>
</div>
<div>
<h1 class="qec-title">QEC Dense</h1>
<p style="margin-top:1rem; color:#a0c4de; font-size:1.3rem; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; text-shadow: 0 0 10px rgba(160,196,222,0.3);">
Noise-Resilient Quantum Superdense Communication
</p>
</div>
</div>
<div style="height:2px; background:linear-gradient(90deg, transparent, rgba(0,242,254,0.6), rgba(255,8,68,0.6), transparent); margin:2rem auto; width: 65%;"></div>
<div style="display:flex; flex-wrap:wrap; gap:1.2rem; align-items:center; justify-content:center;">
<span style="background:linear-gradient(90deg, rgba(255,8,68,0.15), rgba(0,242,254,0.15)); border:1px solid rgba(0,242,254,0.3); color:#ffffff; padding:6px 20px; border-radius:30px; font-size:0.85rem; font-weight:800; letter-spacing:1px; text-transform:uppercase; box-shadow: 0 0 15px rgba(0,242,254,0.2);">QC&#178; Quantum Computing Conclave</span>
<span style="background:rgba(168,85,247,0.15); border:1px solid rgba(168,85,247,0.4); color:#f3a8ff; padding:6px 20px; border-radius:30px; font-size:0.85rem; font-weight:800; letter-spacing:0.8px; box-shadow: 0 0 15px rgba(168,85,247,0.2);">SRM University AP</span>
<a href="mailto:support@qec.com" style="text-decoration:none;">
<span style="color:#00f2fe; font-size:0.9rem; font-weight:600; margin-left:8px; padding: 4px 12px; background: rgba(0,242,254,0.1); border: 1px solid rgba(0,242,254,0.3); border-radius: 8px;">
&#9993; Contact Support
</span>
</a>
</div>
</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Transmit  ",
    "  Noise Analysis  ",
    "  Circuit Viewer  ",
    "  Network Demo  ",
    "  Secure Channel (E91)  ",
])


# ---------------------------------------------------------------------------
# Tab 1: Send Message
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <h2 style="
    margin:0 0 0.3rem 0; font-size:1.55rem; font-weight:800;
    background:linear-gradient(90deg,#ff2e63,#7dd3fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    border:none; padding:0;
  ">Superdense Coding Simulation</h2>
  <p style="margin:0; color:#4a6880; font-size:0.9rem;">
    Send a 2-bit classical message using <strong style="color:#ff2e63;">1 qubit</strong> + shared entanglement.
    Toggle error correction and adjust noise to see the effect.
  </p>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])

    with col1:
        message = st.selectbox("2-Bit Message to Send", ["00", "01", "10", "11"], index=2)
        noise_type = st.selectbox("Noise Type", ["bit_flip", "phase_flip", "depolarizing"])
        noise_level = st.slider("Noise Probability", 0.0, 0.5, 0.15, 0.01, format="%.2f")
        use_qec = st.toggle("Enable 3-Qubit QEC", value=False)
        shots = st.select_slider("Shots", options=[256, 512, 1024, 2048], value=1024)
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    with col2:
        if run_btn:
            with st.spinner("Running quantum simulation ..."):
                circuit = full_superdense_with_qec(message) if use_qec else full_superdense_circuit(message)

                if noise_level == 0.0:
                    sim = AerSimulator()
                    compiled = transpile(circuit, sim, optimization_level=0)
                    job = sim.run(compiled, shots=shots)
                    counts = job.result().get_counts()
                else:
                    # Channel-only noise: apply only to id gates (transmission channel)
                    # optimization_level=0 preserves id gates so noise model takes effect
                    data_qubits = [0, 1, 2] if use_qec else [0]
                    nm = create_channel_noise_model(noise_type, noise_level, data_qubits)
                    sim = AerSimulator(noise_model=nm)
                    compiled = transpile(circuit, sim, optimization_level=0)
                    counts = sim.run(compiled, shots=shots).result().get_counts()

                decoded = decode_result(counts, with_qec=use_qec)

            # Results
            match = decoded == message
            total = sum(counts.values())
            msg_of = lambda label: label.split()[-1][:2]
            delivered = sum(c for k, c in counts.items() if msg_of(k) == message) / total
            corrected = (sum(c for k, c in counts.items()
                             if use_qec and msg_of(k) == message and k.split()[0] != "00") / total)
            col_r1, col_r2, col_r3, col_r4 = st.columns(4)
            col_r1.metric("Sent", message)
            col_r2.metric("Decoded", decoded)
            col_r3.metric("Status", "CORRECT" if match else "ERROR", delta=None)
            col_r4.metric("Delivered", f"{delivered:.1%}",
                          help="Share of shots in which Bob recovered the exact message")

            if match:
                st.success(f"Decoded '{decoded}' correctly!")
            else:
                st.error(f"Decoding error: sent '{message}', got '{decoded}'")

            # Histogram, coloured by whether the message survived:
            #   green = right message, no error detected
            #   teal  = right message after the repetition code corrected an error
            #   red   = wrong message
            labels = sorted(counts.keys())
            probs = [counts[k] / total for k in labels]

            def outcome_color(label):
                if msg_of(label) != message:
                    return "#ff003c"
                if use_qec and label.split()[0] != "00":
                    return "#00d4ff"
                return "#00ff88"

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(labels, probs, color=[outcome_color(k) for k in labels],
                   edgecolor="#1a2d4a", alpha=0.95, linewidth=0.8)
            ax.set_xlabel("Measurement outcome" + ("  (syndrome  message)" if use_qec else ""))
            ax.set_ylabel("Probability")
            ax.set_title(
                f"Measurement Histogram — msg='{message}', noise={noise_level:.0%}, "
                f"QEC={'ON' if use_qec else 'OFF'}"
            )
            from matplotlib.patches import Patch
            legend = [Patch(color="#00ff88", label="correct"),
                      Patch(color="#ff003c", label="wrong message")]
            if use_qec:
                legend.insert(1, Patch(color="#00d4ff", label="correct after QEC fix"))
            ax.legend(handles=legend, loc="upper right", frameon=False, labelcolor="#e2e8f0", fontsize=8)
            ax.set_ylim(0, 1.1)
            ax.tick_params(axis="x", rotation=45)
            fig.patch.set_facecolor("#0b0f20")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            caption = f"Bob recovered '{message}' in {delivered:.1%} of {total} shots"
            if use_qec:
                caption += f", including {corrected:.1%} where the repetition code corrected a bit-flip"
            st.caption(caption + ".")

            if use_qec and noise_type != "bit_flip":
                st.warning(
                    f"**Note:** The 3-qubit repetition code is designed to correct **bit-flip (X) errors only**. "
                    f"With '{noise_type.replace('_', ' ')}' noise, QEC may show reduced benefit or even degraded "
                    f"performance because it cannot correct Z (phase) or Y errors. For full error protection, "
                    f"advanced codes like Shor's 9-qubit code would be needed."
                )
        else:
            st.info("Configure settings and click **Run Simulation**.")


# ---------------------------------------------------------------------------
# Cached sweep helper (avoids re-running expensive simulation on tab switch)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_sweep(message: str, noise_type: str, noise_range_tuple: tuple,
                  with_qec: bool, shots: int) -> list:
    return run_channel_noise_sweep(
        message, noise_type, list(noise_range_tuple), with_qec=with_qec, shots=shots
    )


# ---------------------------------------------------------------------------
# Tab 2: Noise Analysis
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <h2 style="
    margin:0 0 0.3rem 0; font-size:1.55rem; font-weight:800;
    background:linear-gradient(90deg,#ff6b6b,#c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    border:none; padding:0;
  ">Accuracy vs Noise Level</h2>
  <p style="margin:0; color:#4a6880; font-size:0.9rem;">
    Compare decoding accuracy <strong style="color:#ff6b6b;">with</strong> and
    <strong style="color:#ff6b6b;">without</strong> 3-qubit QEC as noise increases from 0% to 50%.
  </p>
</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 3])
    with col_a:
        ana_noise_type = st.selectbox("Noise Type", ["bit_flip", "phase_flip", "depolarizing"], key="ana_noise")
        ana_shots = st.select_slider("Shots per point", [256, 512, 1024], value=512, key="ana_shots")
        ana_msg = st.selectbox("Message to analyse", ["00", "01", "10", "11", "all (average)"], index=4)
        run_analysis = st.button("Run Noise Analysis", type="primary", use_container_width=True)

    with col_b:
        if run_analysis:
            noise_range = [round(i * 0.05, 2) for i in range(11)]
            noise_range_t = tuple(noise_range)  # hashable for cache

            with st.spinner("Running accuracy sweep (this takes ~30s) ..."):
                if ana_msg == "all (average)":
                    messages_to_test = ["00", "01", "10", "11"]
                    no_qec_all = {p: [] for p in noise_range}
                    with_qec_all = {p: [] for p in noise_range}
                    progress = st.progress(0)
                    for i, m in enumerate(messages_to_test):
                        for (p, acc) in _cached_sweep(m, ana_noise_type, noise_range_t, False, ana_shots):
                            no_qec_all[p].append(acc)
                        for (p, acc) in _cached_sweep(m, ana_noise_type, noise_range_t, True, ana_shots):
                            with_qec_all[p].append(acc)
                        progress.progress((i + 1) / 4)
                    data_no_qec = [(p, np.mean(v)) for p, v in sorted(no_qec_all.items())]
                    data_with_qec = [(p, np.mean(v)) for p, v in sorted(with_qec_all.items())]
                else:
                    data_no_qec = _cached_sweep(ana_msg, ana_noise_type, noise_range_t, False, ana_shots)
                    data_with_qec = _cached_sweep(ana_msg, ana_noise_type, noise_range_t, True, ana_shots)

            # Plot
            xs = [p * 100 for p, _ in data_no_qec]
            ys_no = [a for _, a in data_no_qec]
            ys_qec = [a for _, a in data_with_qec]

            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(xs, ys_no, color="#ff6b6b", marker="o", linewidth=2.2,
                    label="No QEC", markersize=6, markerfacecolor="#ff3366",
                    markeredgecolor="#ff6b6b")
            ax.plot(xs, ys_qec, color="#ff2e63", marker="s", linewidth=2.2,
                    label="3-Qubit Repetition QEC", markersize=6,
                    markerfacecolor="#00ff88", markeredgecolor="#ff2e63")
            ax.axhline(0.5, color="#4a6080", linestyle="--", alpha=0.6, label="Random guess")
            ax.set_xlabel("Noise Probability (%)", fontsize=12)
            ax.set_ylabel("Decoding Accuracy", fontsize=12)
            ax.set_title(
                f"Accuracy vs {ana_noise_type.replace('_', ' ').title()} Noise",
                fontsize=13, fontweight="bold"
            )
            ax.set_ylim(-0.05, 1.05)
            ax.legend(fancybox=True, framealpha=0.8)
            fig.patch.set_facecolor("#0b0f20")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Table
            df_data = {"Noise (%)": [p * 100 for p, _ in data_no_qec],
                       "No QEC (%)": [round(a * 100, 1) for _, a in data_no_qec],
                       "With QEC (%)": [round(a * 100, 1) for _, a in data_with_qec]}
            import pandas as pd
            st.dataframe(pd.DataFrame(df_data), use_container_width=True)

            if ana_noise_type != "bit_flip":
                st.info(
                    "**Interpretation note:** The 3-qubit repetition code corrects single bit-flip errors only. "
                    "For phase-flip or depolarizing noise, the QEC advantage may be reduced or absent. "
                    "This is a fundamental limitation of the repetition code, not a bug."
                )
        else:
            st.info("Click **Run Noise Analysis** to start the sweep.")


# ---------------------------------------------------------------------------
# Tab 3: Circuit Viewer
# ---------------------------------------------------------------------------
with tab3:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <h2 style="
    margin:0 0 0.3rem 0; font-size:1.55rem; font-weight:800;
    background:linear-gradient(90deg,#34d399,#6ee7b7);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    border:none; padding:0;
  ">Quantum Circuit Diagrams</h2>
  <p style="margin:0; color:#4a6880; font-size:0.9rem;">
    Visualize the <strong style="color:#34d399;">superdense coding circuit</strong> and Bloch sphere state after encoding.
  </p>
</div>
""", unsafe_allow_html=True)

    col_c1, col_c2 = st.columns([1, 3])
    with col_c1:
        circ_msg = st.selectbox("Message", ["00", "01", "10", "11"], index=2, key="circ_msg")
        circ_qec = st.toggle("Show QEC Circuit", value=False, key="circ_qec")
        show_circ = st.button("Show Circuit", type="primary", use_container_width=True)

    with col_c2:
        if show_circ:
            with st.spinner("Drawing circuit ..."):
                circuit = full_superdense_with_qec(circ_msg) if circ_qec else full_superdense_circuit(circ_msg)
                fig = circuit.draw("mpl", style="iqp", fold=40)
                st.pyplot(fig)
                plt.close(fig)

            mode = "with QEC" if circ_qec else "basic"
            st.caption(f"Superdense coding circuit for message '{circ_msg}' — {mode}")

            # Bloch sphere — state of entangled pair after Alice's encoding
            st.markdown("---")
            st.subheader("Qubit State After Encoding")
            from qiskit.visualization import plot_bloch_multivector
            from qiskit.quantum_info import Statevector
            from quantum_protocols import create_bell_pair, alice_encode
            enc_qc = QuantumCircuit(2)
            create_bell_pair(enc_qc, 0, 1)
            alice_encode(enc_qc, 0, circ_msg)
            sv = Statevector.from_instruction(enc_qc)
            fig_bloch = plot_bloch_multivector(sv)
            st.pyplot(fig_bloch)
            plt.close(fig_bloch)
            st.caption(
                "Bloch sphere of the 2-qubit entangled state after Alice encodes her message. "
                "The center-pointing vectors reflect entanglement — neither qubit has a "
                "definite state on its own. Different messages produce different entangled states."
            )

            if circ_qec:
                st.markdown("""
**Circuit explanation (with QEC):**
1. **Bell pair** — H + CNOT creates entanglement between Alice's qubit (q0) and Bob's qubit (q3)
2. **Alice encodes** — applies I/Z/X/ZX to q0 based on the 2-bit message
3. **QEC encode** — CNOT copies q0 to ancilla qubits q1, q2 (3-qubit repetition code protects the encoded state)
4. **Noisy channel** — identity gates on q0, q1, q2 simulate transmission noise
5. **Syndrome measure** — parity checks q0⊕q1 and q1⊕q2 detect single-qubit bit-flip errors
6. **Error correction** — conditional X gates fix the identified qubit
7. **QEC decode** — undo the repetition encoding
8. **Bell measurement** — CNOT + H + measure on (q0, q3) reveals Alice's message
""")
            else:
                st.markdown("""
**Circuit explanation (basic superdense coding):**
1. **Bell pair** — H + CNOT creates |Φ⁺⟩ on (q0, q1)
2. **Alice encodes** — applies I/Z/X/ZX to q0 (sends 2 bits via 1 qubit!)
3. **Bob decodes** — CNOT + H + measure reveals original 2-bit message
""")
        else:
            st.info("Click **Show Circuit** to render the quantum circuit diagram.")


# ---------------------------------------------------------------------------
# Tab 4: Network Demo
# ---------------------------------------------------------------------------
with tab4:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <h2 style="
    margin:0 0 0.3rem 0; font-size:1.55rem; font-weight:800;
    background:linear-gradient(90deg,#f59e0b,#fcd34d);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    border:none; padding:0;
  ">Alice &amp; Bob — Live Network Demo</h2>
  <p style="margin:0; color:#4a6880; font-size:0.9rem;">
    Run on <strong style="color:#f59e0b;">two laptops</strong> on the same WiFi.
    Alice transmits a 2-bit quantum message to Bob over TCP using superdense coding.
  </p>
</div>
""", unsafe_allow_html=True)

    # Shared state initialisation
    if "network_log" not in st.session_state:
        st.session_state.network_log = []
    if "bob_proc" not in st.session_state:
        st.session_state.bob_proc = None
    if "contacts" not in st.session_state:
        st.session_state.contacts = {"Bob (localhost)": "127.0.0.1"}

    _script_dir = os.path.dirname(os.path.abspath(__file__))

    role = st.radio(
        "Select Your Role",
        ["Alice (Sender)", "Bob (Receiver)"],
        horizontal=True,
        key="net_role",
    )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # BOB MODE
    # -----------------------------------------------------------------------
    if role == "Bob (Receiver)":
        local_ip = get_local_ip()
        st.info(f"This laptop's IP address: **{local_ip}** — Share this with Alice!")

        bob_port = st.number_input(
            "Listening Port", value=5000, step=1,
            min_value=1024, max_value=65535, key="bob_port",
        )

        _bob_logfile = os.path.join(_script_dir, "bob_log.txt")

        with st.expander("Fix Firewall (run once if Alice can't connect)"):
            st.markdown("Run this in **CMD as Administrator** on this laptop to allow incoming connections:")
            st.code(
                f'netsh advfirewall firewall add rule name="QEC Dense Bob Server" '
                f'dir=in action=allow protocol=TCP localport=5000',
                language="batch",
            )
            st.caption("Only needed once. After running this, click Start Listening and tell Alice to send.")

        col_b1, col_b2 = st.columns(2)
        start_bob = col_b1.button("Start Listening", type="primary", use_container_width=True)
        stop_bob  = col_b2.button("Stop Server", use_container_width=True)

        if start_bob:
            # Terminate any previous run
            if st.session_state.bob_proc is not None:
                try:
                    st.session_state.bob_proc.terminate()
                except Exception:
                    pass
            # Clear old log file
            with open(_bob_logfile, "w", encoding="utf-8") as _f:
                _f.write("")
            try:
                proc = subprocess.Popen(
                    [sys.executable,
                     os.path.join(_script_dir, "bob_server_logged.py"),
                     "--port", str(int(bob_port)),
                     "--logfile", _bob_logfile],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    cwd=_script_dir,
                )
                st.session_state.bob_proc = proc
                st.session_state.network_log.append(
                    f"[Bob] Listening on {local_ip}:{int(bob_port)} (PID {proc.pid})"
                )
            except Exception as exc:
                st.session_state.network_log.append(f"[System] Failed to start: {exc}")
            time.sleep(0.4)  # let subprocess write initial log entries before rerun
            st.rerun()

        if stop_bob:
            if st.session_state.bob_proc is not None:
                try:
                    st.session_state.bob_proc.terminate()
                except Exception:
                    pass
                st.session_state.bob_proc = None
                st.session_state.network_log.append("[Bob] Server stopped.")
            else:
                st.session_state.network_log.append("[System] No server running.")
            st.rerun()

        # Always read log file on every render — auto-catches results on any interaction
        if os.path.exists(_bob_logfile):
            try:
                with open(_bob_logfile, "r", encoding="utf-8") as _f:
                    _log_lines = _f.readlines()
                _decoded_msg = None
                for _raw in _log_lines:
                    _raw = _raw.strip()
                    if not _raw:
                        continue
                    try:
                        _entry = json.loads(_raw)
                        _display = f"[Bob] {_entry.get('message', '')}"
                        if _display not in st.session_state.network_log:
                            st.session_state.network_log.append(_display)
                        if _entry.get("type") in ("success", "decoded") and "decoded" in _entry:
                            _decoded_msg = _entry["decoded"]
                    except json.JSONDecodeError:
                        continue

                if _decoded_msg:
                    st.success(f"Message received! Bob decoded: **'{_decoded_msg}'**")
                    st.metric("Decoded Message", _decoded_msg)
                    if st.session_state.bob_proc is not None:
                        st.session_state.bob_proc = None
                elif st.session_state.bob_proc is not None:
                    st.warning("Waiting for Alice to connect...")
                    if st.button("Refresh Status", use_container_width=True, key="refresh_bob_waiting"):
                        st.rerun()
            except Exception as _exc:
                st.error(f"Error reading log: {_exc}")
        elif st.session_state.bob_proc is not None:
            st.warning("Waiting for Alice to connect...")
            if st.button("Refresh Status", use_container_width=True, key="refresh_bob_done"):
                st.rerun()
        else:
            st.info("Click **Start Listening** to begin waiting for Alice's message.")

    # -----------------------------------------------------------------------
    # ALICE MODE
    # -----------------------------------------------------------------------
    else:
        # Contacts manager
        with st.expander("Manage Contacts (Bob's address book)"):
            c1, c2, c3 = st.columns([2, 2, 1])
            new_name = c1.text_input("Name", placeholder="e.g. Lokesh's Laptop", key="new_cname")
            new_ip   = c2.text_input("IP Address", placeholder="e.g. 10.1.7.16", key="new_cip")
            c3.write("")
            c3.write("")
            if c3.button("Add", key="add_contact"):
                if new_name and new_ip:
                    st.session_state.contacts[new_name] = new_ip
                    st.success(f"Added {new_name}!")
            for _name, _ip in list(st.session_state.contacts.items()):
                cr, cd = st.columns([4, 1])
                cr.write(f"**{_name}** -> `{_ip}`")
                if _name != "Bob (localhost)":
                    if cd.button("X", key=f"del_{_name}"):
                        del st.session_state.contacts[_name]
                        st.rerun()

            st.markdown("---")
            st.subheader("Network Scanner")
            st.caption(
                "Scan your local WiFi subnet to find devices. "
                "Devices with Bob's server open on port 5000 are highlighted."
            )
            if st.button("Scan Network", use_container_width=True, key="scan_btn"):
                with st.spinner("Scanning local network (10-20 seconds)..."):
                    _found = scan_network(get_local_ip())
                if _found:
                    st.success(f"Found {len(_found)} device(s)!")
                    for _dev_ip, _is_bob in _found:
                        _status = "Bob server detected!" if _is_bob else "Device found"
                        _sc1, _sc2 = st.columns([3, 1])
                        _sc1.write(f"`{_dev_ip}` — {_status}")
                        if _sc2.button("Add", key=f"scan_add_{_dev_ip}"):
                            _auto_name = f"Scanned-{_dev_ip.split('.')[-1]}"
                            st.session_state.contacts[_auto_name] = _dev_ip
                            st.rerun()
                else:
                    st.warning("No devices found. Make sure both laptops are on the same WiFi.")

        selected = st.selectbox(
            "Select Recipient (Bob)",
            list(st.session_state.contacts.keys()),
            key="sel_bob",
        )
        bob_host = st.session_state.contacts[selected]
        st.caption(f"Will connect to: `{bob_host}`")

        col_a1, col_a2 = st.columns(2)
        alice_msg  = col_a1.selectbox("Message", ["00", "01", "10", "11"],
                                      index=2, key="alice_msg")
        alice_port = col_a2.number_input("Port", value=5000, step=1,
                                         min_value=1024, max_value=65535,
                                         key="alice_port")
        eve_on = st.toggle("Simulate Eavesdropper (Eve)", value=False, key="net_eve")

        st.markdown("---")
        st.markdown("**Channel Simulation**")
        net_noise_on = st.toggle("Simulate Noisy Channel", value=False, key="net_noise_on")
        if net_noise_on:
            net_noise_type = st.selectbox(
                "Noise Type",
                ["bit_flip", "phase_flip", "depolarizing"],
                key="net_noise_type",
            )
            net_noise_prob = st.slider(
                "Noise Probability", 0.01, 0.50, 0.15, 0.01,
                format="%.2f", key="net_noise_prob",
            )
            net_qec_on = st.toggle(
                "Enable QEC (3-Qubit Repetition Code)", value=True, key="net_qec_on"
            )
        else:
            net_noise_type = "bit_flip"
            net_noise_prob = 0.0
            net_qec_on = False

        tc1, tc2 = st.columns(2)
        if tc1.button("Test Connection", use_container_width=True, key="test_conn"):
            with st.spinner(f"Pinging {bob_host} ..."):
                _host_alive = _ping_host(bob_host)
            if _host_alive:
                st.success(f"Host {bob_host} is reachable! Click 'Send Message to Bob' to transmit.")
            else:
                st.warning(
                    f"Ping to {bob_host} failed — but **ICMP ping is often blocked on university/office WiFi "
                    f"even when TCP works fine**. If Bob's server is running, go ahead and click "
                    f"**Send Message to Bob** — it may still work."
                )
                st.code(
                    f'netsh advfirewall firewall add rule name="QEC Dense Bob Server" '
                    f'dir=in action=allow protocol=TCP localport={int(alice_port)}',
                    language="batch",
                )
                st.caption("Run this in CMD as Administrator on BOB's laptop if sending also fails.")

        if tc2.button("Send Message to Bob", type="primary", use_container_width=True, key="send_msg"):
            st.session_state.network_log.append(
                f"[Alice] Connecting to {selected} ({bob_host}:{int(alice_port)}) ..."
            )
            with st.spinner(f"Connecting to {bob_host} ..."):
                try:
                    result = subprocess.run(
                        [sys.executable,
                         os.path.join(_script_dir, "alice_client.py"),
                         "--message", alice_msg,
                         "--host", bob_host,
                         "--port", str(int(alice_port))],
                        capture_output=True, text=True, timeout=15, cwd=_script_dir,
                    )
                except subprocess.TimeoutExpired:
                    st.session_state.network_log.append("[System] Timeout — Bob did not respond in 15s.")
                    st.error("Connection timed out. Is Bob's server running?")
                    result = None
                except Exception as exc:
                    st.session_state.network_log.append(f"[System] Error: {exc}")
                    result = None

            if result is not None:
                for line in result.stdout.strip().splitlines():
                    # alice_client.py already prefixes with [Alice], avoid double prefix
                    if line.startswith("[Alice]") or line.startswith("[System]"):
                        st.session_state.network_log.append(line)
                    else:
                        st.session_state.network_log.append(f"[Alice] {line}")

                if result.returncode == 0:
                    st.session_state.network_log.append(
                        f"[System] SUCCESS — '{alice_msg}' transmitted and decoded correctly!"
                    )
                    st.success(f"Bob decoded '{alice_msg}' correctly!")

                    if net_noise_on:
                        from noise_channel import create_channel_noise_model
                        from quantum_protocols import full_superdense_circuit, full_superdense_with_qec, decode_result as _decode_result
                        st.markdown("---")
                        st.subheader("Noisy Channel Simulation")
                        st.caption(
                            "Simulating the same message through a noisy quantum channel — "
                            "comparing with and without QEC error correction."
                        )
                        with st.spinner("Running noisy channel simulation ..."):
                            circ_basic = full_superdense_circuit(alice_msg)
                            if net_noise_prob > 0:
                                nm_basic = create_channel_noise_model(net_noise_type, net_noise_prob, [0])
                                sim_basic = AerSimulator(noise_model=nm_basic)
                                comp_basic = transpile(circ_basic, sim_basic, optimization_level=0)
                                counts_basic = sim_basic.run(comp_basic, shots=1024).result().get_counts()
                            else:
                                _sb = AerSimulator()
                                comp_basic = transpile(circ_basic, _sb)
                                counts_basic = _sb.run(comp_basic, shots=1024).result().get_counts()
                            decoded_basic = _decode_result(counts_basic, with_qec=False)
                            correct_basic = sum(v for k, v in counts_basic.items() if _decode_result({k: v}, with_qec=False) == alice_msg)
                            acc_basic = correct_basic / 1024

                            circ_qec = full_superdense_with_qec(alice_msg)
                            if net_noise_prob > 0:
                                nm_qec = create_channel_noise_model(net_noise_type, net_noise_prob, [0, 1, 2])
                                sim_qec = AerSimulator(noise_model=nm_qec)
                                comp_qec = transpile(circ_qec, sim_qec, optimization_level=0)
                                counts_qec = sim_qec.run(comp_qec, shots=1024).result().get_counts()
                            else:
                                _sq = AerSimulator()
                                comp_qec = transpile(circ_qec, _sq)
                                counts_qec = _sq.run(comp_qec, shots=1024).result().get_counts()
                            decoded_qec = _decode_result(counts_qec, with_qec=True)
                            correct_qec = sum(v for k, v in counts_qec.items() if _decode_result({k: v}, with_qec=True) == alice_msg)
                            acc_qec = correct_qec / 1024

                        col_n1, col_n2 = st.columns(2)
                        with col_n1:
                            st.markdown(
                                f"<div style='text-align:center;padding:1rem;"
                                f"background:rgba(255,107,107,0.1);border:1px solid rgba(255,107,107,0.3);"
                                f"border-radius:12px;'>"
                                f"<div style='color:#ff6b6b;font-size:0.8rem;font-weight:700;"
                                f"text-transform:uppercase;letter-spacing:1px;'>Without QEC</div>"
                                f"<div style='font-size:2rem;font-weight:800;color:#ff3366;"
                                f"margin:0.5rem 0;'>{acc_basic*100:.1f}%</div>"
                                f"<div style='color:#7a9ab8;font-size:0.85rem;'>Decoded: '{decoded_basic}'</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        with col_n2:
                            st.markdown(
                                f"<div style='text-align:center;padding:1rem;"
                                f"background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);"
                                f"border-radius:12px;'>"
                                f"<div style='color:#00ff88;font-size:0.8rem;font-weight:700;"
                                f"text-transform:uppercase;letter-spacing:1px;'>With QEC</div>"
                                f"<div style='font-size:2rem;font-weight:800;color:#00ff88;"
                                f"margin:0.5rem 0;'>{acc_qec*100:.1f}%</div>"
                                f"<div style='color:#7a9ab8;font-size:0.85rem;'>Decoded: '{decoded_qec}'</div>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                        improvement = acc_qec - acc_basic
                        if improvement > 0:
                            st.success(f"QEC improved accuracy by **{improvement*100:.1f}%** at {net_noise_prob*100:.0f}% {net_noise_type.replace('_', ' ')} noise!")
                        elif improvement < 0 and net_noise_type != "bit_flip":
                            st.warning(f"QEC reduced accuracy — expected for {net_noise_type.replace('_', ' ')} noise. The 3-qubit repetition code only corrects bit-flip errors.")

                        fig_net, (ax_net1, ax_net2) = plt.subplots(1, 2, figsize=(10, 4))
                        labels_b = sorted(counts_basic.keys())
                        ax_net1.bar(labels_b, [counts_basic[k]/1024 for k in labels_b], color="#ff6b6b", edgecolor="#1a2d4a", alpha=0.85)
                        ax_net1.set_title("Without QEC", fontsize=12)
                        ax_net1.set_ylabel("Probability")
                        ax_net1.set_ylim(0, 1.1)
                        ax_net1.tick_params(axis="x", rotation=45)
                        labels_q = sorted(counts_qec.keys())
                        ax_net2.bar(labels_q, [counts_qec[k]/1024 for k in labels_q], color="#00ff88", edgecolor="#1a2d4a", alpha=0.85)
                        ax_net2.set_title("With QEC", fontsize=12)
                        ax_net2.set_ylabel("Probability")
                        ax_net2.set_ylim(0, 1.1)
                        ax_net2.tick_params(axis="x", rotation=45)
                        fig_net.suptitle(
                            f"Network Transmission — msg='{alice_msg}', {net_noise_prob*100:.0f}% {net_noise_type.replace('_', ' ')} noise",
                            fontsize=13, fontweight="bold",
                        )
                        fig_net.patch.set_facecolor("#0b0f20")
                        plt.tight_layout()
                        st.pyplot(fig_net)
                        plt.close(fig_net)
                        st.session_state.network_log.append(
                            f"[Noise] {net_noise_type} @ {net_noise_prob*100:.0f}% — "
                            f"No QEC: {acc_basic*100:.1f}% | With QEC: {acc_qec*100:.1f}% (+{improvement*100:.1f}%)"
                        )

                    if eve_on:
                        from e91_qkd import generate_e91_key, detect_eavesdropper
                        st.markdown("---")
                        st.subheader("E91 Eavesdropper Detection")
                        with st.spinner("Running CHSH inequality test ..."):
                            eve_r   = generate_e91_key(num_bits=16, eavesdropper=True)
                            clean_r = generate_e91_key(num_bits=16, eavesdropper=False)
                        s_eve   = eve_r["chsh_s"]
                        s_clean = clean_r["chsh_s"]
                        m1, m2 = st.columns(2)
                        m1.metric("S-value (with Eve)",
                                  f"{s_eve:.4f}",
                                  delta="COMPROMISED" if s_eve < 2.0 else "SECURE")
                        m2.metric("S-value (clean channel)",
                                  f"{s_clean:.4f}",
                                  delta="SECURE" if s_clean > 2.0 else "CHECK")
                        if s_eve < 2.0:
                            st.error(f"Eve DETECTED! S = {s_eve:.4f} < 2.0 — quantum correlations disturbed.")
                        st.info(
                            "A secure channel shows S > 2 (quantum bound ~2.828). "
                            "Eve's interception collapses entanglement, dropping S below the classical bound."
                        )
                        st.session_state.network_log.append(
                            f"[Eve] CHSH S = {s_eve:.4f} — "
                            f"{'DETECTED' if s_eve < 2.0 else 'undetected'}"
                        )
                else:
                    for line in result.stderr.strip().splitlines():
                        st.session_state.network_log.append(f"[Alice ERR] {line}")
                    st.error(f"Connection failed — could not reach Bob at {bob_host}:{int(alice_port)}.")
                    with st.expander("How to fix this"):
                        st.markdown(f"""
**Step 1 — Firewall fix on Bob's laptop (most common cause)**

Open CMD as Administrator on Bob's laptop and run:
```batch
netsh advfirewall firewall add rule name="QEC Dense Bob Server" dir=in action=allow protocol=TCP localport=5000
```

**Step 2 — Verify Bob is listening**

On Bob's laptop open CMD and run:
```
netstat -an | findstr 5000
```
You should see `0.0.0.0:5000  LISTENING`. If not, click Start Listening again.

**Step 3 — Test basic connectivity**

On this laptop (Alice), open CMD and run:
```
ping {bob_host}
```
If ping fails → laptops are not on the same WiFi, or the network blocks device-to-device traffic.
""")

        # Standalone noise simulation (no network required)
        if net_noise_on:
            st.markdown("---")
            if st.button("Run Noise Simulation Only (no network)", use_container_width=True, key="sim_only"):
                from noise_channel import create_channel_noise_model
                from quantum_protocols import full_superdense_circuit, full_superdense_with_qec, decode_result as _decode_result2
                st.subheader("Noisy Channel Simulation (Local)")
                with st.spinner("Running noisy channel simulation ..."):
                    circ_basic2 = full_superdense_circuit(alice_msg)
                    if net_noise_prob > 0:
                        nm_basic2 = create_channel_noise_model(net_noise_type, net_noise_prob, [0])
                        sim_basic2 = AerSimulator(noise_model=nm_basic2)
                        comp_basic2 = transpile(circ_basic2, sim_basic2, optimization_level=0)
                        counts_basic2 = sim_basic2.run(comp_basic2, shots=1024).result().get_counts()
                    else:
                        _sb2 = AerSimulator()
                        comp_basic2 = transpile(circ_basic2, _sb2)
                        counts_basic2 = _sb2.run(comp_basic2, shots=1024).result().get_counts()
                    decoded_basic2 = _decode_result2(counts_basic2, with_qec=False)
                    correct_basic2 = sum(v for k, v in counts_basic2.items() if _decode_result2({k: v}, with_qec=False) == alice_msg)
                    acc_basic2 = correct_basic2 / 1024

                    circ_qec2 = full_superdense_with_qec(alice_msg)
                    if net_noise_prob > 0:
                        nm_qec2 = create_channel_noise_model(net_noise_type, net_noise_prob, [0, 1, 2])
                        sim_qec2 = AerSimulator(noise_model=nm_qec2)
                        comp_qec2 = transpile(circ_qec2, sim_qec2, optimization_level=0)
                        counts_qec2 = sim_qec2.run(comp_qec2, shots=1024).result().get_counts()
                    else:
                        _sq2 = AerSimulator()
                        comp_qec2 = transpile(circ_qec2, _sq2)
                        counts_qec2 = _sq2.run(comp_qec2, shots=1024).result().get_counts()
                    decoded_qec2 = _decode_result2(counts_qec2, with_qec=True)
                    correct_qec2 = sum(v for k, v in counts_qec2.items() if _decode_result2({k: v}, with_qec=True) == alice_msg)
                    acc_qec2 = correct_qec2 / 1024

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.markdown(
                        f"<div style='text-align:center;padding:1rem;"
                        f"background:rgba(255,107,107,0.1);border:1px solid rgba(255,107,107,0.3);"
                        f"border-radius:12px;'>"
                        f"<div style='color:#ff6b6b;font-size:0.8rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:1px;'>Without QEC</div>"
                        f"<div style='font-size:2rem;font-weight:800;color:#ff3366;"
                        f"margin:0.5rem 0;'>{acc_basic2*100:.1f}%</div>"
                        f"<div style='color:#7a9ab8;font-size:0.85rem;'>Decoded: '{decoded_basic2}'</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_s2:
                    st.markdown(
                        f"<div style='text-align:center;padding:1rem;"
                        f"background:rgba(0,255,136,0.08);border:1px solid rgba(0,255,136,0.3);"
                        f"border-radius:12px;'>"
                        f"<div style='color:#00ff88;font-size:0.8rem;font-weight:700;"
                        f"text-transform:uppercase;letter-spacing:1px;'>With QEC</div>"
                        f"<div style='font-size:2rem;font-weight:800;color:#00ff88;"
                        f"margin:0.5rem 0;'>{acc_qec2*100:.1f}%</div>"
                        f"<div style='color:#7a9ab8;font-size:0.85rem;'>Decoded: '{decoded_qec2}'</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                improvement2 = acc_qec2 - acc_basic2
                if improvement2 > 0:
                    st.success(f"QEC improved accuracy by **{improvement2*100:.1f}%** at {net_noise_prob*100:.0f}% {net_noise_type.replace('_', ' ')} noise!")
                elif improvement2 < 0 and net_noise_type != "bit_flip":
                    st.warning(f"QEC reduced accuracy — expected for {net_noise_type.replace('_', ' ')} noise. The 3-qubit repetition code only corrects bit-flip errors.")

                fig_sim, (ax_sim1, ax_sim2) = plt.subplots(1, 2, figsize=(10, 4))
                labels_b2 = sorted(counts_basic2.keys())
                ax_sim1.bar(labels_b2, [counts_basic2[k]/1024 for k in labels_b2], color="#ff6b6b", edgecolor="#1a2d4a", alpha=0.85)
                ax_sim1.set_title("Without QEC", fontsize=12)
                ax_sim1.set_ylabel("Probability")
                ax_sim1.set_ylim(0, 1.1)
                ax_sim1.tick_params(axis="x", rotation=45)
                labels_q2 = sorted(counts_qec2.keys())
                ax_sim2.bar(labels_q2, [counts_qec2[k]/1024 for k in labels_q2], color="#00ff88", edgecolor="#1a2d4a", alpha=0.85)
                ax_sim2.set_title("With QEC", fontsize=12)
                ax_sim2.set_ylabel("Probability")
                ax_sim2.set_ylim(0, 1.1)
                ax_sim2.tick_params(axis="x", rotation=45)
                fig_sim.suptitle(
                    f"Local Simulation — msg='{alice_msg}', {net_noise_prob*100:.0f}% {net_noise_type.replace('_', ' ')} noise",
                    fontsize=13, fontweight="bold",
                )
                fig_sim.patch.set_facecolor("#0b0f20")
                plt.tight_layout()
                st.pyplot(fig_sim)
                plt.close(fig_sim)
                st.session_state.network_log.append(
                    f"[Sim] {net_noise_type} @ {net_noise_prob*100:.0f}% — "
                    f"No QEC: {acc_basic2*100:.1f}% | With QEC: {acc_qec2*100:.1f}% (+{improvement2*100:.1f}%)"
                )

    # -----------------------------------------------------------------------
    # Shared: log + tips
    # -----------------------------------------------------------------------
    st.markdown("---")
    if st.button("Clear Log", use_container_width=True):
        st.session_state.network_log = []

    log_text = "\n".join(st.session_state.network_log[-30:]) or "(no activity yet)"
    st.code(log_text, language="text")

    with st.expander("Network Setup Tips"):
        st.markdown("""
**Find your IP address:**
- Windows: Open CMD -> type `ipconfig` -> look for "IPv4 Address" under Wi-Fi
- Mac/Linux: Open Terminal -> type `ifconfig` or `ip addr`

**Troubleshooting:**
- Both laptops must be on the **same WiFi network**
- Bob must click "Start Listening" **before** Alice clicks "Send"
- If connection fails: allow Python through Windows Firewall when prompted
- Test connectivity: run `ping <Bob IP>` from Alice's laptop first
- Default port is 5000 — make sure Alice and Bob use the same port number
""")


# ---------------------------------------------------------------------------
# Tab 5: Secure Channel (E91 QKD)
# ---------------------------------------------------------------------------
with tab5:
    st.markdown("""
<div style="margin-bottom:1.4rem;">
  <h2 style="
    margin:0 0 0.3rem 0; font-size:1.55rem; font-weight:800;
    background:linear-gradient(90deg,#f472b6,#fb7185);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
    border:none; padding:0;
  ">Secure Channel &mdash; E91 Quantum Key Distribution</h2>
  <p style="margin:0; color:#4a6880; font-size:0.9rem;">
    The <strong style="color:#f472b6;">E91 protocol</strong> uses Bell pairs to generate a shared secret key.
    CHSH S-value confirms security: <strong style="color:#34d399;">S &gt; 2</strong> means no eavesdropper.
  </p>
</div>
""", unsafe_allow_html=True)

    col_e1, col_e2 = st.columns([1, 2])

    with col_e1:
        qkd_bits = st.slider("Key Length (bits)", 8, 64, 16, 4)
        add_eve = st.toggle("Add Eavesdropper (Eve)", value=False)
        qkd_msg = st.selectbox("Message to Encrypt", ["00", "01", "10", "11"], index=2, key="qkd_msg")
        run_qkd = st.button("Generate Key & Test", type="primary", use_container_width=True)

    with col_e2:
        if run_qkd:
            with st.spinner("Simulating E91 protocol ..."):
                from e91_qkd import (
                    generate_e91_key,
                    detect_eavesdropper,
                    encrypt_with_qkd_key,
                    decrypt_with_qkd_key,
                    simulate_eavesdropper,
                )
                key_result = generate_e91_key(num_bits=qkd_bits, eavesdropper=add_eve)

            s_val = key_result["chsh_s"]
            eve_detected = detect_eavesdropper(key_result)
            alice_key = key_result["alice_key"][:2]
            bob_key = key_result["bob_key"][:2]

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("CHSH S-Value", f"{s_val:.4f}", delta=None)
            m2.metric("Security", "COMPROMISED" if eve_detected else "SECURE")
            m3.metric("Key Match", "YES" if alice_key == bob_key else "NO")

            if eve_detected:
                st.error(f"Eve detected! S = {s_val:.4f} < 2.0 — channel is compromised.")
            else:
                st.success(f"Secure channel confirmed. S = {s_val:.4f} > 2.0 (classical bound).")

            st.markdown(f"**Alice's key (first 2 bits):** `{''.join(str(b) for b in alice_key)}`")
            st.markdown(f"**Bob's key (first 2 bits):** `{''.join(str(b) for b in bob_key)}`")

            # Encrypt and send
            st.markdown("---")
            st.markdown("**Encrypted Superdense Transmission:**")
            msg_bits = [int(b) for b in qkd_msg]
            cipher_bits = encrypt_with_qkd_key(msg_bits, alice_key)
            cipher_str = "".join(str(b) for b in cipher_bits)
            plain_again_bits = decrypt_with_qkd_key(cipher_bits, bob_key)
            plain_again = "".join(str(b) for b in plain_again_bits)

            st.code(
                f"Original message :  {qkd_msg}\n"
                f"QKD key (2 bits)  :  {''.join(str(b) for b in alice_key)}\n"
                f"Encrypted message :  {cipher_str}\n"
                f"Decrypted by Bob  :  {plain_again}\n"
                f"Match             :  {'YES' if plain_again == qkd_msg else 'NO (key mismatch due to noise)'}"
            )

            # CHSH S-value visualization
            fig, ax = plt.subplots(figsize=(5, 3))
            bar_color = "#00ff88" if not eve_detected else "#ff3366"
            ax.bar(["Measured S"], [s_val], color=bar_color, width=0.4,
                   edgecolor="#ff2e63" if not eve_detected else "#ff6b6b", linewidth=1.5)
            ax.axhline(2.0, color="#fbbf24", linestyle="--", linewidth=2, label="Classical bound S=2")
            ax.axhline(2.828, color="#ff6b6b", linestyle="--", linewidth=1.5, label="Quantum max S≈2.828")
            ax.set_ylim(0, 3.2)
            ax.set_ylabel("CHSH S-Value")
            ax.set_title("CHSH Inequality Test")
            ax.legend(fontsize=9)
            fig.patch.set_facecolor("#0b0f20")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Raw trial data table
            with st.expander("View Raw Trial Data"):
                import pandas as pd
                chsh_pairs = {(0, 1), (0, 3), (2, 1), (2, 3)}
                matching_pairs = {(1, 1), (2, 2)}
                basis_labels = {0: "0°", 1: "45°", 2: "90°", 3: "135°"}
                rows = []
                for i, t in enumerate(key_result["basis_data"][:50]):
                    ab = t["alice_basis"]
                    bb = t["bob_basis"]
                    if (ab, bb) in matching_pairs:
                        used = "Key Bit"
                    elif (ab, bb) in chsh_pairs:
                        used = "CHSH Test"
                    else:
                        used = "Discarded"
                    rows.append({
                        "Trial": i + 1,
                        "Alice Basis": basis_labels[ab],
                        "Bob Basis": basis_labels[bb],
                        "Alice Bit": t["alice_bit"],
                        "Bob Bit": t["bob_bit"],
                        "Used For": used,
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Click **Generate Key & Test** to run the E91 protocol.")

        st.markdown("""
---
**How E91 QKD works:**
1. Alice and Bob each receive one qubit from a shared Bell pair
2. They independently measure in randomly chosen bases
3. They publicly compare their basis choices (not results!)
4. **Matching bases** → key bits (kept secret)
5. **Non-matching bases** → used to compute CHSH S-value
6. If S > 2: quantum correlations confirmed → no eavesdropper
""")

st.markdown("""
<div style="margin-top: 5rem; margin-bottom: 3rem; text-align: center;">
  <div style="height:1px; background:linear-gradient(90deg, transparent, rgba(0,242,254,0.4), rgba(255,8,68,0.4), transparent); margin: 0 auto 2.5rem auto; width: 80%;"></div>
  <h2 style="margin-bottom: 1.5rem; font-size: 1.6rem; font-weight: 900; letter-spacing: 3px; text-transform: uppercase;">
    <span style="background: linear-gradient(135deg, #00f2fe, #ff0844); -webkit-background-clip: text; -webkit-text-fill-color: transparent; filter: drop-shadow(0 0 10px rgba(0,242,254,0.3));">
      About Us
    </span>
  </h2>
  <div style="display:flex; justify-content:center;">
    <span style="color:#b2d5f0; font-size:1.15rem; font-weight:600; padding: 14px 28px; background: rgba(11, 15, 32, 0.7); border-radius: 18px; border: 1px solid rgba(0,242,254,0.25); box-shadow: 0 10px 30px rgba(0,0,0,0.6), inset 0 0 20px rgba(255,8,68,0.1); line-height: 1.8;">
      Team: K. Penchala Lokesh &nbsp;&middot;&nbsp; K. Penchala Rakesh &nbsp;&middot;&nbsp; Ch. Joel Judson Paul &nbsp;&middot;&nbsp; M. Paramesh Varma &nbsp;&middot;&nbsp; K. Amareswar Rao
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

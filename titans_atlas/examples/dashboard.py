#!/usr/bin/env python3
"""
Atlas Training Dashboard.

Real-time monitoring dashboard for Atlas model training.
Reads metrics from JSONL files written by train_with_metrics.py.

Usage:
    # Default run directory
    streamlit run dashboard.py

    # Specify run directory
    streamlit run dashboard.py -- --run-dir ./runs/atlas_36m

    # Or use Dash (alternative):
    python dashboard.py --dash --port 8050
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import deque
import numpy as np

# Try to import visualization libraries
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import plotly.graph_objs as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_jsonl(filepath: Path, max_lines: int = 10000) -> List[Dict]:
    """Load JSONL file, returning list of dicts."""
    if not filepath.exists():
        return []

    records = []
    try:
        with open(filepath, "r") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        # Log but don't fail - dashboard should be resilient to transient file issues
        import logging
        logging.debug(f"Could not read {filepath}: {e}")

    return records


def load_metrics(run_dir: Path) -> Dict[str, Any]:
    """Load all metrics from run directory."""
    metrics = {
        "convergence": [],
        "weaver": [],
        "config": {},
    }

    # Load convergence metrics
    conv_file = run_dir / "metrics" / "convergence" / "metrics.jsonl"
    metrics["convergence"] = load_jsonl(conv_file)

    # Load weaver metrics
    weaver_file = run_dir / "metrics" / "weaver" / "weaver_summary.jsonl"
    metrics["weaver"] = load_jsonl(weaver_file)

    # Load config
    config_file = run_dir / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            metrics["config"] = json.load(f)

    return metrics


def create_loss_figure(convergence_data: List[Dict]) -> go.Figure:
    """Create loss over time plot."""
    if not convergence_data:
        return go.Figure()

    steps = [d["step"] for d in convergence_data]
    losses = [d["loss"] for d in convergence_data]
    smoothed = [d.get("smoothed_loss", d["loss"]) for d in convergence_data]
    best = [d.get("best_loss", min(losses[:i+1])) for i, d in enumerate(convergence_data)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=losses,
        mode="lines",
        name="Loss",
        line=dict(color="lightblue", width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=smoothed,
        mode="lines",
        name="Smoothed Loss",
        line=dict(color="blue", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=best,
        mode="lines",
        name="Best Loss",
        line=dict(color="green", width=1, dash="dash"),
    ))

    fig.update_layout(
        title="Training Loss",
        xaxis_title="Step",
        yaxis_title="Loss",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig


def create_perplexity_figure(convergence_data: List[Dict]) -> go.Figure:
    """Create perplexity over time plot."""
    if not convergence_data:
        return go.Figure()

    steps = [d["step"] for d in convergence_data]
    ppl = [min(d.get("perplexity", np.exp(d["loss"])), 1e6) for d in convergence_data]
    smoothed_ppl = [min(d.get("smoothed_perplexity", ppl[i]), 1e6) for i, d in enumerate(convergence_data)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=ppl,
        mode="lines",
        name="Perplexity",
        line=dict(color="lightcoral", width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=smoothed_ppl,
        mode="lines",
        name="Smoothed PPL",
        line=dict(color="red", width=2),
    ))

    fig.update_layout(
        title="Perplexity",
        xaxis_title="Step",
        yaxis_title="Perplexity",
        yaxis_type="log",
        hovermode="x unified",
    )

    return fig


def create_gradient_figure(convergence_data: List[Dict]) -> go.Figure:
    """Create gradient norm over time plot."""
    if not convergence_data:
        return go.Figure()

    steps = [d["step"] for d in convergence_data]
    grad_norms = [d.get("grad_norm", 0) for d in convergence_data]
    grad_smoothed = [d.get("grad_norm_smoothed", grad_norms[i]) for i, d in enumerate(convergence_data)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=grad_norms,
        mode="lines",
        name="Grad Norm",
        line=dict(color="lightgreen", width=1),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=grad_smoothed,
        mode="lines",
        name="Smoothed",
        line=dict(color="green", width=2),
    ))

    fig.update_layout(
        title="Gradient Norm",
        xaxis_title="Step",
        yaxis_title="Gradient Norm",
        hovermode="x unified",
    )

    return fig


def create_convergence_figure(convergence_data: List[Dict]) -> go.Figure:
    """Create convergence score over time plot."""
    if not convergence_data:
        return go.Figure()

    steps = [d["step"] for d in convergence_data]
    scores = [d.get("convergence_score", 0) for d in convergence_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=scores,
        mode="lines",
        name="Convergence Score",
        line=dict(color="purple", width=2),
        fill="tozeroy",
        fillcolor="rgba(128, 0, 128, 0.1)",
    ))

    # Add threshold lines
    fig.add_hline(y=0.9, line_dash="dash", line_color="orange", annotation_text="Good")
    fig.add_hline(y=0.95, line_dash="dash", line_color="green", annotation_text="Converged")

    fig.update_layout(
        title="Convergence Score",
        xaxis_title="Step",
        yaxis_title="Score (0-1)",
        yaxis_range=[0, 1],
        hovermode="x unified",
    )

    return fig


def create_lr_figure(convergence_data: List[Dict]) -> go.Figure:
    """Create learning rate schedule plot."""
    if not convergence_data:
        return go.Figure()

    steps = [d["step"] for d in convergence_data]
    lrs = [d.get("learning_rate", 0) for d in convergence_data]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=lrs,
        mode="lines",
        name="Learning Rate",
        line=dict(color="orange", width=2),
    ))

    fig.update_layout(
        title="Learning Rate Schedule",
        xaxis_title="Step",
        yaxis_title="Learning Rate",
        hovermode="x unified",
    )

    return fig


def run_streamlit_dashboard(run_dir: Path):
    """Run Streamlit dashboard."""
    st.set_page_config(
        page_title="Atlas Training Dashboard",
        page_icon="🔮",
        layout="wide",
    )

    st.title("🔮 Atlas 36M Training Dashboard")

    # Sidebar
    st.sidebar.header("Settings")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (s)", 1, 30, 5)

    # Load metrics
    metrics = load_metrics(run_dir)

    if not metrics["convergence"]:
        st.warning(f"No metrics found in {run_dir}. Make sure training is running.")
        st.info("Expected metrics file: {}/metrics/convergence/metrics.jsonl".format(run_dir))
        time.sleep(2)
        if auto_refresh:
            st.rerun()
        return

    # Config info
    if metrics["config"]:
        with st.expander("Run Configuration"):
            st.json(metrics["config"])

    # Latest stats
    latest = metrics["convergence"][-1]
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Step", f"{latest['step']:,}")
    with col2:
        st.metric("Loss", f"{latest['loss']:.4f}",
                  delta=f"{latest['loss'] - latest.get('smoothed_loss', latest['loss']):.4f}")
    with col3:
        st.metric("Perplexity", f"{latest.get('perplexity', 0):.2f}")
    with col4:
        st.metric("Convergence", f"{latest.get('convergence_score', 0):.2%}")
    with col5:
        status = "🟢 Converging" if latest.get("is_converging") else (
            "🟡 Plateau" if latest.get("is_plateaued") else "🔵 Training")
        st.metric("Status", status)

    # Main plots
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_loss_figure(metrics["convergence"]), use_container_width=True)
        st.plotly_chart(create_gradient_figure(metrics["convergence"]), use_container_width=True)

    with col2:
        st.plotly_chart(create_perplexity_figure(metrics["convergence"]), use_container_width=True)
        st.plotly_chart(create_convergence_figure(metrics["convergence"]), use_container_width=True)

    # Learning rate
    st.plotly_chart(create_lr_figure(metrics["convergence"]), use_container_width=True)

    # Weaver space metrics
    if metrics["weaver"]:
        st.header("🌀 Weaver Space Metrics")

        latest_weaver = metrics["weaver"][-1]

        if "manifold" in latest_weaver:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("D_eff Preservation",
                          f"{latest_weaver['manifold'].get('mean_d_eff_preservation', 0):.2f}")
            with col2:
                st.metric("Min D_eff",
                          f"{latest_weaver['manifold'].get('mean_min_d_eff', 0):.1f}")
            with col3:
                st.metric("Captures",
                          f"{latest_weaver['manifold'].get('num_captures', 0)}")

    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def run_dash_dashboard(run_dir: Path, port: int = 8050):
    """Run Dash dashboard (alternative to Streamlit)."""
    try:
        from dash import Dash, dcc, html
        from dash.dependencies import Input, Output
    except ImportError:
        print("Dash not installed. Install with: pip install dash")
        return

    app = Dash(__name__)

    app.layout = html.Div([
        html.H1("🔮 Atlas 36M Training Dashboard"),

        html.Div(id="metrics-display"),

        dcc.Graph(id="loss-plot"),
        dcc.Graph(id="ppl-plot"),
        dcc.Graph(id="convergence-plot"),

        dcc.Interval(
            id="interval-component",
            interval=5000,  # 5 seconds
            n_intervals=0
        ),
    ])

    @app.callback(
        [Output("metrics-display", "children"),
         Output("loss-plot", "figure"),
         Output("ppl-plot", "figure"),
         Output("convergence-plot", "figure")],
        [Input("interval-component", "n_intervals")]
    )
    def update_dashboard(n):
        metrics = load_metrics(run_dir)

        if not metrics["convergence"]:
            return (
                html.P("No metrics found. Waiting for training to start..."),
                go.Figure(),
                go.Figure(),
                go.Figure(),
            )

        latest = metrics["convergence"][-1]

        metrics_div = html.Div([
            html.P(f"Step: {latest['step']:,} | "
                   f"Loss: {latest['loss']:.4f} | "
                   f"PPL: {latest.get('perplexity', 0):.2f} | "
                   f"Convergence: {latest.get('convergence_score', 0):.2%}")
        ])

        return (
            metrics_div,
            create_loss_figure(metrics["convergence"]),
            create_perplexity_figure(metrics["convergence"]),
            create_convergence_figure(metrics["convergence"]),
        )

    print(f"Starting Dash dashboard at http://localhost:{port}")
    app.run_server(debug=False, host="127.0.0.1", port=port)


def main():
    parser = argparse.ArgumentParser(description="Atlas Training Dashboard")
    parser.add_argument("--run-dir", type=str, default="./runs/atlas_36m",
                        help="Training run directory")
    parser.add_argument("--dash", action="store_true",
                        help="Use Dash instead of Streamlit")
    parser.add_argument("--port", type=int, default=8050,
                        help="Port for Dash server")

    # Parse known args (Streamlit passes extra args)
    args, _ = parser.parse_known_args()

    run_dir = Path(args.run_dir)

    if args.dash:
        run_dash_dashboard(run_dir, args.port)
    else:
        if HAS_STREAMLIT:
            run_streamlit_dashboard(run_dir)
        else:
            print("Streamlit not installed. Install with: pip install streamlit")
            print("Or use --dash flag for Dash dashboard")


if __name__ == "__main__":
    # Check if running via streamlit
    import sys
    if "streamlit" in sys.modules:
        # Running via streamlit run
        run_dir = Path("./runs/atlas_36m")

        # Check for command line override
        if "--" in sys.argv:
            idx = sys.argv.index("--")
            for i, arg in enumerate(sys.argv[idx+1:]):
                if arg == "--run-dir" and i + 1 < len(sys.argv[idx+1:]):
                    run_dir = Path(sys.argv[idx + i + 2])

        run_streamlit_dashboard(run_dir)
    else:
        main()

"""
Streamlit Real-Time Training Dashboard

Three-page dashboard for monitoring Atlas training:
1. Live Training Overview - Loss, gates, progress
2. Memory Deep Dive - Per-layer analysis, heatmaps
3. Alerts & Health Check - History, recommendations

Usage:
    streamlit run streamlit_monitor.py -- --metrics-path runs/experiment/metrics_stream.jsonl
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Page config
st.set_page_config(
    page_title="Atlas Training Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_metrics(metrics_path: str, last_n: int = 10000) -> pd.DataFrame:
    """Load metrics from JSONL file."""
    metrics = []
    try:
        with open(metrics_path, 'r') as f:
            lines = f.readlines()
            # Take last N lines for performance
            for line in lines[-last_n:]:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))
    except FileNotFoundError:
        st.warning(f"Metrics file not found: {metrics_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return pd.DataFrame()

    if not metrics:
        return pd.DataFrame()

    df = pd.DataFrame(metrics)

    # Handle column name variations
    if 'storage_loss_mean' in df.columns and 'loss' not in df.columns:
        df['loss'] = df['storage_loss_mean']

    return df


def get_latest_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Get the most recent metrics."""
    if df.empty:
        return {}
    return df.iloc[-1].to_dict()


def compute_rolling_stats(df: pd.DataFrame, column: str, window: int = 100) -> pd.Series:
    """Compute rolling mean for a column."""
    if column not in df.columns:
        return pd.Series()
    return df[column].rolling(window=window, min_periods=1).mean()


# === Page 1: Live Training Overview ===
def page_live_overview(df: pd.DataFrame, latest: Dict[str, Any]):
    """Live training overview page."""
    st.title("🚀 Live Training Overview")

    # Top metrics row
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        step = latest.get('step', 0)
        st.metric("Step", f"{step:,}")

    with col2:
        loss = latest.get('loss', 0)
        loss_change = None
        if len(df) > 100 and 'loss' in df.columns:
            loss_change = loss - df['loss'].iloc[-100]
        st.metric("Loss", f"{loss:.4f}", delta=f"{loss_change:.4f}" if loss_change else None)

    with col3:
        gate_mean = latest.get('gate_mean', 0.5)
        st.metric("Gate Mean", f"{gate_mean:.2%}")

    with col4:
        collapse_risk = latest.get('gate_collapse_risk', 0)
        st.metric(
            "Collapse Risk",
            f"{collapse_risk:.1%}",
            delta=None,
            delta_color="inverse" if collapse_risk > 0.5 else "normal"
        )

    with col5:
        elapsed = latest.get('elapsed_hours', 0)
        st.metric("Elapsed", f"{elapsed:.2f}h")

    st.divider()

    # Gate Collapse Risk Gauge
    col_gauge, col_progress = st.columns([1, 2])

    with col_gauge:
        st.subheader("Gate Collapse Risk")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=collapse_risk * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 80], 'color': "orange"},
                    {'range': [80, 100], 'color': "red"},
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            },
            title={'text': "Risk %"}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, width='stretch')

    with col_progress:
        st.subheader("Training Progress")
        max_steps = st.session_state.get('max_steps', 100000)
        progress = step / max_steps if max_steps > 0 else 0
        st.progress(progress)
        st.caption(f"Step {step:,} / {max_steps:,} ({progress:.1%})")

        # ETA calculation
        if elapsed > 0 and step > 0:
            steps_per_hour = step / elapsed
            remaining_steps = max_steps - step
            eta_hours = remaining_steps / steps_per_hour if steps_per_hour > 0 else 0
            st.info(f"ETA: {eta_hours:.1f} hours ({steps_per_hour:.0f} steps/hour)")

    st.divider()

    # Main charts
    col_loss, col_gate = st.columns(2)

    with col_loss:
        st.subheader("Loss Curve")
        if 'loss' in df.columns and not df.empty:
            fig_loss = px.line(df, x='step', y='loss', title="Training Loss")
            fig_loss.update_layout(height=400)
            st.plotly_chart(fig_loss, width='stretch')
        else:
            st.info("Waiting for loss data...")

    with col_gate:
        st.subheader("Gate Mean")
        if 'gate_mean' in df.columns and not df.empty:
            fig_gate = px.line(df, x='step', y='gate_mean', title="Gate Mean Over Time")
            fig_gate.add_hline(y=0.1, line_dash="dash", line_color="red",
                              annotation_text="Collapse threshold")
            fig_gate.update_layout(height=400)
            st.plotly_chart(fig_gate, width='stretch')
        else:
            st.info("Waiting for gate data...")

    # Retrieval metrics (if available)
    if 'retrieval_token_accuracy' in df.columns:
        st.subheader("Retrieval Performance")
        col_ret1, col_ret2 = st.columns(2)

        with col_ret1:
            fig_ret = px.line(df, x='step', y='retrieval_token_accuracy',
                             title="Retrieval Token Accuracy")
            fig_ret.add_hline(y=0.9, line_dash="dash", line_color="green",
                             annotation_text="Target (90%)")
            st.plotly_chart(fig_ret, width='stretch')

        with col_ret2:
            ret_acc = latest.get('retrieval_token_accuracy', 0)
            exact_match = latest.get('retrieval_exact_match', 0)
            st.metric("Current Retrieval Accuracy", f"{ret_acc:.1%}")
            st.metric("Exact Match Rate", f"{exact_match:.1%}")


# === Page 2: Memory Deep Dive ===
def page_memory_deep_dive(df: pd.DataFrame, latest: Dict[str, Any]):
    """Memory analysis page."""
    st.title("🧠 Memory Deep Dive")

    # Find per-layer columns
    layer_cols = [c for c in df.columns if c.startswith('layer_')]
    n_layers = len(set(c.split('/')[0] for c in layer_cols))

    if n_layers == 0:
        st.warning("No per-layer metrics found. Enable `track_per_layer=True` in adapter.")
        return

    st.info(f"Found {n_layers} layers with metrics")

    # Memory magnitude per layer
    st.subheader("Memory Magnitude by Layer")
    mag_cols = [c for c in layer_cols if 'memory_magnitude' in c]
    if mag_cols and not df.empty:
        latest_mags = {c.split('/')[0]: latest.get(c, 0) for c in mag_cols}
        fig_mag = px.bar(
            x=list(latest_mags.keys()),
            y=list(latest_mags.values()),
            labels={'x': 'Layer', 'y': 'Magnitude'},
            title="Current Memory Magnitude"
        )
        st.plotly_chart(fig_mag, width='stretch')

    # Memory rank per layer
    st.subheader("Memory Rank by Layer")
    rank_cols = [c for c in layer_cols if 'memory_rank' in c]
    if rank_cols and not df.empty:
        latest_ranks = {c.split('/')[0]: latest.get(c, 0) for c in rank_cols}
        fig_rank = px.bar(
            x=list(latest_ranks.keys()),
            y=list(latest_ranks.values()),
            labels={'x': 'Layer', 'y': 'Effective Rank'},
            title="Memory Effective Rank"
        )
        fig_rank.add_hline(y=10, line_dash="dash", line_color="green",
                          annotation_text="Target minimum")
        st.plotly_chart(fig_rank, width='stretch')

    # Gate values per layer
    st.subheader("Gate Values by Layer")
    gate_cols = [c for c in layer_cols if 'gate_mean' in c]
    if gate_cols and not df.empty:
        latest_gates = {c.split('/')[0]: latest.get(c, 0.5) for c in gate_cols}
        fig_gates = px.bar(
            x=list(latest_gates.keys()),
            y=list(latest_gates.values()),
            labels={'x': 'Layer', 'y': 'Gate Mean'},
            title="Gate Mean by Layer"
        )
        fig_gates.add_hline(y=0.1, line_dash="dash", line_color="red",
                           annotation_text="Collapse threshold")
        st.plotly_chart(fig_gates, width='stretch')

    # Surprise accumulator
    st.subheader("Surprise Accumulator Activity")
    surprise_cols = [c for c in layer_cols if 'surprise_norm' in c]
    if surprise_cols and not df.empty:
        latest_surprise = {c.split('/')[0]: latest.get(c, 0) for c in surprise_cols}
        fig_surprise = px.bar(
            x=list(latest_surprise.keys()),
            y=list(latest_surprise.values()),
            labels={'x': 'Layer', 'y': 'S Norm'},
            title="Surprise Accumulator Norm"
        )
        st.plotly_chart(fig_surprise, width='stretch')

    # Memory sparsity over time
    st.subheader("Memory Sparsity Trend")
    if 'memory_sparsity' in df.columns:
        fig_sparsity = px.line(df, x='step', y='memory_sparsity',
                               title="Memory Sparsity Over Time")
        fig_sparsity.add_hline(y=0.5, line_dash="dash", line_color="orange",
                               annotation_text="Warning threshold")
        st.plotly_chart(fig_sparsity, width='stretch')


# === Page 3: Alerts & Health Check ===
def page_alerts_health(df: pd.DataFrame, latest: Dict[str, Any]):
    """Alerts and health check page."""
    st.title("⚠️ Alerts & Health Check")

    # Current health status
    st.subheader("Current Health Status")

    checks = []

    # Gate collapse check
    collapse_risk = latest.get('gate_collapse_risk', 0)
    if collapse_risk > 0.8:
        checks.append(("🚨 CRITICAL", "Gate collapse risk > 80%", "red"))
    elif collapse_risk > 0.5:
        checks.append(("⚠️ WARNING", f"Gate collapse risk at {collapse_risk:.1%}", "orange"))
    else:
        checks.append(("✅ OK", f"Gate collapse risk at {collapse_risk:.1%}", "green"))

    # Memory health
    memory_sparsity = latest.get('memory_sparsity', 0)
    if memory_sparsity > 0.5:
        checks.append(("⚠️ WARNING", f"High memory sparsity: {memory_sparsity:.1%}", "orange"))
    else:
        checks.append(("✅ OK", f"Memory sparsity: {memory_sparsity:.1%}", "green"))

    # Retrieval accuracy
    ret_acc = latest.get('retrieval_token_accuracy', None)
    if ret_acc is not None:
        if ret_acc < 0.5:
            checks.append(("⚠️ WARNING", f"Low retrieval accuracy: {ret_acc:.1%}", "orange"))
        elif ret_acc >= 0.9:
            checks.append(("✅ EXCELLENT", f"Retrieval accuracy: {ret_acc:.1%}", "green"))
        else:
            checks.append(("INFO", f"Retrieval accuracy: {ret_acc:.1%}", "blue"))

    # Loss stability
    if 'loss' in df.columns and len(df) > 100:
        recent_loss = df['loss'].iloc[-100:].mean()
        earlier_loss = df['loss'].iloc[-200:-100].mean() if len(df) > 200 else recent_loss
        if recent_loss > earlier_loss * 1.5:
            checks.append(("⚠️ WARNING", "Loss increasing", "orange"))
        elif recent_loss < earlier_loss:
            checks.append(("✅ OK", "Loss decreasing", "green"))

    # Display health checks
    for status, message, color in checks:
        st.markdown(f":{color}[{status}] {message}")

    st.divider()

    # Recommendations
    st.subheader("Recommendations")

    recommendations = []

    if collapse_risk > 0.5:
        recommendations.append(
            "⚡ **Increase gate floor**: Consider raising `phase_gate_floor` to prevent further collapse"
        )

    if memory_sparsity > 0.3:
        recommendations.append(
            "🔧 **Check memory updates**: Memory may not be learning. Verify storage phase is working."
        )

    if ret_acc is not None and ret_acc < 0.7:
        recommendations.append(
            "📊 **Review retrieval loss**: Increase `retrieval_contrast_weight` to encourage memory usage"
        )

    gate_mean = latest.get('gate_mean', 0.5)
    if gate_mean < 0.1:
        recommendations.append(
            "🚨 **Gate collapse detected**: Model is bypassing memory. Consider restarting with higher gate floor."
        )

    if not recommendations:
        recommendations.append("✨ Training looks healthy! No immediate actions needed.")

    for rec in recommendations:
        st.markdown(rec)

    st.divider()

    # Detailed metrics table
    st.subheader("Latest Metrics")
    if latest:
        # Filter to important metrics
        important_keys = [
            'step', 'loss', 'gate_mean', 'gate_collapse_risk', 'gate_dead_ratio',
            'memory_sparsity', 'memory_rank_mean', 'retrieval_token_accuracy',
            'elapsed_hours'
        ]
        filtered = {k: v for k, v in latest.items() if k in important_keys}
        st.json(filtered)

    # Raw data explorer
    with st.expander("Raw Data Explorer"):
        if not df.empty:
            st.dataframe(df.tail(100))


# === Page 4: Grokking & Geometry ===
def page_grokking_geometry(df: pd.DataFrame, latest: Dict[str, Any]):
    """Grokking detection and geometric structure page."""
    st.title("🔮 Grokking & Geometry")

    # Check if grokking metrics are available
    grokking_cols = [c for c in df.columns if c.startswith('grokking/')]

    if not grokking_cols:
        st.warning("Grokking metrics not detected. Enable grokking detection in config:")
        st.code("""
monitoring:
  grokking:
    enabled: true
    metrics_interval: 500
        """)
        return

    # Current grokking phase
    st.subheader("Current Grokking Phase")

    phase = latest.get('grokking/phase', 'unknown')
    phase_colors = {
        'memorization': '🟡',
        'circuit_formation': '🟠',
        'cleanup': '🔵',
        'grokked': '🟢',
        'unknown': '⚪',
        'insufficient_data': '⚪',
    }
    phase_icon = phase_colors.get(phase, '⚪')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Phase", f"{phase_icon} {phase}")

    with col2:
        fourier = latest.get('grokking/embedding_fourier_concentration', 0)
        st.metric("Fourier Concentration", f"{fourier:.3f}")

    with col3:
        circular = latest.get('grokking/embedding_circular_fit', 0)
        st.metric("Circular Fit", f"{circular:.3f}")

    with col4:
        dim_ratio = latest.get('grokking/embedding_effective_dim_ratio', 0)
        st.metric("Effective Dim Ratio", f"{dim_ratio:.1%}")

    st.divider()

    # Grokking timeline
    st.subheader("Grokking Metrics Over Time")

    # Prepare data
    grok_df = df[['step', *grokking_cols]].dropna()

    if not grok_df.empty:
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Fourier Concentration",
                "Circular Fit",
                "Effective Dim Ratio",
                "Memory Rank"
            ]
        )

        # Fourier concentration
        if 'grokking/embedding_fourier_concentration' in grok_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=grok_df['step'],
                    y=grok_df['grokking/embedding_fourier_concentration'],
                    name='Fourier',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="green",
                         row=1, col=1, annotation_text="Target")

        # Circular fit
        if 'grokking/embedding_circular_fit' in grok_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=grok_df['step'],
                    y=grok_df['grokking/embedding_circular_fit'],
                    name='Circular',
                    line=dict(color='purple')
                ),
                row=1, col=2
            )
            fig.add_hline(y=0.8, line_dash="dash", line_color="green",
                         row=1, col=2, annotation_text="Target")

        # Effective dimensionality ratio
        if 'grokking/embedding_effective_dim_ratio' in grok_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=grok_df['step'],
                    y=grok_df['grokking/embedding_effective_dim_ratio'],
                    name='Dim Ratio',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
            fig.add_hline(y=0.3, line_dash="dash", line_color="green",
                         row=2, col=1, annotation_text="Target")

        # Memory rank
        if 'grokking/memory_rank' in grok_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=grok_df['step'],
                    y=grok_df['grokking/memory_rank'],
                    name='Mem Rank',
                    line=dict(color='red')
                ),
                row=2, col=2
            )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Grokking interpretation
    st.subheader("Interpretation Guide")

    with st.expander("Understanding Grokking Metrics"):
        st.markdown("""
**Grokking Phases:**
- 🟡 **Memorization**: Model memorizes training data but hasn't developed structure
- 🟠 **Circuit Formation**: Geometric structure emerging (metrics improving)
- 🔵 **Cleanup**: Generalization happening (retrieval accuracy jumping)
- 🟢 **Grokked**: Stable performance + stable geometry (ready for Kakeya extraction)

**Metrics:**
- **Fourier Concentration**: Energy in low-frequency components. Higher = more periodic structure.
- **Circular Fit**: How well embeddings fit a circle in 2D PCA. Higher = more organized geometry.
- **Effective Dim Ratio**: Fraction of dimensions needed for 95% variance. Lower = more compressed representation.
- **Memory Rank**: Numerical rank of memory matrices. Indicates structure in learned associations.

**Targets for Grokking:**
- Fourier Concentration > 0.5
- Circular Fit > 0.8
- Effective Dim Ratio < 0.3
- Stable retrieval accuracy > 70%
        """)

    # Retrieval vs grokking correlation
    st.subheader("Retrieval Accuracy vs Grokking")

    ret_acc_col = None
    for col in ['retrieval_accuracy_mean', 'retrieval_token_accuracy']:
        if col in df.columns:
            ret_acc_col = col
            break

    if ret_acc_col and 'grokking/embedding_fourier_concentration' in df.columns:
        corr_df = df[['step', ret_acc_col, 'grokking/embedding_fourier_concentration']].dropna()

        if not corr_df.empty:
            fig_corr = go.Figure()

            fig_corr.add_trace(go.Scatter(
                x=corr_df['step'],
                y=corr_df[ret_acc_col],
                name='Retrieval Accuracy',
                yaxis='y1'
            ))

            fig_corr.add_trace(go.Scatter(
                x=corr_df['step'],
                y=corr_df['grokking/embedding_fourier_concentration'],
                name='Fourier Concentration',
                yaxis='y2'
            ))

            fig_corr.update_layout(
                title="Retrieval Accuracy & Fourier Concentration",
                yaxis=dict(title='Retrieval Accuracy', side='left'),
                yaxis2=dict(title='Fourier Concentration', side='right', overlaying='y'),
                height=400
            )

            st.plotly_chart(fig_corr, use_container_width=True)


# === Main App ===
def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics-path', type=str,
                        default='runs/experiment/metrics_stream.jsonl',
                        help='Path to metrics JSONL file')
    parser.add_argument('--max-steps', type=int, default=100000,
                        help='Maximum training steps (for progress bar)')
    parser.add_argument('--refresh-interval', type=int, default=30,
                        help='Auto-refresh interval in seconds')

    # Use parse_known_args to handle Streamlit's extra arguments
    args, _ = parser.parse_known_args()

    # Sidebar
    st.sidebar.title("Atlas Training Monitor")
    st.sidebar.markdown("---")

    # Settings
    metrics_path = st.sidebar.text_input(
        "Metrics Path",
        value=args.metrics_path
    )

    max_steps = st.sidebar.number_input(
        "Max Steps",
        value=args.max_steps,
        min_value=1000
    )
    st.session_state['max_steps'] = max_steps

    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)

    # Load data
    df = load_metrics(metrics_path)
    latest = get_latest_metrics(df)

    # Page selection
    page = st.sidebar.radio(
        "Page",
        ["Live Overview", "Memory Deep Dive", "Grokking & Geometry", "Alerts & Health"]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last update: {time.strftime('%H:%M:%S')}")
    st.sidebar.caption(f"Loaded {len(df):,} data points")

    # Render selected page
    if page == "Live Overview":
        page_live_overview(df, latest)
    elif page == "Memory Deep Dive":
        page_memory_deep_dive(df, latest)
    elif page == "Grokking & Geometry":
        page_grokking_geometry(df, latest)
    else:
        page_alerts_health(df, latest)

    # Auto-refresh using Streamlit's native mechanism (non-blocking)
    if auto_refresh:
        st.sidebar.info("Auto-refresh enabled (30s)")
        # Use st.empty() trick for non-blocking auto-refresh
        import streamlit.components.v1 as components
        components.html(
            """
            <script>
                setTimeout(function() {
                    window.parent.location.reload();
                }, 30000);
            </script>
            """,
            height=0
        )


if __name__ == "__main__":
    main()

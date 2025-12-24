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
    """Get the most recent metrics, forward-filling sparse columns.

    Some metrics (like layer_X/memory_rank) are only computed at intervals,
    so we need to get the most recent known value for each column.
    """
    if df.empty:
        return {}
    # Forward-fill to get most recent value for each column
    # Then take the last row
    return df.ffill().iloc[-1].to_dict()


def compute_rolling_stats(df: pd.DataFrame, column: str, window: int = 100) -> pd.Series:
    """Compute rolling mean for a column."""
    if column not in df.columns:
        return pd.Series()
    return df[column].rolling(window=window, min_periods=1).mean()


# === Phase Indicator Helper ===
def render_phase_indicator(phase: str) -> None:
    """Render a color-coded phase indicator."""
    phase_config = {
        "memory_learning": {
            "color": "#3498db",  # Blue
            "icon": "🔵",
            "description": "Model is actively learning (loss decreasing, accuracy increasing)",
        },
        "converged": {
            "color": "#2ecc71",  # Green
            "icon": "🟢",
            "description": "Metrics have plateaued - training has converged",
        },
        "overfitting": {
            "color": "#f39c12",  # Orange
            "icon": "🟠",
            "description": "Train/val gap growing - model is overfitting",
        },
        "grokking": {
            "color": "#9b59b6",  # Purple
            "icon": "🟣",
            "description": "Val accuracy recovering, representations stabilizing",
        },
        "unknown": {
            "color": "#95a5a6",  # Gray
            "icon": "⚪",
            "description": "Insufficient data for phase detection",
        },
    }

    config = phase_config.get(phase, phase_config["unknown"])

    st.markdown(f"""
        <div style="
            background: {config['color']};
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            color: white;
            margin-bottom: 10px;
        ">
            <h3 style="margin: 0;">{config['icon']} Phase: {phase.upper().replace('_', ' ')}</h3>
            <p style="margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9;">{config['description']}</p>
        </div>
    """, unsafe_allow_html=True)


# === Page 1: Live Training Overview ===
def page_live_overview(df: pd.DataFrame, latest: Dict[str, Any]):
    """Live training overview page."""
    st.title("🚀 Live Training Overview")

    # Phase indicator at top (if phase detection enabled)
    detected_phase = latest.get('detected_phase', None)
    if detected_phase:
        render_phase_indicator(detected_phase)

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
            checks.append(("ℹ️ INFO", f"Retrieval accuracy: {ret_acc:.1%}", "blue"))

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

    # ========================================
    # SECTION 1: Primary Metrics (All Tasks)
    # ========================================
    st.subheader("Retrieval Accuracy & Grokking Phase")

    phase = latest.get('grokking/phase', 'unknown')
    phase_colors = {
        'memorization': '🟡',
        'circuit_formation': '🟠',
        'cleanup': '🔵',
        'grokked': '🟢',
        'gate_collapse': '🔴',
        'unknown': '⚪',
        'insufficient_data': '⚪',
    }
    phase_icon = phase_colors.get(phase, '⚪')

    # Primary metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Phase", f"{phase_icon} {phase}")

    with col2:
        # Check for masked word accuracy (language) or math accuracy
        lang_acc = latest.get('masked_word_accuracy', latest.get('grokking/masked_word_accuracy', 0))
        st.metric("Masked Word Accuracy", f"{lang_acc:.1%}")

    with col3:
        dim_ratio = latest.get('grokking/embedding_effective_dim_ratio', 0)
        st.metric("Effective Dim Ratio", f"{dim_ratio:.1%}")

    with col4:
        gate_mean = latest.get('gate_mean', 0)
        gate_status = "✓" if gate_mean > 0.1 else "⚠"
        st.metric("Gate Health", f"{gate_mean:.1%} {gate_status}")

    st.divider()

    # Retrieval accuracy over time (PRIMARY CHART)
    st.subheader("Retrieval Accuracy Over Time")

    # Find accuracy column
    lang_acc_col = next((c for c in ['masked_word_accuracy', 'grokking/masked_word_accuracy'] if c in df.columns), None)

    if lang_acc_col:
        fig_acc = px.line(df, x='step', y=lang_acc_col, title="Masked Word Accuracy")
        fig_acc.add_hline(y=0.15, line_dash="dash", line_color="orange", annotation_text="Learning threshold")
        fig_acc.add_hline(y=0.30, line_dash="dash", line_color="green", annotation_text="Memory usage")
        fig_acc.update_layout(height=350)
        st.plotly_chart(fig_acc, use_container_width=True)
    else:
        st.info("Waiting for accuracy data...")

    # General grokking metrics (effective dim, memory rank, entropy)
    st.subheader("General Grokking Metrics")

    grok_df = df[['step'] + grokking_cols].dropna()

    if not grok_df.empty:
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Effective Dim Ratio", "Memory Rank", "Embedding Entropy"]
        )

        # Effective dimensionality ratio
        if 'grokking/embedding_effective_dim_ratio' in grok_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=grok_df['step'],
                    y=grok_df['grokking/embedding_effective_dim_ratio'],
                    name='Dim Ratio',
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
            fig.add_hline(y=0.3, line_dash="dash", line_color="green", row=1, col=1)

        # Memory rank
        if 'grokking/memory_rank' in grok_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=grok_df['step'],
                    y=grok_df['grokking/memory_rank'],
                    name='Mem Rank',
                    line=dict(color='blue')
                ),
                row=1, col=2
            )

        # Embedding entropy
        if 'grokking/embedding_entropy' in grok_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=grok_df['step'],
                    y=grok_df['grokking/embedding_entropy'],
                    name='Entropy',
                    line=dict(color='purple')
                ),
                row=1, col=3
            )

        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ========================================
    # SECTION 2: Math-Specific Metrics
    # ========================================
    st.subheader("Math-Specific Metrics (Modular Arithmetic Only)")

    st.caption("These metrics detect circular/periodic structure in modular arithmetic. They are NOT meaningful for language tasks.")

    # Check if we have math-specific metrics with non-zero values
    has_fourier = 'grokking/embedding_fourier_concentration' in df.columns
    has_circular = 'grokking/embedding_circular_fit' in df.columns
    fourier_val = latest.get('grokking/embedding_fourier_concentration', 0)
    circular_val = latest.get('grokking/embedding_circular_fit', 0)

    # Only show if we have data and it's non-zero (math task)
    if has_fourier or has_circular:
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fourier Concentration", f"{fourier_val:.3f}",
                     help="Energy in low-frequency components. Higher = more periodic structure. Only meaningful for mod-p arithmetic.")

        with col2:
            st.metric("Circular Fit", f"{circular_val:.3f}",
                     help="How well embeddings fit a circle in 2D PCA. Only meaningful for mod-p arithmetic.")

        # Show charts only if non-zero (indicates math training)
        if fourier_val > 0.01 or circular_val > 0.01:
            if not grok_df.empty:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=["Fourier Concentration", "Circular Fit"]
                )

                if has_fourier:
                    fig.add_trace(
                        go.Scatter(
                            x=grok_df['step'],
                            y=grok_df['grokking/embedding_fourier_concentration'],
                            name='Fourier',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    fig.add_hline(y=0.5, line_dash="dash", line_color="green", row=1, col=1)

                if has_circular:
                    fig.add_trace(
                        go.Scatter(
                            x=grok_df['step'],
                            y=grok_df['grokking/embedding_circular_fit'],
                            name='Circular',
                            line=dict(color='purple')
                        ),
                        row=1, col=2
                    )
                    fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=1, col=2)

                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Fourier/Circular metrics are zero (language-only training). These will populate during math training.")
    else:
        st.info("Math-specific metrics not available. Enable `task_type: math` in grokking config for modular arithmetic training.")

    st.divider()

    # ========================================
    # SECTION 3: Interpretation Guide
    # ========================================
    st.subheader("Interpretation Guide")

    with st.expander("Understanding Grokking Metrics"):
        st.markdown("""
**Grokking Phases:**
- 🟡 **Memorization**: Model memorizes training data but hasn't developed structure
- 🟠 **Circuit Formation**: Geometric structure emerging (metrics improving)
- 🔵 **Cleanup**: Generalization happening (retrieval accuracy jumping)
- 🟢 **Grokked**: Stable performance + stable geometry (ready for Kakeya extraction)
- 🔴 **Gate Collapse**: Memory pathway bypassed (gates < 10%)

**General Metrics (All Tasks):**
- **Masked Word Accuracy**: Primary metric for language episodic memory
- **Effective Dim Ratio**: Fraction of dimensions needed for 95% variance. Lower = more compressed.
- **Memory Rank**: Numerical rank of memory matrices. Indicates structure in learned associations.
- **Embedding Entropy**: Organization of embedding space.

**Math-Specific Metrics (Modular Arithmetic):**
- **Fourier Concentration**: Energy in low-frequency components. Higher = more periodic structure.
- **Circular Fit**: How well embeddings fit a circle in 2D PCA. Detects mod-p circular structure.

**Targets for Grokking:**
- Effective Dim Ratio < 0.3
- Masked Word Accuracy > 20% (language) or > 90% (math)
- Gate Health > 10% (memory not bypassed)
        """)

    # Combined accuracy + effective dim chart (general purpose)
    st.subheader("Accuracy vs Representation Compression")

    # Find accuracy column
    acc_col = next((c for c in ['masked_word_accuracy', 'grokking/masked_word_accuracy',
                                 'math_accuracy', 'retrieval_accuracy_mean'] if c in df.columns), None)
    dim_col = 'grokking/embedding_effective_dim_ratio' if 'grokking/embedding_effective_dim_ratio' in df.columns else None

    if acc_col and dim_col:
        fig_corr = go.Figure()

        fig_corr.add_trace(go.Scatter(
            x=df['step'],
            y=df[acc_col],
            name='Accuracy',
            yaxis='y1',
            line=dict(color='green')
        ))

        fig_corr.add_trace(go.Scatter(
            x=df['step'],
            y=df[dim_col],
            name='Effective Dim Ratio',
            yaxis='y2',
            line=dict(color='orange', dash='dash')
        ))

        fig_corr.update_layout(
            title="Accuracy vs Effective Dimensionality (Compression)",
            yaxis=dict(title='Accuracy', side='left', range=[0, 1]),
            yaxis2=dict(title='Dim Ratio (lower=compressed)', side='right', overlaying='y', range=[0, 1]),
            height=350
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption("When accuracy rises AND dim ratio falls, the model is developing compressed, generalizing representations.")
    else:
        st.info("Waiting for accuracy and dimensionality metrics...")


# === Page: Numerical Stability ===
def page_numerical_stability(df: pd.DataFrame, latest: Dict[str, Any]):
    """
    Page for numerical stability monitoring.

    Based on "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697v2)
    """
    st.header("Numerical Stability")

    # Check if stability metrics are available
    stability_cols = [c for c in df.columns if c.startswith('stability/')]

    if not stability_cols:
        st.warning("Numerical stability metrics not detected. These metrics will appear once integrated into training.")
        st.info("""
**Numerical Stability Metrics** (from arXiv:2501.04697v2):
- **Softmax Collapse (SC)**: Detects floating-point absorption in softmax
- **NLM Gradient Alignment**: Detects naïve loss minimization (weight scaling)

These metrics help understand WHY grokking happens and when the model
is at risk of numerical instability that blocks learning.
        """)
        return

    # Current stability status
    st.subheader("Current Stability Status")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        sc_risk = latest.get('stability/sc_risk', 'unknown')
        risk_colors = {'low': '🟢', 'medium': '🟡', 'high': '🟠', 'critical': '🔴'}
        risk_icon = risk_colors.get(sc_risk, '⚪')
        st.metric("SC Risk", f"{risk_icon} {sc_risk}")

    with col2:
        sc_frac = latest.get('stability/sc_fraction', 0)
        st.metric("SC Fraction", f"{sc_frac:.1%}")

    with col3:
        nlm_active = latest.get('stability/nlm_active', False)
        nlm_icon = '🔴' if nlm_active else '🟢'
        st.metric("NLM Active", f"{nlm_icon} {'Yes' if nlm_active else 'No'}")

    with col4:
        grad_cosine = latest.get('stability/grad_weight_cosine', 0)
        st.metric("Grad-Weight Cosine", f"{grad_cosine:.3f}")

    st.divider()

    # Stability metrics over time
    st.subheader("Stability Metrics Over Time")

    # Filter to rows with stability data
    stability_df = df[df['stability/sc_fraction'].notna()].copy() if 'stability/sc_fraction' in df.columns else pd.DataFrame()

    if not stability_df.empty:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Softmax Collapse Fraction',
                'Gradient-Weight Cosine',
                'Weight Norm',
                'Max Logit Mean'
            ]
        )

        # SC Fraction
        if 'stability/sc_fraction' in stability_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=stability_df['step'],
                    y=stability_df['stability/sc_fraction'],
                    name='SC Fraction',
                    line=dict(color='red')
                ),
                row=1, col=1
            )
            fig.add_hline(y=0.1, line_dash="dash", line_color="orange",
                         row=1, col=1, annotation_text="Warning")
            fig.add_hline(y=0.5, line_dash="dash", line_color="red",
                         row=1, col=1, annotation_text="Critical")

        # Gradient-weight cosine
        if 'stability/grad_weight_cosine' in stability_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=stability_df['step'],
                    y=stability_df['stability/grad_weight_cosine'],
                    name='Grad-Weight Cosine',
                    line=dict(color='purple')
                ),
                row=1, col=2
            )
            fig.add_hline(y=0.7, line_dash="dash", line_color="orange",
                         row=1, col=2, annotation_text="NLM Threshold")

        # Weight norm
        if 'stability/weight_norm' in stability_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=stability_df['step'],
                    y=stability_df['stability/weight_norm'],
                    name='Weight Norm',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )

        # Max logit mean
        if 'stability/max_logit_mean' in stability_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=stability_df['step'],
                    y=stability_df['stability/max_logit_mean'],
                    name='Max Logit',
                    line=dict(color='green')
                ),
                row=2, col=2
            )

        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Interpretation guide
    st.subheader("Interpretation Guide")

    with st.expander("Understanding Numerical Stability (from arXiv:2501.04697v2)"):
        st.markdown("""
**Softmax Collapse (SC):**
When logits become very large, softmax experiences floating-point absorption errors:
- `sum(exp(z_k)) ≈ exp(z_max)` - smaller terms vanish
- Gradients for correctly classified samples become zero
- Learning stops completely

**Naïve Loss Minimization (NLM):**
After 100% train accuracy, gradients align with weight direction:
- Model scales up logits to reduce loss (without learning)
- Eventually triggers Softmax Collapse
- Cosine similarity > 0.7 indicates NLM is active

**Why Weight Decay Causes Grokking:**
- Weight decay fights against NLM's logit scaling
- Keeps model in numerically stable zone
- Forces model to find generalizing circuits

**Solutions (available in this codebase):**
- **StableMax**: Numerically stable softmax replacement
- **PerpGrad Optimizer**: Projects out weight-aligned gradients -> immediate generalization

**References:**
- Doshi et al. (2025): "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697v2)
- Power et al. (2022): "Grokking: Generalization Beyond Overfitting"
        """)


# === Page: Phase Detection ===
def page_phase_detection(df: pd.DataFrame, latest: Dict[str, Any]):
    """Phase detection and train/val comparison page."""
    st.title("📊 Phase Detection")

    # Check if phase detection is enabled
    if 'detected_phase' not in df.columns and 'detected_phase' not in latest:
        st.warning("Phase detection not enabled in training config.")
        st.info("""
To enable phase detection, add to your config:
```yaml
training:
  validation_ratio: 0.15
  validation_interval: 500
  phase_detection_enabled: true
```
        """)
        return

    # Current phase indicator
    st.subheader("Current Training Phase")
    detected_phase = latest.get('detected_phase', 'unknown')
    render_phase_indicator(detected_phase)

    st.divider()

    # Train vs Validation comparison
    st.subheader("Train vs Validation Accuracy")

    has_val = 'val_accuracy' in df.columns
    has_train = 'masked_word_accuracy' in df.columns or 'train_accuracy' in df.columns

    if has_train and has_val:
        train_col = 'masked_word_accuracy' if 'masked_word_accuracy' in df.columns else 'train_accuracy'

        # Current metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            train_acc = latest.get(train_col, 0)
            st.metric("Train Accuracy", f"{train_acc:.1%}")

        with col2:
            val_acc = latest.get('val_accuracy', 0)
            st.metric("Val Accuracy", f"{val_acc:.1%}")

        with col3:
            gap = train_acc - val_acc
            gap_color = "normal" if gap < 0.1 else "inverse"
            st.metric("Train-Val Gap", f"{gap:+.1%}", delta_color=gap_color)

        # Accuracy comparison chart
        fig_acc = go.Figure()

        # Filter to rows with validation data
        val_df = df[df['val_accuracy'].notna()].copy() if 'val_accuracy' in df.columns else df

        fig_acc.add_trace(go.Scatter(
            x=val_df['step'],
            y=val_df[train_col],
            name='Train Accuracy',
            line=dict(color='blue')
        ))

        if 'val_accuracy' in val_df.columns:
            fig_acc.add_trace(go.Scatter(
                x=val_df['step'],
                y=val_df['val_accuracy'],
                name='Val Accuracy',
                line=dict(color='orange')
            ))

        fig_acc.update_layout(
            title="Train vs Validation Accuracy Over Time",
            yaxis=dict(title='Accuracy', range=[0, 1]),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )

        st.plotly_chart(fig_acc, use_container_width=True)

        # Overfitting gap trend
        st.subheader("Overfitting Gap Trend")

        if 'val_accuracy' in val_df.columns:
            val_df['gap'] = val_df[train_col] - val_df['val_accuracy']

            fig_gap = px.line(val_df, x='step', y='gap', title="Train-Val Gap Over Time")
            fig_gap.add_hline(y=0, line_dash="solid", line_color="gray")
            fig_gap.add_hline(y=0.1, line_dash="dash", line_color="orange",
                             annotation_text="Warning: Gap > 10%")
            fig_gap.add_hline(y=0.2, line_dash="dash", line_color="red",
                             annotation_text="Critical: Gap > 20%")
            fig_gap.update_layout(height=300)
            st.plotly_chart(fig_gap, use_container_width=True)
    else:
        st.info("Validation metrics not available. Enable validation split in training config.")

    st.divider()

    # Phase detection metrics
    st.subheader("Phase Detection Metrics")

    col1, col2 = st.columns(2)

    with col1:
        # Loss comparison
        if 'train_loss' in df.columns or 'storage_loss_mean' in df.columns:
            loss_col = 'train_loss' if 'train_loss' in df.columns else 'storage_loss_mean'

            fig_loss = go.Figure()

            fig_loss.add_trace(go.Scatter(
                x=df['step'],
                y=df[loss_col],
                name='Train Loss',
                line=dict(color='blue')
            ))

            if 'val_loss' in df.columns:
                val_df = df[df['val_loss'].notna()]
                fig_loss.add_trace(go.Scatter(
                    x=val_df['step'],
                    y=val_df['val_loss'],
                    name='Val Loss',
                    line=dict(color='orange')
                ))

            fig_loss.update_layout(
                title="Train vs Validation Loss",
                yaxis=dict(title='Loss'),
                height=300
            )
            st.plotly_chart(fig_loss, use_container_width=True)

    with col2:
        # Effective dimension ratio
        if 'grokking/embedding_effective_dim_ratio' in df.columns:
            fig_dim = px.line(
                df[df['grokking/embedding_effective_dim_ratio'].notna()],
                x='step',
                y='grokking/embedding_effective_dim_ratio',
                title="Effective Dimension Ratio"
            )
            fig_dim.add_hline(y=0.3, line_dash="dash", line_color="green",
                             annotation_text="Grokking threshold")
            fig_dim.update_layout(height=300)
            st.plotly_chart(fig_dim, use_container_width=True)
        else:
            st.info("Effective dimension metrics not available.")

    st.divider()

    # Phase transition history
    st.subheader("Phase Transition History")

    # Look for phase transitions in the data
    if 'detected_phase' in df.columns:
        phase_df = df[['step', 'detected_phase']].dropna()

        if not phase_df.empty:
            # Detect transitions
            phase_df['prev_phase'] = phase_df['detected_phase'].shift(1)
            transitions = phase_df[phase_df['detected_phase'] != phase_df['prev_phase']].copy()

            if not transitions.empty:
                transitions = transitions[['step', 'prev_phase', 'detected_phase']].rename(
                    columns={'prev_phase': 'From', 'detected_phase': 'To'}
                )

                st.dataframe(transitions, use_container_width=True)

                # Timeline visualization
                phase_colors = {
                    'memory_learning': 'blue',
                    'converged': 'green',
                    'overfitting': 'orange',
                    'grokking': 'purple',
                    'unknown': 'gray'
                }

                fig_timeline = go.Figure()

                for phase, color in phase_colors.items():
                    phase_steps = phase_df[phase_df['detected_phase'] == phase]['step']
                    if not phase_steps.empty:
                        fig_timeline.add_trace(go.Scatter(
                            x=phase_steps,
                            y=[phase] * len(phase_steps),
                            mode='markers',
                            name=phase.replace('_', ' ').title(),
                            marker=dict(color=color, size=8)
                        ))

                fig_timeline.update_layout(
                    title="Phase Timeline",
                    xaxis_title="Step",
                    yaxis_title="Phase",
                    height=250
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No phase transitions detected yet.")
        else:
            st.info("No phase detection data available yet.")
    else:
        st.info("Phase detection data not found in metrics.")

    st.divider()

    # Interpretation guide
    with st.expander("Understanding Phase Detection"):
        st.markdown("""
**Training Phases:**

- 🔵 **Memory Learning**: Model is actively learning
  - Loss is decreasing OR accuracy is increasing
  - Healthy gradient signals

- 🟢 **Converged**: Training has plateaued
  - Low variance in metrics over convergence window
  - May be at any accuracy level

- 🟠 **Overfitting**: Train/val gap is growing
  - Train accuracy increasing while val accuracy decreases
  - Consider early stopping or more regularization

- 🟣 **Grokking**: Generalization breakthrough
  - Val accuracy recovering after overfitting period
  - Effective dimension stabilizing
  - Model has learned generalizing representations

**Key Thresholds:**
- Convergence: Variance < 0.001 over 2000 steps
- Overfitting: Train-val gap growing at > 0.001 per step
- Grokking: Val accuracy improves > 5% with stable eff_dim
        """)


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

    # Streamlit handles args differently
    try:
        args = parser.parse_args()
    except SystemExit:
        args = argparse.Namespace(
            metrics_path='runs/experiment/metrics_stream.jsonl',
            max_steps=100000,
            refresh_interval=30
        )

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
        ["Live Overview", "Phase Detection", "Memory Deep Dive", "Grokking & Geometry", "Numerical Stability", "Alerts & Health"]
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Last update: {time.strftime('%H:%M:%S')}")
    st.sidebar.caption(f"Loaded {len(df):,} data points")

    # Render selected page
    if page == "Live Overview":
        page_live_overview(df, latest)
    elif page == "Phase Detection":
        page_phase_detection(df, latest)
    elif page == "Memory Deep Dive":
        page_memory_deep_dive(df, latest)
    elif page == "Grokking & Geometry":
        page_grokking_geometry(df, latest)
    elif page == "Numerical Stability":
        page_numerical_stability(df, latest)
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

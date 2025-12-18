#!/usr/bin/env python3
"""
Launch the Streamlit training dashboard.

Usage:
    python scripts/launch_dashboard.py --metrics-path runs/atlas_389m_episodic/metrics_stream.jsonl

Or directly:
    streamlit run training_framework/monitoring/streamlit_monitor.py -- --metrics-path <path>
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Launch Atlas training dashboard")
    parser.add_argument(
        '--metrics-path',
        type=str,
        default='runs/experiment/metrics_stream.jsonl',
        help='Path to metrics JSONL file'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=100000,
        help='Maximum training steps (for progress bar)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run dashboard on'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host to bind to (0.0.0.0 for remote access)'
    )

    args = parser.parse_args()

    # Find the streamlit monitor script
    script_dir = Path(__file__).parent.parent
    monitor_script = script_dir / 'training_framework' / 'monitoring' / 'streamlit_monitor.py'

    if not monitor_script.exists():
        print(f"Error: Monitor script not found at {monitor_script}")
        sys.exit(1)

    # Build command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(monitor_script),
        '--server.port', str(args.port),
        '--server.address', args.host,
        '--',
        '--metrics-path', args.metrics_path,
        '--max-steps', str(args.max_steps),
    ]

    print(f"Starting dashboard...")
    print(f"  Metrics: {args.metrics_path}")
    print(f"  URL: http://{args.host}:{args.port}")
    print()

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


if __name__ == '__main__':
    main()

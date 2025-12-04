"""
GAADP Benchmarks Package
========================
Provides rigorous metrics and trajectory analysis for measuring
system behavior over time.

Modules:
    - metrics: Graph physics calculations (TOVS, ICR, GCR, DSR)
    - trajectory: Execution path analysis from telemetry
"""
from .metrics import GraphMetrics, calculate_metrics, print_metrics_report
from .trajectory import TrajectoryAnalyzer, analyze_telemetry, print_trajectory_report

__all__ = [
    "GraphMetrics",
    "calculate_metrics",
    "print_metrics_report",
    "TrajectoryAnalyzer",
    "analyze_telemetry",
    "print_trajectory_report"
]

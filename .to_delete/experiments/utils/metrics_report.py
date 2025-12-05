#!/usr/bin/env python3
"""
METRICS REPORT CLI
Generate and display metrics reports for GAADP execution analysis.
"""
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.metrics import MetricsCollector


def format_percentage(value: float) -> str:
    """Format a float as percentage."""
    return f"{value * 100:.1f}%"


def print_summary_report(metrics: MetricsCollector):
    """Print a human-readable summary report."""
    summary = metrics.get_summary_report()

    print("=" * 80)
    print("GAADP METRICS REPORT")
    print("=" * 80)
    print(f"\nSession Start: {summary['session_start']}")
    print(f"Total Nodes: {summary['total_nodes']}")

    # Status Breakdown
    print("\n" + "─" * 80)
    print("STATUS BREAKDOWN")
    print("─" * 80)
    status_breakdown = summary['status_breakdown']
    for status, count in status_breakdown.items():
        percentage = count / summary['total_nodes'] * 100 if summary['total_nodes'] > 0 else 0
        print(f"  {status:15s}: {count:4d} ({percentage:5.1f}%)")

    # Success Rates by Type
    print("\n" + "─" * 80)
    print("SUCCESS RATES BY NODE TYPE")
    print("─" * 80)
    for node_type, rate in summary['success_rates_by_type'].items():
        print(f"  {node_type:15s}: {format_percentage(rate)}")

    # Top Status Transitions
    print("\n" + "─" * 80)
    print("TOP STATUS TRANSITIONS")
    print("─" * 80)
    sorted_transitions = sorted(
        summary['status_transitions'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for transition, count in sorted_transitions:
        print(f"  {transition:30s}: {count:4d}")

    # Failure Reasons
    if summary['failure_reasons']:
        print("\n" + "─" * 80)
        print("TOP FAILURE REASONS")
        print("─" * 80)
        for reason, count in list(summary['failure_reasons'].items())[:10]:
            print(f"  {count:4d}x  {reason}")

    # Retry Stats
    print("\n" + "─" * 80)
    print("RETRY STATISTICS")
    print("─" * 80)
    retry_stats = summary['retry_stats']
    print(f"  Total Retries:   {retry_stats['total_retries']}")
    print(f"  Average Retries: {retry_stats['avg_retries']:.2f}")
    print(f"  Max Retries:     {retry_stats['max_retries']}")

    # Performance
    print("\n" + "─" * 80)
    print("PERFORMANCE METRICS")
    print("─" * 80)
    perf = summary['performance']
    print(f"  Avg Processing Time: {perf['avg_processing_time']:.2f}s")
    print(f"  Avg Token Count:     {perf['avg_token_count']:.0f} tokens")

    # Consensus
    if summary['consensus']['verdicts']:
        print("\n" + "─" * 80)
        print("CONSENSUS RESULTS")
        print("─" * 80)
        for verdict, count in summary['consensus']['verdicts'].items():
            print(f"  {verdict:20s}: {count:4d}")
        print(f"\n  Avg Agreement Rate: {format_percentage(summary['consensus']['avg_agreement_rate'])}")

    # Context Pruning
    pruning = summary['context_pruning']
    if pruning['avg_context_size'] > 0:
        print("\n" + "─" * 80)
        print("CONTEXT PRUNING EFFICIENCY")
        print("─" * 80)
        print(f"  Avg Pruning Ratio: {format_percentage(pruning['avg_pruning_ratio'])}")
        print(f"  Avg Context Size:  {pruning['avg_context_size']:.1f} nodes")

    # Failure Patterns
    if summary['failure_patterns']:
        print("\n" + "─" * 80)
        print("FAILURE PATTERNS (with traceability)")
        print("─" * 80)
        for pattern in summary['failure_patterns'][:5]:
            print(f"\n  Category: {pattern['category']}")
            print(f"  Count: {pattern['count']}")
            print(f"  Node Types: {', '.join(pattern['node_types'])}")
            print(f"  Avg Retries: {pattern['avg_retry_count']:.1f}")
            if pattern.get('agent_roles'):
                print(f"  Agent Roles: {', '.join(pattern['agent_roles'])}")
            if pattern.get('operations'):
                print(f"  Operations: {', '.join(pattern['operations'])}")
            if pattern.get('avg_context_size', 0) > 0:
                print(f"  Avg Context Size: {pattern['avg_context_size']:.1f} nodes")
            if pattern.get('context_token_ranges'):
                print(f"  Context Tokens: {pattern['context_token_ranges']}")
            if pattern['examples']:
                print(f"  Examples: {', '.join(pattern['examples'][:3])}")

    # Success Patterns
    if summary['success_patterns']:
        print("\n" + "─" * 80)
        print("SUCCESS PATTERNS")
        print("─" * 80)
        for pattern in summary['success_patterns']:
            print(f"\n  Node Type: {pattern['node_type']}")
            print(f"  Count: {pattern['count']}")
            print(f"  Avg Processing Time: {pattern['avg_processing_time']:.2f}s")
            print(f"  Avg Token Count: {pattern['avg_token_count']:.0f} tokens")
            if pattern['avg_verifier_count'] > 0:
                print(f"  Avg Verifiers: {pattern['avg_verifier_count']:.1f}")

    print("\n" + "=" * 80)


def load_metrics_from_file(filepath: str) -> Optional[MetricsCollector]:
    """Load metrics from a JSON export file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct MetricsCollector from exported data
        metrics = MetricsCollector()

        # Import node metrics
        from infrastructure.metrics import NodeMetric
        for node_id, metric_data in data.get('node_metrics', {}).items():
            metrics.node_metrics[node_id] = NodeMetric(**metric_data)

        # Import aggregated data
        summary = data.get('summary', {})
        if summary:
            metrics.status_transitions = dict(summary.get('status_transitions', {}))
            metrics.failure_reasons = dict(summary.get('failure_reasons', {}))

        return metrics

    except Exception as e:
        print(f"Error loading metrics from {filepath}: {e}", file=sys.stderr)
        return None


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate metrics reports for GAADP execution analysis"
    )
    parser.add_argument(
        'metrics_file',
        nargs='?',
        help='Path to exported metrics JSON file (optional)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output JSON instead of formatted report'
    )
    parser.add_argument(
        '--requirement',
        help='Show traceability report for specific requirement ID'
    )
    parser.add_argument(
        '--agent-role',
        help='Filter by agent role (e.g., builder, architect, verifier)'
    )
    parser.add_argument(
        '--operation',
        help='Filter by operation (e.g., building, verifying, materializing)'
    )

    args = parser.parse_args()

    # Load metrics
    if args.metrics_file:
        metrics = load_metrics_from_file(args.metrics_file)
        if not metrics:
            return 1
    else:
        print("No metrics file provided. Use during live execution or provide exported metrics file.")
        return 1

    # Generate report based on options
    if args.requirement:
        # Traceability report for specific requirement
        report = metrics.get_traceability_report(args.requirement)
        print(json.dumps(report, indent=2, default=str))

    elif args.agent_role or args.operation:
        # Query by agent/operation
        results = metrics.query_by_agent(
            agent_role=args.agent_role,
            operation=args.operation
        )
        print(f"\nFound {len(results)} nodes matching filters:")
        print(f"  Agent Role: {args.agent_role or 'any'}")
        print(f"  Operation: {args.operation or 'any'}")
        print(f"\nResults:")
        for m in results[:20]:  # Show first 20
            print(f"  - {m.node_id} [{m.node_type}] {m.status}")
            if m.failure_reason:
                print(f"    Failure: {m.failure_reason}")

    elif args.json:
        # Full summary as JSON
        summary = metrics.get_summary_report()
        print(json.dumps(summary, indent=2, default=str))

    else:
        # Human-readable summary
        print_summary_report(metrics)

    return 0


if __name__ == '__main__':
    sys.exit(main())

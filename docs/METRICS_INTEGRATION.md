# Metrics System Integration Guide

## Overview

The GAADP metrics system provides comprehensive tracking of execution patterns, failures, and performance. It automatically collects data through the event bus and provides analysis tools for testing and debugging.

## Architecture

```
┌─────────────────┐      Events       ┌─────────────────────┐
│   GraphDB       │─────────────────>│   Event Bus         │
│   Consensus     │                   │                     │
│   Feedback      │                   └─────────┬───────────┘
└─────────────────┘                             │
                                                │ Subscribe
                                                ▼
                                   ┌────────────────────────┐
                                   │ MetricsSubscriber      │
                                   │ (Auto-collect events)  │
                                   └────────┬───────────────┘
                                            │
                                            ▼
                                   ┌────────────────────────┐
                                   │ MetricsCollector       │
                                   │ - Node metrics         │
                                   │ - Failure patterns     │
                                   │ - Success patterns     │
                                   │ - Performance data     │
                                   └────────────────────────┘
```

## Integration Steps

### 1. Initialize Metrics Collector

```python
from infrastructure import MetricsCollector, MetricsSubscriber, EventBus, GraphDB

# Create event bus
event_bus = EventBus()

# Create metrics collector
metrics = MetricsCollector()

# Subscribe to event bus (automatic collection)
subscriber = MetricsSubscriber(metrics, event_bus)

# Create GraphDB with event bus
db = GraphDB(
    persistence_path=".gaadp/graph.json",
    event_bus=event_bus
)
```

### 2. Events Automatically Tracked

The system tracks these events:

- **NODE_CREATED**: New nodes added to graph
- **STATUS_CHANGED**: Node status transitions
- **RETRY_ATTEMPT**: Retry attempts on failures
- **CONSENSUS_ACHIEVED**: Verification consensus results
- **CONTEXT_PRUNED**: Context pruning effectiveness

### 3. Query Metrics During Execution

```python
# Get summary report
summary = metrics.get_summary_report()
print(f"Total nodes: {summary['total_nodes']}")
print(f"Success rate: {summary['success_rates_by_type']}")

# Query specific metrics
failed_codes = metrics.query_metrics(
    node_type="CODE",
    status="FAILED"
)

# Get failure patterns
patterns = metrics.get_failure_patterns()
for pattern in patterns:
    print(f"Failure: {pattern.category} ({pattern.count}x)")
    print(f"  Avg retries: {pattern.avg_retry_count}")
```

### 4. Export Metrics for Analysis

```python
# Export detailed report to file
metrics.export_detailed_report(".gaadp/metrics_report.json")
```

### 5. Generate Human-Readable Reports

```bash
# View formatted report
python utils/metrics_report.py .gaadp/metrics_report.json

# Export as JSON for further analysis
python utils/metrics_report.py .gaadp/metrics_report.json --json > analysis.json
```

## Metrics Collected

### Node Metrics
- Node ID, type, status
- Creation and update timestamps
- Processing time
- Token count
- Retry count
- Failure reason
- Verifier count
- Consensus verdict

### Failure Patterns
- Failure categories and reasons
- Affected node types
- Average retry counts
- Example nodes

### Success Patterns
- Success counts by node type
- Average processing times
- Average token usage
- Average verifier counts

### Performance Metrics
- Processing times by node type
- Token usage patterns
- Context pruning efficiency
- Consensus agreement rates

## Example Report Output

```
================================================================================
GAADP METRICS REPORT
================================================================================

Session Start: 2025-01-15T10:30:00.000000
Total Nodes: 150

────────────────────────────────────────────────────────────────────────────────
STATUS BREAKDOWN
────────────────────────────────────────────────────────────────────────────────
  VERIFIED       :   85 ( 56.7%)
  FAILED         :   45 ( 30.0%)
  PENDING        :   15 ( 10.0%)
  IN_PROGRESS    :    5 (  3.3%)

────────────────────────────────────────────────────────────────────────────────
SUCCESS RATES BY NODE TYPE
────────────────────────────────────────────────────────────────────────────────
  REQ            : 100.0%
  SPEC           : 85.0%
  CODE           : 45.0%
  TEST           : 60.0%

────────────────────────────────────────────────────────────────────────────────
TOP FAILURE REASONS
────────────────────────────────────────────────────────────────────────────────
   12x  Syntax error in generated code
    8x  Type mismatch in implementation
    6x  Missing import statements
    5x  Incomplete implementation
```

## Best Practices

1. **Export metrics regularly** during long runs
2. **Review failure patterns** to identify systemic issues
3. **Monitor retry counts** to catch infinite loops
4. **Track success rates** to validate improvements
5. **Use metrics to guide architecture decisions**

## Integration with GADPEngine

```python
class GADPEngine:
    def __init__(self, ...):
        self.event_bus = EventBus()
        self.metrics = MetricsCollector()
        self.metrics_subscriber = MetricsSubscriber(self.metrics, self.event_bus)

        self.db = GraphDB(
            persistence_path=persistence_path,
            event_bus=self.event_bus
        )

        # ... rest of initialization

    def get_metrics(self):
        """Expose metrics for external queries."""
        return self.metrics

    def export_metrics(self, filepath: str):
        """Export metrics to file."""
        self.metrics.export_detailed_report(filepath)
```

## Testing Support

The metrics system is designed to help identify:

- **Bottlenecks**: Which node types take longest?
- **Failure modes**: What causes the most failures?
- **Retry patterns**: Are we stuck in retry loops?
- **Context effectiveness**: Is semantic pruning working?
- **Consensus quality**: Are verifiers agreeing?

Use these insights to iterate and improve the system!

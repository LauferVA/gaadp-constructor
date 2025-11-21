# Traceability-Enhanced Metrics System

## Overview

The enhanced metrics system provides **complete traceability** from research topics → implementation decisions → agent operations → failures/successes → git commits.

This enables:
- **Root cause analysis**: "Why did this CODE node fail?" → Trace back to REQ/SPEC
- **Pattern detection**: "Which requirements have the most failures?"
- **Agent performance**: "Which agent/operation has highest failure rate?"
- **Learning signals**: Full context for adaptive improvement

## Architecture

```
REQ (research topic)
 │
 ├─ SPEC (implementation decision)
 │   │
 │   ├─ CODE (implementation)
 │   │   │
 │   │   ├─ Agent: builder_1
 │   │   ├─ Operation: building
 │   │   ├─ Context: [req, spec, other_code]
 │   │   ├─ Status: FAILED
 │   │   ├─ Reason: "Missing import"
 │   │   └─ Git: (none)
 │   │
 │   └─ CODE (retry)
 │       │
 │       ├─ Agent: builder_1
 │       ├─ Operation: building
 │       ├─ Context: [req, spec, feedback]
 │       ├─ Status: VERIFIED
 │       └─ Git: commit abc123, files: [main.py]
```

## Enhanced NodeMetric Fields

```python
@dataclass
class NodeMetric:
    # Basic fields
    node_id: str
    node_type: str
    status: str
    created_at: str

    # Agent attribution
    agent_id: Optional[str]           # e.g., "builder_1"
    agent_role: Optional[str]         # e.g., "builder", "architect"
    operation: Optional[str]          # e.g., "building", "verifying"

    # Traceability chain
    traces_to_req: Optional[str]      # REQ node this traces back to
    traces_to_spec: Optional[str]     # SPEC node this implements
    implements_spec: Optional[str]    # Direct IMPLEMENTS edge target
    depends_on: List[str]             # DEPENDS_ON nodes

    # Context snapshot
    context_node_ids: List[str]       # Nodes in context
    context_token_count: int          # Token count
    context_pruning_ratio: str        # "15/50" (selected/considered)

    # Git linkage
    materialized_commit: Optional[str]  # Git commit hash
    materialized_files: List[str]       # Files materialized
    materialization_status: str         # "success"/"failed"
```

## Usage Examples

### 1. Recording Traceability During Node Creation

```python
from utils.traceability_helper import TraceabilityHelper

# When creating a CODE node from a SPEC
spec_id = "spec_abc123"
code_id = "code_xyz789"

# Extract traceability
lineage = TraceabilityHelper.get_traceability_for_new_node(
    graph=db.graph,
    parent_node_id=spec_id,
    implements_node_id=spec_id
)

# Emit NODE_CREATED event with traceability
await event_bus.publish(
    topic="node_lifecycle",
    message_type="NODE_CREATED",
    payload={
        "node_id": code_id,
        "node_type": "CODE",
        "agent_id": "builder_1",
        "agent_role": "builder",
        "traces_to_req": lineage['traces_to_req'],
        "traces_to_spec": lineage['traces_to_spec'],
        "implements_spec": lineage['implements_spec'],
        "depends_on": lineage['depends_on']
    },
    source_id="builder"
)
```

### 2. Recording Operations

```python
# Record what operation is being performed
await event_bus.publish(
    topic="performance",
    message_type="OPERATION",
    payload={
        "node_id": code_id,
        "operation": "building",
        "agent_id": "builder_1",
        "agent_role": "builder"
    },
    source_id="builder"
)
```

### 3. Recording Context Snapshots

```python
# After getting context for agent
context = db.get_context_neighborhood(code_id, radius=2)

await event_bus.publish(
    topic="context",
    message_type="CONTEXT_SNAPSHOT",
    payload={
        "node_id": code_id,
        "context_node_ids": [n['id'] for n in context['nodes']],
        "context_token_count": context['_token_stats']['total_tokens'],
        "pruning_ratio": context['_token_stats']['pruning_ratio']
    },
    source_id="builder"
)
```

### 4. Recording Materialization

```python
# After git commit
await event_bus.publish(
    topic="performance",
    message_type="MATERIALIZATION",
    payload={
        "node_id": code_id,
        "commit": commit_hash,
        "files": ["src/main.py", "tests/test_main.py"],
        "status": "success"
    },
    source_id="materializer"
)
```

## Query Examples

### Query by Requirement

```python
# Get all nodes that trace to a specific requirement
metrics = engine.get_metrics()

# Full traceability report
report = metrics.get_traceability_report("req_auth_system")

print(f"Requirement: {report['requirement_id']}")
print(f"Total nodes: {report['total_nodes']}")
print(f"Verified: {report['status_breakdown']['verified']}")
print(f"Failed: {report['status_breakdown']['failed']}")

# Show failures with full context
for failure in report['failures']:
    print(f"\nFailed: {failure['node_id']}")
    print(f"  Type: {failure['node_type']}")
    print(f"  Reason: {failure['reason']}")
    print(f"  Agent: {failure['agent_role']}")
    print(f"  Operation: {failure['operation']}")
    print(f"  Retries: {failure['retry_count']}")
```

### Query by Agent

```python
# Get all failures from builder agent
failures = metrics.query_by_agent(
    agent_role="builder",
    status="FAILED"
)

print(f"Builder failures: {len(failures)}")
for f in failures:
    print(f"  {f.node_id}: {f.failure_reason}")
    print(f"    Traces to REQ: {f.traces_to_req}")
    print(f"    Context size: {len(f.context_node_ids)} nodes")
```

### Query by Operation

```python
# Get all failures during verification
verifier_failures = metrics.query_by_agent(
    operation="verifying",
    status="FAILED"
)

for f in verifier_failures:
    print(f"Verification failed: {f.node_id}")
    print(f"  Agent: {f.agent_id}")
    print(f"  Reason: {f.failure_reason}")
```

## CLI Usage

### View Full Report

```bash
python utils/metrics_report.py .gaadp/metrics_report.json
```

### Query by Requirement

```bash
# Show all nodes that trace to a requirement
python utils/metrics_report.py .gaadp/metrics_report.json \
  --requirement req_abc123
```

Output:
```json
{
  "requirement_id": "req_abc123",
  "total_nodes": 15,
  "status_breakdown": {
    "verified": 10,
    "failed": 3,
    "pending": 2
  },
  "failures": [
    {
      "node_id": "code_xyz",
      "node_type": "CODE",
      "reason": "Missing import for jwt library",
      "agent_role": "builder",
      "operation": "building",
      "retry_count": 2
    }
  ],
  "successes": [
    {
      "node_id": "code_abc",
      "node_type": "CODE",
      "processing_time": 5.2,
      "verifier_count": 2,
      "materialized_commit": "abc123def"
    }
  ]
}
```

### Filter by Agent

```bash
# Show all builder failures
python utils/metrics_report.py .gaadp/metrics_report.json \
  --agent-role builder \
  --operation building
```

## Failure Pattern Analysis

Enhanced failure patterns now include:

```python
@dataclass
class FailurePattern:
    category: str                    # Failure reason
    count: int                       # How many times
    node_types: List[str]           # Which node types
    avg_retry_count: float          # Average retries

    # NEW: Agent attribution
    agent_roles: List[str]          # Which agents hit this
    operations: List[str]           # During which operations

    # NEW: Context patterns
    avg_context_size: float         # Average context size
    context_token_ranges: str       # Token range
```

Example output:
```
FAILURE PATTERNS (with traceability)
────────────────────────────────────────────────────────────────────────────────

  Category: Missing import statements
  Count: 12
  Node Types: CODE
  Avg Retries: 1.8
  Agent Roles: builder
  Operations: building
  Avg Context Size: 3.2 nodes
  Context Tokens: 800-1500 tokens
  Examples: code_abc, code_def, code_ghi
```

## Integration with Agents

### Builder Agent Example

```python
class BuilderAgent:
    async def process(self, task):
        spec_id = task['spec_id']
        code_id = f"code_{uuid.uuid4().hex[:8]}"

        # Extract traceability
        lineage = TraceabilityHelper.get_traceability_for_new_node(
            graph=self.db.graph,
            parent_node_id=spec_id,
            implements_node_id=spec_id
        )

        # Create node with traceability
        self.db.add_node(
            code_id,
            NodeType.CODE,
            content="...",
            metadata={}
        )

        # Emit creation event with full traceability
        await self.event_bus.publish(
            topic="node_lifecycle",
            message_type="NODE_CREATED",
            payload={
                "node_id": code_id,
                "node_type": "CODE",
                "agent_id": self.agent_id,
                "agent_role": "builder",
                **lineage  # Includes traces_to_req, traces_to_spec, etc.
            },
            source_id=self.agent_id
        )

        # Record operation
        await self.event_bus.publish(
            topic="performance",
            message_type="OPERATION",
            payload={
                "node_id": code_id,
                "operation": "building",
                "agent_id": self.agent_id,
                "agent_role": "builder"
            },
            source_id=self.agent_id
        )

        # Build code...
        start_time = time.time()
        result = await self._build_code(spec_id, code_id)
        processing_time = time.time() - start_time

        # Record processing time
        await self.event_bus.publish(
            topic="performance",
            message_type="PROCESSING_COMPLETE",
            payload={
                "node_id": code_id,
                "processing_time": processing_time,
                "node_type": "CODE"
            },
            source_id=self.agent_id
        )

        return result
```

## Benefits for Testing & Learning

### 1. Root Cause Analysis
```python
# "Why is this requirement failing so much?"
report = metrics.get_traceability_report("req_problematic")

# See which SPECs/CODEs failed
# See which agents/operations failed
# See what context was provided
```

### 2. Agent Performance Tracking
```python
# "Which agent has the highest failure rate?"
builder_failures = metrics.query_by_agent(agent_role="builder", status="FAILED")
builder_successes = metrics.query_by_agent(agent_role="builder", status="VERIFIED")

failure_rate = len(builder_failures) / (len(builder_failures) + len(builder_successes))
```

### 3. Context Quality Analysis
```python
# "Do failures correlate with small context?"
failures = metrics.query_metrics(status="FAILED")
avg_context_size = sum(len(f.context_node_ids) for f in failures) / len(failures)

print(f"Failures average {avg_context_size} nodes in context")
```

### 4. Operation-Specific Issues
```python
# "Are verification failures different from build failures?"
build_failures = metrics.query_by_agent(operation="building", status="FAILED")
verify_failures = metrics.query_by_agent(operation="verifying", status="FAILED")

# Compare failure reasons
build_reasons = Counter(f.failure_reason for f in build_failures)
verify_reasons = Counter(f.failure_reason for f in verify_failures)
```

## Next Steps

With this traceability data, you can:

1. **Identify systemic issues**: "All failures in req_X trace to missing library Y"
2. **Improve agents**: "Builder fails when context <1000 tokens"
3. **Optimize context**: "Successful builds have 3-5 context nodes"
4. **Track progress**: "REQ completion rate by traceability"
5. **Enable adaptive learning**: Feed patterns back to agents

The full chain is now traceable:
```
Research Topic → Implementation Decision → Agent + Context → Success/Failure → Git Commit
      REQ      →         SPEC           →  CODE (w/ metrics) →   VERIFIED   →  abc123def
```

# GAADP Self-Generation Loop

## Overview

The GAADP system is designed to bootstrap itself. Once the core kernel is operational,
the system can generate its own missing infrastructure by treating feature requirements
as input nodes.

## The Bootstrap Process

### Phase 1: Seed Kernel (Complete)
- Core ontology and type system
- Graph database with persistence
- Base agent with cryptographic signing
- Architect, Builder, Verifier agents

### Phase 2: Self-Extension

To have GAADP build its own missing components:

```python
# Example: Have GAADP build a new infrastructure component
from infrastructure.graph_db import GraphDB
from agents.concrete_agents import RealArchitect, RealBuilder, RealVerifier
from core.ontology import NodeType, AgentRole

async def self_generate(feature_description: str):
    db = GraphDB()

    # 1. Inject the feature as a requirement
    req_id = f"req_self_{uuid.uuid4().hex[:8]}"
    db.add_node(req_id, NodeType.REQ, feature_description)

    # 2. Let Architect decompose it
    architect = RealArchitect("arch_self", AgentRole.ARCHITECT, db)
    plan = await architect.process({"nodes": [{"content": feature_description, "id": req_id}]})

    # 3. Let Builder implement each spec
    builder = RealBuilder("build_self", AgentRole.BUILDER, db)
    for node in plan.get('new_nodes', []):
        if node['type'] == 'SPEC':
            code = await builder.process({"nodes": [node]})
            # ... verification and materialization

    # 4. The generated code becomes part of GAADP itself
```

## Self-Generation Targets

The following components can be self-generated:

| Component | REQ Description |
|-----------|-----------------|
| Docker Wrapper | "Create a Python module that wraps Docker commands for container lifecycle management" |
| Neo4j Adapter | "Create a graph database adapter that uses Neo4j instead of NetworkX" |
| REST API | "Create a FastAPI server that exposes the graph database as a REST API" |
| Web Dashboard | "Create a React dashboard that visualizes the knowledge graph" |

## Safety Constraints

1. **Human Review Gate**: Self-generated code MUST pass through Verifier before integration
2. **Sandbox Execution**: All generated code runs in isolation first
3. **Rollback Capability**: Git integration enables reverting failed generations
4. **Budget Limits**: Treasurer halts if generation exceeds cost threshold

## Running the Self-Generation Loop

```bash
# Start with a feature request
python self_generate.py "Create a module for distributed graph synchronization"

# The system will:
# 1. Architect decomposes into specs
# 2. Builder generates code
# 3. Verifier reviews
# 4. Materializer writes to disk
# 5. Git commits the change
# 6. Tests run to validate
```

## Monitoring

Use the Curator agent to monitor graph health during self-generation:

```python
curator = RealCurator("curator_01", AgentRole.CURATOR, db)
health = await curator.process({})
print(f"Graph Health: {health['verdict']}")
```

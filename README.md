# GAADP: Graph-Augmented Autonomous Development Platform

A graph-native architecture for autonomous software generation with cryptographic verification, governance constraints, and human-in-the-loop controls.

## Overview

GAADP is a **factory for building software**. It treats code generation as a graph problem where:
- **Nodes** represent artifacts (requirements, specs, code, tests)
- **Edges** represent relationships (traces to, implements, verifies)
- **Agents** transform nodes according to strict rules
- **Governance** enforced as physics in node metadata

The key insight: **Policy as Physics, not Prompts**. Budget limits, security rules, and state transitions are enforced by the runtime via the TransitionMatrix, not by asking LLMs nicely.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CONTROL PLANE                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Claude Code │  │  MCP Server │  │   Human Loop Controller │ │
│  │  (Interactive)│  │  (API)      │  │   (Pause/Approve)       │ │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘ │
└─────────┼────────────────┼─────────────────────┼───────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GRAPH RUNTIME                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Transition   │  │  Generic     │  │  Visualization       │  │
│  │ Matrix       │  │  Agent       │  │  Server              │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       INFRASTRUCTURE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  GraphDB     │  │  LLM Gateway │  │  Materializer        │  │
│  │  (State)     │  │  (Providers) │  │  (Filesystem)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/LauferVA/gaadp-constructor.git
cd gaadp-constructor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Set API key (for standalone mode)
export ANTHROPIC_API_KEY="your-key"
# or
export OPENAI_API_KEY="your-key"
```

## Usage

### Mode 1: CLI with Real-Time Visualization

```bash
# Basic usage
python main.py "Create a REST API for user authentication"

# With real-time DAG visualization dashboard
python main.py --viz "Create a sorting algorithm"

# From file
python main.py -f requirements.txt --viz
```

The `--viz` flag opens a browser dashboard showing:
- Force-directed graph of nodes and edges
- Agent status (working/idle)
- Real-time event log
- Cost and iteration statistics

### Mode 2: Programmatic

```python
import asyncio
from infrastructure.graph_db import GraphDB
from infrastructure.graph_runtime import GraphRuntime
from infrastructure.llm_gateway import LLMGateway
from core.ontology import NodeType

async def main():
    graph_db = GraphDB(persistence_path=".gaadp/graph.json")
    gateway = LLMGateway()
    runtime = GraphRuntime(graph_db=graph_db, llm_gateway=gateway)

    # Create initial requirement
    import uuid
    req_id = uuid.uuid4().hex
    graph_db.add_node(
        node_id=req_id,
        node_type=NodeType.REQ,
        content="Create a function to validate email addresses",
        metadata={'cost_limit': 5.0}
    )

    # Run until complete
    stats = await runtime.run_until_complete(max_iterations=50)
    print(f"Processed {stats['nodes_processed']} nodes, cost: ${stats['total_cost']:.4f}")

asyncio.run(main())
```

### Mode 3: MCP Server (Claude Code Integration)

```bash
# Start the MCP server
python server.py
```

Configure in `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "gaadp-core": {
      "command": "python",
      "args": ["/path/to/gaadp-constructor/server.py"]
    }
  }
}
```

## Core Concepts

### Ontology

**Node Types:**
| Type | Description |
|------|-------------|
| `REQ` | Root requirement from user |
| `CLARIFICATION` | Questions/clarifications needed |
| `SPEC` | Atomic specification (decomposed from REQ) |
| `PLAN` | Execution plan |
| `CODE` | Implementation |
| `TEST` | Test cases |
| `DOC` | Documentation |
| `ESCALATION` | Issues requiring human intervention |

**Edge Types:**
| Type | Description |
|------|-------------|
| `TRACES_TO` | Provenance link (child → parent) |
| `DEPENDS_ON` | Dependency (blocked until target complete) |
| `IMPLEMENTS` | CODE implements SPEC |
| `VERIFIES` | Verification relationship |
| `FEEDBACK` | Critique from failed verification |

**Node Statuses:**
| Status | Description |
|--------|-------------|
| `PENDING` | Ready for processing |
| `PROCESSING` | Currently being worked on |
| `BLOCKED` | Waiting on dependency or governance |
| `VERIFIED` | Passed verification (terminal for CODE) |
| `FAILED` | Processing failed |

### TransitionMatrix (The Physics)

State transitions are defined in `core/ontology.py`:

```python
TRANSITION_MATRIX: Dict[Tuple[str, str], List[TransitionRule]] = {
    (NodeStatus.PENDING.value, NodeType.SPEC.value): [
        TransitionRule(
            target_status=NodeStatus.PROCESSING,
            required_conditions=["cost_under_limit", "dependencies_verified", "not_blocked"],
            priority=10
        ),
    ],
    # ... more rules
}
```

The runtime **cannot** violate these rules - they are physics, not suggestions.

### NodeMetadata (Governance as Data)

```python
class NodeMetadata(BaseModel):
    cost_limit: Optional[float] = None      # Max cost for this node
    cost_actual: float = 0.0                # Actual cost incurred
    security_level: int = 0                 # Required security clearance
    attempts: int = 0                       # Retry count
    max_attempts: int = 3                   # Max retries before failure
```

Budget and security are **embedded in the data**, not middleware that can be bypassed.

### GenericAgent

One universal agent class that loads personality from YAML:

```yaml
# config/agent_manifest.yaml
agents:
  architect:
    system_prompt_file: ".prompts/architect_system.md"
    output_protocol: "ArchitectOutput"
    context_radius: 3
    tools_allowed: ["read_file", "list_directory", "search_web"]
```

## Key Components

### GraphRuntime
The execution engine that consults TransitionMatrix and dispatches agents.

```python
runtime = GraphRuntime(graph_db=graph_db, llm_gateway=gateway)
stats = await runtime.run_until_complete(max_iterations=100)
```

### GraphDB
Enhanced graph database with 12+ query methods:
- `get_by_status()`, `get_by_type()`
- `dependencies_met()`, `is_blocked()`
- `get_children()`, `get_parents()`
- `topological_order()`, `get_execution_waves()`

### VizServer
WebSocket server for real-time visualization:

```python
from infrastructure.viz_server import start_viz_server
viz_server = await start_viz_server()
runtime = GraphRuntime(graph_db, gateway, viz_server=viz_server)
```

### GraphVisualizer
Export graph to various formats:

```python
viz = GraphVisualizer(db)
viz.export("graph.html", format="html")  # Interactive D3 visualization
viz.export("graph.dot", format="dot")    # Graphviz
viz.export("graph.md", format="mermaid") # Markdown diagrams
```

## Project Structure

```
gaadp-constructor/
├── main.py                     # CLI entry point (--viz flag)
├── server.py                   # MCP Server
├── core/
│   ├── ontology.py             # TransitionMatrix, NodeMetadata, enums
│   └── protocols.py            # UnifiedAgentOutput, GraphContext
├── agents/
│   └── generic_agent.py        # Universal agent (loads from YAML)
├── config/
│   └── agent_manifest.yaml     # Agent personalities
├── infrastructure/
│   ├── graph_db.py             # Persistence, queries
│   ├── graph_runtime.py        # Execution engine
│   ├── llm_gateway.py          # LLM abstraction
│   ├── llm_providers.py        # Anthropic, OpenAI, Claude SDK providers
│   ├── viz_server.py           # WebSocket visualization server
│   ├── viz_dashboard.html      # D3.js interactive dashboard
│   ├── materializer.py         # Atomic file writes
│   ├── checkpoint.py           # Save/restore state
│   └── visualizer.py           # Graph export (DOT, JSON, HTML)
├── orchestration/
│   ├── feedback.py             # Retry loops
│   ├── human_loop.py           # Pause points
│   ├── alerts.py               # Escalation
│   └── consensus.py            # Multi-agent consensus
├── .blueprint/
│   ├── prime_directives.md     # Immutable laws
│   └── topology_config.yaml    # Concurrency, RBAC
└── requirements.txt
```

## Design Principles

1. **Graph is Truth**: Filesystem is a view, not source of truth
2. **Policy as Physics**: Constraints enforced by TransitionMatrix, not prompts
3. **Fail Loud**: Invalid operations raise exceptions, not warnings
4. **Dual Mode**: Headless automation + interactive control
5. **Merkle Everything**: Cryptographic chain of custody
6. **Human Override**: Critical decisions can always pause for approval

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Ensure all tests pass: `pytest tests/`
4. Submit a pull request

## Acknowledgments

Built with Claude Code and the Claude Agent SDK architecture patterns.

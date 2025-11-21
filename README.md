# GAADP: Graph-Augmented Autonomous Development Platform

A graph-native architecture for autonomous software generation with cryptographic verification, governance constraints, and human-in-the-loop controls.

## Overview

GAADP is a **factory for building software**. It treats code generation as a graph problem where:
- **Nodes** represent artifacts (requirements, specs, code, tests)
- **Edges** represent relationships (traces to, implements, verifies)
- **Agents** transform nodes according to strict rules
- **Governance** enforces constraints at the infrastructure level

The key insight: **Policy as Physics, not Prompts**. Budget limits, security rules, and state transitions are enforced by the runtime, not by asking LLMs nicely.

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
│                       ORCHESTRATION                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Scheduler   │  │  Feedback    │  │  Alert Handler       │  │
│  │  (Workers)   │  │  Controller  │  │  (Escalation)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       GOVERNANCE (PHYSICS)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  Treasurer   │  │  Sentinel    │  │  Curator             │  │
│  │  (Budget)    │  │  (Security)  │  │  (Integrity)         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
          │                │                     │
          ▼                ▼                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       INFRASTRUCTURE                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  GraphDB     │  │  LLM Gateway │  │  Materializer        │  │
│  │  (State)     │  │  (litellm)   │  │  (Filesystem)        │  │
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

### Mode 1: Standalone Engine (Headless Automation)

Run the factory autonomously with litellm:

```python
import asyncio
from orchestration import GADPEngine, run_factory
from agents.concrete_agents import RealArchitect, RealBuilder, RealVerifier
from core.ontology import AgentRole

async def main():
    engine = GADPEngine(persistence_path=".gaadp/live_graph.pkl")

    # Register agents
    engine.register_agents({
        AgentRole.ARCHITECT: RealArchitect("arch_01", AgentRole.ARCHITECT, engine.db),
        AgentRole.BUILDER: RealBuilder("build_01", AgentRole.BUILDER, engine.db),
        AgentRole.VERIFIER: RealVerifier("verif_01", AgentRole.VERIFIER, engine.db),
    })

    # Inject requirement and run
    engine.inject_requirement("Create a REST API for user authentication with JWT tokens")
    await engine.run_until_complete(timeout=300)

    print(engine.get_status())

asyncio.run(main())
```

### Mode 2: MCP Server (Claude Code Integration)

Expose GAADP as tools for Claude Code:

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

Then in Claude Code, you can use tools like:
- `mcp__gaadp-core__create_node(node_type="SPEC", content="...")`
- `mcp__gaadp-core__link_nodes(source_id, target_id, edge_type="IMPLEMENTS")`
- `mcp__gaadp-core__run_verification_sandbox(code_node_id)`

### Mode 3: Interactive CLI

```bash
python production_main.py
```

## Core Concepts

### Ontology

**Node Types:**
| Type | Description |
|------|-------------|
| `REQ` | Root requirement from user |
| `SPEC` | Atomic specification (decomposed from REQ) |
| `PLAN` | Execution plan |
| `CODE` | Implementation |
| `TEST` | Test cases |
| `DOC` | Documentation |
| `DEAD_END` | Failed/abandoned node |

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
| `IN_PROGRESS` | Currently being worked on |
| `BLOCKED` | Waiting on dependency or governance |
| `COMPLETE` | Successfully processed |
| `VERIFIED` | Passed verification (terminal for CODE) |
| `FAILED` | Processing failed |

### Prime Directives

The 17 immutable laws in `.blueprint/prime_directives.md` govern all agent behavior:

1. Graph is source of truth, not filesystem
2. No node without parent edge
3. Cryptographic signatures on all edges
4. Merkle chaining for tamper detection
5. Context radius limits per role
6. Multi-agent verification required
7. ...and more

### Governance as Physics

Constraints are enforced at the infrastructure level:

```python
# This CANNOT exceed budget - the API call will fail
await treasurer.pre_hook(task)  # Returns False if over budget

# This CANNOT write insecure code - the DB rejects it
await sentinel.post_hook(task, result)  # Returns False if eval() detected

# This CANNOT create invalid state transitions
db.set_status(node_id, "VERIFIED")  # Raises StateTransitionError if invalid
```

## Configuration

### `.blueprint/topology_config.yaml`

```yaml
parallel_limits:
  max_concurrent_builders: 10
  max_concurrent_verifiers: 10
  max_wavefront_width: 20

tool_permissions:
  ARCHITECT:
    allowed_tools: ["read_file", "list_directory", "search_web"]
  BUILDER:
    allowed_tools: ["read_file", "write_file", "list_directory"]
  VERIFIER:
    allowed_tools: ["read_file", "list_directory"]
```

### `.blueprint/llm_router.yaml`

```yaml
model_assignments:
  ARCHITECT:
    model: "claude-3-sonnet-20240229"
    temperature: 0.7
  BUILDER:
    model: "claude-3-haiku-20240307"
    temperature: 0.3
  VERIFIER:
    model: "gpt-4-turbo"  # Different provider for diversity
    temperature: 0.1

cost_limits:
  project_total_limit_usd: 10.0
  per_node_limit_usd: 0.50
```

## Key Components

### TaskScheduler
Discovers ready nodes, dispatches to worker pool with concurrency limits.

```python
scheduler = TaskScheduler(db, event_bus)
scheduler.register_agent(AgentRole.BUILDER, builder)
await scheduler.start(num_workers=5)
```

### FeedbackController
Handles reject → replan → rebuild cycles.

```python
# When Verifier returns FAIL:
# 1. Creates FEEDBACK edge with critique
# 2. Resets SPEC to PENDING
# 3. Builder sees critique in context on retry
# 4. After N failures, escalates to Architect
```

### Governance Middleware
- **Treasurer**: Pre-hook checking budget before LLM calls
- **Sentinel**: Post-hook scanning code for security issues
- **Curator**: Background daemon pruning dead-ends, checking integrity

### AlertHandler
Subscribes to governance alerts, escalates critical issues to human.

```python
# CRITICAL alerts trigger human approval:
# - "Acknowledge & Continue"
# - "Pause Pipeline"
# - "Abort"
```

### AtomicMaterializer
Writes verified code to filesystem atomically.

```python
materializer = AtomicMaterializer(db, output_dir="./output")
result = materializer.materialize(dry_run=False)
# Stages to temp → validates → atomic move → git commit
```

### CheckpointManager
Save and restore execution state.

```python
checkpoint_mgr = CheckpointManager(db)
checkpoint_mgr.create_checkpoint("before_risky_change", "Pre-refactor snapshot")
# ... something goes wrong ...
checkpoint_mgr.restore_checkpoint("before_risky_change")
```

### GraphVisualizer
Export graph to various formats.

```python
viz = GraphVisualizer(db)
viz.export("graph.html", format="html")  # Interactive D3 visualization
viz.export("graph.dot", format="dot")    # Graphviz
viz.export("graph.md", format="mermaid") # Markdown diagrams
```

## MCP Tools Reference

| Tool | Description |
|------|-------------|
| `create_node` | Create REQ/SPEC/CODE/TEST node |
| `link_nodes` | Create edge (with cycle detection) |
| `get_node` | Retrieve node content/metadata |
| `query_graph` | Find nodes by type/status |
| `run_verification_sandbox` | Execute CODE node safely |
| `update_node_status` | Change node status (validated) |
| `git_commit_work` | Commit with traceability |
| `get_graph_stats` | Node/edge counts, status distribution |

## Project Structure

```
gaadp-constructor/
├── .blueprint/              # Configuration files
│   ├── prime_directives.md  # Immutable laws
│   ├── topology_config.yaml # Concurrency, RBAC
│   ├── llm_router.yaml      # Model assignments
│   └── prompt_templates.yaml
├── core/
│   ├── ontology.py          # Type definitions
│   ├── state_machine.py     # Status transitions
│   └── token_counter.py     # Real token counting
├── infrastructure/
│   ├── graph_db.py          # Persistence, queries
│   ├── llm_gateway.py       # LLM calls via litellm
│   ├── materializer.py      # Atomic file writes
│   ├── checkpoint.py        # Save/restore state
│   └── visualizer.py        # Graph export
├── orchestration/
│   ├── engine.py            # Main GADPEngine
│   ├── scheduler.py         # Task dispatch
│   ├── feedback.py          # Retry loops
│   ├── governance.py        # Treasurer/Sentinel/Curator
│   ├── human_loop.py        # Pause points
│   └── alerts.py            # Escalation
├── agents/
│   ├── base_agent.py        # Crypto, RBAC, tools
│   ├── concrete_agents.py   # Architect/Builder/Verifier
│   └── governance.py        # Treasurer/Sentinel agents
├── server.py                # MCP Server
├── production_main.py       # CLI entry point
└── requirements.txt
```

## Design Principles

1. **Graph is Truth**: Filesystem is a view, not source of truth
2. **Policy as Physics**: Constraints enforced by infrastructure, not prompts
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

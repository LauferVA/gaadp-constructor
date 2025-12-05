# GAADP SYSTEM INSTRUCTIONS (v3.0 - Graph-First Architecture)

You are the Engine of the Graph-Augmented Autonomous Development Platform (GAADP).

## 1. THE SOURCE OF TRUTH

* **Authority:** `.blueprint/global_ontology.md` defines all node types, edge types, and statuses.
* **Rules:** `.blueprint/prime_directives.md` contains the 22 Directives governing system behavior.
* **Physics:** `core/ontology.py` contains the `TRANSITION_MATRIX` - the single source of truth for state transitions.

## 2. CRITICAL OPERATIONAL CONSTRAINTS

### A. Policy as Physics
* Governance is embedded in `NodeMetadata`, not middleware.
* `cost_limit`, `security_level`, `max_attempts` are physics - they cannot be bypassed.
* The `TRANSITION_MATRIX` defines what transitions are possible. The runtime CANNOT violate it.

### B. The "Who Signs?" Rule
* **NEVER** generate a `signature` field.
* **YOU** generate content. **THE RUNTIME** (Python) signs it.

### C. The "Enum" Rule
* **ALWAYS** use `core.ontology` Enums (`NodeType.CODE`, `NodeStatus.VERIFIED`).
* Valid statuses: PENDING, PROCESSING, BLOCKED, VERIFIED, FAILED.

### D. The Agent Rule
* **ONE** agent class: `GenericAgent` in `agents/generic_agent.py`.
* Agent behavior comes from `config/agent_manifest.yaml`, not inheritance.

## 3. DEVELOPMENT WORKFLOW

1. **Read:** Check `config/agent_manifest.yaml` for agent configuration.
2. **Implement:** Write code that imports from `core/ontology`.
3. **Run:** `python main.py "your requirement"` or `python main.py --viz "requirement"`.

## 4. KEY FILES

| File | Purpose |
|------|---------|
| `main.py` | CLI entry point |
| `core/ontology.py` | TransitionMatrix, NodeMetadata, enums |
| `core/protocols.py` | UnifiedAgentOutput, GraphContext |
| `agents/generic_agent.py` | Universal agent class |
| `config/agent_manifest.yaml` | Agent personalities |
| `infrastructure/graph_runtime.py` | Execution engine |
| `infrastructure/graph_db.py` | Graph persistence |

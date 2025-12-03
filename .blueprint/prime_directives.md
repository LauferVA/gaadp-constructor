# PRIME DIRECTIVES: GRAPH-FIRST ARCHITECTURE
> SYSTEM_LEVEL_IMMUTABLE

## I. TRANSITION MATRIX (The Physics)

1. **Single Source of Truth**: The `TRANSITION_MATRIX` in `core/ontology.py` defines ALL valid state transitions. The runtime CONSULTS this matrix; it does not contain transition logic itself.

2. **Conditions as Functions**: Named conditions (e.g., `"cost_under_limit"`, `"dependencies_verified"`) are evaluated by `GraphRuntime.evaluate_condition()`. New conditions require implementation there.

3. **Priority Resolution**: When multiple transitions are valid, the highest `priority` rule wins. Ties are resolved by rule order.

## II. NODE METADATA (Governance as Data)

4. **Cost Limits**: `NodeMetadata.cost_limit` is physics - if `cost_actual > cost_limit`, the node CANNOT transition to PROCESSING. This is not middleware; it's embedded in data.

5. **Attempt Tracking**: `NodeMetadata.attempts` increments on each processing attempt. If `attempts >= max_attempts`, node transitions to FAILED.

6. **Security Level**: `NodeMetadata.security_level` defines required clearance. Agents cannot process nodes above their clearance.

## III. AGENT DISPATCH

7. **Universal Agent**: `GenericAgent` is the ONLY agent class. Behavior comes from `config/agent_manifest.yaml`, not inheritance.

8. **Role from Config**: Agent roles (ARCHITECT, BUILDER, VERIFIER) are loaded from YAML. The manifest defines system prompt, output protocol, and tool permissions.

9. **AGENT_DISPATCH Table**: The `(NodeType, Condition) -> AgentRole` mapping in `core/ontology.py` determines which agent processes which node.

## IV. STATUS SEMANTICS

10. **PENDING**: Node awaits processing. All required conditions must be met.
11. **PROCESSING**: Agent is actively working on node. Only one agent per node.
12. **BLOCKED**: Node waiting on CLARIFICATION or ESCALATION resolution.
13. **VERIFIED**: Terminal success. Code passed verification.
14. **FAILED**: Terminal failure. Exceeded max_attempts or unrecoverable error.

## V. NODE TYPE SEMANTICS

15. **CLARIFICATION**: Ambiguity requiring human input. Blocks dependent nodes until resolved.
16. **ESCALATION**: Failure requiring intervention. Created when node exhausts retry attempts.

## VI. GRAPH INTEGRITY

17. **No Orphans**: Every non-REQ node must have a TRACES_TO edge to its provenance.
18. **No Cycles**: DAG structure enforced by GraphDB. Cycle attempts raise ValueError.
19. **Signed Edges**: Every edge carries `agent_id` and `cryptographic_signature`.

## VII. RUNTIME BEHAVIOR

20. **Iteration Loop**: `GraphRuntime.run_until_complete()` iterates until no PENDING nodes remain or max_iterations reached.

21. **Processable Query**: Each iteration, runtime queries for nodes where:
    - Status is PENDING
    - All dependencies (DEPENDS_ON edges) are VERIFIED
    - Cost is under limit
    - Not blocked by CLARIFICATION/ESCALATION

22. **Visualization Events**: If `viz_server` is connected, runtime emits events for all state changes.

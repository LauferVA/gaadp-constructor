# Golden Failure Report #001: Asteroid Game Scheduler Bug

**Date**: 2025-12-03
**Session**: 3f9394b3
**Cost**: $0.0744
**Status**: FAILED (1 node failed, 0 verified)

## Executive Summary

A stress test with a 5-file Asteroid Game revealed a **critical scheduler bug**: the system builds dependent nodes before their dependencies exist, causing cascading failures.

## The Test

**Prompt**:
```
PROJECT: Asteroid Game (Pygame)
TARGET DIR: game/

OBJECTIVES:
1. Create a fully playable Asteroids clone using pygame.
2. Architecture MUST be decomposed into at least 5 separate files:
   - main.py: The game loop and initialization.
   - entities.py: Base classes for GameObject.
   - player.py: Ship mechanics (thrust, rotation).
   - asteroid.py: Rock mechanics (splitting, drift).
   - physics.py: Collision detection and movement vectors.
3. Constraint: main.py is the ONLY entry point. No circular imports allowed.

GOVERNANCE:
- The Architect must define the DEPENDS_ON edges explicitly in the PLAN.
- The Verifier must fail any code that imports a file not defined in the architecture.
```

## What Happened

### Phase 1: Architecture (SUCCESS)
The ARCHITECT correctly created 5 SPEC nodes with proper DEPENDS_ON edges:

```
physics.py (a3a283b8) - no dependencies [LEAF]
    ↓
entities.py (1af81feb) - depends on physics
    ↓
asteroid.py (4ce113e8) - depends on entities
player.py (f6ee6c27) - depends on entities
    ↓
main.py (3c32c436) - depends on physics, asteroid, player
```

### Phase 2: Build Order (FAILURE)
**Expected order** (topological sort respecting DEPENDS_ON):
1. physics.py
2. entities.py
3. asteroid.py, player.py
4. main.py

**Actual order**:
1. main.py ← WRONG!

### Phase 3: Build Execution (FAILURE)
The BUILDER, given the main.py SPEC, created a **God Class** with all game logic inline:
- No `from physics import ...`
- No `from entities import ...`
- 139 lines of monolithic code
- Complete game in one file

### Phase 4: Verification (PARTIAL)
The VERIFIER gave a "CONDITIONAL" verdict noting:
- "Game lacks specific gameplay mechanics"
- "No transition mechanism to GAME_OVER state"

**But the VERIFIER missed**:
- Architecture violation (no imports from dependency modules)
- DEPENDS_ON edge compliance

## Root Cause Analysis

### Bug #1: DEPENDS_ON Direction Check is Inverted
Location: `infrastructure/graph_runtime.py:116-124` - `evaluate_condition()`

The `dependencies_verified` condition checks the **wrong edge direction**.

**Current (buggy) logic**:
```python
if condition == "dependencies_verified":
    for pred in self.graph.graph.predecessors(node_id):  # <-- BUG: checks INCOMING edges
        edge_data = self.graph.graph.edges[pred, node_id]
        if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
            pred_status = self.graph.graph.nodes[pred].get('status')
            if pred_status != NodeStatus.VERIFIED.value:
                return False
    return True
```

**The Problem**: DEPENDS_ON edges go FROM dependent TO dependency:
```
main.py --DEPENDS_ON--> physics.py
```
From main.py's perspective, physics.py is a **successor** (outgoing edge), not a predecessor.

**Fix needed**:
```python
if condition == "dependencies_verified":
    for succ in self.graph.graph.successors(node_id):  # Check OUTGOING edges
        edge_data = self.graph.graph.edges[node_id, succ]  # Note: edge direction swap
        if edge_data.get('type') == EdgeType.DEPENDS_ON.value:
            succ_status = self.graph.graph.nodes[succ].get('status')
            if succ_status != NodeStatus.VERIFIED.value:
                return False
    return True
```

### Bug #2: Builder Lacks Dependency Context
Location: `agents/generic_agent.py` - prompt construction

When building main.py, the BUILDER should receive the content of its dependencies (physics.py, entities.py, etc.) so it knows what to import.

**Current behavior**: Builder only sees the SPEC content for main.py
**Expected behavior**: Builder sees SPEC + CODE of all DEPENDS_ON nodes

### Bug #3: Verifier Architecture Blindness
Location: `agents/generic_agent.py` - VERIFIER prompt

The VERIFIER doesn't check if the CODE's imports match the DEPENDS_ON edges defined by the ARCHITECT.

**Current behavior**: Only checks code quality
**Expected behavior**: Also checks `imports ⊆ DEPENDS_ON edges`

## Metrics We Now Know We Need

Based on this failure, we need to track:

| Metric | What It Measures | This Test |
|--------|------------------|-----------|
| **Topological Order Violation Score** | Nodes built before their DEPENDS_ON targets | 4 violations |
| **Import Consistency Score** | `imports / DEPENDS_ON edges` | 0/4 = 0% |
| **God Class Detection** | LOC in single file vs architecture spec | 139 vs ~5×30 |
| **Dependency Satisfaction Rate** | DEPENDS_ON nodes VERIFIED before build | 0/4 = 0% |

## Timeline (from logs)

```
23:09:59 | Created REQ node 48c35a12
23:10:14 | ARCHITECT created 5 SPEC nodes + 7 DEPENDS_ON edges ($0.0209)
23:10:14 | Iteration 2: Scheduler picks main.py SPEC ← BUG HERE
23:10:38 | BUILDER creates CODE node (God Class) ($0.0339)
23:10:47 | VERIFIER marks FAILED ($0.0143)
23:10:47 | "No more processable nodes" - execution stops
```

## Files Affected

| File | Status | Issue |
|------|--------|-------|
| physics.py | PENDING | Never built |
| entities.py | PENDING | Never built |
| asteroid.py | PENDING | Never built |
| player.py | PENDING | Never built |
| main.py | FAILED | God Class, no imports |

## Recommendation

**Priority 1**: Fix the scheduler to respect DEPENDS_ON edges
- Implement topological sort on the full graph
- Build leaf nodes (no outgoing DEPENDS_ON) first

**Priority 2**: Enrich BUILDER context
- Pass verified dependency code to BUILDER
- Let BUILDER see what functions/classes are available to import

**Priority 3**: Add architecture compliance check to VERIFIER
- Extract imports from CODE
- Compare against DEPENDS_ON edges
- Fail if mismatch

## Conclusion

This "Golden Failure" proved the value of stress testing before building metrics. We now know exactly which metrics matter because we have concrete evidence of the failure modes.

**Key Insight**: The ARCHITECT did its job perfectly. The system failed at scheduling and context propagation - exactly the kind of infrastructure bug that would be invisible to simple unit tests.

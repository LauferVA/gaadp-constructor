# GAADP Architecture Guardrails

**READ THIS FIRST** when modifying any core GAADP code.

This document captures recurring architectural violations that have caused failures
multiple times. These invariants exist because **we have learned them the hard way**.

---

## Invariant #1: Graph-First, Not Nested JSON

> "The graph isn't documentation of the computation—it IS the computation."

### The Anti-Pattern (DO NOT DO THIS)

```python
# BAD: Nested arrays of complex objects
class ResearcherOutput(BaseModel):
    happy_path_examples: List[Example]      # LLMs serialize incorrectly
    edge_case_examples: List[EdgeCaseExample]  # Returns strings not arrays
    unit_tests: List[UnitTest]              # Missing required fields
```

**Why it fails:** LLMs (especially smaller models like Haiku) incorrectly serialize
nested object arrays. They return JSON strings instead of arrays, omit required
fields, or double-serialize. The `@model_validator` patch in `ResearcherOutput`
is a bandaid, not a fix.

### The Correct Pattern (DO THIS)

```python
# GOOD: Flat structures with graph relationships
class ArchitectOutput(BaseModel):
    new_nodes: List[NodeSpec]  # Flat: {type, content, metadata}
    new_edges: List[EdgeSpec]  # Relationships: {source_id, target_id, relation}
```

**Why it works:**
- `NodeSpec` is flat (type + content + metadata dict)
- Relationships are expressed as `EdgeSpec` (graph edges)
- No nested objects for LLMs to mishandle
- The graph structure IS the data structure

### When Adding New Agent Outputs

**BEFORE** adding `List[ComplexObject]` to any protocol, ask:
1. Can this be a graph node with `content` string + `metadata` dict?
2. Can relationships be expressed as `EdgeSpec` edges?
3. Does `_to_unified_output` in `generic_agent.py` handle this?

**If yes to all three:** Use graph nodes and edges, not nested arrays.

### Files That Need Refactoring

| File | Issue | Status |
|------|-------|--------|
| `core/protocols.py:ResearcherOutput` | Uses nested `List[Example]`, `List[UnitTest]`, etc. | **NEEDS FIX** |
| `core/protocols.py:TesterOutput` | Uses `List[TestResult]` | Review needed |
| `core/protocols.py:DialectorOutput` | Uses nested `AmbiguityMarker.question` | Review needed |

---

## Invariant #2: No Hardcoding, Use Configuration

> "Keep the code generic, express specifics through configuration."

### The Anti-Pattern (DO NOT DO THIS)

```python
# BAD: Hardcoded agent-specific logic
if agent_role == "ARCHITECT":
    do_architect_thing()
elif agent_role == "RESEARCHER":
    do_researcher_thing()  # Forgot to add new agent type!
```

```python
# BAD: Hardcoded values
DEFAULT_COST_LIMIT = 0.50  # Should come from manifest
```

### The Correct Pattern (DO THIS)

```yaml
# GOOD: Configuration in agent_manifest.yaml
architect:
  role_name: ARCHITECT
  default_cost_limit: 0.50
  output_protocol: ArchitectOutput
  system_prompt: |
    You are the Architect agent...
```

```python
# GOOD: Generic code that reads from config
def get_cost_limit(self):
    return self.manifest.get('default_cost_limit', 1.0)
```

### The GenericAgent Pattern

The entire agent system demonstrates this:
- `GenericAgent` is the **generic code**
- `agent_manifest.yaml` provides the **specifics**
- No `ArchitectAgent`, `ResearcherAgent` classes exist

**When adding new behavior:**
1. Can it be a manifest variable? → Add to `agent_manifest.yaml`
2. Can it be a new node/edge type? → Add to `core/ontology.py`
3. Does it require code? → Make the code generic, parameterized

---

## Invariant #3: TransitionMatrix is Single Source of Truth

> "All valid state transitions are defined in TransitionMatrix."

### The Anti-Pattern

```python
# BAD: Inline state transition logic
if node.status == "TESTING" and verdict == "PASS":
    node.status = "VERIFIED"  # Bypasses TransitionMatrix!
```

### The Correct Pattern

```python
# GOOD: All transitions through the matrix
new_status = TransitionMatrix.get_transition(
    current_status=node.status,
    trigger=TransitionTrigger.TEST_PASS
)
```

---

## Checklist: Before Merging Any PR

- [ ] No new `List[ComplexNestedObject]` in protocols without graph-first justification
- [ ] No hardcoded agent names, costs, or behaviors outside manifest
- [ ] All state transitions go through `TransitionMatrix`
- [ ] `_to_unified_output` handles any new output fields
- [ ] Ran Task 16 (PyMark Parser) as a regression test for RESEARCHER

---

## Historical Failures (Learn From These)

### Failure #1: ResearcherOutput Nested Arrays (2025-12)
- **Symptom:** RESEARCHER validation errors, `edge_case_examples - Input should be a valid list`
- **Root Cause:** `List[EdgeCaseExample]` serialized as JSON string by LLM
- **Fix Attempt:** Added `@model_validator` to parse JSON strings (bandaid)
- **Proper Fix:** Refactor to use graph nodes with CONTAINS edges (pending)

### Failure #2: Hardcoded Provider Selection (2025-12)
- **Symptom:** New provider not being used despite configuration
- **Root Cause:** Hardcoded `if provider == "anthropic"` check
- **Fix:** Made provider selection fully configuration-driven

### Failure #3: BuilderOutput follow_up_specs (historical)
- **Symptom:** Builder couldn't create follow-up specifications
- **Root Cause:** `follow_up_specs: List[SpecDraft]` nested array
- **Fix:** Removed field, added comment explaining why (see `protocols.py:139`)

---

## For Claude Code Sessions

When working on GAADP, Claude Code should:

1. **Read this file first** when touching `core/`, `agents/`, or `config/`
2. **Check for nested List[Object]** patterns before approving protocol changes
3. **Ask "is this graph-first?"** before adding new data structures
4. **Reference this document** when the user asks about recurring issues

This document should be included in CLAUDE.md or referenced in system prompts
to ensure architectural invariants survive context window limits.

---

## Appendix: ResearcherOutput Refactoring Proposal

**Status:** Pending implementation

The `ResearcherOutput` in `core/protocols.py:378-511` has 9 fields using `List[ComplexObject]`
that cause LLM serialization failures. Here's the graph-first refactoring plan:

### Current Anti-Pattern (problematic fields)

```python
# In ResearcherOutput - ALL PROBLEMATIC
success_criteria: List[SuccessCriterion]  # Line 403
inputs: List[InputSpec]                    # Line 409
outputs: List[OutputSpec]                  # Line 410
happy_path_examples: List[Example]         # Line 421
edge_case_examples: List[EdgeCaseExample]  # Line 425
error_case_examples: List[ErrorCaseExample]# Line 429
ambiguities: List[AmbiguityCapture]        # Line 435
unit_tests: List[UnitTest]                 # Line 453
files: List[FileSpec]                      # Line 478
```

### Proposed Graph-First Design

```python
class ResearcherOutput(BaseModel):
    """Graph-first Researcher output."""

    # Flat metadata (keep as-is - these are fine)
    maturity_level: Literal["DRAFT", "REVIEWABLE"]
    completeness_score: float
    task_category: Literal["greenfield", "brownfield", ...]
    why: str  # Simple string

    # GRAPH-FIRST: Convert all lists to nodes/edges
    new_nodes: List[NodeSpec]  # All examples, tests, specs become nodes
    new_edges: List[EdgeSpec]  # Relationships via CONTAINS, TRACES_TO

    # Optional flat fields (keep)
    complexity_time: Optional[str]
    complexity_space: Optional[str]
    reasoning: Optional[str]
```

### Node Type Mapping

| Old Field | New Node Type | Content |
|-----------|---------------|---------|
| `success_criteria` | `CRITERION` | criterion.description |
| `inputs` / `outputs` | `CONTRACT` | name + type + constraints (as JSON in metadata) |
| `happy_path_examples` | `EXAMPLE` | input→output as content |
| `edge_case_examples` | `EDGE_CASE` | edge case description |
| `error_case_examples` | `ERROR_CASE` | error scenario |
| `unit_tests` | `TEST_SPEC` | test description |
| `files` | `FILE_SPEC` | path + purpose |

### Edge Relationships

```
RESEARCH_NODE
  ├── CONTAINS → CRITERION (success criteria)
  ├── CONTAINS → CONTRACT (inputs/outputs)
  ├── CONTAINS → EXAMPLE (happy path)
  ├── CONTAINS → EDGE_CASE
  ├── CONTAINS → ERROR_CASE
  ├── CONTAINS → TEST_SPEC
  └── CONTAINS → FILE_SPEC
      └── DEPENDS_ON → FILE_SPEC (file dependencies)
```

### Implementation Steps

1. **Add new NodeTypes** to `core/ontology.py`:
   - `CRITERION`, `CONTRACT`, `EXAMPLE`, `EDGE_CASE`, `ERROR_CASE`, `TEST_SPEC`, `FILE_SPEC`

2. **Update ResearcherOutput** in `core/protocols.py`:
   - Add `new_nodes: List[NodeSpec]` and `new_edges: List[EdgeSpec]`
   - Keep flat metadata fields
   - Remove all `List[ComplexObject]` fields

3. **Update agent_manifest.yaml**:
   - Modify RESEARCHER system prompt to output graph nodes/edges

4. **Update `_to_unified_output`** in `agents/generic_agent.py`:
   - Handle new RESEARCHER node types

5. **Run regression test**:
   - Task 16 (PyMark Parser) must pass

### Why This Works

- **Flat NodeSpec**: `{type: "EXAMPLE", content: "input: 5 → output: 25", metadata: {category: "happy_path"}}`
- **No nested objects**: LLM outputs simple type + string + dict
- **Relationships as edges**: `EdgeSpec(source=research_id, target=example_id, relation="CONTAINS")`
- **Same pattern as ArchitectOutput**: Proven to work reliably

---

## Related Documentation

- **[GRAPH_INVARIANTS.md](./GRAPH_INVARIANTS.md)** - Mathematical graph-theoretic invariants
  (Euler paths, DAG validation, Petri net soundness, cyclomatic complexity)

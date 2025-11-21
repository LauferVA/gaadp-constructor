# GAADP SYSTEM INSTRUCTIONS (v2.0 - Graph-Native)
You are the Engine of the Graph-Augmented Autonomous Development Platform (GAADP).

## 1. THE SOURCE OF TRUTH
* **Authority:** `.blueprint/global_ontology.md` is the absolute reference.
* **Rules:** `.blueprint/prime_directives.md` contains the 17 Immutable Laws.

## 2. CRITICAL OPERATIONAL CONSTRAINTS
### A. The "Who Signs?" Rule
* **NEVER** generate a `signature` field.
* **YOU** generate content. **THE RUNTIME** (Python) signs it.

### B. The "No SQL" Rule
* **NEVER** use `SELECT * FROM`.
* **ALWAYS** use Graph Traversal (`G.predecessors()`).

### C. The "Enum" Rule
* **ALWAYS** use `core.ontology` Enums (`NodeType.CODE`, `NodeStatus.VERIFIED`).

## 3. DEVELOPMENT WORKFLOW
1.  **Consult:** Read `MASTER_BLUEPRINT.tsv` for high-level intent.
2.  **Implement:** Write code in `src/` that imports from `core.ontology`.
3.  **Verify:** Run `validate_blueprint.py`.

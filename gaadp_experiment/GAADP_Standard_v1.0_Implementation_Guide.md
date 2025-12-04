# GAADP Research Artifact Standard v1.0 - Implementation Guide

**Status:** CANONICAL - All 5 Expert Perspectives Unified
**Date:** 2025-12-03
**Authority:** The Architect

---

## Executive Summary

This standard resolves the fundamental tensions between the 5 expert perspectives by recognizing that **specs evolve through maturity levels**. Early-stage specs capture ambiguities (PRAGMATIST), mature specs eliminate them through executable contracts (THEORIST), and all stages enforce security boundaries (SECURITY).

---

## The Resolution Matrix

| Tension | Resolution |
|---------|-----------|
| **THEORIST vs PRAGMATIST**: "Eliminate choices" vs "Capture ambiguities" | **Maturity Levels**: DRAFT captures ambiguities, EXECUTABLE eliminates via Hoare contracts |
| **SECURITY requirements** | **Enforcement Block**: Forbidden patterns + approval gates + attestation chains are mandatory at all maturity levels |
| **PRODUCT spec maturity** | **4-Level Ladder**: DRAFT → REVIEWABLE → EXECUTABLE → PRODUCTION |
| **QA's 6 testability gaps** | **Direct Schema Fields**: test_mapping, state_machine, cost_limit, deadline, max_clarification_rounds, merkle_chain |

---

## The 4 Maturity Levels

### Level 1: DRAFT (Human Clarification Expected)
- **Purpose:** Capture initial understanding + explicit ambiguities
- **Completeness:** 0.25-0.5
- **Requirements:**
  - Domain WHY populated
  - At least 1 typed example
  - Ambiguities array non-empty (PRAGMATIST: "Document that you don't know")
- **Enforcement:** Basic forbidden patterns checked
- **Escalation:** Automatic - all ambiguities trigger clarification loop

### Level 2: REVIEWABLE (Ambiguities Resolved)
- **Purpose:** Human-readable, all choices documented
- **Completeness:** 0.5-0.75
- **Requirements:**
  - All ambiguities.resolution != "escalation_required"
  - Minimum 3 typed examples (happy/edge/error)
  - Failure hierarchy defined
- **Enforcement:** Approval gates evaluated
- **Escalation:** On demand - user can request clarification

### Level 3: EXECUTABLE (Machine-Verifiable)
- **Purpose:** Agent can build without clarification
- **Completeness:** 0.75-0.95
- **Requirements:**
  - Hoare pre/postconditions executable (Python expressions)
  - test_mapping complete (1 test per requirement)
  - State machine deterministic
  - Complexity bounds polynomial-time decidable
- **Enforcement:** Cost limits + deadlines enforced
- **Escalation:** Only on cost/deadline breach

### Level 4: PRODUCTION (Attestation-Complete)
- **Purpose:** Deployment-ready with full audit trail
- **Completeness:** 0.95-1.0
- **Requirements:**
  - Merkle chain intact (REQ→SPEC→PLAN→CODE→TEST)
  - HMAC attestation signed
  - All approval gates PASSED
  - Transitive attestation from dependencies
- **Enforcement:** All gates enforced
- **Escalation:** Security/compliance violations only

---

## Addressing QA's 6 Gaps

| Gap # | Issue | Schema Solution |
|-------|-------|-----------------|
| 1 | Condition evaluation not executable | `test_mapping[].test_condition`: Python expressions validated via AST parse |
| 2 | Merkle integrity not enforced | `enforcement.integrity.merkle_chain`: SHA-256 chain with parent links |
| 3 | Agent dispatch conflicts | `state_machine.transitions[].condition`: Deterministic boolean expressions |
| 4 | Clarification loop termination | `governance.max_clarification_rounds`: Hard limit (default: 3) |
| 5 | Deadline enforcement | `governance.deadline`: ISO8601 timestamp, triggers escalation at T-0 |
| 6 | Security clearance gating | `approval_gates[].required_clearance`: L0-L3 enum + executable condition |

---

## Security Enforcement (Non-Negotiable)

All artifacts MUST include:

1. **Forbidden Patterns Blocklist**
   - Enforced at Verifier stage via regex/AST checks
   - Examples: `eval()`, `exec()`, shell injection patterns
   - Violations = immediate FAIL verdict

2. **Approval Gates with Clearance Levels**
   ```json
   {
     "gate_id": "pii_access",
     "required_clearance": "L2_PRIVILEGED",
     "condition": "any(o.sensitivity == 'PII' for o in outputs)"
   }
   ```

3. **Cost as Hard Boundary**
   - Not a "guideline" - enforced by runtime
   - `metadata.cost_actual` incremented on every LLM call
   - Node transitions BLOCKED if `cost_actual >= cost_limit`

4. **Transitive Attestation**
   - SPEC inherits attestation from REQ
   - CODE inherits from SPEC
   - Broken chain = deployment blocked

---

## THEORIST's Closed Type System

Valid type annotations (Python-based):
- Primitives: `int`, `float`, `str`, `bool`
- Collections: `list[T]`, `dict[K, V]`, `set[T]`, `tuple[T1, T2, ...]`
- Optional: `Optional[T]` or `T | None`
- Custom: Must reference SPEC node that defines the type

Hoare conditions must be:
- Executable Python expressions
- Return boolean
- Reference only inputs/outputs by name
- Example: `len(arr) > 0 AND all(x > 0 for x in arr)`

---

## PRAGMATIST's Environmental Physics

Required fields to prevent "it worked on my machine" failures:

```json
{
  "environment": {
    "runtime": {"language": "Python", "version": "3.10+"},
    "dependencies": [
      {"name": "numpy", "version": ">=1.21.0", "required": true}
    ],
    "failure_hierarchy": [
      {
        "failure_mode": "ImportError: numpy not found",
        "severity": "FATAL",
        "recovery": "Install numpy>=1.21.0",
        "escalation_required": true
      }
    ]
  }
}
```

---

## PRODUCT's Success Criteria as Physics

NFRs are not "nice-to-haves" - they are executable constraints:

```json
{
  "success_criteria": [
    {
      "criterion": "Sorting completes within 1 second for 10^6 elements",
      "measurement": "time.perf_counter() after sort() call",
      "threshold": "elapsed < 1.0",
      "nfr_category": "PERFORMANCE"
    }
  ]
}
```

These become TEST nodes with automated assertions.

---

## Implementation Checklist

When generating a Research Artifact:

1. **Start at DRAFT maturity**
   - Populate domain.why (user story)
   - Add 1 typed example
   - Enumerate ambiguities

2. **Security gate check**
   - Define forbidden_patterns (minimum: `["eval", "exec", "os.system"]`)
   - Set cost_limit (default: $1.00)
   - Classify trust boundaries

3. **Clarification loop**
   - For each ambiguity.resolution == "escalation_required":
     - Create CLARIFICATION node
     - BLOCKS this artifact
   - When answered → update resolution → advance maturity

4. **Executable contract phase** (REVIEWABLE → EXECUTABLE)
   - Convert prose to Hoare conditions
   - Generate test_mapping from requirements
   - Build state_machine for verification flow

5. **Attestation phase** (EXECUTABLE → PRODUCTION)
   - Compute Merkle chain
   - Sign with HMAC
   - Link parent attestations

---

## Validation Rules (Enforced by Verifier)

```python
def validate_research_artifact(artifact: dict) -> bool:
    # Maturity-specific checks
    level = artifact['maturity']['level']

    if level == 'DRAFT':
        assert len(artifact['spec']['contracts']['examples']) >= 1
        assert len(artifact['spec']['contracts']['ambiguities']) > 0

    elif level == 'EXECUTABLE':
        # THEORIST: Hoare conditions must parse as Python AST
        for inp in artifact['spec']['domain']['inputs']:
            ast.parse(inp['hoare_precondition'], mode='eval')

        # QA Gap 1: test_mapping non-empty
        assert len(artifact['spec']['verification']['test_mapping']) > 0

    elif level == 'PRODUCTION':
        # SECURITY: Merkle chain validated
        assert validate_merkle_chain(artifact['enforcement']['integrity']['merkle_chain'])

        # SECURITY: Attestation signature valid
        assert verify_hmac(artifact['enforcement']['integrity']['attestation'])

    return True
```

---

## Usage Example

See `/Users/lauferva/gaadp-constructor/gaadp_experiment/GAADP_Research_Artifact_Standard_v1.0.json` for full JSON Schema.

To validate an artifact:
```python
import jsonschema
schema = json.load(open('GAADP_Research_Artifact_Standard_v1.0.json'))
jsonschema.validate(artifact, schema)
```

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2025-12-03 | Initial synthesis - 5 expert perspectives unified |

---

**This standard is now authoritative for all GAADP Research Phase outputs.**

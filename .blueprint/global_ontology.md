# GAADP GLOBAL ONTOLOGY
> VERSION: 2.0.0
> AUTHORITY: ABSOLUTE
> STATUS: IMMUTABLE

## I. THE AGENT HIERARCHY (ROLES)
| Role ID | Scope | Responsibility |
| :--- | :--- | :--- |
| **ARCHITECT** | System | Decomposes Specs into Task Graphs. Defines Topology. |
| **BUILDER** | Leaf | Implements single nodes (Code/Test/Config). |
| **VERIFIER** | Edge | Reviews Builder output. Signs `verified_by` edges. |
| **CURATOR** | Graph | Manages Graph Integrity (GC, Orphans, Citations). |
| **SOCRATES** | Input | Interrogates User/Docs to resolve ambiguity. |
| **SENTINEL** | Security | Scans Data Flow for Taint/Vulnerabilities. |
| **TREASURER** | Resource | Monitors `cost_ledger`. Halts runaway processes. |
| **LIBRARIAN** | Meta | Manages System Prompts & Optimization. |

## II. THE GRAPH ANATOMY (NODE TYPES)
| Node Type | Definition |
| :--- | :--- |
| **`REQ`** | A distinct, traceable user need or constraint. |
| **`SPEC`** | A technical translation of a Requirement. |
| **`PLAN`** | A unit of work assigned to an Agent. |
| **`CODE`** | Source code, config, or logic. |
| **`TEST`** | Code designed to validate a CODE node. |
| **`DOC`** | Human-readable explanation or citation. |
| **`STATE`** | A frozen snapshot of the system memory. |
| **`DEAD_END`** | Failed branch (Tombstone). |

## III. THE CONNECTIVE TISSUE (EDGE TYPES)
| Edge Type | Meaning | Constraint |
| :--- | :--- | :--- |
| **`TRACES_TO`** | Traceability back to Root Spec. | Mandatory for all nodes. |
| **`DEPENDS_ON`** | Code import or Logical dependency. | **NO CYCLES ALLOWED.** |
| **`IMPLEMENTS`** | "This code satisfies that spec." | One-to-Many allowed. |
| **`VERIFIES`** | "I attest this code is correct." | **MUST** have Signature. |
| **`DEFINES`** | "This plan resulted in this code." | 1:1 Relationship. |

# Project Specification: LWW-Element-Set (CRDT)

## 1. Objective
Implement a **Last-Write-Wins Element Set (LWW-Element-Set)**. This is a data structure used in distributed databases that allows concurrent updates from different nodes to merge without conflict.

## 2. Architecture
The LWW-Element-Set consists of two internal sets:
1.  **Add Set:** Stores `(element, timestamp)` for additions.
2.  **Remove Set:** Stores `(element, timestamp)` for removals.

## 3. Logic (The Rules of Convergence)
* **Lookup:** An element `e` is in the set IF:
    * It is in the Add Set.
    * AND (It is NOT in the Remove Set OR the timestamp in the Add Set is > the timestamp in the Remove Set).
* **Merge:** To merge two CRDTs (Replica A and Replica B):
    * New Add Set = Union of A and B's Add Sets (taking the max timestamp for duplicates).
    * New Remove Set = Union of A and B's Remove Sets.

## 4. Functional Requirements
* **Class:** `CRDTSet`
* **Method:** `add(element, timestamp)`
* **Method:** `remove(element, timestamp)`
* **Method:** `exists(element) -> bool`
* **Method:** `merge(other_crdt)`: Mutates self to include state from `other_crdt`.

## 5. Acceptance Criteria
* **Scenario:**
    1.  Replica A: Adds "Apple" at t=10.
    2.  Replica B: Removes "Apple" at t=20.
    3.  **Merge:** Both replicas sync.
    4.  **Result:** "Apple" should NOT exist (20 > 10).
* **Edge Case (The Bias):**
    1.  Replica A adds "Banana" at t=5.
    2.  Replica B removes "Banana" at t=5.
    3.  **Result:** The spec must define a tie-breaker. (Standard convention: Add wins over Remove, or prefer higher Replica ID). You must implement **Add Wins** for ties.

## 6. Research Instructions
1.  Research "State-based CRDTs" vs "Operation-based CRDTs". This task is State-based.
2.  Understand why we store the timestamp even for removals (otherwise we can't handle "re-adding" an item later).

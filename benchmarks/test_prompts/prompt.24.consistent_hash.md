# Project Specification: Consistent Hashing Ring

## 1. Objective
Implement a `ConsistentHash` class that maps keys (strings) to Nodes (servers) efficiently.
When a Node is added or removed, only K/N keys should move.

## 2. Core Logic (The Ring)
1.  **Ring Space:** Treat the hash output (e.g., 0 to 2^32-1) as a circle.
2.  **Node Placement:** Hash the server names ("Server-A", "Server-B") to place them on the ring.
3.  **Key Placement:** Hash the data key ("User-123"). Walk clockwise on the ring to find the first Server Node.

## 3. Functional Requirements
* **Method:** `add_node(node_name)`
* **Method:** `remove_node(node_name)`
* **Method:** `get_node(key) -> node_name`
* **Virtual Nodes:** To prevent uneven distribution, every physical node must map to `v` points on the ring (e.g., `Server-A-0`, `Server-A-1`...).

## 4. Performance Requirement
* Lookups must be O(log N) (where N is number of nodes/virtual nodes).
* **Hint:** Use `bisect` (Binary Search) on the sorted list of node hashes.

## 5. Acceptance Criteria
* **Test:**
    1.  Add Nodes A, B, C.
    2.  Map Key "data1" -> Maps to Node B.
    3.  Remove Node C.
    4.  Map Key "data1" -> **Must still map to Node B** (Stability check).
    5.  Add Node D.
    6.  Check redistribution: Only keys that fall between C and D should move to D.

## 6. Research Instructions
1.  Research why "Virtual Nodes" are necessary for uniform distribution in consistent hashing.
2.  Identify the correct Python library for sorted search (`bisect`).

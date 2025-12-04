# Project Specification: "PyCask" (Log-Structured Key-Value Store)

## 1. Objective
Implement a persistent key-value store based on the **Bitcask** architecture (used by Riak).
* **Writes:** Append-only to a log file (fast).
* **Reads:** Use an in-memory Hash Map (KeyDir) that points to the file offset where the data lives.

## 2. Architecture
* **The Data File (`store.db`):** A binary sequence of entries.
    * Entry Format: `[timestamp][key_size][val_size][key][value]`
* **The KeyDir (In-Memory):** A Python Dictionary `Key -> {file_id, value_size, value_pos, timestamp}`.
    * When the database starts, it must read `store.db` from beginning to end to rebuild this KeyDir.

## 3. Functional Requirements
* **Class:** `PyCask(filepath)`
* **Method:** `put(key: str, value: str)`
    * Serializes the data into binary format.
    * Appends it to the end of the file.
    * Updates the in-memory KeyDir.
* **Method:** `get(key: str) -> str`
    * Looks up the key in KeyDir.
    * Uses `file.seek(pos)` to jump directly to the data.
    * Reads `value_size` bytes and returns the value.
    * Returns `None` if key not found.
* **Method:** `close()`: Closes the file handle.

## 4. Edge Cases
* **Updates:** If I write Key="A" Val="1", then later write Key="A" Val="2", the file grows. The KeyDir must point to the *latest* entry.
* **Persistence:** If I crash the script and restart it, the `__init__` method must scan the file to recover the state.

## 5. Acceptance Criteria
```python
db = PyCask("test.db")
db.put("user", "Vincent")
db.put("role", "Admin")
db.put("user", "Magnus") # Update
db.close()

db2 = PyCask("test.db") # Re-open
print(db2.get("user")) # Must print "Magnus"
print(db2.get("role")) # Must print "Admin"
```

## 6. Research Instructions
1. Research "Bitcask File Format".
2. Why use `struct.pack` for the header (sizes) but raw bytes for key/value?
3. Understand the trade-off: Fast Writes (Append) vs. Slow Startup (must scan whole file).

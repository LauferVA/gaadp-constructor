# Project Specification: Heap Memory Allocator (Malloc/Free)

## 1. Objective
Create a class `MemoryManager` that manages a fixed-size byte array (the "Heap") and simulates `malloc` and `free`.
You must implement a **Free List** data structure to track available memory blocks.

## 2. Architecture
* **The Heap:** A `bytearray` of size 1024 bytes.
* **Metadata:** Each block needs a header (hidden from the user) indicating:
    * Block Size
    * Is_Free (Boolean)
    * Next_Block_Pointer (Index)

## 3. Functional Requirements
* **Method:** `malloc(size) -> int`
    * Scans the Free List (using "First Fit" or "Best Fit" strategy).
    * Splits the block if it's larger than requested (accounting for header overhead).
    * Returns the **index** of the usable data (after the header).
    * Returns `-1` if OOM (Out of Memory).
* **Method:** `free(ptr)`
    * Marks the block at `ptr` as free.
    * **Crucial Step (Coalescing):** Checks if the *next* or *previous* block is also free. If so, merge them into one large block to reduce fragmentation.

## 4. Acceptance Criteria
* **Scenario:**
    1.  `ptr1 = malloc(100)`
    2.  `ptr2 = malloc(100)`
    3.  `ptr3 = malloc(100)`
    4.  `free(ptr2)` -> Middle block is free.
    5.  `free(ptr3)` -> Last block is free. **Must coalesce** with the middle block.
    6.  `ptr4 = malloc(200)` -> Should succeed (using the merged space of 2+3).
    * *Failure:* If Coalescing is missing, step 6 will fail because we have two 100-byte holes, not one 200-byte hole.

## 5. Research Instructions
1.  Research "Memory Block Header Layout". How much space does the metadata take?
2.  Research "First Fit" vs "Best Fit". First fit is faster; Best fit reduces fragmentation. Pick one and justify.
3.  How do you calculate the address of the "Previous Block" without a doubly linked list? (Hint: Knuth's boundary tag method, or just scan the list).

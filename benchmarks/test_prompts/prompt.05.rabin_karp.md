# Project Specification: Rabin-Karp String Search

## 1. Objective
Implement the **Rabin-Karp algorithm** for string searching in Python. The function must find all occurrences of a pattern P in a text T using a rolling hash.

## 2. Mathematical Definition
You must implement the rolling hash manually using this formula:
H = (c₁ × a^(k-1) + c₂ × a^(k-2) + ... + cₖ × a^0) mod m

Where:
* a is the base (use 256 for ASCII).
* m is a large prime number (use 101 for this exercise).
* c are the integer values of the characters.

**Rolling Logic:**
To move the window one step to the right, you must **update** the hash in O(1) time (remove the leading character, shift, add the trailing character) rather than recalculating the whole string.

## 3. Requirements
* **Function:** `rabin_karp(text: str, pattern: str) -> List[int]`
* **Output:** A list of starting indices where the pattern is found.
* **Collision Handling:** If the hashes match, you must verify the actual strings match (to handle hash collisions).

## 4. Acceptance Criteria
* `rabin_karp("ABABDABACDABABCABAB", "ABABCABAB")` returns `[10]`.
* Code must explicitly calculate hashes. Using `str.find()` is a failure.

## 5. Research Instructions
1.  Derive the formula for "rolling" the hash: H_new = (H_old - c_removed × h) × a + c_added mod m.
2.  Implement the function to be efficient for long strings.

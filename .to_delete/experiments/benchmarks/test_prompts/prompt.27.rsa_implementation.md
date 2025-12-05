# Project Specification: RSA Key Generation & Encryption

## 1. Objective
Implement the RSA (Rivest–Shamir–Adleman) cryptosystem from scratch using Python's built-in arbitrary-precision integers. Do not use any crypto libraries.

## 2. Core Logic (The Math)
1.  **Key Gen:**
    * Select two prime numbers p and q.
    * Compute n = p × q.
    * Compute φ(n) = (p-1)(q-1).
    * Choose public exponent e (usually 65537).
    * Compute private key d such that d × e ≡ 1 (mod φ(n)). (Requires **Modular Multiplicative Inverse**).
2.  **Encrypt:** c = m^e mod n
3.  **Decrypt:** m = c^d mod n

## 3. Functional Requirements
* **Class:** `RSA`
* **Method:** `generate_keys(p, q)`: Stores public (e, n) and private (d, n).
* **Method:** `encrypt(message_int)`: Returns ciphertext integer.
* **Method:** `decrypt(ciphertext_int)`: Returns message integer.

## 4. Helper Algorithms (Must Implement)
* **Extended Euclidean Algorithm:** To find the modular inverse of e.
* **Modular Exponentiation:** `pow(base, exp, mod)` is allowed, but understanding *why* it's needed is part of the research.

## 5. Acceptance Criteria
* **Input:** p=61, q=53.
* **Public Key:** (e=17, n=3233).
* **Private Key:** d=2753.
* **Test:** Encrypt `123`. Decrypt the result. Must return `123`.
* **Constraint:** The implementation must work for standard primes, not just small inputs.

## 6. Research Instructions
1.  Research the "Extended Euclidean Algorithm" for finding `d`.
2.  Research why p and q must be distinct primes.
3.  (Optional) How to convert a string "Hello" into an integer for encryption? (Simple ASCII/UTF-8 bytes-to-int conversion).

# Project Specification: Project Euler #14 (Longest Collatz Sequence)

## 1. Objective
Solve Project Euler Problem 14 efficiently in Python. The script `collatz_solver.py` must find the starting number under 1,000,000 that produces the longest Collatz sequence.

## 2. The Math (The Collatz Conjecture)
The sequence is defined for the set of positive integers:
* n → n/2 (n is even)
* n → 3n + 1 (n is odd)

Using the rule above and starting with 13, we generate the following sequence:
`13 -> 40 -> 20 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1`
This sequence contains 10 terms.

## 3. Requirements
* **Search Space:** Analyze all starting numbers from 1 to 999,999.
* **Metric:** Determine which starting number produces the longest chain before hitting 1.
* **Performance:** The script must run in under 5 seconds on standard hardware.
* **Output:** Print the number and the chain length.

## 4. The Core Challenge
A naive implementation re-calculates the chain for every number.
* *Example:* Calculating the chain for 100 eventually hits 40. We already calculated the chain for 40 in the example above.
* **The Test:** Will the agent implement **Memoization** (caching results) to avoid redundant calculations?

## 5. Acceptance Criteria
* **Correctness:** The code must output the correct known answer for this problem (Starting Number: `837799`).
* **Efficiency:** Code must demonstrate an algorithmic optimization (caching/memoization) rather than brute force.

## 6. Research Instructions
1.  Identify the potential bottleneck in calculating sequences for 1 million numbers.
2.  Propose a caching strategy (e.g., a dictionary or `@functools.lru_cache`) to speed up the process.

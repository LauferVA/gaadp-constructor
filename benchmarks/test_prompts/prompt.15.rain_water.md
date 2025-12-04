# Project Specification: Trapping Rain Water Solver

## 1. Objective
Given `n` non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

## 2. Input/Output
* **Function:** `solve_rain(heights: List[int]) -> int`
* **Input:** `[0,1,0,2,1,0,1,3,2,1,2,1]`
* **Output:** `6`

## 3. The Logic (Visual Physics)
Water can only be trapped at index `i` if there is a higher bar to the *left* AND a higher bar to the *right*.
* The water level at index `i` = `min(max_left_height, max_right_height) - height[i]`.
* If the result is negative, water level is 0.

## 4. Performance Constraints
* **Time Complexity:** Must be O(N).
* **Space Complexity:** O(N) is acceptable, but O(1) (Two Pointer approach) is preferred.
* **Input Size:** The script should handle a list of 1,000,000 integers in under 1 second.

## 5. Edge Cases
* **Ascending/Descending:** `[0, 1, 2, 3]` or `[3, 2, 1, 0]` traps 0 water (no "bowl").
* **Flat:** `[5, 5, 5, 5]` traps 0 water.
* **Single Dip:** `[5, 0, 5]` traps 5 units.

## 6. Acceptance Criteria
* The solution must pass the "Single Dip" test: `[10, 0, 10]` -> `10`.
* The solution must pass the "W" shape test: `[5, 0, 5, 0, 5]` -> `10`.
* Code must not use nested loops (`for i in ...: for j in ...`).

## 7. Research Instructions
1.  Identify the "Two Pointer" technique for this specific problem.
2.  Explain why a naive approach fails on performance.

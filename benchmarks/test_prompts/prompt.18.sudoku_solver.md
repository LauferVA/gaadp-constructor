# Project Specification: Sudoku Solver

## 1. Objective
Write a Python script `sudoku.py` that accepts an unsolved 9x9 Sudoku grid and outputs the solved grid.

## 2. Input Format
* Input is a string of 81 characters.
* `0` represents an empty cell.
* Example: `530070000600195000...`

## 3. The Algorithm (Backtracking)
You cannot use random guessing. You must implement a Depth-First Search (Backtracking) algorithm:
1.  Find the first empty cell (`0`).
2.  Try digits `1-9`.
3.  Check validity: Is this digit valid in the current **Row**, **Column**, and **3x3 Subgrid**?
4.  If valid, place it and recurse to the next cell.
5.  If the recursion returns `True`, bubble up `True`.
6.  If the recursion returns `False` (dead end), reset the cell to `0` (backtrack) and try the next digit.

## 4. Requirements
* **Performance:** Must solve a standard "Hard" Sudoku in under 100ms.
* **Output:** Print the solved 9x9 grid in a readable format (lines separating 3x3 blocks).

## 5. Acceptance Criteria
* **Test Input:**
  `003020600900305001001806400008102900700000008006708200002609500800203009005010300`
* **Verification:** The output must have no duplicates in any row, column, or 3x3 box, and match the non-zero inputs.

## 6. Research Instructions
1.  Research "Sudoku Backtracking Algorithm".
2.  Define helper functions `isValid(board, row, col, num)` to keep the main logic clean.
3.  How do you calculate which 3x3 box a cell `(r, c)` belongs to? (Hint: `(r//3) * 3`).

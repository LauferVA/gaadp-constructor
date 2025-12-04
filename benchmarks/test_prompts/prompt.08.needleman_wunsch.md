# Project Specification: Needleman-Wunsch Alignment

## 1. Objective
Implement the Needleman-Wunsch algorithm to perform global alignment of two DNA sequences.

## 2. Scoring Scheme
* Match: +1
* Mismatch: -1
* Gap: -1

## 3. The Algorithm
1.  **Initialization:** Create a (N+1) × (M+1) matrix. Fill the first row and column with gap penalties (0, -1, -2, ...).
2.  **Fill:** For every cell (i, j), calculate the score based on the max of:
    * Diagonal: Score(i-1, j-1) + (Match/Mismatch)
    * Up: Score(i-1, j) + Gap
    * Left: Score(i, j-1) + Gap
3.  **Traceback:** Start at (N, M) and move towards (0, 0) to reconstruct the alignment strings.

## 4. Requirements
* **Input:** Two strings, e.g., `GATTACA` and `GCATGCU`.
* **Output:** The two aligned strings with `-` for gaps.
    * Example:
        ```
        G-ATTACA
        GCA-TGCU
        ```
* **Visualization:** (Optional but preferred) Print the scoring matrix using tab-separated format.

## 5. Research Instructions
1.  Verify how to handle "branching" during traceback (what if Up and Left scores are equal?). Standard practice: prioritize one direction (e.g., Diagonal > Up > Left).

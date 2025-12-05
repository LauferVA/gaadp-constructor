# Project Specification: Infinite Game of Life

## 1. Objective
Implement Conway's Game of Life, but on an infinitely expanding grid. The simulation must handle patterns that travel thousands of units away from the origin (like "Gliders") without crashing or allocating a massive array.

## 2. Core Logic
* **Rules:**
    1.  Underpopulation: <2 neighbors -> dies.
    2.  Survival: 2 or 3 neighbors -> lives.
    3.  Overpopulation: >3 neighbors -> dies.
    4.  Reproduction: exactly 3 neighbors -> becomes alive.
* **Data Structure:** Do NOT use a 2D Array / NumPy Matrix. Use a **Set of Tuples** representing only the *alive* cells.

## 3. Functional Requirements
* **Class:** `InfiniteLife(initial_state: List[Tuple[int, int]])`
* **Method:** `tick()`: Advances the simulation by one generation.
    * **Crucial Optimization:** You only need to check the neighbors of currently alive cells and their immediate dead neighbors. Do not check the whole universe.
* **Method:** `get_bounds()`: Returns the min/max X and Y of current live cells (to show expansion).

## 4. Acceptance Criteria
* **Input:** A "Glider" pattern at `(0, 0)`.
* **Action:** Run 100 ticks.
* **Result:** The Glider should still exist (same shape) but be located at `(25, 25)` (approx).
* **Constraint:** The code must be efficient enough to run 1000 ticks in under 1 second.

## 5. Research Instructions
1.  How to efficiently calculate neighbors using a Dictionary or Counter?
    * Strategy: Iterate through all live cells. For each live cell, increment the "neighbor count" for all 8 surrounding cells in a `Counter`. Then, apply rules based on the counts.

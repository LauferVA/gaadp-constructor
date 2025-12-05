# Project Specification: Hexagonal Snake Game

## 1. Objective
Create a playable version of the classic game "Snake" using Python and `pygame`. However, instead of a square grid (`x, y`), the game must operate on a **Hexagonal Grid**.

## 2. Core Mechanics
* **The Grid:** A honeycomb pattern (Hexagons flat-topped or pointy-topped—agent's choice).
* **Movement:** The snake does not move Up/Down/Left/Right. It moves in 6 directions (e.g., NE, E, SE, SW, W, NW).
* **Input:** Map keyboard keys to these 6 directions. (e.g., Q, W, E, A, S, D or standard arrows modified for diagonal logic).
* **Growth:** Eating food lengthens the snake by 1 hex.
* **Collision:** Die if hitting the wall or self.

## 3. The Core Challenge (Research Phase Focus)
* **Coordinate System:** The agent must implement a coordinate system for hexagons (e.g., Axial coordinates `q, r` or Cube coordinates `x, y, z`). Standard Cartesian `x, y` makes movement logic very difficult.
* **Pixel Mapping:** Converting the abstract grid coordinates to pixel coordinates for drawing the polygons on screen.

## 4. Requirements
* **Library:** `pygame`.
* **Visuals:** Draw simple polygons for the snake segments and the food.
* **Scoring:** Display score on screen.

## 5. Acceptance Criteria
* The game runs without crashing.
* The snake moves seamlessly between hexes (visual alignment is correct).
* The controls logically map to the 6 directions.
* The "tail" follows the "head" correctly through the hexagonal path.

## 6. Research Instructions
1.  Research "Hexagonal Grids" (specifically Red Blob Games' guide is the gold standard).
2.  Select a coordinate system (Axial vs. Offset).
3.  Define the 6 neighbor vectors for that coordinate system.
4.  Derive the formula to convert Grid(q,r) -> Screen(x,y).

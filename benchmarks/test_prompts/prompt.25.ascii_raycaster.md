# Project Specification: ASCII Raycaster (Wolfenstein 3D Style)

## 1. Objective
Create a script `raycaster.py` that renders a 3D first-person view of a simple maze using ASCII characters in the terminal.

## 2. Core Logic (Ray Casting)
* **Map:** A 2D grid (List of Strings) where `#` is a wall and `.` is empty space.
* **Player:** Has `x, y` position and `angle` (direction).
* **Rendering Loop:**
    1.  Field of View (FOV): 60 degrees.
    2.  Screen Width: 80 columns.
    3.  For every column `x` (0 to 79):
        * Calculate ray angle relative to player angle.
        * "March" the ray forward until it hits a `#`.
        * Calculate distance to wall.
        * Draw a vertical line of characters whose height is inversely proportional to distance (`Height = Constant / Distance`).

## 3. Graphics
Use text shading for depth:
* Close wall: `█`
* Medium wall: `▓`
* Far wall: `░`

## 4. Functional Requirements
* **Input:** Static Map.
* **Output:** Print the 3D frame to stdout *once* (animation not required, just a static render of the view).

## 5. The Math Challenge (Research Phase)
* **Fishbowl Correction:** Raw distance calculation causes straight walls to look curved.
* **Correction Formula:** CorrectedDist = RawDist × cos(RayAngle - PlayerAngle).

## 6. Acceptance Criteria
* The output must look like a corridor.
* Walls further away must be shorter in character height.
* Walls must appear flat (not spherical).

## 7. Research Instructions
1.  Research "DDA Algorithm" (Digital Differential Analyzer) vs "Step Raycasting". DDA is more precise for grid traversal.
2.  Understand how to map "Wall Height" to "Number of ASCII Lines".

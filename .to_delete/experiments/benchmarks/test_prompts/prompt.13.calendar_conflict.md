# Project Specification: Calendar Conflict Detector

## 1. Objective
Write a Python class `CalendarManager` that manages a schedule of meetings and efficiently detects conflicts.

## 2. Data Structure
A "Meeting" is defined as a tuple or object with:
* `start_time`: Integer (0 to 2400) or DateTime.
* `end_time`: Integer or DateTime.
* `meeting_id`: String.

## 3. Functional Requirements
* **Method:** `add_meeting(start, end, id) -> bool`
    * Returns `True` if the meeting was added successfully.
    * Returns `False` if it overlaps with *any* existing meeting (do not add it).
* **Method:** `merge_all_intervals() -> List[Meeting]`
    * Returns a compacted list where touching or overlapping blocks are merged (e.g., [9-10] and [10-11] becomes [9-11]).
    * *Note:* This method operates on a hypothetical list of *all* proposed meetings, ignoring the conflict check, to show "busy blocks."

## 4. Constraints
* **Efficiency:** Assume the calendar has 10,000 meetings. `add_meeting` must be efficient.
* **Strict Overlap:**
    * `[10, 11]` and `[11, 12]` do NOT overlap (touching is fine).
    * `[10, 11]` and `[10:30, 11:30]` DO overlap.

## 5. Edge Cases
* **Encapsulation:** A short meeting `[10:15, 10:45]` completely inside a long one `[10:00, 11:00]`.
* **Unsorted Input:** The user may add meetings in random order (e.g., a morning meeting added after an evening meeting). The internal storage must handle this.

## 6. Acceptance Criteria
* **Correctness:**
    ```python
    c = CalendarManager()
    c.add_meeting(100, 200, "A") # True
    c.add_meeting(150, 250, "B") # False (Overlaps A)
    c.add_meeting(200, 300, "C") # True (Touches A)
    ```
* **Performance:** Use `bisect` module or keep the list sorted to allow O(log N) or O(N) checks, rather than iterating the whole list every time.

## 7. Research Phase Instructions
1.  Research "Interval Trees" vs. "Sorted Lists" for this problem. A Sorted List is simpler to implement in Python using `bisect`. Justify your choice.
2.  Research the "Sweep Line Algorithm" if you were to process a static list of 100k events.

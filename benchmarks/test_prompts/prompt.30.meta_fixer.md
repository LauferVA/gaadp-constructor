# Project Specification: The Self-Correcting Python Script

## 1. Objective
Create a Python script `autofix.py` that takes a "broken" python file as input, attempts to run it, captures the error, and uses an "LLM Mock" (heuristics) to patch the file until it runs successfully.

## 2. The "Broken" Input
Create a file `broken.py` containing:
```python
def add(a, b)
    return a + b  # Syntax Error: Missing colon

print(add(5, 10))
```

## 3. Functional Requirements
1. **The Runner:** Use `subprocess.run` to execute `broken.py`.
2. **The Monitor:** Capture `stderr`.
3. **The "Brain" (Heuristic Matcher):**
    * Note: Since we can't call an actual LLM API inside this specific task, you must implement a Regex-based Patcher that acts as the "Brain".
    * If `stderr` contains `SyntaxError` and `def`, insert a colon.
    * If `stderr` contains `NameError`, define the variable = 0.
    * If `stderr` contains `IndentationError`, fix indentation.
4. **The Loop:**
    * Run Code.
    * If Exit Code 0 -> Success! Stop.
    * If Error -> Parse Error -> Apply Patch -> Rewrite File -> Goto 1.
    * Max Retries: 5 (to prevent infinite loops).

## 4. Acceptance Criteria
1. Run `python autofix.py broken.py`.
2. Initial Output: "SyntaxError detected..."
3. Action: Script modifies `broken.py`.
4. Final Output: "Execution Successful: 15".
5. Verification: The `broken.py` file on disk should now contain the colon `:`.

## 5. Research Instructions
1. Research `subprocess.run` arguments: `capture_output=True`, `text=True`.
2. Research how Python formats stack traces in `stderr` to extract the line number of the error.

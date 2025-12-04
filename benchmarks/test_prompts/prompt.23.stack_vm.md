# Project Specification: "PyVM" (Stack Machine Interpreter)

## 1. Objective
Build a Virtual Machine (VM) that executes a custom bytecode language. The VM must be stack-based (like the JVM or Python's internal VM), not register-based.

## 2. Architecture
* **Stack:** A dynamic list of integers.
* **Instruction Pointer (IP):** An integer tracking the current line of code.
* **Memory:** An array of instruction dictionaries.

## 3. Instruction Set (The Spec)
The VM must support these instructions:
* `PUSH <val>`: Pushes value onto stack.
* `POP`: Removes top value.
* `ADD`: Pops top 2 values, adds them, pushes result.
* `SUB`: Pops top 2 (b - a), pushes result.
* `PRINT`: Pops top value and prints to stdout.
* `JMP <target>`: Sets IP to `<target>`.
* `JEQ <target>`: Pops top 2 values. If equal, jumps to `<target>`.
* `HALT`: Stops execution.

## 4. Acceptance Criteria (The Test Program)
The agent must implement the VM *and* verify it by running this bytecode program (calculating 5 + 5 = 10):

```python
program = [
    {"op": "PUSH", "val": 5},
    {"op": "PUSH", "val": 5},
    {"op": "ADD"},
    {"op": "PUSH", "val": 10},
    {"op": "JEQ", "val": 6},  # If 10 == 10, Jump to line 6
    {"op": "HALT"},           # Line 5 (Should be skipped)
    {"op": "PRINT"},          # Line 6 (Should print 10)
    {"op": "HALT"}
]
```

## 5. Constraints
* **Error Handling:** If `ADD` is called with <2 items on stack, raise `StackUnderflowError`.
* **Loop Protection:** Ensure the fetch-execute loop terminates on `HALT`.

## 6. Research Instructions
1. Research "Fetch-Decode-Execute" cycle.
2. Understand the order of operands in subtraction on a stack.
    * Example: Stack is `[10, 3]`. `SUB` usually means `10 - 3` (if 3 is top) or `3 - 10`?
    * Standard Convention: Pop B, Pop A, Result = A - B.

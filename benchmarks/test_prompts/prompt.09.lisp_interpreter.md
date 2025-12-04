# Project Specification: "PyLisp" Interpreter

## 1. Objective
Write a Python script that parses and evaluates a subset of LISP/Scheme syntax.

## 2. Supported Features
* **Atoms:** Integers (`1`, `42`).
* **Operations:** `+`, `-`, `*`, `/`.
* **Functions:**
    * `(define x 10)`: Sets a variable in the global scope.
    * `(if test then else)`: Conditional logic.
* **Structure:** Nested lists, e.g., `(+ 1 (* 2 3))` evaluates to `7`.

## 3. The Core Challenge
You must write a **Parser** that converts the string `"(+ 1 2)"` into a Python list `['+', 1, 2]`. Then write an **Evaluator** that processes that list.
* *Hint:* Tokenize by adding spaces around parentheses `(` and `)`, then `split()`.

## 4. Acceptance Criteria
* Input: `(begin (define r 10) (* 3.14 (* r r)))`
* Output: `314.0`

## 5. Research Instructions
1.  Look up "Peter Norvig's Lispy" for a minimal reference architecture.
2.  Define an `Env` class (environment) to store variable definitions (like a dict).

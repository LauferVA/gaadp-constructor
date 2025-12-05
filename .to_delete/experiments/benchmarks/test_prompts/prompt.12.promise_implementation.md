# Project Specification: "PyPromise" Implementation

## 1. Objective
Implement a Python class named `PyPromise` that mimics the behavior of a JavaScript `Promise`. This is a cleanroom implementation; do not use Python's `asyncio.Future` or `concurrent.futures` to do the heavy lifting. You must build the logic (state, callbacks, resolution) yourself.

## 2. Core Concepts
A Promise has three states:
1.  **PENDING**: Initial state.
2.  **FULFILLED**: Operation completed successfully.
3.  **REJECTED**: Operation failed.

## 3. Functional Requirements
* **Constructor:** `PyPromise(executor)`
    * The `executor` is a function that takes two arguments: `resolve` and `reject`.
    * Example: `PyPromise(lambda resolve, reject: resolve(10))`
* **Methods:**
    * `.then(on_success, on_failure=None)`: Returns a **new** PyPromise.
    * `.catch(on_failure)`: Sugar for `.then(None, on_failure)`.
    * `.resolve(value)`: Static method returning a fulfilled promise.
    * `.reject(reason)`: Static method returning a rejected promise.
* **Chaining:**
    * If `.then()` returns a value, the next Promise in the chain receives that value.
    * If `.then()` returns a *PyPromise*, the next Promise waits for that inner Promise to settle.

## 4. The Core Challenge (Logic)
The hard part is the **"Thenable" Chaining**.
* If I do: `p = PyPromise(...).then(lambda x: x + 1).then(lambda y: y * 2)`
* The internal logic must queue the callbacks and trigger them sequentially only when the state transitions to FULFILLED.

## 5. Edge Cases
* **Async Resolution:** The executor might resolve immediately (sync) or after a delay (async). Your logic must handle `resolve` being called *after* `.then` has already been attached.
* **Error Bubbling:** If an error occurs and there is no `on_failure` handler in the current `.then()`, the error must propagate down the chain to the next `.catch()`.

## 6. Acceptance Criteria
```python
def test_flow():
    p = PyPromise(lambda res, rej: res(5))
    p.then(lambda x: x * 2) \
     .then(lambda x: x + 10) \
     .then(lambda x: print(f"Result: {x}"))
    # Output must be "Result: 20"
```

## 7. Research Phase Instructions
1. Research the "Promise Resolution Procedure" (simpler version).
2. Determine how to store callbacks. Should they be a list of tuples `(on_fulfilled, on_rejected, next_promise)`?
3. How do you handle exceptions thrown inside a `.then` handler? (They should convert the promise state to REJECTED).

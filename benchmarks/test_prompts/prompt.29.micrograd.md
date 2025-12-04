# Project Specification: Scalar Autograd Engine (Backpropagation)

## 1. Objective
Implement a tiny "Autograd" engine similar to PyTorch or Andrej Karpathy's `micrograd`.
It must allow users to build mathematical expressions using `Value` objects, and then automatically compute gradients (derivatives) with respect to all inputs.

## 2. Constraints
* **No NumPy/PyTorch:** You must use standard Python `float` and `math` library.
* **Core Logic:** You must implement the **Chain Rule** of Calculus manually.

## 3. Functional Requirements
* **Class:** `Value(data, _children=())`
    * Stores the scalar value (`data`).
    * Stores the `grad` (derivative), initialized to 0.0.
    * Stores a `_backward` function (lambda) that propagates gradients to children.
* **Operations:** Implement `__add__`, `__mul__`, `__pow__`, `relu`, and `tanh`.
    * *Crucial:* Each operation must define its own local derivative logic and append it to the topological sort.
* **Method:** `backward()`
    * Topological sort the graph.
    * Set `self.grad = 1.0` (seed).
    * Call `_backward()` on nodes in reverse order.

## 4. Acceptance Criteria (The "Neuron" Test)
The agent must include a test script that constructs a single neuron and updates it:
```python
# Inputs
x1 = Value(2.0); x2 = Value(0.0)
# Weights
w1 = Value(-3.0); w2 = Value(1.0)
# Bias
b = Value(6.8813735)

# Forward Pass
n = x1*w1 + x2*w2 + b
o = n.tanh()

# Backward Pass
o.backward()

# Check Gradients
print(w1.grad) # Should be roughly 0.5 (depending on math derivation)
```

## 5. Research Instructions
1. Research "Reverse-Mode Automatic Differentiation".
2. Derive the gradient for Tanh: d/dx tanh(x) = 1 - tanh²(x).
3. Derive the gradient for Addition: If z = x + y, then dL/dx = dL/dz × 1. Gradients just flow through.

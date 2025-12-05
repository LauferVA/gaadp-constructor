# Project Specification: Black-Scholes Option Calculator

## 1. Objective
Create a Python class `OptionCalculator` that calculates the theoretical price and risk metrics ("Greeks") of European options using the Black-Scholes model.

## 2. Core Formulas
The price of a Call option C is:
C = S₀ × N(d₁) - K × e^(-rT) × N(d₂)

Where:
* d₁ = (ln(S₀/K) + (r + σ²/2)T) / (σ × √T)
* d₂ = d₁ - σ × √T
* N(x) is the standard normal cumulative distribution function.

## 3. Requirements
* **Inputs:** Stock Price (S), Strike Price (K), Time to Maturity (T in years), Risk-free Rate (r), Volatility (σ).
* **Outputs:** A dictionary containing:
    * `price`: The calculated option price.
    * `delta`: ∂C/∂S (First derivative wrt Price)
    * `gamma`: ∂²C/∂S² (Second derivative wrt Price)
* **Dependencies:** Use `scipy.stats.norm` for the CDF.

## 4. Acceptance Criteria
* Calculate the price of a Call where S=100, K=100, T=1, r=0.05, σ=0.2.
* Expected Price ≈ 10.45.
* Expected Delta ≈ 0.637.

## 5. Research Instructions
1.  Research the formula for **Gamma** (Γ) in the Black-Scholes model.
2.  Ensure strict handling of division by zero if T=0.

PROJECT: Calculator Package
TARGET DIR: calculator/

OBJECTIVES:
1. Create a Python calculator package with proper module separation:
   - math_ops.py: Basic operations (add, subtract, multiply, divide) with error handling
   - calculator.py: Calculator class that imports math_ops, maintains history
   - cli.py: Command-line interface that imports Calculator

2. Division by zero must raise a clear exception with helpful message.

3. All functions must have type hints and docstrings.

GOVERNANCE:
- cli.py depends on calculator.py depends on math_ops.py (no circular imports)
- All modules must have docstrings
- Follow PEP 8 style guidelines

# CPG BUILDER TEST
# Tests that the AST-based Code Property Graph builder correctly parses Python code.
# Expected: Classes, functions, calls, and imports are extracted and linked.

PROJECT: CPG Builder Validation Test
TARGET FILE(S): core/test_cpg_sample.py

## CONTEXT
This test validates the Code Property Graph (CPG) builder can correctly parse
Python code using AST analysis. The generated code should include multiple
classes, functions, and imports to exercise all CPG extraction capabilities.

## OBJECTIVES
1. Create a Python file with at least 2 classes
2. Include inheritance between classes (class Child(Parent))
3. Include at least 3 functions (including methods)
4. Include at least 2 import statements
5. Include function calls between the defined functions

## CONSTRAINTS
- Output must be syntactically valid Python
- Classes must have docstrings
- Functions must have at least one line of implementation
- No external dependencies beyond standard library

## EXPECTED OUTPUT
A Python file similar to:
```python
import os
from typing import List

class BaseProcessor:
    """Base class for data processing."""

    def process(self, data: str) -> str:
        return data.upper()

class AdvancedProcessor(BaseProcessor):
    """Advanced processor with additional features."""

    def process(self, data: str) -> str:
        result = super().process(data)
        return self.enhance(result)

    def enhance(self, data: str) -> str:
        return f"[ENHANCED] {data}"

def main():
    processor = AdvancedProcessor()
    result = processor.process("hello")
    print(result)

if __name__ == "__main__":
    main()
```

## ACCEPTANCE CRITERIA
- [ ] Code passes Python syntax check (py_compile)
- [ ] CPG builder extracts all classes
- [ ] CPG builder extracts all functions
- [ ] CPG builder identifies inheritance relationships
- [ ] CPG builder identifies function calls
- [ ] Generated graph can be visualized

## ANTI-PATTERNS
- Do NOT create overly complex code
- Do NOT use async/await (keep it simple)
- Do NOT use decorators (keep AST parsing straightforward)

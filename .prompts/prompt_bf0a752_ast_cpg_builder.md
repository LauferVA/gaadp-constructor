PROJECT: Upgrade Static Analysis to AST
TARGET FILE: core/cpg_builder.py

OBJECTIVES:
1. Replace Regex-based analysis in core/cpg_builder.py with Python ast module.
2. Extract Imports, Class Defs, Function Defs, and Calls using AST traversal.
3. Preserves the existing GraphDB node structure.

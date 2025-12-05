# PROTOCOL VALIDATION TEST
# Tests that agents correctly use Pydantic protocols with forced tool_choice.
# Expected: All agents produce structured output via submit_* tools.

PROJECT: Protocol Validation Test
TARGET FILE(S): core/test_protocol_output.py

## CONTEXT
This is a validation test to ensure the protocol-based communication system
is working correctly. The task is intentionally simple to isolate protocol
behavior from complex code generation.

## OBJECTIVES
1. Create a simple Python function that returns the string "Hello, Protocol!"
2. The function should be named `greet_protocol()`
3. Include a docstring explaining this is a protocol test

## CONSTRAINTS
- Output must be a single Python file
- No external dependencies
- Function must be callable with no arguments

## EXPECTED OUTPUT
```python
def greet_protocol():
    """Protocol validation test function."""
    return "Hello, Protocol!"
```

## ACCEPTANCE CRITERIA
- [ ] Architect produces ArchitectOutput via submit_architecture tool
- [ ] Builder produces BuilderOutput via submit_code tool
- [ ] Verifier produces VerifierOutput via submit_verdict tool
- [ ] No parse_error flags in output
- [ ] Code executes without errors

## ANTI-PATTERNS
- Do NOT add unnecessary complexity
- Do NOT include unit tests (this is just the function)
- Do NOT add type hints (keep it minimal)

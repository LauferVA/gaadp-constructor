# LOGGING VALIDATION TEST
# Tests that the logging system correctly captures debug info, errors, and traces.
# Expected: Logs show trace IDs, phase transitions, and agent activity.

PROJECT: Logging Validation Test
TARGET FILE(S): utils/log_validator.py

## CONTEXT
This test validates the logging infrastructure by creating code that will
intentionally trigger different log levels and phases. We want to verify
that trace IDs propagate correctly and that errors are surfaced clearly.

## OBJECTIVES
1. Create a Python module with functions that log at different levels
2. Include a function that raises an exception (to test error logging)
3. Include a function that performs a multi-step operation (to test phase logging)
4. The module should be self-contained and runnable

## CONSTRAINTS
- Use Python's standard logging module
- Include DEBUG, INFO, WARNING, and ERROR level logs
- Functions must have descriptive names indicating their purpose
- No external dependencies

## EXPECTED OUTPUT
```python
import logging

logger = logging.getLogger(__name__)

def log_info_message():
    """Log an info-level message."""
    logger.info("This is an info message")
    return "info logged"

def log_warning_message():
    """Log a warning-level message."""
    logger.warning("This is a warning message")
    return "warning logged"

def log_error_with_exception():
    """Log an error with exception details."""
    try:
        raise ValueError("Intentional error for testing")
    except ValueError as e:
        logger.error(f"Caught exception: {e}", exc_info=True)
        return "error logged"

def multi_step_operation():
    """Perform a multi-step operation with logging."""
    logger.debug("Step 1: Starting operation")
    logger.debug("Step 2: Processing data")
    logger.info("Step 3: Operation complete")
    return "multi-step complete"

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    log_info_message()
    log_warning_message()
    log_error_with_exception()
    multi_step_operation()
```

## ACCEPTANCE CRITERIA
- [ ] Code compiles without errors
- [ ] Running the code produces log output at all levels
- [ ] GAADP logs show trace ID for this run
- [ ] GAADP logs show phase transitions (INIT -> ARCHITECT -> BUILDER -> VERIFIER)
- [ ] Agent decisions are logged

## ANTI-PATTERNS
- Do NOT use print() for logging
- Do NOT create overly complex logging hierarchies
- Do NOT add file handlers (let GAADP handle that)

# GAADP Constructor - Test Suite

This directory contains comprehensive tests for the GAADP Constructor system.

## Quick Start

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all unit tests (fast, no API calls)
pytest tests/test_integration.py -v

# Run critical fixes validation
pytest tests/test_critical_fixes.py -v

# Run LLM Gateway tests (requires API key, ~$0.05 cost)
ANTHROPIC_API_KEY=sk-... pytest tests/test_llm_gateway.py -v
```

## Test Organization

### `test_integration.py` - Existing Unit Tests ✅
- GraphDB persistence and cycle detection
- Sandbox code execution
- Event bus pub/sub
- Test runner functionality
- **No external dependencies, runs offline**

### `test_critical_fixes.py` - Critical Architecture Fixes ⚠️
Tests for the 5 critical improvements:
1. Docker security enforcement (no unsafe fallback)
2. Vector DB graceful degradation with warnings
3. Janitor daemon orphan cleanup
4. Non-blocking human intervention
5. Architect strategy change on escalation

**No API calls, minimal cost**

### `test_llm_gateway.py` - Real API Integration ⚠️
Tests real Anthropic Claude API:
- API connectivity and authentication
- Token tracking and cost estimation
- Tool use handling
- Error handling and retries
- Multi-turn conversations

**Requires ANTHROPIC_API_KEY, costs ~$0.05-0.10**

## Test Markers

Use pytest markers to run specific test categories:

```bash
# Run only fast unit tests
pytest -m unit

# Run integration tests (may require services)
pytest -m integration

# Skip slow tests
pytest -m "not slow"

# Skip tests that cost money
pytest -m "not costs_money"

# Run tests requiring Docker
pytest -m requires_docker
```

## Environment Setup

### Required Environment Variables

```bash
# For LLM integration tests
export ANTHROPIC_API_KEY="sk-ant-..."

# For Docker tests (or Docker must be installed)
# docker --version
```

### Optional Docker Setup

Some tests require Docker for security validation:

```bash
# Install Docker
# https://docs.docker.com/get-docker/

# Verify installation
docker --version
```

## Running Test Levels

### Level 1: Unit Tests (Always Run First)
```bash
pytest tests/test_integration.py -v
```
- **Time:** ~30 seconds
- **Cost:** $0
- **Requirements:** None

### Level 2: Critical Fixes (Pre-Deployment Validation)
```bash
pytest tests/test_critical_fixes.py -v
```
- **Time:** ~1 minute
- **Cost:** $0
- **Requirements:** None (Docker optional)

### Level 3: LLM Gateway (Real API)
```bash
ANTHROPIC_API_KEY=sk-... pytest tests/test_llm_gateway.py -v
```
- **Time:** ~2 minutes
- **Cost:** ~$0.05-0.10
- **Requirements:** ANTHROPIC_API_KEY

### Full Test Suite
```bash
# Run everything (except slow E2E tests)
pytest tests/ -m "not slow" -v
```
- **Time:** ~5 minutes
- **Cost:** ~$0.10

## Test Fixtures

Tests use temporary directories for isolation:
- `.gaadp_test/` - Test graph persistence
- `.sandbox_test/` - Sandbox execution tests
- `.test_runner_test/` - Test runner tests

These are automatically cleaned up after each test.

## Expected Results

### Minimum Viable (Must Pass)
- [ ] All unit tests in `test_integration.py` pass
- [ ] Docker security enforcement works
- [ ] LLM Gateway connects to Claude API
- [ ] Token tracking accurate

### Production Ready (Should Pass)
- [ ] Janitor cleans orphaned nodes
- [ ] Non-blocking human loop works
- [ ] Escalation prompts include strategy changes
- [ ] Cost estimation within 5% of actual

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### "Docker is required but not available"
Either:
1. Install Docker: https://docs.docker.com/get-docker/
2. Or skip Docker tests: `pytest -m "not requires_docker"`

### "sentence-transformers not found"
Embeddings are optional, tests should pass in fallback mode.
To enable: `pip install sentence-transformers`

### Tests hang indefinitely
Check for:
- Network connectivity issues (API calls)
- Infinite loops in async code
- Event bus not properly shut down

Use pytest timeout:
```bash
pytest --timeout=60 tests/
```

## Coverage Analysis

To measure test coverage:

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# View report
open htmlcov/index.html
```

## Continuous Integration

Suggested CI/CD workflow:

```yaml
# .github/workflows/test.yml
- name: Run Unit Tests
  run: pytest tests/test_integration.py -v

- name: Run Critical Fixes
  run: pytest tests/test_critical_fixes.py -v

- name: Run LLM Tests (if API key available)
  if: env.ANTHROPIC_API_KEY
  run: pytest tests/test_llm_gateway.py -v
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Test Development

When adding new tests:

1. Follow existing patterns in test files
2. Use appropriate markers (`@pytest.mark.slow`, etc.)
3. Clean up resources in fixtures
4. Document expected costs if API calls involved
5. Add to this README if new test category

## Support

For issues with tests:
1. Check test output for specific errors
2. Verify environment setup (API keys, Docker, etc.)
3. Check pytest.ini configuration
4. Review test plan in `TEST_PLAN.md`

---

**Last Updated:** 2025-11-21
**Test Suite Version:** 1.0
**Total Tests:** 50+ across all files

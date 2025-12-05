# PROMPT SKELETON
# Copy this to ./prompt.md and fill in the sections below.
# When done, archive the completed prompt with:
#   mv prompt.md .prompts/prompt_<commit>_<short_description>.md

# =============================================================================
# REQUIRED SECTIONS
# =============================================================================

PROJECT: <Short project/task name>
TARGET FILE(S): <path/to/file.py> [, additional files...]

## CONTEXT
<!-- Why are we doing this? What problem does it solve? -->


## OBJECTIVES
<!-- Numbered list of specific, measurable goals -->
1.
2.
3.

## CONSTRAINTS
<!-- What must NOT change? What are the boundaries? -->
-
-

# =============================================================================
# OPTIONAL SECTIONS (include if relevant)
# =============================================================================

## REFERENCE FILES
<!-- Existing files the agent should read for context -->
-

## EXPECTED OUTPUT
<!-- What does success look like? Example output format? -->


## ACCEPTANCE CRITERIA
<!-- How do we know when this is done? -->
- [ ]
- [ ]

## ANTI-PATTERNS
<!-- What should the agent explicitly avoid? -->
-

# =============================================================================
# NOTES
# =============================================================================
# - Keep objectives atomic and testable
# - Reference specific files/classes/functions by name
# - Include expected output format if JSON/structured
# - List constraints explicitly to prevent scope creep

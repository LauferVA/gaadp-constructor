"""
GAADP COMMUNICATION PROTOCOLS

Typed contracts for agent communication. Each protocol corresponds to a
NodeType or operation in the ontology, ensuring structural guarantees.

These Pydantic models serve dual purposes:
1. Validation - Runtime enforcement of agent output structure
2. Tool Schemas - Converted to JSON Schema for forced tool_choice

Usage:
    # Validate agent output
    output = ArchitectOutput.model_validate(tool_call_args)

    # Get JSON schema for tool definition
    schema = ArchitectOutput.model_json_schema()
"""
from pydantic import BaseModel, Field, field_validator, model_validator
import json
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum
from pathlib import Path

from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# ATOMIC BUILDING BLOCKS
# =============================================================================

class NodeSpec(BaseModel):
    """Specification for creating a new node in the graph."""
    id: Optional[str] = Field(
        default=None,
        description="Placeholder ID for referencing this node in edges (e.g., 'spec_utils', 'spec_main')"
    )
    type: Literal["REQ", "RESEARCH", "SPEC", "PLAN", "CODE", "TEST", "TEST_SUITE", "DOC"] = Field(
        description="Node type from ontology"
    )
    content: str = Field(
        description="The actual artifact content (code, spec text, etc.)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (file_path, language, etc.)"
    )


class EdgeSpec(BaseModel):
    """Specification for creating a new edge in the graph."""
    source_id: str = Field(
        description="ID of the source node"
    )
    target_id: str = Field(
        description="ID of the target node"
    )
    relation: Literal[
        "TRACES_TO", "DEPENDS_ON", "IMPLEMENTS",
        "VERIFIES", "DEFINES", "FEEDBACK",
        "CONTAINS", "REFERENCES", "INHERITS"
    ] = Field(
        description="Edge type from ontology"
    )


class FileReference(BaseModel):
    """Reference to a file that should be read or modified."""
    path: str = Field(description="Relative path to file")
    reason: str = Field(description="Why this file is relevant")


# =============================================================================
# ARCHITECT PROTOCOLS
# =============================================================================

class ArchitectOutput(BaseModel):
    """
    Protocol for Architect agent responses.

    The Architect decomposes requirements into atomic specs and plans.
    This is the ONLY valid output format for the Architect.
    """
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the design approach (optional)"
    )
    new_nodes: List[NodeSpec] = Field(
        description="List of nodes to create (SPEC, PLAN, etc.)"
    )
    new_edges: Optional[List[EdgeSpec]] = Field(
        default=None,
        description="Relationships between nodes"
    )
    files_to_read: Optional[List[FileReference]] = Field(
        default=None,
        description="Files that should be examined before implementation"
    )
    risks: Optional[List[str]] = Field(
        default=None,
        description="Potential risks or concerns identified"
    )


class ArchitectToolCall(BaseModel):
    """Wrapper for Architect tool calls during ReAct loop."""
    tool_name: Literal["read_file", "list_directory", "search_web", "fetch_url"]
    arguments: Dict[str, Any]
    reason: str = Field(description="Why this tool call is needed")


# =============================================================================
# BUILDER PROTOCOLS
# =============================================================================

class BuilderOutput(BaseModel):
    """
    Protocol for Builder agent responses.

    Flat structure (no nested objects) ensures LLM returns proper JSON.
    """
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of implementation choices"
    )
    file_path: str = Field(
        description="Relative path where code should be saved"
    )
    language: Literal["python", "javascript", "typescript", "yaml", "json", "markdown"] = Field(
        default="python",
        description="Programming language"
    )
    content: str = Field(
        description="The complete source code"
    )
    dependencies: Optional[List[str]] = Field(
        default=None,
        description="External dependencies/imports required"
    )
    # Note: follow_up_specs removed - LLMs incorrectly serialize nested arrays as strings.
    # If the Builder discovers additional specs, it should use the output tool_choice
    # mechanism or the agent can create them via new_nodes in UnifiedAgentOutput.


# =============================================================================
# VERIFIER PROTOCOLS
# =============================================================================

class VerificationIssue(BaseModel):
    """A single issue found during verification."""
    severity: Literal["error", "warning", "info"] = Field(
        description="How serious is this issue"
    )
    category: Literal[
        "syntax", "import", "type", "security",
        "logic", "style", "missing_implementation"
    ] = Field(
        description="Category of the issue"
    )
    description: str = Field(
        description="Human-readable description of the issue"
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number where issue occurs"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="How to fix this issue"
    )


class VerifierOutput(BaseModel):
    """
    Protocol for Verifier agent responses.

    The Verifier reviews code and produces a structured verdict.
    """
    verdict: Literal["PASS", "FAIL", "CONDITIONAL"] = Field(
        description="Overall verdict"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of the verdict"
    )
    issues: Optional[List[VerificationIssue]] = Field(
        default=None,
        description="List of issues found (required if FAIL)"
    )
    verified_aspects: Optional[List[str]] = Field(
        default=None,
        description="What was successfully verified"
    )
    recommendations: Optional[List[str]] = Field(
        default=None,
        description="Suggestions for improvement (even if PASS)"
    )


# =============================================================================
# SOCRATES PROTOCOLS
# =============================================================================

class ClarifyingQuestion(BaseModel):
    """A question to clarify ambiguous requirements."""
    question: str = Field(
        description="The clarifying question"
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Possible answers if multiple choice"
    )
    context: Optional[str] = Field(
        default=None,
        description="Why this question matters"
    )
    default: Optional[str] = Field(
        default=None,
        description="Suggested default if user doesn't answer"
    )


class SocratesOutput(BaseModel):
    """
    Protocol for Socrates agent responses.

    Socrates identifies ambiguity and asks clarifying questions.
    """
    ambiguities_found: List[str] = Field(
        description="List of ambiguous aspects identified"
    )
    questions: List[ClarifyingQuestion] = Field(
        description="Clarifying questions to ask"
    )
    research_summary: Optional[str] = Field(
        default=None,
        description="Summary of research conducted"
    )
    assumptions: Optional[List[str]] = Field(
        default=None,
        description="Assumptions being made if questions aren't answered"
    )


# =============================================================================
# DIALECTOR PROTOCOLS (Pre-Research Ambiguity Detection)
# =============================================================================

class AmbiguityMarker(BaseModel):
    """An identified ambiguity marker in user input."""
    phrase: str = Field(
        description="The exact phrase that is ambiguous (e.g., 'simple', 'fast', 'good')"
    )
    category: Literal["subjective", "comparative", "pronoun", "undefined_term", "missing_context"] = Field(
        description="Type of ambiguity detected"
    )
    impact: Literal["blocking", "clarifying"] = Field(
        description="blocking = cannot proceed without answer, clarifying = nice to know"
    )
    question: ClarifyingQuestion = Field(
        description="The clarifying question to resolve this ambiguity"
    )


class DialectorOutput(BaseModel):
    """
    Protocol for Dialector agent responses.

    The Dialector runs BEFORE the Researcher on raw REQ input.
    It detects subjective/ambiguous language that should NOT propagate
    into specifications, and creates CLARIFICATION nodes to block progress
    until the user provides concrete answers.

    The dialectic philosophy: User input may contain subjective language,
    but specifications must be objective and unambiguous.
    """
    verdict: Literal["CLEAR", "NEEDS_CLARIFICATION"] = Field(
        description="CLEAR if no blocking ambiguities, NEEDS_CLARIFICATION otherwise"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Analysis of the input for ambiguity"
    )
    ambiguities: List[AmbiguityMarker] = Field(
        default_factory=list,
        description="List of ambiguity markers found in the input"
    )
    clarifications_required: int = Field(
        default=0,
        ge=0,
        description="Number of blocking clarifications needed before proceeding"
    )


# =============================================================================
# RESEARCHER PROTOCOLS (Research Standard v1.0)
# =============================================================================

class InputSpec(BaseModel):
    """Specification for a single input parameter."""
    name: str = Field(description="Parameter name")
    type: str = Field(description="Python type annotation (e.g., 'List[int]')")
    validation: str = Field(description="Executable Python expression for validation")
    trust_boundary: Optional[Literal["trusted", "untrusted", "mixed"]] = Field(
        default="untrusted",
        description="Trust classification for security"
    )


class OutputSpec(BaseModel):
    """Specification for a single output."""
    name: str = Field(description="Output name")
    type: str = Field(description="Python type annotation")
    postcondition: Optional[str] = Field(
        default=None,
        description="Executable Python expression"
    )


class Example(BaseModel):
    """A single concrete example."""
    input: Dict[str, Any] = Field(description="Input values as key-value pairs")
    expected_output: Any = Field(description="Expected result")
    explanation: Optional[str] = Field(default=None, description="Why this example matters")


class EdgeCaseExample(BaseModel):
    """An edge case example with explanation of why it's an edge case."""
    input: Dict[str, Any] = Field(description="Input values")
    expected_output: Any = Field(description="Expected result")
    why_edge: str = Field(description="Explanation of why this is an edge case")


class ErrorCaseExample(BaseModel):
    """An error case example."""
    input: Dict[str, Any] = Field(description="Input values that should cause an error")
    expected_exception: str = Field(description="Expected exception type/message")
    explanation: Optional[str] = Field(default=None)


class AmbiguityCapture(BaseModel):
    """Explicit capture of an ambiguity (PRAGMATIST requirement)."""
    description: str = Field(description="What is ambiguous")
    options: List[str] = Field(description="Possible interpretations")
    resolution_status: Literal["open", "resolved", "deferred"] = Field(
        description="Current resolution state"
    )
    chosen_option: Optional[str] = Field(default=None, description="Selected option if resolved")
    rationale: Optional[str] = Field(default=None, description="Why this option was chosen")


class SuccessCriterion(BaseModel):
    """A single success criterion (PRODUCT OWNER requirement)."""
    criterion: str = Field(description="What must be true for success")
    test_method: str = Field(description="How to verify this criterion")
    is_automated: bool = Field(description="Can this be tested automatically?")


class UnitTest(BaseModel):
    """A unit test specification."""
    name: str = Field(description="Test function name")
    assertion: str = Field(description="Executable Python assertion")
    traces_to_criterion: int = Field(description="Index into success_criteria array")
    priority: Optional[Literal["critical", "high", "medium", "low"]] = Field(
        default="medium"
    )


class FileSpec(BaseModel):
    """File architecture specification."""
    path: str = Field(description="Relative file path")
    purpose: str = Field(description="What this file is responsible for")
    depends_on: Optional[List[str]] = Field(
        default=None,
        description="Other files this depends on"
    )


class ResearcherOutput(BaseModel):
    """
    Protocol for Researcher agent responses.

    Transforms raw prompts into structured Research Artifacts following
    the Research Standard v1.0. Output achieves REVIEWABLE maturity level.
    """
    # === Metadata ===
    maturity_level: Literal["DRAFT", "REVIEWABLE"] = Field(
        default="REVIEWABLE",
        description="DRAFT if ambiguities remain open, REVIEWABLE if resolved"
    )
    completeness_score: float = Field(
        ge=0.0, le=1.0,
        description="Self-assessed completeness (0.0-1.0)"
    )

    # === Domain Context (PRODUCT OWNER) ===
    task_category: Literal["greenfield", "brownfield", "algorithmic", "systems", "debug"] = Field(
        description="Type of task"
    )
    why: str = Field(
        min_length=20,
        description="Business context - why this exists, who benefits"
    )
    success_criteria: List[SuccessCriterion] = Field(
        min_length=1,
        description="Measurable success criteria"
    )

    # === Contracts (THEORIST) ===
    inputs: List[InputSpec] = Field(description="Input specifications")
    outputs: List[OutputSpec] = Field(description="Output specifications")
    preconditions: Optional[List[str]] = Field(
        default=None,
        description="Executable precondition expressions"
    )
    postconditions: Optional[List[str]] = Field(
        default=None,
        description="Executable postcondition expressions"
    )

    # === Examples (PRAGMATIST - minimum 3) ===
    happy_path_examples: List[Example] = Field(
        min_length=1,
        description="Happy path examples"
    )
    edge_case_examples: List[EdgeCaseExample] = Field(
        min_length=1,
        description="Edge case examples"
    )
    error_case_examples: List[ErrorCaseExample] = Field(
        min_length=1,
        description="Error case examples"
    )

    # === Ambiguity Capture (PRAGMATIST) ===
    ambiguities: Optional[List[AmbiguityCapture]] = Field(
        default=None,
        description="Explicitly captured ambiguities and their resolutions"
    )

    # === Verification (THEORIST + QA) ===
    complexity_time: Optional[str] = Field(
        default=None,
        description="Time complexity in Big-O notation"
    )
    complexity_space: Optional[str] = Field(
        default=None,
        description="Space complexity in Big-O notation"
    )
    complexity_justification: Optional[str] = Field(
        default=None,
        description="Why these bounds are expected"
    )
    unit_tests: List[UnitTest] = Field(
        description="Unit test specifications tracing to success criteria"
    )

    # === Security (SECURITY ENGINEER) ===
    forbidden_patterns: Optional[List[str]] = Field(
        default=None,
        description="Patterns that must NOT appear in generated code"
    )
    trust_boundary: Optional[Literal["trusted", "untrusted", "mixed"]] = Field(
        default="untrusted",
        description="Overall trust classification"
    )

    # === Governance ===
    cost_limit: Optional[float] = Field(
        default=1.0,
        description="Maximum USD cost for implementation"
    )
    max_attempts: Optional[int] = Field(
        default=3,
        description="Maximum implementation attempts"
    )

    # === Architecture ===
    files: Optional[List[FileSpec]] = Field(
        default=None,
        description="File structure if multi-file project"
    )
    entry_point: Optional[str] = Field(
        default=None,
        description="Main entry point file"
    )
    dependencies: Optional[List[str]] = Field(
        default=None,
        description="Required external packages"
    )

    # === Reasoning Trace ===
    reasoning: Optional[str] = Field(
        default=None,
        description="Research reasoning and methodology"
    )

    @model_validator(mode='before')
    @classmethod
    def parse_json_strings(cls, data):
        """Parse JSON strings to lists for fields that LLMs sometimes double-serialize."""
        if isinstance(data, dict):
            for field in ['success_criteria', 'inputs', 'outputs', 'preconditions', 'postconditions',
                          'happy_path_examples', 'edge_case_examples', 'error_case_examples',
                          'ambiguities', 'unit_tests', 'forbidden_patterns', 'files', 'dependencies']:
                if field in data and isinstance(data[field], str):
                    try:
                        data[field] = json.loads(data[field])
                    except (json.JSONDecodeError, TypeError):
                        pass
        return data


class ResearchVerifierOutput(BaseModel):
    """
    Protocol for Research Verifier agent responses.

    Validates Research Artifacts against the 10-criterion checklist.
    Requires 8/10 criteria to pass.
    """
    verdict: Literal["PASS", "FAIL"] = Field(
        description="Overall verdict (PASS requires 8/10 criteria)"
    )
    completeness_score: float = Field(
        ge=0.0, le=1.0,
        description="Calculated completeness score"
    )
    criteria_passed: int = Field(
        ge=0, le=10,
        description="Number of criteria passed (out of 10)"
    )

    # === Per-Criterion Results ===
    criterion_1_input_types: bool = Field(
        description="All input types fully specified with validation"
    )
    criterion_2_output_types: bool = Field(
        description="All output types fully specified"
    )
    criterion_3_examples: bool = Field(
        description="At least 3 examples (happy, edge, error)"
    )
    criterion_4_complexity: bool = Field(
        description="Complexity bounds stated with justification"
    )
    criterion_5_dependencies: bool = Field(
        description="Dependencies declared"
    )
    criterion_6_security: bool = Field(
        description="Security posture defined with trust boundary"
    )
    criterion_7_files: bool = Field(
        description="File structure mapped (if multi-file)"
    )
    criterion_8_tests: bool = Field(
        description="Acceptance tests defined tracing to criteria"
    )
    criterion_9_rationale: bool = Field(
        description="Research rationale documented"
    )
    criterion_10_no_ambiguity: bool = Field(
        description="No ambiguous pronouns or unclear referents"
    )

    # === Feedback ===
    issues: Optional[List[str]] = Field(
        default=None,
        description="Specific issues found (required if FAIL)"
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Improvement suggestions"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Verification reasoning"
    )


# =============================================================================
# TESTER PROTOCOLS (Gen-2 TDD Pipeline)
# =============================================================================

class TestResult(BaseModel):
    """Result of a single test execution."""
    name: str = Field(description="Test function name")
    layer: Literal["unit", "property", "contract", "static", "integration", "e2e"] = Field(
        description="Which test layer this belongs to"
    )
    passed: bool = Field(description="Whether the test passed")
    duration_ms: int = Field(description="Execution time in milliseconds")
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if test failed"
    )
    stdout: Optional[str] = Field(
        default=None,
        description="Standard output from test"
    )
    stderr: Optional[str] = Field(
        default=None,
        description="Standard error from test"
    )
    coverage: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Code coverage for this test (0.0-1.0)"
    )


class StaticAnalysisResult(BaseModel):
    """Result of AST/security analysis."""
    passed: bool = Field(description="Whether static analysis passed")
    forbidden_imports_found: List[str] = Field(
        default_factory=list,
        description="List of forbidden imports detected"
    )
    forbidden_patterns_found: List[str] = Field(
        default_factory=list,
        description="List of forbidden code patterns detected"
    )
    cyclomatic_complexity: Dict[str, int] = Field(
        default_factory=dict,
        description="function_name -> complexity score"
    )
    security_issues: List[str] = Field(
        default_factory=list,
        description="Security vulnerabilities detected"
    )


class TesterOutput(BaseModel):
    """
    Protocol for Tester agent responses.

    The Tester is the GATEKEEPER - code cannot reach the Verifier
    without passing the Tester's test suite.
    """
    # === Overall Verdict ===
    verdict: Literal["PASS", "FAIL", "NEEDS_REVISION"] = Field(
        description="PASS=all tests pass, FAIL=blocking issues, NEEDS_REVISION=send back to builder"
    )

    # === Test Code Generated ===
    test_file_path: str = Field(
        description="Path to pytest file (e.g., test_fibonacci.py)"
    )
    test_code: str = Field(
        description="Complete pytest test suite content"
    )

    # === Execution Results ===
    unit_results: List[TestResult] = Field(
        default_factory=list,
        description="Results from unit tests"
    )
    property_results: List[TestResult] = Field(
        default_factory=list,
        description="Results from property-based tests (Hypothesis)"
    )
    contract_results: List[TestResult] = Field(
        default_factory=list,
        description="Results from contract/schema tests"
    )
    integration_results: List[TestResult] = Field(
        default_factory=list,
        description="Results from integration tests"
    )

    # === Static Analysis ===
    static_analysis: StaticAnalysisResult = Field(
        description="Results from static code analysis"
    )

    # === Coverage ===
    overall_coverage: float = Field(
        ge=0.0, le=1.0,
        description="Combined code coverage (0.0-1.0)"
    )
    uncovered_lines: List[int] = Field(
        default_factory=list,
        description="Line numbers not covered by tests"
    )

    # === Feedback for Builder (if NEEDS_REVISION) ===
    revision_feedback: Optional[str] = Field(
        default=None,
        description="Specific instructions for Builder to fix issues"
    )
    failed_test_details: Optional[List[str]] = Field(
        default=None,
        description="Detailed failure messages"
    )

    # === Metrics ===
    total_tests: int = Field(description="Total number of tests executed")
    tests_passed: int = Field(description="Number of tests that passed")
    tests_failed: int = Field(description="Number of tests that failed")
    execution_time_ms: int = Field(description="Total execution time in milliseconds")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _inline_refs(schema: Dict) -> Dict:
    """
    Recursively inline all $ref references in a JSON schema.

    LLMs often misinterpret $ref pointers, returning JSON strings instead
    of properly nested structures. This function resolves all references
    to produce a self-contained schema that LLMs handle correctly.

    Args:
        schema: JSON schema dict (potentially with $defs and $ref)

    Returns:
        Schema with all $ref inlined
    """
    defs = schema.pop('$defs', {})

    def resolve(obj):
        if isinstance(obj, dict):
            if '$ref' in obj:
                ref_path = obj['$ref']
                # Extract definition name from "#/$defs/NodeSpec"
                if ref_path.startswith('#/$defs/'):
                    def_name = ref_path[len('#/$defs/'):]
                    if def_name in defs:
                        # Return a resolved copy of the definition
                        resolved = resolve(defs[def_name].copy())
                        # Remove title from inlined definitions
                        resolved.pop('title', None)
                        return resolved
                return obj
            return {k: resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve(item) for item in obj]
        return obj

    return resolve(schema)


def protocol_to_tool_schema(protocol_class: type[BaseModel], tool_name: str) -> Dict:
    """
    Convert a Pydantic protocol to an Anthropic tool definition.

    Args:
        protocol_class: The Pydantic model class
        tool_name: Name for the tool

    Returns:
        Tool definition dict for Anthropic API
    """
    schema = protocol_class.model_json_schema()

    # Remove Pydantic-specific keys that Anthropic doesn't need
    schema.pop('title', None)

    # Inline all $ref to produce a self-contained schema
    # This prevents LLMs from misinterpreting nested structures
    schema = _inline_refs(schema)

    return {
        "name": tool_name,
        "description": protocol_class.__doc__ or f"Submit {tool_name} output",
        "input_schema": schema
    }


def get_agent_tools(agent_role: str) -> List[Dict]:
    """
    Get the required output tool for an agent role.

    This is used with tool_choice to force structured output.

    Args:
        agent_role: One of ARCHITECT, BUILDER, VERIFIER, SOCRATES, RESEARCHER, RESEARCH_VERIFIER, TESTER

    Returns:
        List with the output tool definition
    """
    tool_map = {
        "ARCHITECT": (ArchitectOutput, "submit_architecture"),
        "BUILDER": (BuilderOutput, "submit_code"),
        "VERIFIER": (VerifierOutput, "submit_verdict"),
        "SOCRATES": (SocratesOutput, "submit_questions"),
        "RESEARCHER": (ResearcherOutput, "submit_research"),
        "RESEARCH_VERIFIER": (ResearchVerifierOutput, "submit_research_verdict"),
        "TESTER": (TesterOutput, "submit_tests"),
        "DIALECTOR": (DialectorOutput, "submit_dialectic"),
    }

    if agent_role not in tool_map:
        raise ValueError(f"Unknown agent role: {agent_role}")

    protocol_class, tool_name = tool_map[agent_role]
    return [protocol_to_tool_schema(protocol_class, tool_name)]


def validate_agent_output(agent_role: str, output: Dict) -> BaseModel:
    """
    Validate and parse agent output against its protocol.

    Args:
        agent_role: The agent role
        output: The raw output dict (from tool call arguments)

    Returns:
        Validated Pydantic model instance

    Raises:
        ValidationError: If output doesn't match protocol
    """
    protocol_map = {
        "ARCHITECT": ArchitectOutput,
        "BUILDER": BuilderOutput,
        "VERIFIER": VerifierOutput,
        "SOCRATES": SocratesOutput,
        "RESEARCHER": ResearcherOutput,
        "RESEARCH_VERIFIER": ResearchVerifierOutput,
        "TESTER": TesterOutput,
        "DIALECTOR": DialectorOutput,
    }

    protocol_class = protocol_map.get(agent_role)
    if not protocol_class:
        raise ValueError(f"Unknown agent role: {agent_role}")

    return protocol_class.model_validate(output)


# =============================================================================
# AGENT CONFIGURATION (For GenericAgent)
# =============================================================================

class AgentConfig(BaseModel):
    """
    Configuration for a GenericAgent loaded from agent_manifest.yaml.

    This replaces hardcoded agent classes with declarative configuration.
    Adding a new agent type = adding a new entry in the manifest.
    """
    role_name: str = Field(
        description="Agent role identifier (ARCHITECT, BUILDER, etc.)"
    )
    description: str = Field(
        description="Human-readable description of what this agent does"
    )
    input_node_types: List[str] = Field(
        description="Node types that trigger this agent"
    )
    output_node_types: List[str] = Field(
        description="Node types this agent can produce"
    )
    dispatch_conditions: List[str] = Field(
        default_factory=list,
        description="Conditions from AGENT_DISPATCH that trigger this agent"
    )
    system_prompt: str = Field(
        description="System prompt for the agent (inline in YAML)"
    )
    allowed_tools: List[str] = Field(
        default_factory=list,
        description="Tools this agent can use"
    )
    output_protocol: str = Field(
        description="Name of the Pydantic output model (e.g., 'ArchitectOutput')"
    )
    output_tool_name: str = Field(
        description="Name of the output tool for forced tool_choice"
    )
    default_cost_limit: float = Field(
        default=1.0,
        description="Default max USD cost per invocation"
    )
    default_max_attempts: int = Field(
        default=3,
        description="Default max retry attempts"
    )


class AgentManifest(BaseModel):
    """Complete agent manifest loaded from YAML."""
    architect: Optional[AgentConfig] = None
    builder: Optional[AgentConfig] = None
    verifier: Optional[AgentConfig] = None
    socrates: Optional[AgentConfig] = None
    dialector: Optional[AgentConfig] = None  # Pre-research ambiguity detection
    researcher: Optional[AgentConfig] = None
    research_verifier: Optional[AgentConfig] = None
    tester: Optional[AgentConfig] = None  # Gen-2 TDD
    global_settings: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="global"
    )

    class Config:
        populate_by_name = True

    def get_config(self, role: str) -> Optional[AgentConfig]:
        """Get config by role name (case-insensitive)."""
        role_lower = role.lower().replace("_", "")
        # Handle special cases
        role_map = {
            "researchverifier": "research_verifier",
            "researcher": "researcher",
            "tester": "tester",
        }
        attr_name = role_map.get(role_lower, role_lower)
        return getattr(self, attr_name, None)


# =============================================================================
# UNIFIED AGENT OUTPUT (For GenericAgent)
# =============================================================================

class UnifiedAgentOutput(BaseModel):
    """
    Standardized output structure for ALL agents via GenericAgent.

    This provides a common interface while allowing agent-specific
    details in the protocol_output field.
    """
    # Reasoning trace
    thought: Optional[str] = Field(
        default=None,
        description="Agent's reasoning process (for debugging/audit)"
    )
    plan: Optional[str] = Field(
        default=None,
        description="What the agent intends to do"
    )

    # Graph mutations
    new_nodes: List[NodeSpec] = Field(
        default_factory=list,
        description="Nodes to create in the graph"
    )
    new_edges: List[EdgeSpec] = Field(
        default_factory=list,
        description="Edges to create in the graph"
    )
    status_updates: Dict[str, str] = Field(
        default_factory=dict,
        description="Node ID -> new status"
    )
    metadata_updates: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Node ID -> metadata key-value pairs to update"
    )

    # File artifacts
    artifacts: Dict[str, str] = Field(
        default_factory=dict,
        description="Files to write: path -> content"
    )

    # Original protocol output (for backwards compatibility)
    protocol_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw output from agent-specific protocol"
    )

    # Metadata
    agent_role: Optional[str] = Field(
        default=None,
        description="Which agent produced this output"
    )
    cost_incurred: float = Field(
        default=0.0,
        description="USD cost of this agent invocation"
    )
    tokens_used: Dict[str, int] = Field(
        default_factory=dict,
        description="Token counts: input_tokens, output_tokens"
    )


# =============================================================================
# GRAPH CONTEXT (Passed to Agents)
# =============================================================================

class GraphContext(BaseModel):
    """
    Context from the graph passed to an agent for processing.

    The runtime builds this by querying the graph neighborhood.
    """
    # The node being processed
    node_id: str
    node_type: str
    node_content: str
    node_status: str
    node_metadata: Dict[str, Any] = Field(default_factory=dict)

    # Graph neighborhood
    parent_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes this node traces to"
    )
    child_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes that trace to this node"
    )
    dependency_nodes: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Nodes this depends on (DEPENDS_ON edges)"
    )
    dependency_code: Dict[str, str] = Field(
        default_factory=dict,
        description="Verified CODE content of dependencies (file_path -> code)"
    )

    # The root requirement
    requirement_content: Optional[str] = Field(
        default=None,
        description="Content of the root REQ node"
    )
    requirement_id: Optional[str] = Field(
        default=None,
        description="ID of the root REQ node"
    )

    # Research artifact (if RESEARCH phase completed for this REQ)
    research_artifact: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Verified RESEARCH artifact for the requirement"
    )

    # Feedback from previous attempts
    feedback: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="FEEDBACK edges pointing to this node"
    )
    previous_attempts: int = Field(
        default=0,
        description="How many times this node has been processed"
    )


# =============================================================================
# MANIFEST LOADING
# =============================================================================

def load_agent_manifest(path: str = "config/agent_manifest.yaml") -> AgentManifest:
    """
    Load agent manifest from YAML file.

    Args:
        path: Path to agent_manifest.yaml

    Returns:
        Parsed AgentManifest
    """
    import yaml

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Agent manifest not found: {path}")

    with open(manifest_path) as f:
        data = yaml.safe_load(f)

    return AgentManifest.model_validate(data)


def get_protocol_class(protocol_name: str) -> type[BaseModel]:
    """
    Get a protocol class by name.

    Args:
        protocol_name: Name like 'ArchitectOutput'

    Returns:
        The Pydantic model class
    """
    protocol_map = {
        "ArchitectOutput": ArchitectOutput,
        "BuilderOutput": BuilderOutput,
        "VerifierOutput": VerifierOutput,
        "SocratesOutput": SocratesOutput,
        "ResearcherOutput": ResearcherOutput,
        "ResearchVerifierOutput": ResearchVerifierOutput,
        "TesterOutput": TesterOutput,
        "DialectorOutput": DialectorOutput,
    }

    if protocol_name not in protocol_map:
        raise ValueError(f"Unknown protocol: {protocol_name}")

    return protocol_map[protocol_name]


# =============================================================================
# SCHEMA EXPORT (for debugging / documentation)
# =============================================================================

if __name__ == "__main__":
    import json

    print("=== ARCHITECT OUTPUT SCHEMA ===")
    print(json.dumps(ArchitectOutput.model_json_schema(), indent=2))

    print("\n=== BUILDER OUTPUT SCHEMA ===")
    print(json.dumps(BuilderOutput.model_json_schema(), indent=2))

    print("\n=== VERIFIER OUTPUT SCHEMA ===")
    print(json.dumps(VerifierOutput.model_json_schema(), indent=2))

    print("\n=== TOOL DEFINITION EXAMPLE ===")
    print(json.dumps(get_agent_tools("ARCHITECT"), indent=2))

    print("\n=== AGENT CONFIG SCHEMA ===")
    print(json.dumps(AgentConfig.model_json_schema(), indent=2))

    print("\n=== UNIFIED AGENT OUTPUT SCHEMA ===")
    print(json.dumps(UnifiedAgentOutput.model_json_schema(), indent=2))

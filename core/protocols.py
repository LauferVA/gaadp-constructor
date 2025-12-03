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
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Union
from enum import Enum
from pathlib import Path

from core.ontology import NodeType, EdgeType, NodeStatus


# =============================================================================
# ATOMIC BUILDING BLOCKS
# =============================================================================

class NodeSpec(BaseModel):
    """Specification for creating a new node in the graph."""
    type: Literal["REQ", "SPEC", "PLAN", "CODE", "TEST", "DOC"] = Field(
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

class CodeOutput(BaseModel):
    """
    Protocol for Builder agent code generation.

    The Builder produces code artifacts from specs.
    """
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
    test_hints: Optional[str] = Field(
        default=None,
        description="Suggestions for how to test this code"
    )


class BuilderOutput(BaseModel):
    """
    Full Builder response including reasoning.
    """
    reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of implementation choices"
    )
    code: CodeOutput = Field(
        description="The generated code artifact"
    )
    follow_up_specs: Optional[List[NodeSpec]] = Field(
        default=None,
        description="Additional specs discovered during implementation"
    )


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
# UTILITY FUNCTIONS
# =============================================================================

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
        agent_role: One of ARCHITECT, BUILDER, VERIFIER, SOCRATES

    Returns:
        List with the output tool definition
    """
    tool_map = {
        "ARCHITECT": (ArchitectOutput, "submit_architecture"),
        "BUILDER": (BuilderOutput, "submit_code"),
        "VERIFIER": (VerifierOutput, "submit_verdict"),
        "SOCRATES": (SocratesOutput, "submit_questions"),
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
    system_prompt_path: str = Field(
        description="Path to system prompt markdown file"
    )
    system_prompt_fallback: Optional[str] = Field(
        default=None,
        description="Inline prompt used if file not found"
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
    global_settings: Optional[Dict[str, Any]] = Field(
        default=None,
        alias="global"
    )

    class Config:
        populate_by_name = True

    def get_config(self, role: str) -> Optional[AgentConfig]:
        """Get config by role name (case-insensitive)."""
        role_lower = role.lower()
        return getattr(self, role_lower, None)


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

    # The root requirement
    requirement_content: Optional[str] = Field(
        default=None,
        description="Content of the root REQ node"
    )
    requirement_id: Optional[str] = Field(
        default=None,
        description="ID of the root REQ node"
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

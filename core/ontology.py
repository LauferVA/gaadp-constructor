"""
GAADP ONTOLOGY DEFINITIONS
"""
from enum import Enum

class AgentRole(str, Enum):
    ARCHITECT = "ARCHITECT"
    BUILDER = "BUILDER"
    VERIFIER = "VERIFIER"
    CURATOR = "CURATOR"
    SOCRATES = "SOCRATES"
    SENTINEL = "SENTINEL"
    TREASURER = "TREASURER"
    LIBRARIAN = "LIBRARIAN"

class NodeType(str, Enum):
    # High-level artifact types
    REQ = "REQ"
    SPEC = "SPEC"
    PLAN = "PLAN"
    CODE = "CODE"
    TEST = "TEST"
    DOC = "DOC"
    STATE = "STATE"
    DEAD_END = "DEAD_END"
    # AST-level types for Code Property Graph
    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    CALL = "CALL"
    IMPORT = "IMPORT"

class EdgeType(str, Enum):
    # High-level relationships
    TRACES_TO = "TRACES_TO"
    DEPENDS_ON = "DEPENDS_ON"
    IMPLEMENTS = "IMPLEMENTS"
    VERIFIES = "VERIFIES"
    DEFINES = "DEFINES"
    FEEDBACK = "FEEDBACK"  # Critique from Verifier back to SPEC for retry
    # AST-level relationships for Code Property Graph
    CONTAINS = "CONTAINS"      # File contains class/function
    REFERENCES = "REFERENCES"  # Code references/calls another entity
    INHERITS = "INHERITS"      # Class inherits from another

class NodeStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"

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
    REQ = "REQ"
    SPEC = "SPEC"
    PLAN = "PLAN"
    CODE = "CODE"
    TEST = "TEST"
    DOC = "DOC"
    STATE = "STATE"
    DEAD_END = "DEAD_END"

class EdgeType(str, Enum):
    TRACES_TO = "TRACES_TO"
    DEPENDS_ON = "DEPENDS_ON"
    IMPLEMENTS = "IMPLEMENTS"
    VERIFIES = "VERIFIES"
    DEFINES = "DEFINES"
    FEEDBACK = "FEEDBACK"  # Critique from Verifier back to SPEC for retry

class NodeStatus(str, Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    COMPLETE = "COMPLETE"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"

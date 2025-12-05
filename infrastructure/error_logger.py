"""
ERROR LOGGER - Runtime Error Tracking with Git Commit Tagging

Catches and classifies errors at occurrence time, tagging each with:
- Current git commit hash (HEAD)
- Error category (derived from ontology)
- Node context (type, status, id)
- Timestamp

Errors are stored per-commit in .gaadp/errors/{commit_hash}.jsonl
This enables:
1. Grouping errors by factory version
2. Regression testing (compare error rates across commits)
3. Pattern detection for recurring error modes
"""
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from core.ontology import NodeType, NodeStatus, EdgeType

logger = logging.getLogger("ErrorLogger")


# =============================================================================
# ERROR CATEGORIES (Derived from System Ontology)
# =============================================================================

class ErrorCategory(str, Enum):
    """
    Categories of errors, aligned with system ontology.

    These map to the phases/components where errors occur.
    """
    # Pipeline phase errors
    DIALECTIC = "DIALECTIC"          # Ambiguity detection failures
    RESEARCH = "RESEARCH"            # Research transformation failures
    DECOMPOSITION = "DECOMPOSITION"  # SPEC generation failures
    BUILD = "BUILD"                  # CODE generation failures
    TEST = "TEST"                    # Test execution failures (TDD loop)
    VERIFICATION = "VERIFICATION"    # Final verification failures

    # System errors
    TRANSITION = "TRANSITION"        # Invalid state transition
    DISPATCH = "DISPATCH"            # Agent dispatch failures
    GOVERNANCE = "GOVERNANCE"        # Cost/security/timeout violations
    GRAPH = "GRAPH"                  # Graph integrity errors

    # External errors
    LLM = "LLM"                      # LLM API errors
    IO = "IO"                        # File/network errors
    UNKNOWN = "UNKNOWN"              # Unclassified errors


# =============================================================================
# ERROR SEVERITY (Aligned with TDD Verdicts)
# =============================================================================

class ErrorSeverity(str, Enum):
    """
    Severity levels aligned with TDD verdict system.
    """
    CRITICAL = "CRITICAL"    # Security/safety issues - immediate halt
    ERROR = "ERROR"          # Functional failures - escalation
    WARNING = "WARNING"      # Non-blocking issues - logged
    INFO = "INFO"            # Informational - for pattern analysis


# =============================================================================
# ERROR RECORD (The Unit of Error Tracking)
# =============================================================================

@dataclass
class ErrorRecord:
    """
    A single error occurrence with full context.

    Designed for append-only logging (JSONL format).
    """
    # Git context
    git_commit: str
    git_branch: str

    # Error classification
    category: str  # ErrorCategory.value
    severity: str  # ErrorSeverity.value
    message: str

    # Node context (if applicable)
    node_id: Optional[str] = None
    node_type: Optional[str] = None  # NodeType.value
    node_status: Optional[str] = None  # NodeStatus.value

    # Agent context
    agent_role: Optional[str] = None

    # Timing
    timestamp: str = None

    # Stack trace (if exception)
    stack_trace: Optional[str] = None

    # Additional context
    extra: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.extra is None:
            self.extra = {}


# =============================================================================
# GIT HELPER
# =============================================================================

def get_git_info() -> Dict[str, str]:
    """
    Get current git commit and branch.

    Returns:
        Dict with 'commit' and 'branch' keys
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True
        ).stdout.strip()

        return {"commit": commit, "branch": branch}
    except subprocess.CalledProcessError:
        return {"commit": "unknown", "branch": "unknown"}


# =============================================================================
# ERROR CLASSIFIER
# =============================================================================

def classify_error(
    error: Exception,
    node_type: Optional[str] = None,
    agent_role: Optional[str] = None
) -> ErrorCategory:
    """
    Classify an error into a category based on context.

    Classification priority:
    1. Agent role (if known)
    2. Node type (if known)
    3. Exception type
    4. Error message patterns
    """
    error_msg = str(error).lower()

    # Agent-based classification
    if agent_role:
        role_map = {
            "DIALECTOR": ErrorCategory.DIALECTIC,
            "RESEARCHER": ErrorCategory.RESEARCH,
            "RESEARCH_VERIFIER": ErrorCategory.RESEARCH,
            "ARCHITECT": ErrorCategory.DECOMPOSITION,
            "BUILDER": ErrorCategory.BUILD,
            "TESTER": ErrorCategory.TEST,
            "VERIFIER": ErrorCategory.VERIFICATION,
            "SOCRATES": ErrorCategory.DIALECTIC,
        }
        if agent_role.upper() in role_map:
            return role_map[agent_role.upper()]

    # Node type classification
    if node_type:
        type_map = {
            NodeType.REQ.value: ErrorCategory.DIALECTIC,
            NodeType.RESEARCH.value: ErrorCategory.RESEARCH,
            NodeType.SPEC.value: ErrorCategory.DECOMPOSITION,
            NodeType.PLAN.value: ErrorCategory.DECOMPOSITION,
            NodeType.CODE.value: ErrorCategory.BUILD,
            NodeType.TEST.value: ErrorCategory.TEST,
            NodeType.TEST_SUITE.value: ErrorCategory.TEST,
            NodeType.CLARIFICATION.value: ErrorCategory.DIALECTIC,
            NodeType.ESCALATION.value: ErrorCategory.GOVERNANCE,
        }
        if node_type in type_map:
            return type_map[node_type]

    # Exception type classification
    error_type = type(error).__name__
    type_patterns = {
        "TransitionError": ErrorCategory.TRANSITION,
        "DispatchError": ErrorCategory.DISPATCH,
        "CostLimitExceeded": ErrorCategory.GOVERNANCE,
        "TimeoutError": ErrorCategory.GOVERNANCE,
        "SecurityViolation": ErrorCategory.GOVERNANCE,
        "GraphIntegrityError": ErrorCategory.GRAPH,
        "APIError": ErrorCategory.LLM,
        "RateLimitError": ErrorCategory.LLM,
        "IOError": ErrorCategory.IO,
        "FileNotFoundError": ErrorCategory.IO,
    }
    for pattern, category in type_patterns.items():
        if pattern.lower() in error_type.lower():
            return category

    # Message pattern classification
    msg_patterns = {
        "cost": ErrorCategory.GOVERNANCE,
        "timeout": ErrorCategory.GOVERNANCE,
        "security": ErrorCategory.GOVERNANCE,
        "clearance": ErrorCategory.GOVERNANCE,
        "api": ErrorCategory.LLM,
        "rate limit": ErrorCategory.LLM,
        "model": ErrorCategory.LLM,
        "transition": ErrorCategory.TRANSITION,
        "status": ErrorCategory.TRANSITION,
        "dispatch": ErrorCategory.DISPATCH,
        "agent": ErrorCategory.DISPATCH,
        "graph": ErrorCategory.GRAPH,
        "node": ErrorCategory.GRAPH,
        "edge": ErrorCategory.GRAPH,
    }
    for pattern, category in msg_patterns.items():
        if pattern in error_msg:
            return category

    return ErrorCategory.UNKNOWN


def classify_severity(
    error: Exception,
    category: ErrorCategory
) -> ErrorSeverity:
    """
    Determine error severity based on category and error content.
    """
    error_msg = str(error).lower()

    # Critical: Security or safety issues
    critical_patterns = ["security", "injection", "unauthorized", "critical"]
    if any(p in error_msg for p in critical_patterns):
        return ErrorSeverity.CRITICAL

    # Category-based defaults
    category_severity = {
        ErrorCategory.GOVERNANCE: ErrorSeverity.ERROR,
        ErrorCategory.TRANSITION: ErrorSeverity.ERROR,
        ErrorCategory.LLM: ErrorSeverity.WARNING,  # Often transient
        ErrorCategory.IO: ErrorSeverity.WARNING,
        ErrorCategory.UNKNOWN: ErrorSeverity.INFO,
    }

    return category_severity.get(category, ErrorSeverity.ERROR)


# =============================================================================
# ERROR LOGGER CLASS
# =============================================================================

class ErrorLogger:
    """
    Logs errors at occurrence time with git commit tagging.

    Storage format: .gaadp/errors/{commit_hash}.jsonl
    Each line is a JSON-encoded ErrorRecord.
    """

    def __init__(self, storage_dir: str = ".gaadp/errors"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._git_info = get_git_info()

        # In-memory buffer for current session
        self._session_errors: List[ErrorRecord] = []

    @property
    def current_commit(self) -> str:
        return self._git_info["commit"]

    @property
    def current_branch(self) -> str:
        return self._git_info["branch"]

    def _get_log_path(self, commit: str = None) -> Path:
        """Get log file path for a commit."""
        commit = commit or self.current_commit
        # Use short hash for filename readability
        short_hash = commit[:8] if len(commit) > 8 else commit
        return self.storage_dir / f"{short_hash}.jsonl"

    def log_error(
        self,
        error: Exception,
        node_id: Optional[str] = None,
        node_type: Optional[str] = None,
        node_status: Optional[str] = None,
        agent_role: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """
        Log an error at occurrence time.

        This is the primary interface - call this when an error occurs.

        Args:
            error: The exception or error message
            node_id: ID of the node being processed (if applicable)
            node_type: Type of node (from NodeType enum)
            node_status: Status of node (from NodeStatus enum)
            agent_role: Role of the agent that encountered the error
            extra: Additional context

        Returns:
            The ErrorRecord that was logged
        """
        import traceback

        # Classify the error
        category = classify_error(error, node_type, agent_role)
        severity = classify_severity(error, category)

        # Get stack trace if exception
        stack_trace = None
        if isinstance(error, Exception):
            stack_trace = "".join(traceback.format_exception(
                type(error), error, error.__traceback__
            ))

        # Create record
        record = ErrorRecord(
            git_commit=self.current_commit,
            git_branch=self.current_branch,
            category=category.value,
            severity=severity.value,
            message=str(error),
            node_id=node_id,
            node_type=node_type,
            node_status=node_status,
            agent_role=agent_role,
            stack_trace=stack_trace,
            extra=extra or {}
        )

        # Add to session buffer
        self._session_errors.append(record)

        # Append to file
        self._write_record(record)

        # Log to standard logger
        log_level = {
            ErrorSeverity.CRITICAL.value: logging.CRITICAL,
            ErrorSeverity.ERROR.value: logging.ERROR,
            ErrorSeverity.WARNING.value: logging.WARNING,
            ErrorSeverity.INFO.value: logging.INFO,
        }.get(severity.value, logging.ERROR)

        logger.log(
            log_level,
            f"[{category.value}] {record.message} "
            f"(commit={record.git_commit[:8]}, node={node_id or 'N/A'})"
        )

        return record

    def _write_record(self, record: ErrorRecord):
        """Append a record to the log file."""
        log_path = self._get_log_path()
        with open(log_path, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def get_session_errors(self) -> List[ErrorRecord]:
        """Get all errors from current session."""
        return list(self._session_errors)

    def get_commit_errors(self, commit: str = None) -> List[ErrorRecord]:
        """
        Get all errors for a specific commit.

        Args:
            commit: Git commit hash (default: current commit)

        Returns:
            List of ErrorRecord objects
        """
        log_path = self._get_log_path(commit)
        if not log_path.exists():
            return []

        errors = []
        with open(log_path, "r") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    errors.append(ErrorRecord(**data))
        return errors

    def get_error_summary(self, commit: str = None) -> Dict[str, Any]:
        """
        Get error summary for a commit.

        Returns:
            Dict with counts by category and severity
        """
        errors = self.get_commit_errors(commit)

        summary = {
            "commit": commit or self.current_commit,
            "total_errors": len(errors),
            "by_category": {},
            "by_severity": {},
            "by_agent": {},
            "by_node_type": {},
        }

        for err in errors:
            # By category
            cat = err.category
            summary["by_category"][cat] = summary["by_category"].get(cat, 0) + 1

            # By severity
            sev = err.severity
            summary["by_severity"][sev] = summary["by_severity"].get(sev, 0) + 1

            # By agent
            if err.agent_role:
                summary["by_agent"][err.agent_role] = summary["by_agent"].get(err.agent_role, 0) + 1

            # By node type
            if err.node_type:
                summary["by_node_type"][err.node_type] = summary["by_node_type"].get(err.node_type, 0) + 1

        return summary

    def compare_commits(self, baseline_commit: str, treatment_commit: str) -> Dict[str, Any]:
        """
        Compare error profiles between two commits.

        This is the key function for regression testing.

        Args:
            baseline_commit: The reference commit (typically HEAD~1)
            treatment_commit: The current commit (typically HEAD)

        Returns:
            Dict with comparison metrics
        """
        baseline = self.get_error_summary(baseline_commit)
        treatment = self.get_error_summary(treatment_commit)

        comparison = {
            "baseline_commit": baseline_commit,
            "treatment_commit": treatment_commit,
            "baseline_total": baseline["total_errors"],
            "treatment_total": treatment["total_errors"],
            "delta": treatment["total_errors"] - baseline["total_errors"],
            "regression": treatment["total_errors"] > baseline["total_errors"],
            "category_delta": {},
            "new_categories": [],
            "resolved_categories": [],
        }

        # Category-level comparison
        all_categories = set(baseline["by_category"].keys()) | set(treatment["by_category"].keys())
        for cat in all_categories:
            base_count = baseline["by_category"].get(cat, 0)
            treat_count = treatment["by_category"].get(cat, 0)
            comparison["category_delta"][cat] = treat_count - base_count

            if base_count == 0 and treat_count > 0:
                comparison["new_categories"].append(cat)
            elif base_count > 0 and treat_count == 0:
                comparison["resolved_categories"].append(cat)

        return comparison

    def list_commits_with_errors(self) -> List[Dict[str, Any]]:
        """
        List all commits that have error logs.

        Returns:
            List of dicts with commit hash and error count
        """
        commits = []
        for log_file in self.storage_dir.glob("*.jsonl"):
            commit_hash = log_file.stem
            errors = self.get_commit_errors(commit_hash)
            commits.append({
                "commit": commit_hash,
                "error_count": len(errors),
                "categories": list(set(e.category for e in errors))
            })
        return sorted(commits, key=lambda x: x["error_count"], reverse=True)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Singleton pattern for easy access from anywhere in the codebase
_error_logger: Optional[ErrorLogger] = None


def get_error_logger() -> ErrorLogger:
    """Get the global ErrorLogger instance."""
    global _error_logger
    if _error_logger is None:
        _error_logger = ErrorLogger()
    return _error_logger


def log_error(
    error: Exception,
    node_id: Optional[str] = None,
    node_type: Optional[str] = None,
    node_status: Optional[str] = None,
    agent_role: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> ErrorRecord:
    """
    Convenience function to log an error using the global logger.

    This is the primary interface for the rest of the codebase.
    """
    return get_error_logger().log_error(
        error=error,
        node_id=node_id,
        node_type=node_type,
        node_status=node_status,
        agent_role=agent_role,
        extra=extra
    )

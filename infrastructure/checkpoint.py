"""
CHECKPOINT MANAGER
Save and restore execution state for pause/resume capability.

Extended (Gen-3) to include:
- Run results storage per git commit
- Integration with ErrorLogger for regression testing
- Meaningful baseline/treatment comparison
"""
import os
import json
import shutil
import logging
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field

from infrastructure.graph_db import GraphDB
from core.ontology import NodeStatus, NodeType

logger = logging.getLogger("Checkpoint")


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    id: str
    created_at: str
    description: str
    node_count: int
    edge_count: int
    pending_count: int
    verified_count: int
    failed_count: int


# =============================================================================
# RUN RESULT (Per-Commit Results for Regression Testing)
# =============================================================================

@dataclass
class RunResult:
    """
    Complete results from a single factory run.

    Indexed by git commit for meaningful baseline/treatment comparison.
    This is the unit of comparison for regression testing.
    """
    # Git context
    git_commit: str
    git_branch: str

    # Timing
    started_at: str
    completed_at: str
    duration_seconds: float

    # Execution metrics
    iterations: int = 0
    nodes_processed: int = 0
    total_cost: float = 0.0

    # Node status counts (using NodeStatus enum values)
    pending_count: int = 0
    processing_count: int = 0
    blocked_count: int = 0
    testing_count: int = 0
    tested_count: int = 0
    verified_count: int = 0
    failed_count: int = 0

    # Node type counts (using NodeType enum values)
    node_type_counts: Dict[str, int] = field(default_factory=dict)

    # Error summary (from ErrorLogger)
    error_count: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_severity: Dict[str, int] = field(default_factory=dict)

    # Success indicators
    success: bool = False
    failure_reason: Optional[str] = None

    # Extra context
    requirement_hash: Optional[str] = None  # Hash of input requirement
    extra: Dict[str, Any] = field(default_factory=dict)


def _get_git_info() -> Dict[str, str]:
    """Get current git commit and branch."""
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


class RunResultStore:
    """
    Stores and retrieves run results indexed by git commit.

    Storage: .gaadp/runs/{commit_hash}.json

    This enables:
    1. Loading baseline results from HEAD~1 (or any prior commit)
    2. Comparing treatment (current run) vs baseline (stored)
    3. Regression detection based on actual stored data
    """

    def __init__(self, storage_dir: str = ".gaadp/runs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._git_info = _get_git_info()

    @property
    def current_commit(self) -> str:
        return self._git_info["commit"]

    @property
    def current_branch(self) -> str:
        return self._git_info["branch"]

    def _get_result_path(self, commit: str) -> Path:
        """Get result file path for a commit."""
        short_hash = commit[:8] if len(commit) > 8 else commit
        return self.storage_dir / f"{short_hash}.json"

    def save_result(self, result: RunResult) -> Path:
        """
        Save a run result.

        Args:
            result: The RunResult to save

        Returns:
            Path to saved file
        """
        path = self._get_result_path(result.git_commit)
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)

        logger.info(f"Run result saved: {path.name}")
        return path

    def load_result(self, commit: str) -> Optional[RunResult]:
        """
        Load a run result for a specific commit.

        Args:
            commit: Git commit hash (full or short)

        Returns:
            RunResult if found, None otherwise
        """
        path = self._get_result_path(commit)
        if not path.exists():
            # Try to find by partial match
            for p in self.storage_dir.glob("*.json"):
                if p.stem.startswith(commit[:8]):
                    path = p
                    break
            else:
                return None

        with open(path, "r") as f:
            data = json.load(f)
            return RunResult(**data)

    def get_baseline_commit(self, steps_back: int = 1) -> Optional[str]:
        """
        Get the commit hash for N commits back from HEAD.

        Args:
            steps_back: How many commits back (default 1 = HEAD~1)

        Returns:
            Commit hash or None if not available
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", f"HEAD~{steps_back}"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def load_baseline(self, steps_back: int = 1) -> Optional[RunResult]:
        """
        Load the baseline result from N commits back.

        This is the key function for comparison mode.

        Args:
            steps_back: How many commits back (default 1)

        Returns:
            RunResult from that commit, or None if not available
        """
        baseline_commit = self.get_baseline_commit(steps_back)
        if not baseline_commit:
            logger.warning(f"No commit found at HEAD~{steps_back}")
            return None

        result = self.load_result(baseline_commit)
        if not result:
            logger.warning(f"No run result found for commit {baseline_commit[:8]}")

        return result

    def compare(
        self,
        baseline: RunResult,
        treatment: RunResult
    ) -> Dict[str, Any]:
        """
        Compare baseline and treatment results for regression testing.

        Args:
            baseline: Results from prior commit
            treatment: Results from current run

        Returns:
            Dict with comparison metrics
        """
        comparison = {
            "baseline_commit": baseline.git_commit[:8],
            "treatment_commit": treatment.git_commit[:8],

            # Success comparison
            "baseline_success": baseline.success,
            "treatment_success": treatment.success,
            "success_regression": baseline.success and not treatment.success,

            # Cost comparison
            "baseline_cost": baseline.total_cost,
            "treatment_cost": treatment.total_cost,
            "cost_delta": treatment.total_cost - baseline.total_cost,
            "cost_regression": treatment.total_cost > baseline.total_cost * 1.2,  # 20% threshold

            # Error comparison
            "baseline_errors": baseline.error_count,
            "treatment_errors": treatment.error_count,
            "error_delta": treatment.error_count - baseline.error_count,
            "error_regression": treatment.error_count > baseline.error_count,

            # Node status comparison
            "baseline_verified": baseline.verified_count,
            "treatment_verified": treatment.verified_count,
            "verified_delta": treatment.verified_count - baseline.verified_count,

            "baseline_failed": baseline.failed_count,
            "treatment_failed": treatment.failed_count,
            "failed_delta": treatment.failed_count - baseline.failed_count,
            "failure_regression": treatment.failed_count > baseline.failed_count,

            # Duration comparison
            "baseline_duration": baseline.duration_seconds,
            "treatment_duration": treatment.duration_seconds,
            "duration_delta": treatment.duration_seconds - baseline.duration_seconds,

            # Overall regression flag
            "has_regression": False,
            "regression_reasons": [],
        }

        # Determine if there's a regression
        reasons = []
        if comparison["success_regression"]:
            reasons.append("SUCCESS: Baseline succeeded but treatment failed")
        if comparison["error_regression"]:
            reasons.append(f"ERRORS: +{comparison['error_delta']} errors")
        if comparison["failure_regression"]:
            reasons.append(f"FAILURES: +{comparison['failed_delta']} failed nodes")
        if comparison["cost_regression"]:
            reasons.append(f"COST: +${comparison['cost_delta']:.4f} (>{20}% increase)")

        comparison["has_regression"] = len(reasons) > 0
        comparison["regression_reasons"] = reasons

        return comparison

    def list_results(self) -> List[Dict[str, Any]]:
        """
        List all stored run results.

        Returns:
            List of dicts with commit, success, timestamp
        """
        results = []
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                    results.append({
                        "commit": data.get("git_commit", "unknown")[:8],
                        "success": data.get("success", False),
                        "completed_at": data.get("completed_at"),
                        "error_count": data.get("error_count", 0),
                        "total_cost": data.get("total_cost", 0),
                    })
            except json.JSONDecodeError:
                continue

        # Sort by completion time
        results.sort(key=lambda x: x.get("completed_at", ""), reverse=True)
        return results


class CheckpointManager:
    """
    Manages checkpoints for pause/resume functionality.

    Features:
    - Create named checkpoints
    - List available checkpoints
    - Restore from checkpoint
    - Auto-checkpoint on critical operations
    """

    def __init__(
        self,
        db: GraphDB,
        checkpoint_dir: str = ".gaadp/checkpoints"
    ):
        self.db = db
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def create_checkpoint(
        self,
        checkpoint_id: str,
        description: str = ""
    ) -> CheckpointMetadata:
        """
        Create a checkpoint of current state.

        Args:
            checkpoint_id: Unique ID for checkpoint
            description: Human-readable description

        Returns:
            CheckpointMetadata
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Copy graph JSON
        graph_src = Path(self.db.persistence_path)
        graph_dst = checkpoint_path / "graph.json"
        if graph_src.exists():
            shutil.copy2(graph_src, graph_dst)

        # Calculate stats
        stats = self._calculate_stats()

        # Create metadata
        metadata = CheckpointMetadata(
            id=checkpoint_id,
            created_at=datetime.utcnow().isoformat(),
            description=description,
            **stats
        )

        # Save metadata
        meta_path = checkpoint_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

        logger.info(f"Checkpoint created: {checkpoint_id}")
        return metadata

    def _calculate_stats(self) -> Dict:
        """Calculate graph statistics."""
        stats = {
            'node_count': self.db.graph.number_of_nodes(),
            'edge_count': self.db.graph.number_of_edges(),
            'pending_count': 0,
            'verified_count': 0,
            'failed_count': 0
        }

        for _, data in self.db.graph.nodes(data=True):
            status = data.get('status', '')
            if status == NodeStatus.PENDING.value:
                stats['pending_count'] += 1
            elif status == NodeStatus.VERIFIED.value:
                stats['verified_count'] += 1
            elif status == NodeStatus.FAILED.value:
                stats['failed_count'] += 1

        return stats

    def list_checkpoints(self) -> List[CheckpointMetadata]:
        """List all available checkpoints."""
        checkpoints = []

        for item in self.checkpoint_dir.iterdir():
            if item.is_dir():
                meta_path = item / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        data = json.load(f)
                        checkpoints.append(CheckpointMetadata(**data))

        # Sort by creation time
        checkpoints.sort(key=lambda x: x.created_at, reverse=True)
        return checkpoints

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore state from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore

        Returns:
            True if successful
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id

        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return False

        graph_src = checkpoint_path / "graph.json"
        if not graph_src.exists():
            logger.error(f"Graph file not found in checkpoint: {checkpoint_id}")
            return False

        # Backup current state first
        current_backup = f"pre_restore_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.create_checkpoint(current_backup, f"Auto-backup before restoring {checkpoint_id}")

        # Restore graph
        graph_dst = Path(self.db.persistence_path)
        shutil.copy2(graph_src, graph_dst)

        # Reload graph
        self.db._load()

        logger.info(f"Restored checkpoint: {checkpoint_id}")
        return True

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if successful
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_id

        if not checkpoint_path.exists():
            return False

        shutil.rmtree(checkpoint_path)
        logger.info(f"Deleted checkpoint: {checkpoint_id}")
        return True

    def get_checkpoint_info(self, checkpoint_id: str) -> Optional[CheckpointMetadata]:
        """Get metadata for a specific checkpoint."""
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        meta_path = checkpoint_path / "metadata.json"

        if not meta_path.exists():
            return None

        with open(meta_path, 'r') as f:
            data = json.load(f)
            return CheckpointMetadata(**data)

    def auto_checkpoint(self, trigger: str) -> CheckpointMetadata:
        """
        Create an automatic checkpoint with generated ID.

        Args:
            trigger: What triggered this checkpoint (for description)

        Returns:
            CheckpointMetadata
        """
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        checkpoint_id = f"auto_{timestamp}"
        description = f"Auto-checkpoint: {trigger}"

        return self.create_checkpoint(checkpoint_id, description)

    def cleanup_old_checkpoints(self, keep_count: int = 10):
        """
        Remove old checkpoints, keeping the most recent N.

        Args:
            keep_count: Number of checkpoints to keep
        """
        checkpoints = self.list_checkpoints()

        if len(checkpoints) <= keep_count:
            return

        # Delete oldest checkpoints
        to_delete = checkpoints[keep_count:]
        for cp in to_delete:
            self.delete_checkpoint(cp.id)

        logger.info(f"Cleaned up {len(to_delete)} old checkpoints")

"""
CHECKPOINT MANAGER
Save and restore execution state for pause/resume capability.
"""
import os
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

from infrastructure.graph_db import GraphDB

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

        # Copy graph pickle
        graph_src = Path(self.db.persistence_path)
        graph_dst = checkpoint_path / "graph.pkl"
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
            if status == 'PENDING':
                stats['pending_count'] += 1
            elif status == 'VERIFIED':
                stats['verified_count'] += 1
            elif status == 'FAILED':
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

        graph_src = checkpoint_path / "graph.pkl"
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

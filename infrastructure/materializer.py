"""
ATOMIC MATERIALIZER
Writes verified code nodes to filesystem atomically.
Supports multi-file projects with rollback on failure.
"""
import os
import shutil
import tempfile
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

from infrastructure.graph_db import GraphDB
from infrastructure.sandbox import CodeSandbox
from infrastructure.version_control import GitController
from core.ontology import NodeType, NodeStatus

logger = logging.getLogger("Materializer")


@dataclass
class MaterializationResult:
    """Result of a materialization operation."""
    success: bool
    files_written: List[str]
    errors: List[str]
    test_results: Optional[Dict] = None
    commit_hash: Optional[str] = None


class AtomicMaterializer:
    """
    Materializes verified CODE nodes to the filesystem atomically.

    Process:
    1. Collect all VERIFIED CODE nodes
    2. Write to temp staging directory
    3. Validate (syntax check, optional tests)
    4. Atomic move to final location
    5. Git commit if successful
    6. Rollback temp on any failure
    """

    def __init__(
        self,
        db: GraphDB,
        output_dir: str = ".",
        run_tests: bool = True,
        auto_commit: bool = True
    ):
        self.db = db
        self.output_dir = Path(output_dir).absolute()
        self.run_tests = run_tests
        self.auto_commit = auto_commit
        self.sandbox = CodeSandbox(use_docker=False)
        self.git = GitController()

    def _get_verified_nodes(self) -> List[Dict]:
        """Get all VERIFIED CODE nodes that haven't been materialized."""
        nodes = []
        for node_id, data in self.db.graph.nodes(data=True):
            if (data.get('type') == NodeType.CODE.value and
                data.get('status') == NodeStatus.VERIFIED.value and
                not data.get('materialized', False)):
                nodes.append({
                    'id': node_id,
                    'content': data.get('content', ''),
                    'file_path': data.get('metadata', {}).get('file_path', f'{node_id}.py'),
                    'metadata': data.get('metadata', {})
                })
        return nodes

    def _validate_syntax(self, file_path: str) -> bool:
        """Check Python syntax of a file."""
        import py_compile
        try:
            py_compile.compile(file_path, doraise=True)
            return True
        except py_compile.PyCompileError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            return False

    def _write_to_staging(self, nodes: List[Dict], staging_dir: Path) -> List[str]:
        """
        Write all nodes to staging directory.

        Returns:
            List of relative file paths written
        """
        written = []

        for node in nodes:
            rel_path = node['file_path']
            full_path = staging_dir / rel_path

            # Create parent directories
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            with open(full_path, 'w') as f:
                f.write(node['content'])

            written.append(rel_path)
            logger.debug(f"Staged: {rel_path}")

        return written

    def _validate_staged(self, staging_dir: Path, files: List[str]) -> List[str]:
        """
        Validate all staged files.

        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []

        for rel_path in files:
            full_path = staging_dir / rel_path

            # Skip non-Python files
            if not rel_path.endswith('.py'):
                continue

            if not self._validate_syntax(str(full_path)):
                errors.append(f"Syntax error in {rel_path}")

        return errors

    def _run_staged_tests(self, staging_dir: Path) -> Dict:
        """
        Run tests on staged code.

        Returns:
            Test results dict
        """
        # Look for test files in staging
        test_files = list(staging_dir.glob('**/test_*.py'))
        test_files.extend(staging_dir.glob('**/*_test.py'))

        if not test_files:
            return {'status': 'skipped', 'reason': 'No test files found'}

        # Run pytest on staging directory
        import subprocess
        result = subprocess.run(
            ['python', '-m', 'pytest', str(staging_dir), '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=120
        )

        return {
            'status': 'pass' if result.returncode == 0 else 'fail',
            'exit_code': result.returncode,
            'stdout': result.stdout[:2000],
            'stderr': result.stderr[:2000]
        }

    def _atomic_move(self, staging_dir: Path, files: List[str]) -> List[str]:
        """
        Atomically move files from staging to output directory.

        Returns:
            List of final file paths
        """
        final_paths = []

        for rel_path in files:
            src = staging_dir / rel_path
            dst = self.output_dir / rel_path

            # Create parent directories
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Move (atomic on same filesystem, copy+delete otherwise)
            shutil.move(str(src), str(dst))
            final_paths.append(str(dst))
            logger.info(f"Materialized: {rel_path}")

        return final_paths

    def _mark_materialized(self, nodes: List[Dict]):
        """Mark nodes as materialized in the graph."""
        for node in nodes:
            if node['id'] in self.db.graph:
                self.db.graph.nodes[node['id']]['materialized'] = True
                self.db.graph.nodes[node['id']]['materialized_path'] = node['file_path']
        self.db._persist()

    def materialize(self, dry_run: bool = False) -> MaterializationResult:
        """
        Materialize all verified code nodes atomically.

        Args:
            dry_run: If True, validate but don't write to final location

        Returns:
            MaterializationResult with status and details
        """
        nodes = self._get_verified_nodes()

        if not nodes:
            return MaterializationResult(
                success=True,
                files_written=[],
                errors=["No verified nodes to materialize"]
            )

        logger.info(f"Materializing {len(nodes)} verified nodes")

        # Create temp staging directory
        staging_dir = Path(tempfile.mkdtemp(prefix='gaadp_staging_'))

        try:
            # Stage all files
            staged_files = self._write_to_staging(nodes, staging_dir)

            # Validate
            validation_errors = self._validate_staged(staging_dir, staged_files)
            if validation_errors:
                return MaterializationResult(
                    success=False,
                    files_written=[],
                    errors=validation_errors
                )

            # Run tests if enabled
            test_results = None
            if self.run_tests:
                test_results = self._run_staged_tests(staging_dir)
                if test_results.get('status') == 'fail':
                    return MaterializationResult(
                        success=False,
                        files_written=[],
                        errors=[f"Tests failed: {test_results.get('stderr', 'Unknown error')[:500]}"],
                        test_results=test_results
                    )

            if dry_run:
                return MaterializationResult(
                    success=True,
                    files_written=staged_files,
                    errors=[],
                    test_results=test_results
                )

            # Atomic move to final location
            final_paths = self._atomic_move(staging_dir, staged_files)

            # Mark nodes as materialized
            self._mark_materialized(nodes)

            # Git commit if enabled
            commit_hash = None
            if self.auto_commit and final_paths:
                try:
                    commit_hash = self.git.commit_work(
                        "materializer",
                        ",".join([n['id'][:8] for n in nodes]),
                        f"Materialize {len(nodes)} verified nodes"
                    )
                except Exception as e:
                    logger.warning(f"Git commit failed: {e}")

            return MaterializationResult(
                success=True,
                files_written=final_paths,
                errors=[],
                test_results=test_results,
                commit_hash=commit_hash
            )

        except Exception as e:
            logger.error(f"Materialization failed: {e}")
            return MaterializationResult(
                success=False,
                files_written=[],
                errors=[str(e)]
            )

        finally:
            # Clean up staging directory
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

    def get_pending_count(self) -> int:
        """Get count of verified nodes awaiting materialization."""
        return len(self._get_verified_nodes())

    def get_materialized_files(self) -> List[Dict]:
        """Get list of already materialized files."""
        files = []
        for node_id, data in self.db.graph.nodes(data=True):
            if data.get('materialized'):
                files.append({
                    'node_id': node_id,
                    'path': data.get('materialized_path'),
                    'type': data.get('type')
                })
        return files

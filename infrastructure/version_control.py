"""
GAADP VERSION CONTROL
Wraps git commands to provide Atomic Graph Commits.
"""
import subprocess
import logging
import os

logger = logging.getLogger("GitWrapper")

class GitController:
    def __init__(self, repo_path="."):
        self.repo_path = repo_path
        self._init_repo()

    def _run(self, args):
        return subprocess.run(
            args, cwd=self.repo_path, capture_output=True, text=True, check=False
        )

    def _init_repo(self):
        if not os.path.exists(os.path.join(self.repo_path, ".git")):
            self._run(["git", "init"])
            self._run(["git", "checkout", "-b", "main"])
            logger.info("Initialized Git Repository")

    def commit_work(self, agent_id: str, task_id: str, message: str):
        """
        Creates an atomic commit for a specific task.
        """
        # 1. Add all changes
        self._run(["git", "add", "."])

        # 2. Commit with strict formatting
        commit_msg = f"feat({agent_id}): {message}\n\nTask-ID: {task_id}"
        result = self._run(["git", "commit", "-m", commit_msg])

        if result.returncode == 0:
            logger.info(f"Committed: {commit_msg.splitlines()[0]}")
            return True
        return False

    def get_current_hash(self):
        res = self._run(["git", "rev-parse", "HEAD"])
        return res.stdout.strip() if res.returncode == 0 else "GENESIS"

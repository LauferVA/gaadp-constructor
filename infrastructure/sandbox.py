"""
GAADP EXECUTION SANDBOX
Safely runs generated code. Prefers Docker, falls back to Subprocess.
"""
import subprocess
import sys
import logging
import os

logger = logging.getLogger("Sandbox")

class CodeSandbox:
    def __init__(self, use_docker=False):
        self.use_docker = use_docker
        # Check if docker is available
        if self.use_docker:
            res = subprocess.run(["docker", "--version"], capture_output=True)
            if res.returncode != 0:
                logger.warning("Docker not found. Falling back to local subprocess.")
                self.use_docker = False

    def run_code(self, file_path: str, timeout=10):
        """
        Executes the python script at file_path and captures stdout/stderr.
        """
        abs_path = os.path.abspath(file_path)
        work_dir = os.path.dirname(abs_path)

        if self.use_docker:
            # Docker execution (Mounts current dir as read-only volume)
            cmd = [
                "docker", "run", "--rm",
                "--network", "none", # No internet access
                "-v", f"{work_dir}:/app:ro",
                "python:3.11-slim",
                "python", f"/app/{os.path.basename(file_path)}"
            ]
        else:
            # Local execution (Less safe, but functional for seed)
            cmd = [sys.executable, file_path]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir
            )
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "stdout": "", "stderr": "Execution Timed Out"}
        except Exception as e:
            return {"exit_code": -1, "stdout": "", "stderr": str(e)}

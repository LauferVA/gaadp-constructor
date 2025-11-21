"""
GAADP EXECUTION SANDBOX
Safely runs generated code in isolated Docker containers.

SECURITY POLICY:
- Docker is REQUIRED for autonomous agent code execution
- Local subprocess execution is DISABLED by default (security risk)
- Containers run with: no network, read-only filesystem, resource limits
"""
import subprocess
import sys
import logging
import os

logger = logging.getLogger("Sandbox")


class SandboxSecurityError(Exception):
    """Raised when sandbox security requirements are not met."""
    pass


class CodeSandbox:
    def __init__(self, use_docker=True, allow_local_fallback=False):
        """
        Initialize code sandbox.

        Args:
            use_docker: If True, require Docker (recommended for autonomous agents)
            allow_local_fallback: If True, allow unsafe local execution (DANGEROUS)

        Raises:
            SandboxSecurityError: If Docker required but not available
        """
        self.use_docker = use_docker
        self.allow_local_fallback = allow_local_fallback

        # Check if docker is available
        if self.use_docker:
            if not self._check_docker():
                if not self.allow_local_fallback:
                    raise SandboxSecurityError(
                        "Docker is required for safe code execution but is not available. "
                        "Install Docker: https://docs.docker.com/get-docker/ "
                        "OR set allow_local_fallback=True (NOT RECOMMENDED for autonomous agents)"
                    )
                else:
                    logger.critical(
                        "Docker not found. Falling back to UNSAFE local subprocess execution. "
                        "This is a SECURITY RISK for autonomous agent-generated code!"
                    )
                    self.use_docker = False

    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def run_code(self, file_path: str, timeout=10):
        """
        Executes the python script at file_path and captures stdout/stderr.

        Args:
            file_path: Path to Python script to execute
            timeout: Maximum execution time in seconds

        Returns:
            Dict with exit_code, stdout, stderr
        """
        abs_path = os.path.abspath(file_path)
        work_dir = os.path.dirname(abs_path)

        if self.use_docker:
            # Docker execution with security hardening
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",  # No internet access
                "--memory", "512m",   # Memory limit
                "--cpus", "1.0",      # CPU limit
                "--read-only",        # Read-only root filesystem
                "-v", f"{work_dir}:/app:ro",  # Mount code as read-only
                "--tmpfs", "/tmp",    # Writable tmpfs for temporary files
                "python:3.11-slim",
                "python", f"/app/{os.path.basename(file_path)}"
            ]
        else:
            # UNSAFE: Local execution (only if explicitly allowed)
            logger.warning(
                f"Executing agent-generated code locally (UNSAFE): {file_path}"
            )
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

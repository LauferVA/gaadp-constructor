"""
GAADP TEST RUNNER
Executes TEST nodes and captures results.
"""
import subprocess
import sys
import logging
import os
import tempfile
from typing import Dict, List, Any

logger = logging.getLogger("TestRunner")


class TestRunner:
    """
    Executes test code associated with TEST nodes.
    Wraps pytest for Python tests.
    """

    def __init__(self, work_dir: str = "."):
        self.work_dir = work_dir
        self.results_history: List[Dict] = []

    def run_pytest(self, test_file: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Run pytest on a specific test file.
        """
        abs_path = os.path.abspath(test_file)

        if not os.path.exists(abs_path):
            return {
                "status": "ERROR",
                "exit_code": -1,
                "message": f"Test file not found: {abs_path}",
                "passed": 0,
                "failed": 0
            }

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", abs_path, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir
            )

            # Parse pytest output
            output = result.stdout + result.stderr
            passed = output.count(" PASSED")
            failed = output.count(" FAILED")
            errors = output.count(" ERROR")

            status = "PASS" if result.returncode == 0 else "FAIL"

            test_result = {
                "status": status,
                "exit_code": result.returncode,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

            self.results_history.append(test_result)
            logger.info(f"Test {test_file}: {status} (P:{passed} F:{failed} E:{errors})")

            return test_result

        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "exit_code": 124,
                "message": f"Test timed out after {timeout}s",
                "passed": 0,
                "failed": 0
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "exit_code": -1,
                "message": str(e),
                "passed": 0,
                "failed": 0
            }

    def run_inline_test(self, test_code: str, timeout: int = 30) -> Dict[str, Any]:
        """
        Run inline test code (from a TEST node's content).
        """
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='_test.py',
            dir=self.work_dir,
            delete=False
        ) as f:
            f.write(test_code)
            temp_path = f.name

        try:
            result = self.run_pytest(temp_path, timeout)
            return result
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def run_all_tests(self, test_dir: str = "tests", timeout: int = 120) -> Dict[str, Any]:
        """
        Run all tests in a directory.
        """
        if not os.path.exists(test_dir):
            return {
                "status": "ERROR",
                "message": f"Test directory not found: {test_dir}",
                "total_passed": 0,
                "total_failed": 0
            }

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_dir, "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.work_dir
            )

            output = result.stdout + result.stderr
            passed = output.count(" PASSED")
            failed = output.count(" FAILED")

            return {
                "status": "PASS" if result.returncode == 0 else "FAIL",
                "exit_code": result.returncode,
                "total_passed": passed,
                "total_failed": failed,
                "stdout": result.stdout,
                "stderr": result.stderr
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "TIMEOUT",
                "message": f"Tests timed out after {timeout}s"
            }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of all test runs in this session.
        """
        if not self.results_history:
            return {"total_runs": 0}

        total_passed = sum(r.get('passed', 0) for r in self.results_history)
        total_failed = sum(r.get('failed', 0) for r in self.results_history)
        total_errors = sum(r.get('errors', 0) for r in self.results_history)

        return {
            "total_runs": len(self.results_history),
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "success_rate": total_passed / max(total_passed + total_failed, 1)
        }

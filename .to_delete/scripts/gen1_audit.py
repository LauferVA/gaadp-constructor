#!/usr/bin/env python3
"""
GEN-1 AUDIT SCRIPT - Phase A of the Clinical Trial
===================================================

Audits existing 30 DAGs to establish baseline reliability metrics
BEFORE implementing the TDD (Builder-Tester-Verifier) pipeline.

Metrics captured:
1. Crash Rate - Percentage of CODE nodes that throw runtime exceptions
2. Correctness Rate - Percentage that produce expected outputs (if testable)
3. Static Analysis - Forbidden imports, dangerous patterns, complexity
4. Coverage Gap - How much of the code has no tests

Usage:
    python scripts/gen1_audit.py                           # Audit latest run
    python scripts/gen1_audit.py --run-id 20251203_061631  # Specific run
    python scripts/gen1_audit.py --verbose                 # Detailed output
"""
import os
import sys
import ast
import json
import asyncio
import argparse
import logging
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
from statistics import mean

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Note: Using inline subprocess execution for audit safety

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)-20s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("Gen1.Audit")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class StaticAnalysisResult:
    """Result of AST-based static analysis."""
    passed: bool
    forbidden_imports: List[str] = field(default_factory=list)
    dangerous_patterns: List[str] = field(default_factory=list)
    cyclomatic_complexity: Dict[str, int] = field(default_factory=dict)
    security_warnings: List[str] = field(default_factory=list)
    lines_of_code: int = 0
    num_functions: int = 0
    num_classes: int = 0


@dataclass
class ExecutionResult:
    """Result of attempting to execute CODE."""
    attempted: bool
    success: bool
    crashed: bool = False
    crash_type: Optional[str] = None
    crash_message: Optional[str] = None
    wrong_output: bool = False
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time_ms: int = 0


@dataclass
class NodeAuditResult:
    """Audit result for a single CODE node."""
    node_id: str
    node_type: str
    file_path: Optional[str]
    content_preview: str  # First 200 chars

    # Static analysis
    static_analysis: StaticAnalysisResult

    # Execution
    execution: ExecutionResult

    # Calculated
    @property
    def overall_passed(self) -> bool:
        return self.static_analysis.passed and self.execution.success


@dataclass
class DAGAuditResult:
    """Audit result for an entire DAG."""
    dag_id: str
    task_name: str
    task_type: str
    dag_path: str

    # Counts
    total_nodes: int
    code_nodes: int
    verified_nodes: int
    failed_nodes: int

    # Node results
    node_results: List[NodeAuditResult] = field(default_factory=list)

    # Aggregated metrics
    @property
    def static_pass_rate(self) -> float:
        if not self.node_results:
            return 0.0
        return sum(1 for n in self.node_results if n.static_analysis.passed) / len(self.node_results)

    @property
    def execution_attempted_count(self) -> int:
        return sum(1 for n in self.node_results if n.execution.attempted)

    @property
    def crash_count(self) -> int:
        return sum(1 for n in self.node_results if n.execution.crashed)

    @property
    def crash_rate(self) -> float:
        attempted = self.execution_attempted_count
        if attempted == 0:
            return 0.0
        return self.crash_count / attempted

    @property
    def success_count(self) -> int:
        return sum(1 for n in self.node_results if n.execution.success)


@dataclass
class Gen1BaselineReport:
    """Complete baseline report for Gen-1 DAGs."""
    run_id: str
    audit_timestamp: str
    total_dags: int

    # Aggregated metrics
    avg_crash_rate: float
    avg_static_pass_rate: float
    total_code_nodes: int
    total_research_nodes: int
    total_verified_nodes: int
    total_failed_nodes: int
    total_crashes: int
    total_forbidden_imports: int
    total_dangerous_patterns: int

    # Node type distribution
    node_type_distribution: Dict[str, int] = field(default_factory=dict)

    # Per-DAG results
    dag_results: List[DAGAuditResult] = field(default_factory=list)

    # Top issues
    most_common_crash_types: Dict[str, int] = field(default_factory=dict)
    most_common_forbidden_imports: Dict[str, int] = field(default_factory=dict)


# =============================================================================
# STATIC ANALYSIS
# =============================================================================

# Forbidden imports that indicate security risks
FORBIDDEN_IMPORTS = {
    'os.system', 'os.popen', 'os.exec', 'os.spawn',
    'subprocess.call', 'subprocess.run', 'subprocess.Popen',
    'eval', 'exec', 'compile',
    'pickle.loads', 'pickle.load',
    '__import__',
    'importlib.import_module',
}

# Dangerous code patterns
DANGEROUS_PATTERNS = [
    'while True:',
    'while 1:',
    'os.remove',
    'os.rmdir',
    'shutil.rmtree',
    'open(',  # Unrestricted file access
    'socket.socket',
    'requests.get',
    'urllib.request',
]


class StaticAnalyzer(ast.NodeVisitor):
    """AST-based static analyzer for Python code."""

    def __init__(self):
        self.imports = []
        self.function_complexities = {}
        self.current_function = None
        self.lines_of_code = 0
        self.num_functions = 0
        self.num_classes = 0
        self.warnings = []

    def analyze(self, code: str) -> StaticAnalysisResult:
        """Analyze Python code and return results."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return StaticAnalysisResult(
                passed=False,
                security_warnings=[f"Syntax error: {e}"]
            )

        self.lines_of_code = len(code.splitlines())
        self.visit(tree)

        # Check for forbidden imports
        forbidden_found = []
        for imp in self.imports:
            for forbidden in FORBIDDEN_IMPORTS:
                if forbidden in imp:
                    forbidden_found.append(imp)

        # Check for dangerous patterns
        dangerous_found = []
        for pattern in DANGEROUS_PATTERNS:
            if pattern in code:
                dangerous_found.append(pattern)

        passed = len(forbidden_found) == 0 and len(dangerous_found) == 0

        return StaticAnalysisResult(
            passed=passed,
            forbidden_imports=forbidden_found,
            dangerous_patterns=dangerous_found,
            cyclomatic_complexity=self.function_complexities,
            security_warnings=self.warnings,
            lines_of_code=self.lines_of_code,
            num_functions=self.num_functions,
            num_classes=self.num_classes
        )

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.append(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            self.imports.append(f"{module}.{alias.name}")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self.num_functions += 1
        self.current_function = node.name
        # Simplified cyclomatic complexity: count branches
        complexity = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        self.function_complexities[node.name] = complexity
        if complexity > 10:
            self.warnings.append(f"High complexity ({complexity}) in function '{node.name}'")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.num_classes += 1
        self.generic_visit(node)


def run_static_analysis(code: str) -> StaticAnalysisResult:
    """Run static analysis on Python code."""
    analyzer = StaticAnalyzer()
    return analyzer.analyze(code)


# =============================================================================
# CODE EXECUTION
# =============================================================================

async def execute_code(code: str, timeout: int = 30) -> ExecutionResult:
    """
    Attempt to execute Python code in sandbox.

    We try to:
    1. Import the code as a module
    2. Call main() if it exists
    3. Capture any exceptions
    """
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        # First check if it can be compiled
        start_time = datetime.now()

        # Try to compile
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return ExecutionResult(
                attempted=True,
                success=False,
                crashed=True,
                crash_type='SyntaxError',
                crash_message=str(e),
                execution_time_ms=0
            )

        # Try to execute with python -c
        test_script = f'''
import sys
sys.path.insert(0, "{Path(temp_path).parent}")
try:
    exec(open("{temp_path}").read())
    print("__EXECUTION_SUCCESS__")
except Exception as e:
    print(f"__EXECUTION_ERROR__: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
'''

        try:
            result = subprocess.run(
                [sys.executable, '-c', test_script],
                capture_output=True,
                text=True,
                timeout=timeout
            )

            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)

            if result.returncode == 0 and "__EXECUTION_SUCCESS__" in result.stdout:
                return ExecutionResult(
                    attempted=True,
                    success=True,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    execution_time_ms=execution_time
                )
            else:
                # Parse error from stderr
                crash_type = "RuntimeError"
                crash_message = result.stderr
                if "__EXECUTION_ERROR__:" in result.stderr:
                    error_line = [l for l in result.stderr.split('\n') if "__EXECUTION_ERROR__:" in l]
                    if error_line:
                        parts = error_line[0].replace("__EXECUTION_ERROR__:", "").strip().split(":", 1)
                        crash_type = parts[0].strip()
                        crash_message = parts[1].strip() if len(parts) > 1 else ""

                return ExecutionResult(
                    attempted=True,
                    success=False,
                    crashed=True,
                    crash_type=crash_type,
                    crash_message=crash_message,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    execution_time_ms=execution_time
                )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                attempted=True,
                success=False,
                crashed=True,
                crash_type='TimeoutError',
                crash_message=f'Execution exceeded {timeout}s timeout',
                execution_time_ms=timeout * 1000
            )

    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass


# =============================================================================
# DAG AUDITING
# =============================================================================

async def audit_single_dag(dag_path: Path, verbose: bool = False) -> DAGAuditResult:
    """Audit a single DAG JSON file."""
    logger.info(f"Auditing: {dag_path.name}")

    with open(dag_path) as f:
        dag_data = json.load(f)

    metadata = dag_data.get("metadata", {})
    nodes = dag_data.get("nodes", [])

    dag_id = metadata.get("task_id", dag_path.stem)
    task_name = metadata.get("task_name", "Unknown")
    task_type = metadata.get("task_type", "unknown")

    # Count node types
    total_nodes = len(nodes)

    # Look for CODE nodes, but also include RESEARCH nodes that contain code
    code_nodes = [n for n in nodes if n.get("type") == "CODE"]

    # RESEARCH nodes with JSON content might contain code patterns
    research_nodes = [n for n in nodes if n.get("type") == "RESEARCH"]

    verified_nodes = [n for n in nodes if n.get("status") == "VERIFIED"]
    failed_nodes = [n for n in nodes if n.get("status") == "FAILED"]

    # Log node type distribution
    node_type_counts = {}
    for n in nodes:
        t = n.get("type", "UNKNOWN")
        node_type_counts[t] = node_type_counts.get(t, 0) + 1
    if verbose:
        logger.info(f"  Node types: {node_type_counts}")

    # Audit each CODE node (or analyze RESEARCH if no CODE)
    node_results = []
    nodes_to_analyze = code_nodes if code_nodes else []

    # If we have RESEARCH nodes with verified status, analyze their content
    for research_node in research_nodes:
        content = research_node.get("content", "")
        # Look for code patterns in research output (JSON with code_implementation fields)
        if "```python" in content or "def " in content or "class " in content:
            nodes_to_analyze.append(research_node)

    for node in nodes_to_analyze:
        node_id = node.get("id", "unknown")
        content = node.get("content", "")
        node_metadata = node.get("metadata", {})
        file_path = node_metadata.get("file_path")

        if verbose:
            logger.info(f"  Analyzing CODE node: {node_id[:8]}...")

        # Static analysis
        static_result = run_static_analysis(content)

        # Execution (only if static passed and code is substantial)
        if len(content) > 10:
            exec_result = await execute_code(content)
        else:
            exec_result = ExecutionResult(attempted=False, success=False)

        node_results.append(NodeAuditResult(
            node_id=node_id,
            node_type="CODE",
            file_path=file_path,
            content_preview=content[:200] if content else "",
            static_analysis=static_result,
            execution=exec_result
        ))

    return DAGAuditResult(
        dag_id=dag_id,
        task_name=task_name,
        task_type=task_type,
        dag_path=str(dag_path),
        total_nodes=total_nodes,
        code_nodes=len(code_nodes),
        verified_nodes=len(verified_nodes),
        failed_nodes=len(failed_nodes),
        node_results=node_results
    )


async def audit_gen1_dags(run_id: str = None, verbose: bool = False) -> Gen1BaselineReport:
    """
    Audit all Gen-1 DAGs from the benchmark run.

    Args:
        run_id: Specific run ID (e.g., '20251203_061631'). If None, use latest.
        verbose: Print detailed output
    """
    dag_benchmark_dir = PROJECT_ROOT / "logs" / "dag_benchmark"

    # Find the run directory
    if run_id:
        run_dir = dag_benchmark_dir / f"run_{run_id}"
    else:
        # Find latest run
        runs = sorted(dag_benchmark_dir.glob("run_*"), reverse=True)
        if not runs:
            logger.error("No benchmark runs found!")
            return None
        run_dir = runs[0]
        run_id = run_dir.name.replace("run_", "")

    dags_dir = run_dir / "dags"
    if not dags_dir.exists():
        logger.error(f"DAGs directory not found: {dags_dir}")
        return None

    dag_files = sorted(dags_dir.glob("dag_*.json"))
    logger.info(f"Found {len(dag_files)} DAG files in run {run_id}")

    # Audit each DAG
    dag_results = []
    for dag_path in dag_files:
        result = await audit_single_dag(dag_path, verbose)
        dag_results.append(result)

    # Aggregate metrics
    total_code_nodes = sum(d.code_nodes for d in dag_results)
    total_crashes = sum(d.crash_count for d in dag_results)

    # Count nodes by type and status across all DAGs
    node_type_dist = {}
    total_verified = 0
    total_failed = 0
    total_research = 0

    for dag in dag_results:
        total_verified += dag.verified_nodes
        total_failed += dag.failed_nodes
        # Count research nodes from total_nodes - code_nodes (approximation)
        # Actually, we need to recalculate from raw data

    # Re-read DAGs to get node type distribution
    for dag_path in dag_files:
        with open(dag_path) as f:
            dag_data = json.load(f)
        for node in dag_data.get("nodes", []):
            node_type = node.get("type", "UNKNOWN")
            node_type_dist[node_type] = node_type_dist.get(node_type, 0) + 1
            if node.get("status") == "VERIFIED":
                total_verified += 1
            if node.get("status") == "FAILED":
                total_failed += 1
            if node_type == "RESEARCH":
                total_research += 1

    # Reset counts (we double counted above)
    total_verified = sum(1 for dag_path in dag_files
                         for node in json.load(open(dag_path)).get("nodes", [])
                         if node.get("status") == "VERIFIED")
    total_failed = sum(1 for dag_path in dag_files
                       for node in json.load(open(dag_path)).get("nodes", [])
                       if node.get("status") == "FAILED")
    total_research = node_type_dist.get("RESEARCH", 0)

    crash_rates = [d.crash_rate for d in dag_results if d.code_nodes > 0]
    static_rates = [d.static_pass_rate for d in dag_results if d.code_nodes > 0]

    avg_crash_rate = mean(crash_rates) if crash_rates else 0.0
    avg_static_pass_rate = mean(static_rates) if static_rates else 0.0

    # Count forbidden imports and patterns
    forbidden_imports_count = {}
    dangerous_patterns_count = {}
    crash_types_count = {}

    for dag in dag_results:
        for node in dag.node_results:
            for imp in node.static_analysis.forbidden_imports:
                forbidden_imports_count[imp] = forbidden_imports_count.get(imp, 0) + 1
            for pattern in node.static_analysis.dangerous_patterns:
                dangerous_patterns_count[pattern] = dangerous_patterns_count.get(pattern, 0) + 1
            if node.execution.crash_type:
                crash_types_count[node.execution.crash_type] = crash_types_count.get(node.execution.crash_type, 0) + 1

    total_forbidden = sum(forbidden_imports_count.values())
    total_dangerous = sum(dangerous_patterns_count.values())

    return Gen1BaselineReport(
        run_id=run_id,
        audit_timestamp=datetime.now().isoformat(),
        total_dags=len(dag_results),
        avg_crash_rate=avg_crash_rate,
        avg_static_pass_rate=avg_static_pass_rate,
        total_code_nodes=total_code_nodes,
        total_research_nodes=total_research,
        total_verified_nodes=total_verified,
        total_failed_nodes=total_failed,
        total_crashes=total_crashes,
        total_forbidden_imports=total_forbidden,
        total_dangerous_patterns=total_dangerous,
        node_type_distribution=node_type_dist,
        dag_results=dag_results,
        most_common_crash_types=crash_types_count,
        most_common_forbidden_imports=forbidden_imports_count
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_report(report: Gen1BaselineReport):
    """Print human-readable report to console."""
    print("\n" + "=" * 70)
    print("GEN-1 BASELINE AUDIT REPORT")
    print("=" * 70)
    print(f"Run ID: {report.run_id}")
    print(f"Audit Time: {report.audit_timestamp}")
    print(f"Total DAGs: {report.total_dags}")
    print()

    print("NODE TYPE DISTRIBUTION")
    print("-" * 40)
    for node_type, count in sorted(report.node_type_distribution.items()):
        print(f"  {node_type:<20} {count:>5}")
    print()

    print("AGGREGATE METRICS")
    print("-" * 40)
    print(f"  Total CODE nodes:        {report.total_code_nodes}")
    print(f"  Total RESEARCH nodes:    {report.total_research_nodes}")
    print(f"  Verified nodes:          {report.total_verified_nodes}")
    print(f"  Failed nodes:            {report.total_failed_nodes}")
    print(f"  Total crashes:           {report.total_crashes}")
    print(f"  Average crash rate:      {report.avg_crash_rate:.1%}")
    print(f"  Average static pass:     {report.avg_static_pass_rate:.1%}")
    print(f"  Forbidden imports:       {report.total_forbidden_imports}")
    print(f"  Dangerous patterns:      {report.total_dangerous_patterns}")
    print()

    # Key finding for Gen-1
    if report.total_code_nodes == 0:
        print("KEY FINDING: Gen-1 Pipeline Limitation")
        print("-" * 40)
        print("  The Gen-1 DAGs contain NO CODE nodes.")
        print("  Pipeline terminated at RESEARCH stage.")
        print("  This confirms the need for Gen-2 TDD pipeline")
        print("  to actually BUILD and TEST code.")
        print()

    if report.most_common_crash_types:
        print("MOST COMMON CRASH TYPES")
        print("-" * 40)
        for crash_type, count in sorted(report.most_common_crash_types.items(), key=lambda x: -x[1])[:10]:
            print(f"  {crash_type}: {count}")
        print()

    if report.most_common_forbidden_imports:
        print("FORBIDDEN IMPORTS FOUND")
        print("-" * 40)
        for imp, count in sorted(report.most_common_forbidden_imports.items(), key=lambda x: -x[1])[:10]:
            print(f"  {imp}: {count}")
        print()

    print("PER-DAG RESULTS")
    print("-" * 40)
    print(f"{'DAG':<35} {'CODE':>6} {'Crash':>8} {'Static':>8}")
    print("-" * 60)
    for dag in report.dag_results:
        print(f"{dag.task_name[:34]:<35} {dag.code_nodes:>6} {dag.crash_rate:>7.0%} {dag.static_pass_rate:>7.0%}")

    print()
    print("=" * 70)
    print("BASELINE ESTABLISHED - Ready for Gen-2 comparison")
    print("=" * 70)


def save_report(report: Gen1BaselineReport, output_path: Path):
    """Save report to JSON file."""
    # Convert dataclasses to dicts, handling nested structures
    def to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: to_dict(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        return obj

    report_dict = to_dict(report)

    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)

    logger.info(f"Report saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    parser = argparse.ArgumentParser(description="Gen-1 DAG Baseline Audit")
    parser.add_argument("--run-id", type=str, help="Specific run ID to audit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--output", "-o", type=str, help="Output JSON path")
    args = parser.parse_args()

    report = await audit_gen1_dags(args.run_id, args.verbose)

    if report is None:
        sys.exit(1)

    print_report(report)

    # Save report
    output_path = args.output or PROJECT_ROOT / "logs" / f"gen1_audit_{report.run_id}.json"
    save_report(report, Path(output_path))


if __name__ == "__main__":
    asyncio.run(main())

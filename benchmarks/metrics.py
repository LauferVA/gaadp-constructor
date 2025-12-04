"""
GAADP METRICS - Graph Physics Measurement
==========================================
No targets. No thresholds. Only observation.

These metrics emerged from GOLDEN_FAILURE_001.md - the Asteroid Game failure analysis.
They quantify what went wrong so we can track improvement over time.

Metrics:
    - TOVS: Topological Order Violation Score
    - ICR: Import Consistency Ratio
    - GCR: God Class Ratio
    - DSR: Dependency Satisfaction Rate

Usage:
    from benchmarks.metrics import GraphMetrics

    metrics = GraphMetrics(graph_db, telemetry_path=".gaadp/logs/telemetry.jsonl")
    report = metrics.calculate_all()
"""
import ast
import json
import re
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("GAADP.Metrics")


@dataclass
class TOVSResult:
    """Topological Order Violation Score result."""
    score: float  # 0.0 = perfect, 1.0 = all violations
    violations: List[Dict[str, Any]]  # Details of each violation
    total_nodes_built: int
    nodes_violating: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ICRResult:
    """Import Consistency Ratio result."""
    score: float  # 1.0 = perfect consistency
    phantom_imports: List[Dict[str, str]]  # Imports without DEPENDS_ON edges
    unused_edges: List[Dict[str, str]]  # DEPENDS_ON edges without imports
    total_imports: int
    matching_imports: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GCRResult:
    """God Class Ratio result."""
    score: float  # max_loc / median_loc
    largest_module: Dict[str, Any]
    median_loc: float
    module_sizes: List[Dict[str, Any]]  # All modules with LOC

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DSRResult:
    """Dependency Satisfaction Rate result."""
    score: float  # verified_deps / declared_deps
    unsatisfied: List[Dict[str, str]]  # Deps that aren't VERIFIED
    total_declared: int
    total_satisfied: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsReport:
    """Complete metrics report for a run."""
    run_id: str
    timestamp: str

    # Graph Physics
    tovs: TOVSResult
    icr: ICRResult
    gcr: GCRResult
    dsr: DSRResult

    # Summary
    prompt: Optional[str] = None
    commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "prompt": self.prompt,
            "commit": self.commit,
            "metrics": {
                "tovs": self.tovs.score,
                "icr": self.icr.score,
                "gcr": self.gcr.score,
                "dsr": self.dsr.score
            },
            "details": {
                "tovs": self.tovs.to_dict(),
                "icr": self.icr.to_dict(),
                "gcr": self.gcr.to_dict(),
                "dsr": self.dsr.to_dict()
            }
        }


class GraphMetrics:
    """
    Calculate graph physics metrics from a GraphDB instance.

    These metrics measure the structural integrity of the dependency graph
    and the execution path taken to build it.
    """

    def __init__(self, graph_db, telemetry_path: str = None):
        """
        Initialize metrics calculator.

        Args:
            graph_db: GraphDB instance with the graph to analyze
            telemetry_path: Path to telemetry.jsonl for trajectory analysis
        """
        self.graph_db = graph_db
        self.graph = graph_db.graph
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self._telemetry_events = None

    def _load_telemetry(self) -> List[Dict]:
        """Load telemetry events from JSONL file."""
        if self._telemetry_events is not None:
            return self._telemetry_events

        if not self.telemetry_path or not self.telemetry_path.exists():
            logger.warning(f"Telemetry file not found: {self.telemetry_path}")
            return []

        events = []
        with open(self.telemetry_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        self._telemetry_events = events
        return events

    def calculate_tovs(self) -> TOVSResult:
        """
        Calculate Topological Order Violation Score.

        TOVS = nodes_built_before_dependencies / total_nodes_built

        A violation occurs when a node transitions to BUILDING before
        all its DEPENDS_ON targets are VERIFIED.

        Returns:
            TOVSResult with score and violation details
        """
        from core.ontology import EdgeType, NodeStatus

        events = self._load_telemetry()
        if not events:
            # Fall back to static analysis if no telemetry
            return self._calculate_tovs_static()

        # Track when nodes reached VERIFIED status
        verified_at_step: Dict[str, int] = {}
        # Track when nodes started BUILDING
        building_at_step: Dict[str, int] = {}

        for event in events:
            if event.get("event_type") == "state_transition":
                payload = event.get("payload", {})
                node_id = payload.get("node_id")
                to_status = payload.get("to_status")
                step = event.get("step", 0)

                if to_status == "VERIFIED":
                    verified_at_step[node_id] = step
                elif to_status == "BUILDING":
                    building_at_step[node_id] = step

        # Check each node that was built
        violations = []
        total_built = len(building_at_step)

        for node_id, build_step in building_at_step.items():
            # Find full node ID from graph (telemetry truncates to 8 chars)
            full_node_id = self._find_node_by_prefix(node_id)
            if not full_node_id:
                continue

            # Get DEPENDS_ON predecessors
            missing_deps = []
            for pred in self.graph.predecessors(full_node_id):
                edge_data = self.graph.edges[pred, full_node_id]
                if edge_data.get("type") == EdgeType.DEPENDS_ON.value:
                    pred_prefix = pred[:8]
                    verified_step = verified_at_step.get(pred_prefix)

                    if verified_step is None or verified_step > build_step:
                        # Dependency wasn't verified before we started building
                        missing_deps.append(pred[:8])

            if missing_deps:
                violations.append({
                    "node": node_id,
                    "build_step": build_step,
                    "missing_deps": missing_deps
                })

        score = len(violations) / total_built if total_built > 0 else 0.0

        return TOVSResult(
            score=score,
            violations=violations,
            total_nodes_built=total_built,
            nodes_violating=len(violations)
        )

    def _calculate_tovs_static(self) -> TOVSResult:
        """
        Static TOVS calculation when telemetry is unavailable.

        Checks current graph state for dependency ordering issues.
        """
        from core.ontology import EdgeType, NodeStatus, NodeType

        violations = []
        code_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("type") == NodeType.CODE.value
        ]

        for node_id in code_nodes:
            node_status = self.graph.nodes[node_id].get("status")
            if node_status not in [NodeStatus.VERIFIED.value, NodeStatus.BUILDING.value]:
                continue

            # Check DEPENDS_ON predecessors
            missing = []
            for pred in self.graph.predecessors(node_id):
                edge_data = self.graph.edges[pred, node_id]
                if edge_data.get("type") == EdgeType.DEPENDS_ON.value:
                    pred_status = self.graph.nodes[pred].get("status")
                    if pred_status != NodeStatus.VERIFIED.value:
                        missing.append(pred[:8])

            if missing:
                violations.append({
                    "node": node_id[:8],
                    "build_step": -1,  # Unknown from static
                    "missing_deps": missing
                })

        total = len(code_nodes)
        score = len(violations) / total if total > 0 else 0.0

        return TOVSResult(
            score=score,
            violations=violations,
            total_nodes_built=total,
            nodes_violating=len(violations)
        )

    def _find_node_by_prefix(self, prefix: str) -> Optional[str]:
        """Find full node ID from 8-char prefix."""
        for node_id in self.graph.nodes():
            if node_id.startswith(prefix) or node_id[:8] == prefix:
                return node_id
        return None

    def calculate_icr(self) -> ICRResult:
        """
        Calculate Import Consistency Ratio.

        ICR = imports_matching_DEPENDS_ON / total_imports

        Compares Python import statements in CODE nodes against
        DEPENDS_ON edges in the graph.

        Returns:
            ICRResult with score and discrepancy details
        """
        from core.ontology import EdgeType, NodeType, NodeStatus

        phantom_imports = []  # Imports without edges
        unused_edges = []  # Edges without imports
        total_imports = 0
        matching_imports = 0

        # Get all CODE nodes
        code_nodes = [
            (n, d) for n, d in self.graph.nodes(data=True)
            if d.get("type") == NodeType.CODE.value
        ]

        for node_id, node_data in code_nodes:
            content = node_data.get("content", "")
            file_path = node_data.get("metadata", {}).get("file_path", "")

            if not content or not file_path:
                continue

            # Extract imports from code
            local_imports = self._extract_local_imports(content, file_path)

            # Get declared DEPENDS_ON edges
            declared_deps = set()
            for pred in self.graph.predecessors(node_id):
                edge_data = self.graph.edges[pred, node_id]
                if edge_data.get("type") == EdgeType.DEPENDS_ON.value:
                    pred_path = self.graph.nodes[pred].get("metadata", {}).get("file_path", "")
                    if pred_path:
                        declared_deps.add(self._normalize_module_name(pred_path))

            # Compare
            for imp in local_imports:
                total_imports += 1
                if imp in declared_deps:
                    matching_imports += 1
                else:
                    phantom_imports.append({
                        "file": file_path,
                        "import": imp
                    })

            # Check for unused edges
            for dep in declared_deps:
                if dep not in local_imports:
                    unused_edges.append({
                        "file": file_path,
                        "edge_to": dep
                    })

        score = matching_imports / total_imports if total_imports > 0 else 1.0

        return ICRResult(
            score=score,
            phantom_imports=phantom_imports,
            unused_edges=unused_edges,
            total_imports=total_imports,
            matching_imports=matching_imports
        )

    def _extract_local_imports(self, code: str, file_path: str) -> Set[str]:
        """
        Extract local module imports from Python code.

        Only returns imports that look like local project imports,
        not standard library or third-party packages.
        """
        local_imports = set()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to regex for unparseable code
            return self._extract_imports_regex(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Heuristic: local imports are usually short names
                    # without dots or common stdlib prefixes
                    name = alias.name.split(".")[0]
                    if self._is_likely_local_import(name):
                        local_imports.add(name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    name = node.module.split(".")[0]
                    if self._is_likely_local_import(name):
                        local_imports.add(name)

        return local_imports

    def _extract_imports_regex(self, code: str) -> Set[str]:
        """Regex fallback for import extraction."""
        imports = set()

        # Match: import foo, from foo import bar
        patterns = [
            r"^import\s+(\w+)",
            r"^from\s+(\w+)"
        ]

        for line in code.split("\n"):
            line = line.strip()
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    name = match.group(1)
                    if self._is_likely_local_import(name):
                        imports.add(name)

        return imports

    def _is_likely_local_import(self, name: str) -> bool:
        """Heuristic to detect local imports vs stdlib/third-party."""
        # Common stdlib/third-party prefixes to exclude
        EXCLUDE_PREFIXES = {
            "os", "sys", "re", "json", "time", "datetime", "pathlib",
            "typing", "collections", "itertools", "functools", "logging",
            "asyncio", "threading", "multiprocessing", "subprocess",
            "math", "random", "hashlib", "uuid", "copy", "io",
            "unittest", "pytest", "numpy", "pandas", "requests",
            "pygame", "flask", "django", "fastapi", "pydantic",
            "networkx", "yaml", "toml", "dataclasses", "enum",
            "abc", "contextlib", "warnings", "traceback", "inspect"
        }

        return name.lower() not in EXCLUDE_PREFIXES

    def _normalize_module_name(self, file_path: str) -> str:
        """Convert file path to module name."""
        # "game/player.py" -> "player"
        name = Path(file_path).stem
        return name

    def calculate_gcr(self) -> GCRResult:
        """
        Calculate God Class Ratio.

        GCR = max(module_loc) / median(module_loc)

        A high GCR indicates one module is doing too much work
        relative to others.

        Returns:
            GCRResult with score and module size details
        """
        from core.ontology import NodeType

        # Get all CODE nodes with their content
        module_sizes = []

        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("type") != NodeType.CODE.value:
                continue

            content = node_data.get("content", "")
            file_path = node_data.get("metadata", {}).get("file_path", "unknown")

            loc = len(content.split("\n")) if content else 0

            module_sizes.append({
                "file": file_path,
                "loc": loc,
                "node_id": node_id[:8]
            })

        if not module_sizes:
            return GCRResult(
                score=0.0,
                largest_module={},
                median_loc=0.0,
                module_sizes=[]
            )

        # Sort by LOC
        module_sizes.sort(key=lambda x: x["loc"], reverse=True)

        # Calculate median
        locs = [m["loc"] for m in module_sizes]
        n = len(locs)
        if n % 2 == 0:
            median = (locs[n//2 - 1] + locs[n//2]) / 2
        else:
            median = locs[n//2]

        # Calculate GCR
        max_loc = locs[0] if locs else 0
        gcr = max_loc / median if median > 0 else 0.0

        return GCRResult(
            score=gcr,
            largest_module=module_sizes[0] if module_sizes else {},
            median_loc=median,
            module_sizes=module_sizes
        )

    def calculate_dsr(self) -> DSRResult:
        """
        Calculate Dependency Satisfaction Rate.

        DSR = verified_dependencies / declared_dependencies

        Measures whether dependencies are actually verified
        when we claim they are.

        Returns:
            DSRResult with score and unsatisfied dependency details
        """
        from core.ontology import EdgeType, NodeStatus, NodeType

        total_declared = 0
        total_satisfied = 0
        unsatisfied = []

        # Check all DEPENDS_ON edges
        for u, v, edge_data in self.graph.edges(data=True):
            if edge_data.get("type") != EdgeType.DEPENDS_ON.value:
                continue

            total_declared += 1

            # The source (u) should be VERIFIED for the edge to be satisfied
            source_status = self.graph.nodes[u].get("status")
            target_file = self.graph.nodes[v].get("metadata", {}).get("file_path", v[:8])
            source_file = self.graph.nodes[u].get("metadata", {}).get("file_path", u[:8])

            if source_status == NodeStatus.VERIFIED.value:
                total_satisfied += 1
            else:
                unsatisfied.append({
                    "target": target_file,
                    "depends_on": source_file,
                    "dep_status": source_status
                })

        score = total_satisfied / total_declared if total_declared > 0 else 1.0

        return DSRResult(
            score=score,
            unsatisfied=unsatisfied,
            total_declared=total_declared,
            total_satisfied=total_satisfied
        )

    def calculate_all(self, run_id: str = None, prompt: str = None) -> MetricsReport:
        """
        Calculate all metrics and return a complete report.

        Args:
            run_id: Optional identifier for this run
            prompt: Optional prompt text that generated this graph

        Returns:
            MetricsReport with all metrics
        """
        from datetime import datetime
        import subprocess

        # Get git commit if available
        commit = None
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
        except Exception:
            pass

        return MetricsReport(
            run_id=run_id or datetime.now().strftime("%Y%m%d_%H%M%S"),
            timestamp=datetime.now().isoformat(),
            prompt=prompt[:500] if prompt else None,
            commit=commit,
            tovs=self.calculate_tovs(),
            icr=self.calculate_icr(),
            gcr=self.calculate_gcr(),
            dsr=self.calculate_dsr()
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_metrics(graph_db, telemetry_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to calculate all metrics.

    Args:
        graph_db: GraphDB instance
        telemetry_path: Optional path to telemetry JSONL

    Returns:
        Dict with all metrics
    """
    calculator = GraphMetrics(graph_db, telemetry_path)
    report = calculator.calculate_all()
    return report.to_dict()


def print_metrics_report(report: MetricsReport):
    """Pretty-print a metrics report."""
    print("\n" + "=" * 60)
    print("METRICS REPORT")
    print("=" * 60)
    print(f"Run ID: {report.run_id}")
    print(f"Commit: {report.commit or 'unknown'}")
    print()

    print("Graph Physics:")
    print(f"  TOVS: {report.tovs.score:.3f}  ({report.tovs.nodes_violating}/{report.tovs.total_nodes_built} violations)")
    print(f"  ICR:  {report.icr.score:.3f}  ({report.icr.matching_imports}/{report.icr.total_imports} matching)")
    print(f"  GCR:  {report.gcr.score:.2f}  (max {report.gcr.largest_module.get('loc', 0)} LOC, median {report.gcr.median_loc:.0f})")
    print(f"  DSR:  {report.dsr.score:.3f}  ({report.dsr.total_satisfied}/{report.dsr.total_declared} satisfied)")

    if report.tovs.violations:
        print("\nTOVS Violations:")
        for v in report.tovs.violations[:5]:
            print(f"  - {v['node']}: missing {v['missing_deps']}")

    if report.icr.phantom_imports:
        print("\nPhantom Imports (no DEPENDS_ON edge):")
        for p in report.icr.phantom_imports[:5]:
            print(f"  - {p['file']}: import {p['import']}")

    if report.dsr.unsatisfied:
        print("\nUnsatisfied Dependencies:")
        for u in report.dsr.unsatisfied[:5]:
            print(f"  - {u['target']} depends on {u['depends_on']} (status: {u['dep_status']})")

    print("=" * 60)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO)

    print("=== Testing Metrics Module ===\n")

    # Try to load existing graph
    from infrastructure.graph_db import GraphDB

    graph_path = ".gaadp/graph.json"
    if Path(graph_path).exists():
        graph_db = GraphDB(persistence_path=graph_path)
        calculator = GraphMetrics(
            graph_db,
            telemetry_path=".gaadp/logs/telemetry.jsonl"
        )

        report = calculator.calculate_all()
        print_metrics_report(report)

        # Save report
        report_path = Path("reports") / f"metrics_{report.run_id}.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved: {report_path}")
    else:
        print(f"No graph found at {graph_path}")
        print("Run a GAADP task first to generate a graph.")

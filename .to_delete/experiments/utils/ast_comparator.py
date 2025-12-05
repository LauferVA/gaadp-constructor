"""
AST COMPARATOR - Structural Code Comparison
============================================
Compares Python code at the AST level to measure semantic similarity,
independent of formatting, variable names, and superficial differences.

Based on research into pyastsim and Tree Edit Distance algorithms.

Usage:
    from utils.ast_comparator import ASTComparator

    comparator = ASTComparator()
    result = comparator.compare(code1, code2)
    print(f"Similarity: {result.similarity}")
"""
import ast
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

logger = logging.getLogger("GAADP.ASTComparator")


@dataclass
class InterfaceSignature:
    """Represents a function or class signature."""
    name: str
    kind: str  # "function", "class", "method"
    parameters: List[str]
    return_annotation: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    is_async: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Result of comparing two pieces of code."""
    similarity: float  # 0.0 to 1.0
    structural_similarity: float  # Based on AST node types
    interface_match: float  # Based on function/class signatures

    # Details
    nodes_in_a: int
    nodes_in_b: int
    matching_nodes: int

    interfaces_in_a: List[InterfaceSignature]
    interfaces_in_b: List[InterfaceSignature]
    matching_interfaces: List[str]
    missing_interfaces: List[str]
    extra_interfaces: List[str]

    # Parse status
    a_parsed: bool
    b_parsed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity": self.similarity,
            "structural_similarity": self.structural_similarity,
            "interface_match": self.interface_match,
            "nodes": {
                "a": self.nodes_in_a,
                "b": self.nodes_in_b,
                "matching": self.matching_nodes
            },
            "interfaces": {
                "matching": self.matching_interfaces,
                "missing": self.missing_interfaces,
                "extra": self.extra_interfaces
            },
            "parsed": {
                "a": self.a_parsed,
                "b": self.b_parsed
            }
        }


class ASTComparator:
    """
    Compare Python code using AST analysis.

    Provides structural similarity scores that are invariant to:
    - Whitespace and formatting
    - Comment differences
    - Variable naming (when normalized)
    """

    def __init__(self, normalize_names: bool = False):
        """
        Initialize comparator.

        Args:
            normalize_names: If True, normalize variable/function names
                           for comparison. This allows detecting equivalent
                           code with different naming conventions.
        """
        self.normalize_names = normalize_names

    def compare(self, code_a: str, code_b: str) -> ComparisonResult:
        """
        Compare two pieces of Python code.

        Args:
            code_a: First code string (typically reference)
            code_b: Second code string (typically generated)

        Returns:
            ComparisonResult with similarity scores
        """
        # Try to parse both
        tree_a, parsed_a = self._parse(code_a)
        tree_b, parsed_b = self._parse(code_b)

        if not parsed_a or not parsed_b:
            # Can't compare unparseable code
            return ComparisonResult(
                similarity=0.0,
                structural_similarity=0.0,
                interface_match=0.0,
                nodes_in_a=0,
                nodes_in_b=0,
                matching_nodes=0,
                interfaces_in_a=[],
                interfaces_in_b=[],
                matching_interfaces=[],
                missing_interfaces=[],
                extra_interfaces=[],
                a_parsed=parsed_a,
                b_parsed=parsed_b
            )

        # Calculate structural similarity
        nodes_a = self._count_node_types(tree_a)
        nodes_b = self._count_node_types(tree_b)
        structural_sim, matching, total_a, total_b = self._compare_node_distributions(nodes_a, nodes_b)

        # Extract and compare interfaces
        interfaces_a = self._extract_interfaces(tree_a)
        interfaces_b = self._extract_interfaces(tree_b)
        interface_match, matching_names, missing, extra = self._compare_interfaces(
            interfaces_a, interfaces_b
        )

        # Combined similarity (weighted average)
        similarity = 0.6 * structural_sim + 0.4 * interface_match

        return ComparisonResult(
            similarity=similarity,
            structural_similarity=structural_sim,
            interface_match=interface_match,
            nodes_in_a=total_a,
            nodes_in_b=total_b,
            matching_nodes=matching,
            interfaces_in_a=interfaces_a,
            interfaces_in_b=interfaces_b,
            matching_interfaces=matching_names,
            missing_interfaces=missing,
            extra_interfaces=extra,
            a_parsed=True,
            b_parsed=True
        )

    def _parse(self, code: str) -> Tuple[Optional[ast.AST], bool]:
        """Parse code to AST."""
        try:
            tree = ast.parse(code)
            return tree, True
        except SyntaxError as e:
            logger.debug(f"Parse error: {e}")
            return None, False

    def _count_node_types(self, tree: ast.AST) -> Dict[str, int]:
        """Count occurrences of each AST node type."""
        counts: Dict[str, int] = defaultdict(int)

        for node in ast.walk(tree):
            node_type = type(node).__name__
            counts[node_type] += 1

        return dict(counts)

    def _compare_node_distributions(
        self,
        nodes_a: Dict[str, int],
        nodes_b: Dict[str, int]
    ) -> Tuple[float, int, int, int]:
        """
        Compare node type distributions.

        Uses a variant of Jaccard similarity that accounts for counts.
        """
        all_types = set(nodes_a.keys()) | set(nodes_b.keys())

        if not all_types:
            return 1.0, 0, 0, 0

        total_a = sum(nodes_a.values())
        total_b = sum(nodes_b.values())

        # Calculate overlap
        matching = 0
        for node_type in all_types:
            count_a = nodes_a.get(node_type, 0)
            count_b = nodes_b.get(node_type, 0)
            matching += min(count_a, count_b)

        # Similarity as intersection over union
        total = total_a + total_b - matching
        similarity = matching / total if total > 0 else 1.0

        return similarity, matching, total_a, total_b

    def _extract_interfaces(self, tree: ast.AST) -> List[InterfaceSignature]:
        """Extract function and class signatures from AST."""
        interfaces = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                sig = self._function_to_signature(node)
                interfaces.append(sig)

            elif isinstance(node, ast.ClassDef):
                sig = self._class_to_signature(node)
                interfaces.append(sig)

        return interfaces

    def _function_to_signature(self, node) -> InterfaceSignature:
        """Extract signature from function definition."""
        params = []
        for arg in node.args.args:
            param_name = arg.arg
            if self.normalize_names:
                param_name = f"param_{len(params)}"
            params.append(param_name)

        # Return annotation
        return_ann = None
        if node.returns:
            return_ann = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)

        # Decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)

        return InterfaceSignature(
            name=node.name,
            kind="function",
            parameters=params,
            return_annotation=return_ann,
            decorators=decorators,
            is_async=isinstance(node, ast.AsyncFunctionDef)
        )

    def _class_to_signature(self, node: ast.ClassDef) -> InterfaceSignature:
        """Extract signature from class definition."""
        # Get methods as "parameters"
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)

        # Decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)

        return InterfaceSignature(
            name=node.name,
            kind="class",
            parameters=methods,  # methods listed here
            decorators=decorators
        )

    def _compare_interfaces(
        self,
        interfaces_a: List[InterfaceSignature],
        interfaces_b: List[InterfaceSignature]
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Compare interface signatures.

        Returns similarity score and lists of matching/missing/extra.
        """
        names_a = {sig.name for sig in interfaces_a}
        names_b = {sig.name for sig in interfaces_b}

        matching = list(names_a & names_b)
        missing = list(names_a - names_b)  # In A but not B
        extra = list(names_b - names_a)  # In B but not A

        # Calculate similarity
        total = len(names_a | names_b)
        if total == 0:
            return 1.0, matching, missing, extra

        # For matching names, check if signatures actually match
        sig_matches = 0
        sigs_a = {sig.name: sig for sig in interfaces_a}
        sigs_b = {sig.name: sig for sig in interfaces_b}

        for name in matching:
            sig_a = sigs_a[name]
            sig_b = sigs_b[name]

            if self._signatures_match(sig_a, sig_b):
                sig_matches += 1

        # Score is signature matches / total unique names
        similarity = sig_matches / total

        return similarity, matching, missing, extra

    def _signatures_match(
        self,
        sig_a: InterfaceSignature,
        sig_b: InterfaceSignature
    ) -> bool:
        """Check if two signatures match."""
        # Kind must match
        if sig_a.kind != sig_b.kind:
            return False

        # Parameter count must match
        if len(sig_a.parameters) != len(sig_b.parameters):
            return False

        # Async must match for functions
        if sig_a.kind == "function" and sig_a.is_async != sig_b.is_async:
            return False

        return True

    def compare_files(
        self,
        file_a: str,
        file_b: str
    ) -> ComparisonResult:
        """
        Compare two Python files.

        Args:
            file_a: Path to first file
            file_b: Path to second file

        Returns:
            ComparisonResult
        """
        from pathlib import Path

        try:
            code_a = Path(file_a).read_text()
        except Exception as e:
            logger.error(f"Failed to read {file_a}: {e}")
            code_a = ""

        try:
            code_b = Path(file_b).read_text()
        except Exception as e:
            logger.error(f"Failed to read {file_b}: {e}")
            code_b = ""

        return self.compare(code_a, code_b)


def compare_code(code_a: str, code_b: str) -> Dict[str, Any]:
    """
    Convenience function to compare two code strings.

    Args:
        code_a: Reference code
        code_b: Generated code

    Returns:
        Dict with comparison results
    """
    comparator = ASTComparator()
    result = comparator.compare(code_a, code_b)
    return result.to_dict()


def print_comparison(result: ComparisonResult):
    """Pretty-print comparison result."""
    print("\n" + "=" * 60)
    print("AST COMPARISON RESULT")
    print("=" * 60)

    print(f"Overall Similarity: {result.similarity:.3f}")
    print(f"  Structural: {result.structural_similarity:.3f}")
    print(f"  Interface:  {result.interface_match:.3f}")
    print()

    print(f"Nodes: {result.nodes_in_a} vs {result.nodes_in_b} ({result.matching_nodes} matching)")

    if result.matching_interfaces:
        print(f"\nMatching interfaces ({len(result.matching_interfaces)}):")
        for name in result.matching_interfaces[:5]:
            print(f"  - {name}")

    if result.missing_interfaces:
        print(f"\nMissing from generated ({len(result.missing_interfaces)}):")
        for name in result.missing_interfaces[:5]:
            print(f"  - {name}")

    if result.extra_interfaces:
        print(f"\nExtra in generated ({len(result.extra_interfaces)}):")
        for name in result.extra_interfaces[:5]:
            print(f"  - {name}")

    if not result.a_parsed:
        print("\nWARNING: Reference code failed to parse")
    if not result.b_parsed:
        print("\nWARNING: Generated code failed to parse")

    print("=" * 60)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Testing AST Comparator ===\n")

    # Test 1: Identical code
    code1 = '''
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    return a - b
'''

    code2 = '''
def add(x, y):
    # Add two numbers
    return x + y

def subtract(x, y):
    return x - y
'''

    print("Test 1: Semantically equivalent code")
    comparator = ASTComparator()
    result = comparator.compare(code1, code2)
    print_comparison(result)

    # Test 2: Missing function
    code3 = '''
def add(a, b):
    return a + b
'''

    print("\nTest 2: Missing function")
    result = comparator.compare(code1, code3)
    print_comparison(result)

    # Test 3: Different structure
    code4 = '''
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
'''

    print("\nTest 3: Function vs Class")
    result = comparator.compare(code1, code4)
    print_comparison(result)

    # Test 4: Async vs sync
    code5 = '''
async def add(a, b):
    return a + b
'''

    print("\nTest 4: Async vs Sync")
    result = comparator.compare(code1, code5)
    print_comparison(result)

    print("\n=== Tests complete ===")

"""
DEPENDENCY RESOLVER
Topological sorting of specs based on DEPENDS_ON edges for correct build order.

The Architect emits DEPENDS_ON edges to indicate that one spec requires another
to be built first. This module resolves those dependencies into a valid build order.

Example:
    If Spec A depends on Spec B (A --DEPENDS_ON--> B), then B must be built before A.

Usage:
    resolver = DependencyResolver()
    resolver.add_spec("spec_1", {"content": "..."})
    resolver.add_spec("spec_2", {"content": "..."})
    resolver.add_dependency("spec_2", "spec_1")  # spec_2 depends on spec_1

    build_order = resolver.get_build_order()
    # Returns ["spec_1", "spec_2"]
"""
import logging
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


logger = logging.getLogger("GAADP.DependencyResolver")


class CyclicDependencyError(Exception):
    """Raised when a cycle is detected in the dependency graph."""

    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Cyclic dependency detected: {' -> '.join(cycle)}")


@dataclass
class SpecNode:
    """A specification node with its metadata."""
    id: str
    content: str
    node_type: str = "SPEC"
    metadata: Dict = field(default_factory=dict)
    dependencies: Set[str] = field(default_factory=set)  # IDs this spec depends on
    dependents: Set[str] = field(default_factory=set)    # IDs that depend on this spec


class DependencyResolver:
    """
    Resolves build order based on DEPENDS_ON relationships.

    Uses Kahn's algorithm for topological sorting to determine
    a valid build order that respects all dependencies.
    """

    def __init__(self):
        self.specs: Dict[str, SpecNode] = {}
        self._sorted_order: Optional[List[str]] = None

    def add_spec(self, spec_id: str, spec_data: Dict) -> None:
        """
        Add a specification to the resolver.

        Args:
            spec_id: Unique identifier for the spec
            spec_data: Dict containing 'content', 'type', 'metadata'
        """
        if spec_id in self.specs:
            logger.warning(f"Spec {spec_id} already exists, updating")

        self.specs[spec_id] = SpecNode(
            id=spec_id,
            content=spec_data.get("content", ""),
            node_type=spec_data.get("type", "SPEC"),
            metadata=spec_data.get("metadata", {})
        )
        self._sorted_order = None  # Invalidate cache

    def add_dependency(self, dependent_id: str, dependency_id: str) -> None:
        """
        Add a DEPENDS_ON relationship.

        Args:
            dependent_id: The spec that has the dependency (needs the other)
            dependency_id: The spec being depended upon (must be built first)

        Example:
            add_dependency("user_repo", "db_connection")
            # user_repo depends on db_connection
            # db_connection will be built BEFORE user_repo
        """
        if dependent_id not in self.specs:
            logger.warning(f"Dependent spec {dependent_id} not found, creating placeholder")
            self.specs[dependent_id] = SpecNode(id=dependent_id, content="(placeholder)")

        if dependency_id not in self.specs:
            logger.warning(f"Dependency spec {dependency_id} not found, creating placeholder")
            self.specs[dependency_id] = SpecNode(id=dependency_id, content="(placeholder)")

        self.specs[dependent_id].dependencies.add(dependency_id)
        self.specs[dependency_id].dependents.add(dependent_id)
        self._sorted_order = None  # Invalidate cache

        logger.debug(f"Added dependency: {dependent_id} --DEPENDS_ON--> {dependency_id}")

    def get_build_order(self) -> List[str]:
        """
        Get the topologically sorted build order.

        Returns:
            List of spec IDs in the order they should be built.
            Specs with no dependencies come first.

        Raises:
            CyclicDependencyError: If a cycle is detected
        """
        if self._sorted_order is not None:
            return self._sorted_order

        # Kahn's algorithm for topological sort
        # https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm

        # Calculate in-degree (number of dependencies) for each node
        in_degree: Dict[str, int] = {
            spec_id: len(spec.dependencies)
            for spec_id, spec in self.specs.items()
        }

        # Start with nodes that have no dependencies
        queue: List[str] = [
            spec_id for spec_id, degree in in_degree.items() if degree == 0
        ]
        result: List[str] = []

        while queue:
            # Process node with no remaining dependencies
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for all dependents
            for dependent_id in self.specs[current].dependents:
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        # Check for cycles
        if len(result) != len(self.specs):
            # Find a cycle for the error message
            cycle = self._find_cycle()
            raise CyclicDependencyError(cycle)

        self._sorted_order = result
        logger.info(f"Resolved build order: {result}")
        return result

    def _find_cycle(self) -> List[str]:
        """Find a cycle in the dependency graph for error reporting."""
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node: str) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in self.specs[node].dependencies:
                if dep not in visited:
                    cycle = dfs(dep)
                    if cycle:
                        return cycle
                elif dep in rec_stack:
                    # Found cycle
                    cycle_start = path.index(dep)
                    return path[cycle_start:] + [dep]

            path.pop()
            rec_stack.remove(node)
            return None

        for spec_id in self.specs:
            if spec_id not in visited:
                cycle = dfs(spec_id)
                if cycle:
                    return cycle

        return ["unknown"]  # Should not reach here

    def get_build_waves(self) -> List[List[str]]:
        """
        Get specs grouped into parallel build waves.

        Returns a list of waves, where each wave contains specs that
        can be built in parallel (all their dependencies are in previous waves).

        Example:
            Wave 0: [db_connection, config]  # No dependencies
            Wave 1: [user_repo, cache]       # Depend on wave 0
            Wave 2: [user_service]           # Depends on wave 1
        """
        build_order = self.get_build_order()
        waves: List[List[str]] = []
        spec_to_wave: Dict[str, int] = {}

        for spec_id in build_order:
            spec = self.specs[spec_id]

            if not spec.dependencies:
                # No dependencies - goes in wave 0
                wave_num = 0
            else:
                # Goes in wave after all dependencies
                dep_waves = [spec_to_wave.get(d, 0) for d in spec.dependencies]
                wave_num = max(dep_waves) + 1

            spec_to_wave[spec_id] = wave_num

            # Ensure we have enough waves
            while len(waves) <= wave_num:
                waves.append([])

            waves[wave_num].append(spec_id)

        logger.info(f"Build waves: {len(waves)} waves, specs per wave: {[len(w) for w in waves]}")
        return waves

    def get_spec(self, spec_id: str) -> Optional[SpecNode]:
        """Get a spec by ID."""
        return self.specs.get(spec_id)

    def get_dependencies(self, spec_id: str) -> Set[str]:
        """Get the direct dependencies of a spec."""
        spec = self.specs.get(spec_id)
        return spec.dependencies if spec else set()

    def get_dependents(self, spec_id: str) -> Set[str]:
        """Get the specs that depend on this spec."""
        spec = self.specs.get(spec_id)
        return spec.dependents if spec else set()

    def get_all_dependencies(self, spec_id: str) -> Set[str]:
        """Get all transitive dependencies of a spec."""
        result = set()
        to_process = list(self.get_dependencies(spec_id))

        while to_process:
            dep = to_process.pop()
            if dep not in result:
                result.add(dep)
                to_process.extend(self.get_dependencies(dep))

        return result

    def clear(self) -> None:
        """Clear all specs and dependencies."""
        self.specs.clear()
        self._sorted_order = None


def resolve_architect_output(
    new_nodes: List[Dict],
    new_edges: List[Dict],
    node_id_map: Dict[str, str]
) -> Tuple[List[str], List[List[str]]]:
    """
    Resolve Architect output into build order.

    Args:
        new_nodes: List of node specs from Architect
        new_edges: List of edge specs from Architect
        node_id_map: Mapping from temp IDs ("new_0") to actual node IDs

    Returns:
        Tuple of (linear_build_order, parallel_waves)
    """
    resolver = DependencyResolver()

    # Add all SPEC nodes
    for idx, node in enumerate(new_nodes):
        if node.get("type") == "SPEC":
            temp_id = f"new_{idx}"
            actual_id = node_id_map.get(temp_id, node_id_map.get(idx, temp_id))
            resolver.add_spec(actual_id, node)
            logger.debug(f"Added spec {temp_id} -> {actual_id}")

    # Add DEPENDS_ON edges
    for edge in new_edges or []:
        if edge.get("relation") == "DEPENDS_ON":
            src_temp = edge.get("source_id")
            tgt_temp = edge.get("target_id")

            src_actual = node_id_map.get(src_temp, src_temp)
            tgt_actual = node_id_map.get(tgt_temp, tgt_temp)

            if src_actual in resolver.specs and tgt_actual in resolver.specs:
                resolver.add_dependency(src_actual, tgt_actual)
            else:
                logger.warning(f"Skipping dependency edge: {src_temp} -> {tgt_temp} (nodes not found)")

    try:
        linear_order = resolver.get_build_order()
        waves = resolver.get_build_waves()
        return linear_order, waves
    except CyclicDependencyError as e:
        logger.error(f"Cyclic dependency detected: {e.cycle}")
        # Fall back to original order
        return list(resolver.specs.keys()), [list(resolver.specs.keys())]


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # Test basic dependency resolution
    print("=== Test: Basic Dependency Resolution ===")
    resolver = DependencyResolver()

    resolver.add_spec("db_connection", {"content": "Database connection class"})
    resolver.add_spec("user_model", {"content": "User model class"})
    resolver.add_spec("user_repo", {"content": "User repository class"})
    resolver.add_spec("user_service", {"content": "User service class"})

    # user_repo depends on db_connection and user_model
    resolver.add_dependency("user_repo", "db_connection")
    resolver.add_dependency("user_repo", "user_model")

    # user_service depends on user_repo
    resolver.add_dependency("user_service", "user_repo")

    order = resolver.get_build_order()
    print(f"Build order: {order}")
    assert order.index("db_connection") < order.index("user_repo")
    assert order.index("user_model") < order.index("user_repo")
    assert order.index("user_repo") < order.index("user_service")

    waves = resolver.get_build_waves()
    print(f"Build waves: {waves}")
    assert len(waves) == 3
    assert set(waves[0]) == {"db_connection", "user_model"}
    assert waves[1] == ["user_repo"]
    assert waves[2] == ["user_service"]

    print("\n=== Test: Cycle Detection ===")
    resolver2 = DependencyResolver()
    resolver2.add_spec("a", {"content": "A"})
    resolver2.add_spec("b", {"content": "B"})
    resolver2.add_spec("c", {"content": "C"})

    resolver2.add_dependency("a", "b")
    resolver2.add_dependency("b", "c")
    resolver2.add_dependency("c", "a")  # Creates cycle

    try:
        resolver2.get_build_order()
        print("ERROR: Should have raised CyclicDependencyError")
    except CyclicDependencyError as e:
        print(f"Correctly detected cycle: {e.cycle}")

    print("\n✅ All tests passed!")

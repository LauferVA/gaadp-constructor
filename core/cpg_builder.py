"""
CODE PROPERTY GRAPH BUILDER
"""
import ast
import networkx as nx
from typing import List
from core.ontology import NodeType, EdgeType, NodeStatus

class CPGBuilder:
    def build_from_source(self, source_code: str, file_path: str) -> nx.DiGraph:
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return nx.DiGraph()

        graph = nx.DiGraph()
        imports = self._extract_imports(tree)
        root_id = f"code_{hash(file_path)}"

        graph.add_node(
            root_id,
            type=NodeType.CODE.value,
            status=NodeStatus.PENDING.value,
            content=source_code,
            metadata={"file_path": file_path, "language": "python", "imports": imports}
        )

        for imp in imports:
            import_id = f"phantom_{imp}"
            graph.add_node(import_id, type=NodeType.CODE.value, status=NodeStatus.COMPLETE.value)
            graph.add_edge(root_id, import_id, type=EdgeType.DEPENDS_ON.value)

        return graph

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

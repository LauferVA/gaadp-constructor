# Code Property Graph Builder using Python AST

import ast
import networkx as nx
from typing import List, Dict, Any
from core.ontology import NodeType, EdgeType, NodeStatus

class CPGBuilder:
    def build_from_source(self, source_code: str, file_path: str) -> nx.DiGraph:
        """
        Build a Code Property Graph from source code using AST traversal
        
        Args:
            source_code (str): Source code to analyze
            file_path (str): Path of the source file
        
        Returns:
            nx.DiGraph: A graph representing the code structure
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return nx.DiGraph()

        graph = nx.DiGraph()
        root_id = f"code_{hash(file_path)}"

        # Extract code elements
        imports = self._extract_imports(tree)
        classes = self._extract_classes(tree)
        functions = self._extract_functions(tree)
        calls = self._extract_calls(tree)

        # Add root node with source code and metadata
        graph.add_node(
            root_id,
            type=NodeType.CODE.value,
            status=NodeStatus.PENDING.value,
            content=source_code,
            metadata={
                "file_path": file_path, 
                "language": "python", 
                "imports": imports,
                "classes": list(classes.keys()),
                "functions": list(functions.keys())
            }
        )

        # Add import nodes and edges
        for imp in imports:
            import_id = f"phantom_import_{imp}"
            graph.add_node(
                import_id, 
                type=NodeType.CODE.value, 
                status=NodeStatus.COMPLETE.value,
                content=imp
            )
            graph.add_edge(root_id, import_id, type=EdgeType.DEPENDS_ON.value)

        # Add class nodes and edges
        for class_name, class_details in classes.items():
            class_id = f"class_{class_name}"
            graph.add_node(
                class_id,
                type=NodeType.CLASS.value,
                status=NodeStatus.COMPLETE.value,
                content=class_details['content'],
                metadata=class_details
            )
            graph.add_edge(root_id, class_id, type=EdgeType.CONTAINS.value)

        # Add function nodes and edges
        for func_name, func_details in functions.items():
            func_id = f"function_{func_name}"
            graph.add_node(
                func_id,
                type=NodeType.FUNCTION.value,
                status=NodeStatus.COMPLETE.value,
                content=func_details['content'],
                metadata=func_details
            )
            graph.add_edge(root_id, func_id, type=EdgeType.CONTAINS.value)

        # Add call nodes and edges
        for call_name, call_details in calls.items():
            call_id = f"call_{call_name}"
            graph.add_node(
                call_id,
                type=NodeType.CALL.value,
                status=NodeStatus.COMPLETE.value,
                content=call_name,
                metadata=call_details
            )
            graph.add_edge(root_id, call_id, type=EdgeType.REFERENCES.value)

        return graph

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """
        Extract import statements from the AST
        
        Args:
            tree (ast.AST): Abstract Syntax Tree
        
        Returns:
            List[str]: List of imported module names
        """
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _extract_classes(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        """
        Extract class definitions from the AST
        
        Args:
            tree (ast.AST): Abstract Syntax Tree
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of class details
        """
        classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes[node.name] = {
                    'name': node.name,
                    'lineno': node.lineno,
                    'col_offset': node.col_offset,
                    'bases': [base.id for base in node.bases if isinstance(base, ast.Name)],
                    'content': ast.get_source_segment(tree, node) or ''
                }
        return classes

    def _extract_functions(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        """
        Extract function definitions from the AST
        
        Args:
            tree (ast.AST): Abstract Syntax Tree
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of function details
        """
        functions = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions[node.name] = {
                    'name': node.name,
                    'lineno': node.lineno,
                    'col_offset': node.col_offset,
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'args': [arg.arg for arg in node.args.args],
                    'content': ast.get_source_segment(tree, node) or ''
                }
        return functions

    def _extract_calls(self, tree: ast.AST) -> Dict[str, Dict[str, Any]]:
        """
        Extract function and method calls from the AST
        
        Args:
            tree (ast.AST): Abstract Syntax Tree
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of call details
        """
        calls = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Handle function and method calls
                if isinstance(node.func, ast.Name):
                    call_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    call_name = node.func.attr
                else:
                    continue

                calls[call_name] = {
                    'lineno': node.lineno,
                    'col_offset': node.col_offset,
                    'args_count': len(node.args)
                }
        return calls

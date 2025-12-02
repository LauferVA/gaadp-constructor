#!/usr/bin/env python3
"""
CPG Visualization Script

Builds a Code Property Graph from a Python file and exports visualizations.

Usage:
    python scripts/visualize_cpg.py <file.py>
    python scripts/visualize_cpg.py agents/base_agent.py
    python scripts/visualize_cpg.py .  # Analyze all Python files in directory
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cpg_builder import CPGBuilder
from infrastructure.visualizer import GraphVisualizer
import networkx as nx


def analyze_file(file_path: str, output_dir: str = ".gaadp/viz") -> dict:
    """Analyze a single Python file and return stats."""
    with open(file_path, 'r') as f:
        source = f.read()

    builder = CPGBuilder()
    graph = builder.build_from_source(source, file_path)

    # Count by type
    type_counts = {}
    for _, data in graph.nodes(data=True):
        t = data.get('type', 'UNKNOWN')
        type_counts[t] = type_counts.get(t, 0) + 1

    return {
        'file': file_path,
        'nodes': graph.number_of_nodes(),
        'edges': graph.number_of_edges(),
        'types': type_counts,
        'graph': graph
    }


def merge_graphs(graphs: list) -> nx.DiGraph:
    """Merge multiple CPG graphs into one."""
    merged = nx.DiGraph()
    for g in graphs:
        merged.add_nodes_from(g.nodes(data=True))
        merged.add_edges_from(g.edges(data=True))
    return merged


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    output_dir = ".gaadp/viz"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(target):
        # Single file
        print(f"Analyzing: {target}")
        stats = analyze_file(target)

        print(f"\n=== CPG ANALYSIS: {target} ===")
        print(f"Nodes: {stats['nodes']}")
        print(f"Edges: {stats['edges']}")
        print("\nNode Types:")
        for t, count in sorted(stats['types'].items()):
            print(f"  {t}: {count}")

        # Export
        base_name = Path(target).stem
        viz = GraphVisualizer(stats['graph'])
        html_path = viz.export(f"{output_dir}/{base_name}_cpg.html", format='html')
        dot_path = viz.export(f"{output_dir}/{base_name}_cpg.dot", format='dot')

        print(f"\nExported:")
        print(f"  HTML: {html_path}")
        print(f"  DOT:  {dot_path}")

    elif os.path.isdir(target):
        # Directory - analyze all Python files
        py_files = list(Path(target).rglob("*.py"))
        py_files = [f for f in py_files if not any(p in str(f) for p in ['__pycache__', '.gaadp', 'venv'])]

        print(f"Analyzing {len(py_files)} Python files in {target}...")

        all_stats = []
        all_graphs = []

        for py_file in py_files:
            try:
                stats = analyze_file(str(py_file))
                all_stats.append(stats)
                all_graphs.append(stats['graph'])
                print(f"  {py_file}: {stats['nodes']} nodes")
            except Exception as e:
                print(f"  {py_file}: ERROR - {e}")

        # Merge all graphs
        merged = merge_graphs(all_graphs)

        print(f"\n=== MERGED CPG ===")
        print(f"Total Nodes: {merged.number_of_nodes()}")
        print(f"Total Edges: {merged.number_of_edges()}")

        # Count types across all files
        total_types = {}
        for stats in all_stats:
            for t, count in stats['types'].items():
                total_types[t] = total_types.get(t, 0) + count

        print("\nNode Types (all files):")
        for t, count in sorted(total_types.items()):
            print(f"  {t}: {count}")

        # Export merged graph
        viz = GraphVisualizer(merged)
        html_path = viz.export(f"{output_dir}/project_cpg.html", format='html')

        print(f"\nExported merged visualization: {html_path}")

    else:
        print(f"Error: {target} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()

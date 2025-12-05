import os
import json
import inspect
import importlib.util
import sys
import glob
from datetime import datetime
from pathlib import Path

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
OUTPUT_FILENAME = "GAADP_Formal_Structure_XRay.md"
# Tries to find the Desktop; falls back to current directory if not found
DESKTOP_PATH = os.path.join(os.path.expanduser("~"), "Desktop")
if not os.path.exists(DESKTOP_PATH):
    DESKTOP_PATH = os.getcwd()
OUTPUT_PATH = os.path.join(DESKTOP_PATH, OUTPUT_FILENAME)

# Target file paths (Relative to Project Root)
PATH_ONTOLOGY = "core/ontology.py"
PATH_RUNTIME = "infrastructure/graph_runtime.py"
PATH_MANIFEST = "config/agent_manifest.yaml"

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def load_module_from_path(name, path):
    """
    Dynamically loads a python module from a file path to inspect its classes.
    """
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        return f"Error loading module {name}: {e}"

def get_class_hierarchy(module):
    """
    Introspects a module to find Class definitions (Pydantic models),
    representing the 'State Space' of the system.
    """
    if module is None: 
        return "*Module not found or could not be loaded.*"
    
    if isinstance(module, str): 
        return module # Return error message
        
    classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and obj.__module__ == module.__name__:
            # Attempt to get Pydantic fields or docstrings
            fields = "No schema detected"
            if hasattr(obj, "__annotations__"):
                fields = {k: str(v).replace("<class '", "").replace("'>", "") 
                          for k, v in obj.__annotations__.items()}
            
            doc = inspect.getdoc(obj) or "No description."
            classes.append(f"### Class: {name}\n* **Doc:** {doc}\n* **Schema:** `{fields}`\n")
            
    return "\n".join(classes) if classes else "*No classes found in module.*"

def extract_transition_logic(file_path):
    """
    Naively extracts dictionaries or logic blocks related to State Transitions
    from the source code text.
    """
    if not os.path.exists(file_path):
        return "*File not found.*"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        captured_blocks = []
        is_capturing = False
        current_block = []
        
        # Heuristic: Look for Dispatch dictionaries or Transition definitions
        keywords = ["DISPATCH", "TRANSITION", "MATRIX", "conditions", "def dispatch"]
        
        for line in lines:
            if any(k in line for k in keywords):
                is_capturing = True
            
            if is_capturing:
                current_block.append(line)
                # End capture on unindented closing brace or function end (heuristic)
                if line.strip() == "}" or (line.strip() == "" and len(current_block) > 10):
                    captured_blocks.append("".join(current_block))
                    current_block = []
                    is_capturing = False
                    
        return "```python\n" + "\n...\n".join(captured_blocks) + "\n```" if captured_blocks else "*No explicit transition matrices found via heuristic scan.*"
    except Exception as e:
        return f"*Error reading file: {e}*"

def analyze_dag_topology(dag_file):
    """
    Analyzes a JSON DAG artifact to extract graph-theoretic properties.
    """
    try:
        with open(dag_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        
        if not nodes: return f"*File {os.path.basename(dag_file)} contains no nodes.*"
        
        node_count = len(nodes)
        edge_count = len(edges)
        
        # Build Adjacency List
        adj = {n['id']: [] for n in nodes}
        in_degree = {n['id']: 0 for n in nodes}
        out_degree = {n['id']: 0 for n in nodes}
        
        for edge in edges:
            src = edge.get('source')
            tgt = edge.get('target')
            if src in adj: 
                adj[src].append(tgt)
                out_degree[src] += 1
            if tgt in in_degree:
                in_degree[tgt] += 1
                
        # Calculate Metrics
        sources = [n for n, d in in_degree.items() if d == 0]
        sinks = [n for n, d in out_degree.items() if d == 0]
        density = edge_count / (node_count * (node_count - 1)) if node_count > 1 else 0
        
        return f"""
#### Artifact: {os.path.basename(dag_file)}
* **Vertices ($|V|$):** {node_count}
* **Edges ($|E|$):** {edge_count}
* **Graph Density:** {density:.4f}
* **Topology Type:** {'Connected' if edge_count > 0 else 'Disconnected/Sparse'}
* **Source Nodes (Inputs):** {len(sources)} (IDs: {', '.join(sources[:3])}...)
* **Sink Nodes (Outputs):** {len(sinks)} (IDs: {', '.join(sinks[:3])}...)
"""
    except Exception as e:
        return f"Error analyzing DAG {dag_file}: {str(e)}"

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Starting Formal Structure X-Ray...")
    report = []
    
    # HEADER
    report.append(f"# GAADP Formal Structure Analysis (The X-Ray)")
    report.append(f"**Date:** {datetime.now().isoformat()}")
    report.append(f"**Context:** This document provides the set-theoretic and topological definitions required for formal verification of the GAADP system.")
    report.append("\n---\n")

    # 1. ONTOLOGY (Set S)
    report.append("## 1. The State Space ($S$) and Alphabet ($\Sigma$)")
    report.append("### Source: `core/ontology.py`")
    report.append("Defines the formal types (nodes) and allowed relations (edges).")
    ont_module = load_module_from_path("ontology", PATH_ONTOLOGY)
    report.append(get_class_hierarchy(ont_module))
    report.append("\n---\n")

    # 2. TRANSITION FUNCTION (Delta)
    report.append("## 2. The Transition Function ($\delta: S \\times \Sigma \\rightarrow S$)")
    report.append("### Source: `infrastructure/graph_runtime.py`")
    report.append("Defines the deterministic logic that governs state changes based on agent triggers.")
    report.append(extract_transition_logic(PATH_RUNTIME))
    report.append("\n---\n")

    # 3. TOPOLOGY (Graph G)
    report.append("## 3. Topological Realization ($G = (V, E)$)")
    report.append("Topological analysis of actual graph artifacts generated by the system.")
    
    # Search for JSON DAGs recursively
    json_files = glob.glob("**/*dag*.json", recursive=True)
    # Filter out small configs, keep substantial DAGs
    valid_dags = [f for f in json_files if os.path.getsize(f) > 500][:3] 
    
    if valid_dags:
        for dag in valid_dags:
            report.append(analyze_dag_topology(dag))
    else:
        report.append("*No generated DAG artifacts found to analyze.*")
    report.append("\n---\n")

    # 4. AGENTS (Operators)
    report.append("## 4. The Operator Set")
    report.append("### Source: `config/agent_manifest.yaml`")
    if os.path.exists(PATH_MANIFEST):
        with open(PATH_MANIFEST, 'r') as f:
            report.append("```yaml\n" + f.read() + "\n```")
    else:
        report.append("*Agent manifest not found.*")

    # WRITE FILE
    try:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print(f"\n✅ SUCCESS: X-Ray generated.")
        print(f"📂 Location: {OUTPUT_PATH}")
        print("You can now upload this file to a Mathematician/Logician Agent.")
    except Exception as e:
        print(f"\n❌ ERROR: Could not write to desktop. {e}")

if __name__ == "__main__":
    main()

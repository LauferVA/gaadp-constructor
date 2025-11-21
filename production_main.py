#!/usr/bin/env python3
"""
GAADP PRODUCTION RUNTIME (Hardened)
"""
import asyncio
import os
import uuid
import subprocess
import networkx as nx
from infrastructure.graph_db import GraphDB
from infrastructure.version_control import GitController
from infrastructure.sandbox import CodeSandbox
from agents.concrete_agents import RealArchitect, RealBuilder, RealVerifier
from core.ontology import AgentRole, NodeType, EdgeType, NodeStatus

if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: Missing API Keys.")
    exit(1)

def introspect_graph(db: GraphDB):
    """Print the knowledge graph state for debugging."""
    print("\n📊 KNOWLEDGE GRAPH STATE:")
    print(f"  Nodes: {db.graph.number_of_nodes()} | Edges: {db.graph.number_of_edges()}")

    for n, d in db.graph.nodes(data=True):
        status = d.get('status', 'UNKNOWN')
        print(f"  [{status.ljust(8)}] {n[:15]}... : {d.get('type')}")

    try:
        cycles = list(nx.simple_cycles(db.graph))
        if cycles:
            print(f"  ⚠️ CRITICAL: {len(cycles)} Cycles Detected!")
        else:
            print("  ✅ Integrity: DAG Compliant (No Cycles)")
    except: pass

async def main():
    print("🚀 INITIALIZING GAADP PRODUCTION SWARM (HARDENED)...")
    db = GraphDB(persistence_path=".gaadp/live_graph.pkl")

    architect = RealArchitect("arch_01", AgentRole.ARCHITECT, db)
    builder = RealBuilder("build_01", AgentRole.BUILDER, db)
    verifier = RealVerifier("verif_01", AgentRole.VERIFIER, db)

    # 1. Inject Requirement
    req_id = "req_live_001"
    print(f"📝 Injecting Root Requirement: {req_id}")
    db.add_node(req_id, NodeType.REQ, "Create a Python function that calculates the Fibonacci sequence recursively with memoization.")

    # 2. Architect Plan
    print("🧠 Architect is thinking...")
    arch_output = await architect.process({"nodes": [{"content": "Create Fibonacci function", "id": req_id}]})

    plan_id, spec_id = None, None
    for n in arch_output.get('new_nodes', []):
        print(f"   + Created Node: {n['type']}")
        db.add_node(uuid.uuid4().hex, NodeType(n['type']), n['content'])
        if n['type'] == 'PLAN': plan_id = "plan_fib_01"
        if n['type'] == 'SPEC': spec_id = "spec_fib_01"

    # 3. Builder Execution
    print("🔨 Builder is coding...")
    build_output = await builder.process({"nodes": [{"content": "Implement Fibonacci", "id": spec_id}]})

    code_id = uuid.uuid4().hex
    db.add_node(code_id, NodeType(build_output['type']), build_output['content'], metadata=build_output.get('metadata'))
    print(f"   > Generated Code ID: {code_id}")

    # 4. Verification & Merkle Signing
    print("⚖️ Verifier is judging...")
    verify_output = await verifier.process({"nodes": [{"id": code_id, "content": build_output['content']}]})
    print(f"   > Verdict: {verify_output['verdict']}")

    if verify_output['verdict'] == 'PASS':
        print("✅ SUCCESS: Code Verified. Linking Chain...")

        # A. Get Previous Hash for Chain
        prev_hash = db.get_last_node_hash()

        # B. Sign with Chain Context
        sig = verifier.sign_content(code_id, previous_hash=prev_hash)

        # C. Update Graph
        db.add_edge(code_id, code_id, EdgeType.VERIFIES, verifier.agent_id, sig, previous_hash=prev_hash)
        db.graph.nodes[code_id]['status'] = NodeStatus.VERIFIED.value

        # D. MATERIALIZE (Write to Disk)
        file_path = build_output['metadata'].get('file_path', 'output.py')
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        with open(file_path, "w") as f:
            f.write(build_output['content'])
        print(f"💾 Saved to disk: {file_path}")

        # E. EXECUTE (Sandbox)
        print("⚙️ Executing Code in Sandbox...")
        try:
            test_wrapper = f"from {file_path.replace('.py', '')} import fibonacci\nprint(f'FIB(10)={{fibonacci(10)}}')"
            with open("temp_run.py", "w") as f: f.write(test_wrapper)

            result = subprocess.run(["python3", "temp_run.py"], capture_output=True, text=True, timeout=5)
            print(f"🚀 Output: {result.stdout.strip()}")
            if result.stderr: print(f"⚠️ Stderr: {result.stderr}")
        except Exception as e:
            print(f"💥 Execution Failed: {e}")

        # F. GIT COMMIT (Version Control)
        git = GitController()
        git.commit_work("builder_01", code_id, f"Implemented {file_path}")

        # G. SANDBOX EXECUTION (Safe Run)
        sandbox = CodeSandbox(use_docker=False)  # Set True if Docker installed
        run_result = sandbox.run_code(file_path)
        print(f"📦 Sandbox Output: {run_result['stdout']}")
    else:
        print("❌ FAILURE: Code rejected.")

    introspect_graph(db)

if __name__ == "__main__":
    asyncio.run(main())

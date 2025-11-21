#!/usr/bin/env python3
"""
GAADP PRODUCTION RUNTIME (Full Featured)
Includes: Domain Discovery, Data Loading, MCP Tools, RBAC
"""
import asyncio
import os
import uuid
import subprocess
import networkx as nx
from infrastructure.graph_db import GraphDB
from infrastructure.version_control import GitController
from infrastructure.sandbox import CodeSandbox
from infrastructure.semantic_memory import SemanticMemory
from infrastructure.data_loader import DataLoader
from infrastructure.mcp_hub import MCPHub
from core.domain_discovery import DomainResearcher
from agents.concrete_agents import RealArchitect, RealBuilder, RealVerifier
from core.ontology import AgentRole, NodeType, EdgeType, NodeStatus

# Check for API keys
if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: Missing API Keys (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
    exit(1)


def introspect_graph(db: GraphDB):
    """Print the knowledge graph state for debugging."""
    print("\n📊 KNOWLEDGE GRAPH STATE:")
    print(f"  Nodes: {db.graph.number_of_nodes()} | Edges: {db.graph.number_of_edges()}")

    for n, d in db.graph.nodes(data=True):
        status = d.get('status', 'UNKNOWN')
        node_preview = n[:15] if len(n) > 15 else n
        print(f"  [{status.ljust(8)}] {node_preview}... : {d.get('type')}")

    try:
        cycles = list(nx.simple_cycles(db.graph))
        if cycles:
            print(f"  ⚠️ CRITICAL: {len(cycles)} Cycles Detected!")
        else:
            print("  ✅ Integrity: DAG Compliant (No Cycles)")
    except:
        pass


def get_requirement(interactive: bool = True) -> str:
    """Get requirement from file or user input."""
    # Try to read from prompt.md
    if os.path.exists("prompt.md"):
        with open("prompt.md", "r") as f:
            return f.read().strip()

    if interactive:
        return input("📝 Enter Requirement: ").strip()

    return "Create a Python function that calculates the Fibonacci sequence recursively with memoization."


async def main(interactive: bool = True):
    print("🚀 INITIALIZING GAADP PRODUCTION SWARM (FULL FEATURED)...")

    # Initialize core infrastructure
    db = GraphDB(persistence_path=".gaadp/live_graph.pkl")
    memory = SemanticMemory()
    mcp_hub = MCPHub()

    print("🧠 Semantic Memory Online")
    print(f"🔧 MCP Hub Online ({len(mcp_hub.list_all_tools())} tools available)")

    # --- PHASE 0: GET REQUIREMENT ---
    request = get_requirement(interactive)
    print(f"\n📋 Requirement: {request[:100]}...")

    # --- PHASE 0.5: DOMAIN DISCOVERY ---
    if interactive:
        do_discovery = input("\n🌍 Run Domain Discovery? (Y/n): ").strip().lower()
        if do_discovery in ['', 'y', 'yes']:
            researcher = DomainResearcher()
            researcher.run_discovery(request, interactive=True)

    # --- PHASE 0.75: DATA INGESTION ---
    if interactive:
        do_data = input("\n📂 Load source data/datasets? (y/N): ").strip().lower()
        if do_data == 'y':
            loader = DataLoader(db)
            loader.interactive_load()

    # --- PHASE 1: INITIALIZE AGENTS WITH MCP ---
    architect = RealArchitect("arch_01", AgentRole.ARCHITECT, db, mcp_hub=mcp_hub)
    builder = RealBuilder("build_01", AgentRole.BUILDER, db, mcp_hub=mcp_hub)
    verifier = RealVerifier("verif_01", AgentRole.VERIFIER, db, mcp_hub=mcp_hub)

    # --- PHASE 2: INJECT REQUIREMENT ---
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    print(f"\n📝 Injecting Root Requirement: {req_id}")
    db.add_node(req_id, NodeType.REQ, request)

    # Embed requirement into semantic memory
    memory.embed_node(req_id, request)

    # --- PHASE 3: ARCHITECT PLANNING ---
    print("🧠 Architect is thinking...")
    arch_output = await architect.process({"nodes": [{"content": request, "id": req_id}]})

    plan_id, spec_id = None, None
    for n in arch_output.get('new_nodes', []):
        node_id = uuid.uuid4().hex
        print(f"   + Created Node: {n['type']} ({node_id[:8]}...)")
        db.add_node(node_id, NodeType(n['type']), n['content'])
        memory.embed_node(node_id, n['content'])

        if n['type'] == 'PLAN':
            plan_id = node_id
        if n['type'] == 'SPEC':
            spec_id = node_id

    # --- PHASE 4: BUILDER EXECUTION ---
    print("🔨 Builder is coding...")
    spec_content = arch_output.get('new_nodes', [{}])[0].get('content', request)
    build_output = await builder.process({"nodes": [{"content": spec_content, "id": spec_id}]})

    code_id = uuid.uuid4().hex
    db.add_node(
        code_id,
        NodeType(build_output['type']),
        build_output['content'],
        metadata=build_output.get('metadata', {})
    )
    print(f"   > Generated Code ID: {code_id[:8]}...")

    # Embed code into semantic memory
    memory.embed_node(code_id, build_output['content'])
    print(f"   > Embedded into Vector Space")

    # --- PHASE 5: VERIFICATION ---
    print("⚖️ Verifier is judging...")
    verify_output = await verifier.process({
        "nodes": [{"id": code_id, "content": build_output['content']}]
    })
    print(f"   > Verdict: {verify_output.get('verdict', 'UNKNOWN')}")

    if verify_output.get('verdict') == 'PASS':
        print("✅ SUCCESS: Code Verified. Linking Chain...")

        # A. Get Previous Hash for Chain
        prev_hash = db.get_last_node_hash()

        # B. Sign with Chain Context
        sig = verifier.sign_content(code_id, previous_hash=prev_hash)

        # C. Update Graph
        db.add_edge(code_id, code_id, EdgeType.VERIFIES, verifier.agent_id, sig, previous_hash=prev_hash)
        db.graph.nodes[code_id]['status'] = NodeStatus.VERIFIED.value

        # D. MATERIALIZE (Write to Disk)
        file_path = build_output.get('metadata', {}).get('file_path', 'generated_output.py')
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        with open(file_path, "w") as f:
            f.write(build_output['content'])
        print(f"💾 Saved to disk: {file_path}")

        # E. SANDBOX EXECUTION
        print("⚙️ Executing Code in Sandbox...")
        sandbox = CodeSandbox(use_docker=False)
        run_result = sandbox.run_code(file_path)

        if run_result['exit_code'] == 0:
            print(f"🚀 Output: {run_result['stdout'][:200]}")
        else:
            print(f"⚠️ Execution issue: {run_result['stderr'][:200]}")

        # F. GIT COMMIT
        git = GitController()
        git.commit_work("builder_01", code_id, f"Implemented {file_path}")
        print("📦 Changes committed to Git")

    else:
        print("❌ FAILURE: Code rejected.")
        critique = verify_output.get('critique', 'No details provided')
        print(f"   Critique: {critique}")

    # --- FINAL: INTROSPECTION ---
    introspect_graph(db)

    # Show semantic search capability
    if memory.vectors:
        print("\n🔍 Semantic Search Demo:")
        similar = memory.find_similar("fibonacci recursive", top_k=2)
        for score, node_id in similar:
            print(f"   [{score:.3f}] {node_id[:15]}...")


async def run_batch(requirement: str):
    """Run in non-interactive batch mode."""
    # Write requirement to prompt.md for batch processing
    with open("prompt.md", "w") as f:
        f.write(requirement)
    await main(interactive=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        # Batch mode: python production_main.py --batch "requirement"
        req = sys.argv[2] if len(sys.argv) > 2 else "Create a hello world function"
        asyncio.run(run_batch(req))
    else:
        # Interactive mode
        asyncio.run(main(interactive=True))

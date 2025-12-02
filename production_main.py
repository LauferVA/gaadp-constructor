#!/usr/bin/env python3
"""
GAADP PRODUCTION RUNTIME (Full Featured)
Includes: Domain Discovery, Data Loading, MCP Tools, RBAC
"""
# Disable tokenizer parallelism warning BEFORE any other imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import uuid
import subprocess
import logging
import json
from datetime import datetime
from typing import Optional
import networkx as nx

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for file output."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, 'trace_id', None),
            "phase": getattr(record, 'phase', None),
            "agent": getattr(record, 'agent', None),
        }
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Remove None values
        log_entry = {k: v for k, v in log_entry.items() if v is not None}
        return json.dumps(log_entry)


class ConsoleFormatter(logging.Formatter):
    """Colored console formatter with trace context."""

    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        trace_id = getattr(record, 'trace_id', None)
        phase = getattr(record, 'phase', None)

        # Build prefix
        prefix_parts = []
        if trace_id:
            prefix_parts.append(f"[{trace_id[:8]}]")
        if phase:
            prefix_parts.append(f"[{phase}]")
        prefix = " ".join(prefix_parts)

        # Format: [LEVEL] [logger] prefix message
        formatted = f"{color}[{record.levelname:7}]{self.RESET} [{record.name}] {prefix} {record.getMessage()}"
        return formatted


class TraceContext:
    """Thread-local trace context for request tracing."""
    _current_trace_id: Optional[str] = None
    _current_phase: Optional[str] = None

    @classmethod
    def set_trace(cls, trace_id: str):
        cls._current_trace_id = trace_id

    @classmethod
    def set_phase(cls, phase: str):
        cls._current_phase = phase

    @classmethod
    def get_trace_id(cls) -> Optional[str]:
        return cls._current_trace_id

    @classmethod
    def get_phase(cls) -> Optional[str]:
        return cls._current_phase

    @classmethod
    def clear(cls):
        cls._current_trace_id = None
        cls._current_phase = None


class ContextFilter(logging.Filter):
    """Injects trace context into log records."""

    def filter(self, record):
        record.trace_id = TraceContext.get_trace_id()
        record.phase = TraceContext.get_phase()
        return True


def setup_logging(log_level: str = "INFO", log_dir: str = ".gaadp/logs"):
    """
    Configure logging with console and file handlers.

    Args:
        log_level: Console log level (DEBUG, INFO, WARNING, ERROR)
        log_dir: Directory for log files
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"gaadp_{timestamp}.jsonl")

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler (human-readable)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(ConsoleFormatter())
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)

    # File handler (structured JSON)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Capture everything to file
    file_handler.setFormatter(JSONFormatter())
    file_handler.addFilter(ContextFilter())
    root_logger.addHandler(file_handler)

    # Log startup
    logger = logging.getLogger("GAADP")
    logger.info(f"Logging initialized. File: {log_file}")

    return log_file


# Initialize logging immediately
LOG_FILE = setup_logging(
    log_level=os.getenv("GAADP_LOG_LEVEL", "INFO")
)
from infrastructure.graph_db import GraphDB
from infrastructure.version_control import GitController
from infrastructure.sandbox import CodeSandbox
from infrastructure.semantic_memory import SemanticMemory
from infrastructure.data_loader import DataLoader
from infrastructure.mcp_hub import MCPHub
from core.domain_discovery import DomainResearcher
from agents.concrete_agents import RealArchitect, RealBuilder, RealVerifier
from core.ontology import AgentRole, NodeType, EdgeType, NodeStatus

# Check for API keys (skip if using manual mode)
if os.getenv("LLM_PROVIDER", "").lower() != "manual":
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: Missing API Keys (ANTHROPIC_API_KEY or OPENAI_API_KEY)")
        print("   TIP: Set LLM_PROVIDER=manual to use human-in-the-loop mode")
        exit(1)
else:
    print("🧑‍💻 MANUAL MODE: You will provide LLM responses interactively")


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
    logger = logging.getLogger("GAADP.Main")

    # Generate trace ID for this run
    trace_id = uuid.uuid4().hex
    TraceContext.set_trace(trace_id)
    TraceContext.set_phase("INIT")

    logger.info("=" * 60)
    logger.info(f"GAADP RUN STARTED - Trace ID: {trace_id}")
    logger.info("=" * 60)

    print("🚀 INITIALIZING GAADP PRODUCTION SWARM (FULL FEATURED)...")

    # Initialize core infrastructure
    db = GraphDB(persistence_path=".gaadp/live_graph.json")
    memory = SemanticMemory()
    mcp_hub = MCPHub()

    logger.info(f"Infrastructure initialized: GraphDB, SemanticMemory, MCPHub ({len(mcp_hub.list_all_tools())} tools)")
    print("🧠 Semantic Memory Online")
    print(f"🔧 MCP Hub Online ({len(mcp_hub.list_all_tools())} tools available)")

    # --- PHASE 0: GET REQUIREMENT ---
    TraceContext.set_phase("REQUIREMENT")
    request = get_requirement(interactive)
    logger.info(f"Requirement received: {request[:200]}...")
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
    TraceContext.set_phase("AGENT_INIT")
    architect = RealArchitect("arch_01", AgentRole.ARCHITECT, db, mcp_hub=mcp_hub)
    builder = RealBuilder("build_01", AgentRole.BUILDER, db, mcp_hub=mcp_hub)
    verifier = RealVerifier("verif_01", AgentRole.VERIFIER, db, mcp_hub=mcp_hub)
    logger.info("Agents initialized: Architect, Builder, Verifier")

    # --- PHASE 2: INJECT REQUIREMENT ---
    TraceContext.set_phase("INJECT_REQ")
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    logger.info(f"Injecting requirement node: {req_id}")
    print(f"\n📝 Injecting Root Requirement: {req_id}")
    db.add_node(req_id, NodeType.REQ, request)

    # Embed requirement into semantic memory
    memory.embed_node(req_id, request)

    # --- PHASE 3: ARCHITECT PLANNING ---
    TraceContext.set_phase("ARCHITECT")
    logger.info("Architect processing started")
    print("🧠 Architect is thinking...")
    arch_output = await architect.process({"nodes": [{"content": request, "id": req_id}]})
    logger.info(f"Architect output: {len(arch_output.get('new_nodes', []))} new nodes")

    # Track created nodes: store both plan_id, spec_id and a mapping for edge creation
    plan_id, spec_id = None, None
    spec_ids = []  # Track all SPECs (there may be multiple)
    node_id_map = {}  # Maps index/temp_id -> actual node_id for edge resolution

    for idx, n in enumerate(arch_output.get('new_nodes', [])):
        node_id = uuid.uuid4().hex
        node_id_map[f"new_{idx}"] = node_id  # For edge resolution if Architect uses temp IDs
        node_id_map[idx] = node_id

        print(f"   + Created Node: {n['type']} ({node_id[:8]}...)")
        db.add_node(node_id, NodeType(n['type']), n['content'])
        memory.embed_node(node_id, n['content'])

        # Track key node types
        if n['type'] == 'PLAN':
            plan_id = node_id
        if n['type'] == 'SPEC':
            spec_id = node_id  # Last SPEC becomes the primary one for Builder
            spec_ids.append(node_id)

        # === IMPLICIT EDGE: All new nodes TRACE_TO the requirement ===
        sig = architect.sign_content(node_id, previous_hash=req_id)
        db.add_edge(node_id, req_id, EdgeType.TRACES_TO, architect.agent_id, sig)
        logger.debug(f"Created edge: {node_id[:8]} --[TRACES_TO]--> {req_id}")

    # === PLAN → SPEC edges: Plan DEFINES the specs ===
    if plan_id and spec_ids:
        for sid in spec_ids:
            sig = architect.sign_content(f"{plan_id}:{sid}")
            db.add_edge(plan_id, sid, EdgeType.DEFINES, architect.agent_id, sig)
            logger.debug(f"Created edge: {plan_id[:8]} --[DEFINES]--> {sid[:8]}")

    # === Process explicit new_edges from Architect ===
    for edge in arch_output.get('new_edges', []):
        try:
            # Resolve source/target IDs (may be temp IDs or actual IDs)
            src = node_id_map.get(edge.get('source_id'), edge.get('source_id'))
            tgt = node_id_map.get(edge.get('target_id'), edge.get('target_id'))

            # Handle special targets like "req" or "requirement"
            if tgt in ('req', 'requirement', 'REQ'):
                tgt = req_id

            edge_type = EdgeType(edge.get('relation', 'DEPENDS_ON'))
            sig = architect.sign_content(f"{src}:{tgt}:{edge_type.value}")
            db.add_edge(src, tgt, edge_type, architect.agent_id, sig)
            logger.info(f"Created explicit edge: {src[:8] if len(src) > 8 else src} --[{edge_type.value}]--> {tgt[:8] if len(tgt) > 8 else tgt}")
        except Exception as e:
            logger.warning(f"Failed to create edge {edge}: {e}")

    logger.info(f"Graph after Architect: {db.graph.number_of_nodes()} nodes, {db.graph.number_of_edges()} edges")

    # --- PHASE 4: BUILDER EXECUTION ---
    TraceContext.set_phase("BUILDER")
    logger.info("Builder processing started")
    print("🔨 Builder is coding...")
    spec_content = arch_output.get('new_nodes', [{}])[0].get('content', request)
    build_output = await builder.process({"nodes": [{"content": spec_content, "id": spec_id}]})
    logger.info(f"Builder output: type={build_output.get('type')}, content_len={len(build_output.get('content', ''))}")

    code_id = uuid.uuid4().hex
    db.add_node(
        code_id,
        NodeType(build_output['type']),
        build_output['content'],
        metadata=build_output.get('metadata', {})
    )
    logger.info(f"Code node created: {code_id}")
    print(f"   > Generated Code ID: {code_id[:8]}...")

    # === EDGE: CODE --[IMPLEMENTS]--> SPEC ===
    if spec_id:
        sig = builder.sign_content(code_id, previous_hash=spec_id)
        db.add_edge(code_id, spec_id, EdgeType.IMPLEMENTS, builder.agent_id, sig)
        logger.info(f"Created edge: {code_id[:8]} --[IMPLEMENTS]--> {spec_id[:8]}")

    # === EDGE: CODE --[TRACES_TO]--> REQ (for full traceability) ===
    sig = builder.sign_content(code_id, previous_hash=req_id)
    db.add_edge(code_id, req_id, EdgeType.TRACES_TO, builder.agent_id, sig)
    logger.debug(f"Created edge: {code_id[:8]} --[TRACES_TO]--> {req_id}")

    logger.info(f"Graph after Builder: {db.graph.number_of_nodes()} nodes, {db.graph.number_of_edges()} edges")

    # Embed code into semantic memory
    memory.embed_node(code_id, build_output['content'])
    print(f"   > Embedded into Vector Space")

    # --- PHASE 5: VERIFICATION ---
    TraceContext.set_phase("VERIFIER")
    logger.info("Verifier processing started")
    print("⚖️ Verifier is judging...")
    verify_output = await verifier.process({
        "nodes": [{"id": code_id, "content": build_output['content']}]
    })
    verdict = verify_output.get('verdict', 'UNKNOWN')
    logger.info(f"Verifier verdict: {verdict}")
    if verify_output.get('critique'):
        logger.info(f"Verifier critique: {verify_output.get('critique')}")
    print(f"   > Verdict: {verdict}")

    if verify_output.get('verdict') == 'PASS':
        TraceContext.set_phase("MATERIALIZE")
        logger.info("Verification PASSED - materializing code")
        print("✅ SUCCESS: Code Verified. Linking Chain...")

        # A. Get Previous Hash for Chain
        prev_hash = db.get_last_node_hash()

        # B. Sign with Chain Context
        sig = verifier.sign_content(code_id, previous_hash=prev_hash)

        # C. Create verification node and link to code
        verification_id = f"verify_{code_id[:8]}"
        db.add_node(
            verification_id,
            NodeType.TEST,  # Verification is a TEST-type node
            json.dumps(verify_output),  # Store the verification output
            metadata={
                "verdict": verdict,
                "verifier_id": verifier.agent_id,
                "code_id": code_id
            }
        )
        db.add_edge(verification_id, code_id, EdgeType.VERIFIES, verifier.agent_id, sig, previous_hash=prev_hash)
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
            logger.info(f"Sandbox execution SUCCESS: {run_result['stdout'][:100]}")
            print(f"🚀 Output: {run_result['stdout'][:200]}")
        else:
            logger.warning(f"Sandbox execution FAILED: {run_result['stderr'][:200]}")
            print(f"⚠️ Execution issue: {run_result['stderr'][:200]}")

        # F. GIT COMMIT
        git = GitController()
        git.commit_work("builder_01", code_id, f"Implemented {file_path}")
        logger.info(f"Git commit created for {file_path}")
        print("📦 Changes committed to Git")

    else:
        TraceContext.set_phase("FAILED")
        logger.warning(f"Verification FAILED: {verify_output.get('critique', 'No details')}")
        print("❌ FAILURE: Code rejected.")
        critique = verify_output.get('critique', 'No details provided')
        print(f"   Critique: {critique}")

    # --- FINAL: INTROSPECTION ---
    TraceContext.set_phase("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"GAADP RUN COMPLETED - Trace ID: {trace_id}")
    logger.info(f"Final graph state: {db.graph.number_of_nodes()} nodes, {db.graph.number_of_edges()} edges")
    logger.info("=" * 60)

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

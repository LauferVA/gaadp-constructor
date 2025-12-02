#!/usr/bin/env python3
"""
GAADP PRODUCTION RUNTIME (Full Featured)
Includes: Domain Discovery, Data Loading, MCP Tools, RBAC, Socratic Research
"""
# Disable tokenizer parallelism warning BEFORE any other imports
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import uuid
import subprocess
import logging
import json
import argparse
from datetime import datetime
from typing import Optional, Literal
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
from requirements.socratic_agent import InteractiveSocraticPhase, DevelopmentSocraticPhase, SocraticConfig
from orchestration.dependency_resolver import DependencyResolver, resolve_architect_output, CyclicDependencyError
from orchestration.escalation_controller import EscalationController, EscalationLevel

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


async def main(
    interactive: bool = True,
    socratic_mode: Literal["interactive", "development", "skip"] = "skip",
    source_path: Optional[str] = None
):
    """
    Main GAADP production runtime.

    Args:
        interactive: Whether to prompt for user input
        socratic_mode: How to handle the Socratic research phase:
            - "interactive": Ask user clarifying questions before development
            - "development": Auto-extract specs from existing source code (for benchmarking)
            - "skip": Skip Socratic phase entirely (current default behavior)
        source_path: For development mode - path to source code to analyze
    """
    logger = logging.getLogger("GAADP.Main")

    # Generate trace ID for this run
    trace_id = uuid.uuid4().hex
    TraceContext.set_trace(trace_id)
    TraceContext.set_phase("INIT")

    logger.info("=" * 60)
    logger.info(f"GAADP RUN STARTED - Trace ID: {trace_id}")
    logger.info(f"Socratic Mode: {socratic_mode}")
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

    # --- PHASE 0.9: SOCRATIC RESEARCH / DEVELOPMENT SPEC PHASE ---
    TraceContext.set_phase("SOCRATIC")
    enriched_request = request  # May be enriched by Socratic phase
    socratic_artifacts = {}  # Store any artifacts from Socratic phase

    if socratic_mode == "interactive":
        # Interactive Socratic phase: ask user clarifying questions
        logger.info("Starting Interactive Socratic Phase")
        print("\n🎓 SOCRATIC RESEARCH PHASE")
        print("   Analyzing requirement for ambiguities...")

        socratic_config = SocraticConfig(
            max_questions=5,
            question_timeout=120,
            min_confidence=0.7
        )
        socratic_phase = InteractiveSocraticPhase(db, source_path or ".")
        socratic_result = await socratic_phase.run(request, config=socratic_config)

        if socratic_result.get("enriched_requirement"):
            enriched_request = socratic_result["enriched_requirement"]
            logger.info(f"Requirement enriched with {len(socratic_result.get('clarifications', []))} clarifications")
            print(f"   ✅ Requirement enriched with {len(socratic_result.get('clarifications', []))} clarifications")

        socratic_artifacts = socratic_result
        logger.info(f"Socratic phase complete: confidence={socratic_result.get('confidence', 'N/A')}")

    elif socratic_mode == "development":
        # Development Spec phase: auto-extract specs from existing code
        logger.info("Starting Development Specification Phase")
        print("\n🔬 DEVELOPMENT SPECIFICATION PHASE")
        print(f"   Analyzing source code at: {source_path or '.'}")

        socratic_phase = DevelopmentSocraticPhase(db, source_path or ".")
        socratic_result = await socratic_phase.run(request)

        if socratic_result.get("enriched_requirement"):
            enriched_request = socratic_result["enriched_requirement"]
            specs_found = len(socratic_result.get("extracted_specs", []))
            logger.info(f"Extracted {specs_found} specifications from existing code")
            print(f"   ✅ Extracted {specs_found} specifications from existing code")

            # Log what was discovered
            for spec in socratic_result.get("extracted_specs", [])[:5]:
                print(f"      - {spec.get('file_path', 'unknown')}: {len(spec.get('components', []))} components")

        socratic_artifacts = socratic_result
        logger.info(f"Development spec phase complete: {len(socratic_result.get('extracted_specs', []))} files analyzed")

    else:
        # Skip Socratic phase entirely
        logger.info("Socratic phase skipped (mode=skip)")

    # Use enriched request for downstream phases
    request = enriched_request

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

    # --- PHASE 3.5: RESOLVE BUILD ORDER FROM DEPENDS_ON EDGES ---
    TraceContext.set_phase("DEPENDENCY_RESOLUTION")

    # Build a resolver with all SPEC nodes
    spec_data_map = {}  # Maps spec_id -> node data
    for idx, n in enumerate(arch_output.get('new_nodes', [])):
        if n.get('type') == 'SPEC':
            actual_id = node_id_map.get(f"new_{idx}", node_id_map.get(idx))
            spec_data_map[actual_id] = n

    # Resolve build order using DEPENDS_ON edges
    if spec_ids:
        try:
            build_order, build_waves = resolve_architect_output(
                arch_output.get('new_nodes', []),
                arch_output.get('new_edges', []),
                node_id_map
            )
            logger.info(f"Resolved build order: {len(build_order)} specs in {len(build_waves)} waves")
            print(f"\n📊 Build Order Resolved: {len(spec_ids)} specs")
            for wave_idx, wave in enumerate(build_waves):
                wave_display = [f"{sid[:8]}..." for sid in wave]
                print(f"   Wave {wave_idx}: {wave_display}")
        except CyclicDependencyError as e:
            logger.error(f"Cyclic dependency detected: {e.cycle}")
            print(f"⚠️ WARNING: Cyclic dependency detected - using original order")
            build_order = spec_ids
            build_waves = [spec_ids]
    else:
        build_order = spec_ids
        build_waves = [spec_ids] if spec_ids else []

    # --- PHASE 4 & 5: BUILD → VERIFY LOOP (with retry/feedback) ---
    # Now iterate through specs in dependency order
    MAX_BUILD_RETRIES = 3
    ESCALATE_AFTER = 2  # After this many failures, go back to Architect
    MAX_ARCHITECT_REPLANS = 2  # Max Architect re-plans per spec before giving up

    all_verified = True
    built_artifacts = {}  # Maps spec_id -> code_id for dependency tracking

    # Initialize escalation controller for failure tracking
    escalation_controller = EscalationController(
        max_retries=MAX_BUILD_RETRIES,
        escalate_after=ESCALATE_AFTER,
        max_architect_attempts=MAX_ARCHITECT_REPLANS
    )

    for wave_idx, wave in enumerate(build_waves):
        TraceContext.set_phase(f"BUILD_WAVE_{wave_idx}")
        logger.info(f"Processing build wave {wave_idx + 1}/{len(build_waves)}: {len(wave)} specs")
        print(f"\n🔨 BUILD WAVE {wave_idx + 1}/{len(build_waves)}")

        for current_spec_id in wave:
            # Get spec content
            spec_node = spec_data_map.get(current_spec_id)
            if not spec_node:
                # Fall back to graph lookup
                spec_content = db.graph.nodes.get(current_spec_id, {}).get('content', request)
            else:
                spec_content = spec_node.get('content', request)

            # Register spec with escalation controller
            escalation_ctx = escalation_controller.get_or_create_context(
                spec_id=current_spec_id,
                spec_content=spec_content,
                requirement_id=req_id
            )

            # Add context about already-built dependencies
            dependency_context = ""
            deps = db.graph.predecessors(current_spec_id) if current_spec_id in db.graph else []
            built_deps = [(d, built_artifacts.get(d)) for d in deps if d in built_artifacts]
            if built_deps:
                dep_info = "\n".join([f"- {d[:8]}... -> code: {c[:8]}..." for d, c in built_deps])
                dependency_context = f"\n\n[AVAILABLE DEPENDENCIES (already built)]:\n{dep_info}"

            feedback_context = ""  # Accumulates critique for retries
            verified = False
            final_code_id = None
            final_build_output = None
            current_spec_content = spec_content  # May be updated by Architect re-plan

            print(f"\n   📋 Building SPEC: {current_spec_id[:8]}...")

            for attempt in range(MAX_BUILD_RETRIES):
                # --- PHASE 4: BUILDER EXECUTION ---
                TraceContext.set_phase("BUILDER")
                attempt_label = f"(attempt {attempt + 1}/{MAX_BUILD_RETRIES})" if attempt > 0 else ""
                logger.info(f"Builder processing spec {current_spec_id[:8]} {attempt_label}")
                print(f"      🔨 Building... {attempt_label}")

                # Include feedback from previous failure if any
                builder_context = current_spec_content + dependency_context
                if feedback_context:
                    builder_context = f"""{current_spec_content}{dependency_context}

[FEEDBACK FROM PREVIOUS ATTEMPT]:
{feedback_context}

Please fix the issues mentioned above and try again."""

                build_output = await builder.process({"nodes": [{"content": builder_context, "id": current_spec_id}]})
                logger.info(f"Builder output: type={build_output.get('type')}, content_len={len(build_output.get('content', ''))}")

                code_id = uuid.uuid4().hex
                db.add_node(
                    code_id,
                    NodeType(build_output['type']),
                    build_output['content'],
                    metadata={**build_output.get('metadata', {}), "attempt": attempt + 1, "spec_id": current_spec_id}
                )
                logger.info(f"Code node created: {code_id}")
                print(f"      > Generated Code ID: {code_id[:8]}...")

                # === EDGE: CODE --[IMPLEMENTS]--> SPEC ===
                sig = builder.sign_content(code_id, previous_hash=current_spec_id)
                db.add_edge(code_id, current_spec_id, EdgeType.IMPLEMENTS, builder.agent_id, sig)
                logger.info(f"Created edge: {code_id[:8]} --[IMPLEMENTS]--> {current_spec_id[:8]}")

                # === EDGE: CODE --[TRACES_TO]--> REQ (for full traceability) ===
                sig = builder.sign_content(code_id, previous_hash=req_id)
                db.add_edge(code_id, req_id, EdgeType.TRACES_TO, builder.agent_id, sig)
                logger.debug(f"Created edge: {code_id[:8]} --[TRACES_TO]--> {req_id}")

                logger.info(f"Graph after Builder: {db.graph.number_of_nodes()} nodes, {db.graph.number_of_edges()} edges")

                # Embed code into semantic memory
                memory.embed_node(code_id, build_output['content'])

                # --- PHASE 5: VERIFICATION ---
                TraceContext.set_phase("VERIFIER")
                logger.info("Verifier processing started")
                print(f"      ⚖️ Verifying...")
                verify_output = await verifier.process({
                    "nodes": [{"id": code_id, "content": build_output['content']}]
                })
                verdict = verify_output.get('verdict', 'UNKNOWN')
                logger.info(f"Verifier verdict: {verdict}")
                if verify_output.get('critique'):
                    logger.info(f"Verifier critique: {verify_output.get('critique')}")
                print(f"      > Verdict: {verdict}")

                if verdict == 'PASS':
                    verified = True
                    final_code_id = code_id
                    final_build_output = build_output
                    break  # Success! Exit the retry loop
                else:
                    # FAILED: Mark code node, store critique, prepare for retry
                    critique = verify_output.get('critique', 'No details provided')
                    issues = verify_output.get('issues', [])
                    issues_text = "\n".join([f"- [{i.get('severity', 'error')}] {i.get('description', '')}" for i in issues]) if issues else str(critique)

                    db.graph.nodes[code_id]['status'] = NodeStatus.FAILED.value
                    db.graph.nodes[code_id]['critique'] = critique

                    # Create FEEDBACK edge from failed CODE to SPEC
                    feedback_sig = f"feedback_{code_id[:8]}"
                    db.add_edge(code_id, current_spec_id, EdgeType.FEEDBACK, "feedback_controller", feedback_sig)
                    logger.info(f"Created FEEDBACK edge: {code_id[:8]} --[FEEDBACK]--> {current_spec_id[:8]}")

                    # Record failure in escalation controller
                    escalation_level = escalation_controller.record_failure(
                        spec_id=current_spec_id,
                        code_id=code_id,
                        verify_output=verify_output
                    )

                    logger.warning(f"Verification FAILED (attempt {attempt + 1}): {str(critique)[:100]}...")
                    print(f"      ❌ FAILED (attempt {attempt + 1}/{MAX_BUILD_RETRIES})")

                    # Accumulate feedback for next attempt
                    feedback_context = f"""Attempt {attempt + 1} failed with these issues:
{issues_text}

Verifier reasoning: {verify_output.get('reasoning', 'N/A')}"""

                    # Check escalation level and act accordingly
                    if escalation_level == EscalationLevel.ARCHITECT:
                        logger.warning(f"Escalating to Architect for spec {current_spec_id[:8]}")
                        print(f"      ⚠️ Escalating to Architect for re-planning...")

                        # Execute Architect escalation
                        TraceContext.set_phase("ARCHITECT_ESCALATION")
                        escalation_result = await escalation_controller.execute_escalation(
                            spec_id=current_spec_id,
                            level=escalation_level,
                            architect_agent=architect,
                            db=db
                        )

                        if escalation_result.get("action") == "REPLAN":
                            # Architect produced new nodes - extract new spec content
                            arch_output_new = escalation_result.get("architect_output", {})
                            new_nodes = arch_output_new.get("new_nodes", [])
                            if new_nodes:
                                # Use the first SPEC node from Architect's new output
                                for node in new_nodes:
                                    if node.get("type") == "SPEC":
                                        current_spec_content = node.get("content", current_spec_content)
                                        logger.info(f"Architect provided revised spec for {current_spec_id[:8]}")
                                        print(f"      🔄 Architect revised the specification")
                                        break
                        elif escalation_result.get("action") == "HUMAN_INTERVENTION_REQUIRED":
                            logger.error(f"Human intervention required for spec {current_spec_id[:8]}")
                            print(f"      🚨 HUMAN INTERVENTION REQUIRED")
                            print(f"         {escalation_result.get('message', 'System cannot solve this')}")
                            break  # Exit retry loop for this spec

                    elif escalation_level == EscalationLevel.HUMAN:
                        logger.error(f"Human intervention required for spec {current_spec_id[:8]}")
                        print(f"      🚨 HUMAN INTERVENTION REQUIRED")
                        escalation_result = await escalation_controller.execute_escalation(
                            spec_id=current_spec_id,
                            level=escalation_level
                        )
                        print(f"         Patterns: {escalation_result.get('patterns', [])}")
                        break  # Exit retry loop

            # --- POST-SPEC: Handle result for this spec ---
            if verified and final_code_id and final_build_output:
                # Track successful build for dependency context
                built_artifacts[current_spec_id] = final_code_id

                TraceContext.set_phase("MATERIALIZE")
                logger.info(f"Verification PASSED for spec {current_spec_id[:8]} - materializing code")
                print(f"      ✅ SUCCESS: Code Verified")

                # A. Get Previous Hash for Chain
                prev_hash = db.get_last_node_hash()

                # B. Sign with Chain Context
                sig = verifier.sign_content(final_code_id, previous_hash=prev_hash)

                # C. Create verification node and link to code
                verification_id = f"verify_{final_code_id[:8]}"
                db.add_node(
                    verification_id,
                    NodeType.TEST,  # Verification is a TEST-type node
                    json.dumps(verify_output),  # Store the verification output
                    metadata={
                        "verdict": verdict,
                        "verifier_id": verifier.agent_id,
                        "code_id": final_code_id,
                        "spec_id": current_spec_id,
                        "attempts": attempt + 1
                    }
                )
                db.add_edge(verification_id, final_code_id, EdgeType.VERIFIES, verifier.agent_id, sig, previous_hash=prev_hash)
                db.graph.nodes[final_code_id]['status'] = NodeStatus.VERIFIED.value

                # D. MATERIALIZE (Write to Disk)
                file_path = final_build_output.get('metadata', {}).get('file_path', f'generated_{current_spec_id[:8]}.py')
                os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
                with open(file_path, "w") as f:
                    f.write(final_build_output['content'])
                print(f"      💾 Saved: {file_path}")

            else:
                all_verified = False
                logger.error(f"All {MAX_BUILD_RETRIES} attempts failed for spec {current_spec_id[:8]}")
                print(f"      ❌ FINAL FAILURE for spec {current_spec_id[:8]}")

    # --- POST-BUILD: Final summary ---
    TraceContext.set_phase("POST_BUILD")
    if all_verified and built_artifacts:
        print(f"\n✅ ALL SPECS VERIFIED: {len(built_artifacts)} artifacts built")

        # SANDBOX EXECUTION (run the last/main artifact)
        if final_build_output:
            print("⚙️ Executing Code in Sandbox...")
            sandbox = CodeSandbox(use_docker=False)
            file_path = final_build_output.get('metadata', {}).get('file_path', 'generated_output.py')
            run_result = sandbox.run_code(file_path)

            if run_result['exit_code'] == 0:
                logger.info(f"Sandbox execution SUCCESS: {run_result['stdout'][:100]}")
                print(f"🚀 Output: {run_result['stdout'][:200]}")
            else:
                logger.warning(f"Sandbox execution FAILED: {run_result['stderr'][:200]}")
                print(f"⚠️ Execution issue: {run_result['stderr'][:200]}")

        # GIT COMMIT (commit all built artifacts)
        git = GitController()
        artifact_summary = ", ".join([f"{sid[:8]}" for sid in built_artifacts.keys()])
        git.commit_work("builder_01", list(built_artifacts.values())[0], f"Implemented specs: {artifact_summary}")
        logger.info(f"Git commit created for {len(built_artifacts)} artifacts")
        print("📦 Changes committed to Git")

    elif built_artifacts:
        failed_count = len(spec_ids) - len(built_artifacts)
        print(f"\n⚠️ PARTIAL SUCCESS: {len(built_artifacts)}/{len(spec_ids)} specs verified, {failed_count} failed")
    else:
        print(f"\n❌ COMPLETE FAILURE: No specs were successfully verified")

    # --- ESCALATION STATS ---
    escalation_stats = escalation_controller.get_stats()
    if escalation_stats['total_failures'] > 0:
        print(f"\n📊 ESCALATION SUMMARY:")
        print(f"   Total failures: {escalation_stats['total_failures']}")
        print(f"   Architect escalations: {escalation_stats['architect_escalations']}")
        print(f"   Human escalations: {escalation_stats['human_escalations']}")
        if escalation_stats['success_after_escalation'] > 0:
            print(f"   ✅ Success after Architect re-plan: {escalation_stats['success_after_escalation']}")
        logger.info(f"Escalation stats: {escalation_stats}")

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


async def run_batch(
    requirement: str,
    socratic_mode: Literal["interactive", "development", "skip"] = "skip",
    source_path: Optional[str] = None
):
    """Run in non-interactive batch mode."""
    # Write requirement to prompt.md for batch processing
    with open("prompt.md", "w") as f:
        f.write(requirement)
    await main(interactive=False, socratic_mode=socratic_mode, source_path=source_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GAADP Production Runtime - Graph-Augmented Autonomous Development Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (default)
  python production_main.py

  # Batch mode with requirement
  python production_main.py --batch "Create a hello world function"

  # With interactive Socratic phase (asks clarifying questions)
  python production_main.py --socratic interactive

  # Development mode - extract specs from existing code (for benchmarking)
  python production_main.py --socratic development --source ./my_project

  # Batch + development mode for reconstruction benchmarks
  python production_main.py --batch "Reconstruct the module" --socratic development --source ./target_code
        """
    )

    parser.add_argument(
        "--batch",
        metavar="REQUIREMENT",
        help="Run in non-interactive batch mode with the given requirement"
    )
    parser.add_argument(
        "--socratic",
        choices=["interactive", "development", "skip"],
        default="skip",
        help="Socratic phase mode: 'interactive' asks user questions, 'development' extracts specs from code, 'skip' (default) bypasses"
    )
    parser.add_argument(
        "--source",
        metavar="PATH",
        help="Source path for development mode (where to find existing code to analyze)"
    )

    args = parser.parse_args()

    if args.batch:
        # Batch mode
        asyncio.run(run_batch(
            requirement=args.batch,
            socratic_mode=args.socratic,
            source_path=args.source
        ))
    else:
        # Interactive mode
        asyncio.run(main(
            interactive=True,
            socratic_mode=args.socratic,
            source_path=args.source
        ))

"""
GENERIC AGENT - Universal Agent Implementation

This module provides a single GenericAgent class that replaces all
specific agent classes (RealArchitect, RealBuilder, etc.).

The agent loads its "personality" (prompt, tools, output protocol)
from config/agent_manifest.yaml. It does not know WHAT it is building -
it simply follows its configuration.

Key Principles:
1. Configuration-driven: All behavior comes from YAML, not Python
2. Protocol-enforced: Output is validated against Pydantic models
3. ReAct loop: Reason -> Act -> Observe cycle for tool use
4. Graph-aware: Receives context from graph, returns graph mutations

Adding a new agent type = adding a new entry in agent_manifest.yaml.
No Python code changes required.
"""
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any
from pathlib import Path

from core.protocols import (
    AgentConfig,
    AgentManifest,
    UnifiedAgentOutput,
    GraphContext,
    NodeSpec,
    EdgeSpec,
    load_agent_manifest,
    get_protocol_class,
    protocol_to_tool_schema,
)
from core.ontology import NodeType, EdgeType, NodeStatus


logger = logging.getLogger("GAADP.GenericAgent")

# Maximum ReAct iterations to prevent infinite loops
MAX_REACT_ITERATIONS = 5


class GenericAgent:
    """
    Universal agent that loads its behavior from configuration.

    This single class replaces RealArchitect, RealBuilder, RealVerifier, etc.
    The agent's "personality" is defined in config/agent_manifest.yaml.
    """

    def __init__(
        self,
        role: str,
        agent_id: Optional[str] = None,
        manifest_path: str = "config/agent_manifest.yaml",
        llm_gateway=None
    ):
        """
        Initialize a GenericAgent for a specific role.

        Args:
            role: Agent role (ARCHITECT, BUILDER, VERIFIER, SOCRATES)
            agent_id: Unique identifier for this agent instance
            manifest_path: Path to agent_manifest.yaml
            llm_gateway: LLM gateway instance (injected)
        """
        self.role = role.upper()
        self.agent_id = agent_id or f"{role.lower()}_{hashlib.md5(role.encode()).hexdigest()[:8]}"

        # Load configuration from manifest
        self.manifest = load_agent_manifest(manifest_path)
        self.config = self.manifest.get_config(role)

        if not self.config:
            raise ValueError(f"No configuration found for role: {role}")

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

        # LLM gateway (injected or lazy-loaded)
        self._gateway = llm_gateway

        # Output protocol class
        self.output_protocol = get_protocol_class(self.config.output_protocol)

        logger.info(
            f"GenericAgent initialized: role={self.role}, "
            f"inputs={self.config.input_node_types}, "
            f"outputs={self.config.output_node_types}"
        )

    @property
    def gateway(self):
        """Lazy-load LLM gateway if not injected."""
        if self._gateway is None:
            from infrastructure.llm_gateway import LLMGateway
            self._gateway = LLMGateway()
        return self._gateway

    def _load_system_prompt(self) -> str:
        """Load system prompt from file or fallback."""
        prompt_path = Path(self.config.system_prompt_path)

        if prompt_path.exists():
            return prompt_path.read_text()
        elif self.config.system_prompt_fallback:
            logger.warning(f"Prompt file not found, using fallback: {prompt_path}")
            return self.config.system_prompt_fallback
        else:
            raise FileNotFoundError(
                f"System prompt not found and no fallback: {prompt_path}"
            )

    def _build_tools_schema(self) -> List[Dict]:
        """Build tool schemas for this agent."""
        tools = []

        # Add allowed tools from MCP or built-in
        for tool_name in self.config.allowed_tools:
            tool_schema = self._get_tool_schema(tool_name)
            if tool_schema:
                tools.append(tool_schema)

        # Add the output protocol tool (forced via tool_choice)
        output_tool = protocol_to_tool_schema(
            self.output_protocol,
            self.config.output_tool_name
        )
        tools.append(output_tool)

        return tools

    def _get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Get schema for a built-in or MCP tool."""
        # Built-in tool schemas
        builtin_tools = {
            "read_file": {
                "name": "read_file",
                "description": "Read the contents of a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"}
                    },
                    "required": ["path"]
                }
            },
            "write_file": {
                "name": "write_file",
                "description": "Write content to a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to file"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                }
            },
            "list_directory": {
                "name": "list_directory",
                "description": "List contents of a directory",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            },
            "search_web": {
                "name": "search_web",
                "description": "Search the web for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            },
            "fetch_url": {
                "name": "fetch_url",
                "description": "Fetch content from a URL",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"]
                }
            },
        }

        return builtin_tools.get(tool_name)

    def _build_user_prompt(self, context: GraphContext) -> str:
        """Build user prompt from graph context."""
        prompt_parts = [
            f"## Node to Process",
            f"- ID: {context.node_id}",
            f"- Type: {context.node_type}",
            f"- Status: {context.node_status}",
            f"- Content:",
            f"```",
            context.node_content,
            f"```",
            ""
        ]

        # Add requirement context
        if context.requirement_content:
            prompt_parts.extend([
                f"## Root Requirement",
                f"```",
                context.requirement_content,
                f"```",
                ""
            ])

        # Add dependency context
        if context.dependency_nodes:
            prompt_parts.append("## Dependencies (already completed)")
            for dep in context.dependency_nodes:
                prompt_parts.append(f"- {dep.get('id', 'unknown')[:8]}: {dep.get('type', 'unknown')}")
            prompt_parts.append("")

        # Add feedback from previous attempts
        if context.feedback:
            prompt_parts.extend([
                f"## Feedback from Previous Attempts ({context.previous_attempts} attempts)",
                ""
            ])
            for fb in context.feedback:
                prompt_parts.append(f"- {fb.get('content', 'No details')[:200]}")
            prompt_parts.append("")

        # Add instructions
        prompt_parts.extend([
            f"## Instructions",
            f"Process this node and use the `{self.config.output_tool_name}` tool to submit your output.",
            ""
        ])

        return "\n".join(prompt_parts)

    async def process(self, context: GraphContext) -> UnifiedAgentOutput:
        """
        Process a node and return graph mutations.

        This implements the ReAct loop:
        1. Reason about the task
        2. Optionally call tools to gather information
        3. Produce final output via protocol tool

        Args:
            context: Graph context with node info and neighborhood

        Returns:
            UnifiedAgentOutput with new nodes, edges, status updates
        """
        logger.info(f"{self.role} processing node {context.node_id[:8]}...")

        tools = self._build_tools_schema()
        user_prompt = self._build_user_prompt(context)

        # Track cost
        total_cost = 0.0
        total_tokens = {"input": 0, "output": 0}

        # ReAct loop
        for iteration in range(MAX_REACT_ITERATIONS):
            logger.debug(f"{self.role} ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            # Force protocol output on final iteration
            force_output = iteration == MAX_REACT_ITERATIONS - 1

            # Call LLM
            response = self.gateway.call_model(
                role=self.role,
                system_prompt=self.system_prompt,
                user_context=user_prompt,
                tools=tools,
                force_tool=self.config.output_tool_name if force_output else None
            )

            # Track tokens/cost (if available)
            if hasattr(response, 'usage'):
                total_tokens["input"] += getattr(response.usage, 'input_tokens', 0)
                total_tokens["output"] += getattr(response.usage, 'output_tokens', 0)

            # Parse response
            try:
                parsed = json.loads(response) if isinstance(response, str) else response
            except json.JSONDecodeError:
                logger.warning(f"Response not JSON, forcing output")
                parsed = {"content": response}

            # Check for tool calls
            tool_calls = parsed.get("tool_calls", [])

            for tc in tool_calls:
                tool_name = tc.get("name") or tc.get("function", {}).get("name", "")

                # Check if this is the output tool
                if tool_name == self.config.output_tool_name:
                    # Extract and validate output
                    args = tc.get("input") or tc.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args)

                    # Validate against protocol
                    validated = self.output_protocol.model_validate(args)

                    # Convert to UnifiedAgentOutput
                    return self._to_unified_output(
                        validated,
                        context,
                        total_cost,
                        total_tokens
                    )

                # Execute other tools
                tool_result = await self._execute_tool(tool_name, tc.get("input", {}))
                user_prompt += f"\n\n[Tool Result: {tool_name}]\n{tool_result}"

            # If no tool calls and not output, continue loop
            if not tool_calls:
                if force_output:
                    logger.error(f"{self.role} failed to produce output")
                    break
                user_prompt += f"\n\nPlease use the {self.config.output_tool_name} tool to submit your output."

        # Fallback: return empty output on failure
        logger.error(f"{self.role} exhausted ReAct iterations without output")
        return UnifiedAgentOutput(
            thought="Agent failed to produce valid output",
            agent_role=self.role,
            cost_incurred=total_cost,
            tokens_used={"input_tokens": total_tokens["input"], "output_tokens": total_tokens["output"]}
        )

    async def _execute_tool(self, tool_name: str, args: Dict) -> str:
        """Execute a tool and return result."""
        try:
            if tool_name == "read_file":
                path = args.get("path", "")
                if Path(path).exists():
                    return Path(path).read_text()[:10000]  # Limit size
                return f"File not found: {path}"

            elif tool_name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                Path(path).write_text(content)
                return f"Wrote {len(content)} bytes to {path}"

            elif tool_name == "list_directory":
                path = args.get("path", ".")
                if Path(path).is_dir():
                    entries = list(Path(path).iterdir())[:50]  # Limit
                    return "\n".join(str(e) for e in entries)
                return f"Not a directory: {path}"

            elif tool_name == "search_web":
                # Placeholder - would integrate with actual search
                return f"Web search not implemented: {args.get('query', '')}"

            elif tool_name == "fetch_url":
                # Placeholder - would integrate with actual fetch
                return f"URL fetch not implemented: {args.get('url', '')}"

            else:
                return f"Unknown tool: {tool_name}"

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return f"Tool error: {e}"

    def _to_unified_output(
        self,
        protocol_output: Any,
        context: GraphContext,
        cost: float,
        tokens: Dict[str, int]
    ) -> UnifiedAgentOutput:
        """Convert protocol-specific output to UnifiedAgentOutput."""
        output = UnifiedAgentOutput(
            agent_role=self.role,
            cost_incurred=cost,
            tokens_used={"input_tokens": tokens["input"], "output_tokens": tokens["output"]},
            protocol_output=protocol_output.model_dump() if hasattr(protocol_output, 'model_dump') else protocol_output
        )

        # Extract reasoning
        if hasattr(protocol_output, 'reasoning'):
            output.thought = protocol_output.reasoning

        # Extract new_nodes (Architect, Builder)
        if hasattr(protocol_output, 'new_nodes'):
            output.new_nodes = protocol_output.new_nodes

        # Extract new_edges (Architect)
        if hasattr(protocol_output, 'new_edges') and protocol_output.new_edges:
            output.new_edges = protocol_output.new_edges

        # Extract code artifact (Builder)
        if hasattr(protocol_output, 'code'):
            code = protocol_output.code
            output.artifacts[code.file_path] = code.content
            # Create CODE node
            output.new_nodes.append(NodeSpec(
                type="CODE",
                content=code.content,
                metadata={
                    "file_path": code.file_path,
                    "language": code.language,
                    "dependencies": code.dependencies
                }
            ))

        # Extract verdict (Verifier)
        if hasattr(protocol_output, 'verdict'):
            verdict = protocol_output.verdict
            # Update the CODE node status based on verdict
            if verdict == "PASS":
                output.status_updates[context.node_id] = NodeStatus.VERIFIED.value
            else:
                output.status_updates[context.node_id] = NodeStatus.FAILED.value

            # Create TEST node
            output.new_nodes.append(NodeSpec(
                type="TEST",
                content=json.dumps(protocol_output.model_dump()),
                metadata={"verdict": verdict}
            ))

        return output

    def sign_content(self, content: str, previous_hash: Optional[str] = None) -> str:
        """Generate signature for content integrity."""
        data = f"{self.agent_id}:{content}:{previous_hash or ''}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_agent(role: str, **kwargs) -> GenericAgent:
    """
    Factory function to create a GenericAgent.

    Args:
        role: Agent role (ARCHITECT, BUILDER, VERIFIER, SOCRATES)
        **kwargs: Additional arguments passed to GenericAgent

    Returns:
        Configured GenericAgent instance
    """
    return GenericAgent(role=role, **kwargs)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.DEBUG)

    async def test():
        print("=== Testing GenericAgent ===")

        # Test instantiation for each role
        for role in ["ARCHITECT", "BUILDER", "VERIFIER", "SOCRATES"]:
            try:
                agent = GenericAgent(role=role)
                print(f"✅ {role}: config loaded, prompt={len(agent.system_prompt)} chars")
            except Exception as e:
                print(f"❌ {role}: {e}")

        # Test context building
        ctx = GraphContext(
            node_id="test_node_123",
            node_type="REQ",
            node_content="Create a hello world function",
            node_status="PENDING",
            requirement_content="Create a hello world function",
            previous_attempts=0
        )

        agent = GenericAgent(role="ARCHITECT")
        prompt = agent._build_user_prompt(ctx)
        print(f"\n=== User Prompt Preview ===")
        print(prompt[:500] + "...")

        print("\n✅ GenericAgent tests passed")

    asyncio.run(test())

"""
CONCRETE AGENTS
Production implementations with optional MCP tool support.
"""
import json
from typing import Dict, Optional
from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus


class RealArchitect(BaseAgent):
    """
    Decomposes requirements into atomic specs and plans.
    Has read-only filesystem access and web search capability.
    """

    async def process(self, context: Dict) -> Dict:
        req_node = context['nodes'][0]

        # Check for escalation context (from feedback controller)
        escalation_context = req_node.get('escalation_context')
        if escalation_context:
            # This is a re-planning after failures - use escalation prompt
            system_prompt = self._hydrate_prompt("architect_core_v1", {})
            user_prompt = self._build_escalation_prompt(req_node, escalation_context)
            self.logger.warning(f"Processing escalated REQ {req_node['id']} with strategy change")
        else:
            # Normal first-time planning
            system_prompt = self._hydrate_prompt("architect_core_v1", {})
            user_prompt = f"REQUIREMENT: {req_node['content']}\nDecompose into Atomic Specs and Plans."

        # Get filtered tools for this role
        tools_schema = self.get_tools_schema()

        raw_response = self.gateway.call_model(
            role="ARCHITECT",
            system_prompt=system_prompt,
            user_context=user_prompt,
            tools=tools_schema if tools_schema else None
        )

        # Check if response contains tool calls
        try:
            parsed = json.loads(raw_response)
            if 'tool_calls' in parsed and parsed['tool_calls']:
                # Execute tools and get results
                tool_results = await self.execute_tool_calls(parsed)
                # For now, log tool results - could feed back to LLM
                self.logger.info(f"Tool Results: {tool_results}")
                # If there's also content, parse that
                if parsed.get('content'):
                    return self._parse_json_response(parsed['content'])
        except json.JSONDecodeError:
            pass

        return self._parse_json_response(raw_response)

    def _build_escalation_prompt(self, req_node: Dict, escalation_context: str) -> str:
        """
        Build prompt for re-planning after failures.
        Includes failure analysis and strategy change instructions.
        """
        # Analyze failure patterns from escalation context
        failure_hints = []

        if "Missing import" in escalation_context or "import" in escalation_context.lower():
            failure_hints.append("- Add explicit dependency specifications for all external libraries")
            failure_hints.append("- Include setup/installation instructions in the spec")

        if "Type mismatch" in escalation_context or "type" in escalation_context.lower():
            failure_hints.append("- Add explicit type annotations to the specification")
            failure_hints.append("- Specify input/output types clearly")

        if "incomplete" in escalation_context.lower():
            failure_hints.append("- Break this requirement into smaller, more atomic specifications")
            failure_hints.append("- Create sub-tasks for each major component")

        if "complex" in escalation_context.lower():
            failure_hints.append("- Simplify the approach - prefer simpler implementation strategies")
            failure_hints.append("- Reduce dependencies and coupling")

        hints_text = "\n".join(failure_hints) if failure_hints else "- Consider a fundamentally different approach"

        return f"""
{escalation_context}

STRATEGY CHANGE REQUIRED:
Previous specifications led to implementation failures. You must re-plan with a DIFFERENT approach.

Suggested adjustments based on failure analysis:
{hints_text}

REQUIREMENT: {req_node['content']}

Create NEW specifications that address the root causes of the previous failures.
DO NOT simply reproduce the failed specification - change the approach.
"""


class RealBuilder(BaseAgent):
    """
    Implements code nodes from specs.
    Has read/write filesystem access.
    """

    async def process(self, context: Dict) -> Dict:
        spec_content = context.get('nodes', [{}])[0].get('content', "No Spec Found")
        system_prompt = self._hydrate_prompt("builder_core_v1", {
            "target_node_id": "CURRENT_TASK",
            "target_node_spec": spec_content,
            "language": "python",
            "file_path": "generated_module.py"
        })

        # Get filtered tools for this role
        tools_schema = self.get_tools_schema()

        raw_response = self.gateway.call_model(
            role="BUILDER",
            system_prompt=system_prompt,
            user_context="Implement the spec now.",
            tools=tools_schema if tools_schema else None
        )

        # Handle potential tool calls
        try:
            parsed = json.loads(raw_response)
            if 'tool_calls' in parsed and parsed['tool_calls']:
                tool_results = await self.execute_tool_calls(parsed)
                self.logger.info(f"Tool Results: {tool_results}")
                if parsed.get('content'):
                    result = self._parse_json_response(parsed['content'])
                    result['type'] = NodeType.CODE.value
                    result['status'] = NodeStatus.PENDING.value
                    return result
        except json.JSONDecodeError:
            pass

        result = self._parse_json_response(raw_response)
        result['type'] = NodeType.CODE.value
        result['status'] = NodeStatus.PENDING.value
        return result


class RealVerifier(BaseAgent):
    """
    Reviews and verifies code nodes.
    Has read-only filesystem access.
    """

    async def process(self, context: Dict) -> Dict:
        code_node = context['nodes'][0]
        system_prompt = self._hydrate_prompt("verifier_core_v1", {
            "builder_id": "Unknown_Builder",
            "spec_id": "Linked_Spec"
        })
        user_prompt = f"CODE TO VERIFY:\n{code_node['content']}"

        # Get filtered tools for this role
        tools_schema = self.get_tools_schema()

        raw_response = self.gateway.call_model(
            role="VERIFIER",
            system_prompt=system_prompt,
            user_context=user_prompt,
            tools=tools_schema if tools_schema else None
        )

        # Handle potential tool calls
        try:
            parsed = json.loads(raw_response)
            if 'tool_calls' in parsed and parsed['tool_calls']:
                tool_results = await self.execute_tool_calls(parsed)
                self.logger.info(f"Tool Results: {tool_results}")
                if parsed.get('content'):
                    return self._parse_json_response(parsed['content'])
        except json.JSONDecodeError:
            pass

        return self._parse_json_response(raw_response)


class RealSocrates(BaseAgent):
    """
    Researches and resolves ambiguity in requirements.
    Has filesystem and web search access.
    """

    async def process(self, context: Dict) -> Dict:
        question = context.get('question', "What needs clarification?")
        parent_req = context.get('parent_requirement', "")

        system_prompt = """You are Socrates, the philosophical questioner.
Your role is to identify ambiguity and gaps in requirements.
Ask probing questions to clarify intent.
Search for relevant standards and best practices."""

        user_prompt = f"""REQUIREMENT: {parent_req}
QUESTION: {question}

Identify what is unclear and propose specific questions to resolve ambiguity.
If you need to search for standards, use the available tools."""

        tools_schema = self.get_tools_schema()

        raw_response = self.gateway.call_model(
            role="SOCRATES",
            system_prompt=system_prompt,
            user_context=user_prompt,
            tools=tools_schema if tools_schema else None
        )

        try:
            parsed = json.loads(raw_response)
            if 'tool_calls' in parsed and parsed['tool_calls']:
                tool_results = await self.execute_tool_calls(parsed)
                return {
                    "questions": parsed.get('content', ''),
                    "research": tool_results
                }
        except json.JSONDecodeError:
            pass

        return {
            "questions": raw_response,
            "research": None
        }

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

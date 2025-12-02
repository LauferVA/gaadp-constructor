"""
CONCRETE AGENTS
Production implementations with optional MCP tool support.
Implements ReAct (Reason + Act) loop for tool-augmented reasoning.
"""
import json
from typing import Dict, Optional
from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus

# Maximum iterations for ReAct loop to prevent infinite loops
MAX_REACT_ITERATIONS = 5


class RealArchitect(BaseAgent):
    """
    Decomposes requirements into atomic specs and plans.
    Has read-only filesystem access and web search capability.

    Implements ReAct loop: Agent can call tools (e.g., read_file) and
    receive results back to inform its final plan.
    """

    async def process(self, context: Dict) -> Dict:
        req_node = context['nodes'][0]

        # Check for escalation context (from feedback controller)
        escalation_context = req_node.get('escalation_context')
        template_vars = {'req_id': req_node.get('id', 'unknown')}

        if escalation_context:
            # This is a re-planning after failures - use escalation prompt
            system_prompt = self._hydrate_prompt("architect_core_v1", template_vars)
            user_prompt = self._build_escalation_prompt(req_node, escalation_context)
            self.logger.warning(f"Processing escalated REQ {req_node['id']} with strategy change")
        else:
            # Normal first-time planning
            system_prompt = self._hydrate_prompt("architect_core_v1", template_vars)
            user_prompt = f"""REQUIREMENT: {req_node['content']}

INSTRUCTIONS:
1. If the requirement references specific files (e.g., "inherit from X in file Y"), use the read_file tool to examine those files FIRST.
2. After gathering necessary context, decompose into Atomic Specs and Plans.
3. When you have enough information, output your final plan as JSON (no tool calls).

Available tools: read_file, list_directory, fetch_url, search_web"""

        # Get filtered tools for this role
        tools_schema = self.get_tools_schema()

        # === ReAct Loop: Iterate until LLM produces final output (no tool calls) ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Architect ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            raw_response = self.gateway.call_model(
                role="ARCHITECT",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=tools_schema if tools_schema else None
            )

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Execute tools and get results
                    tool_results = await self.execute_tool_calls(parsed)
                    self.logger.info(f"Architect tool calls executed, feeding results back to LLM")

                    # CRITICAL: Append tool results to user_prompt for next iteration
                    user_prompt += f"\n\n[TOOL RESULTS from iteration {iteration + 1}]:\n{tool_results}"

                    # If there's partial content alongside tool calls, note it
                    if parsed.get('content'):
                        user_prompt += f"\n\n[YOUR PREVIOUS THOUGHTS]:\n{parsed['content']}"

                    # Continue loop - let LLM see the tool results
                    continue

                # No tool calls - this is the final response
                # Check if it has the expected structure (new_nodes, new_edges)
                if 'new_nodes' in parsed or 'content' in parsed:
                    self.logger.info(f"Architect produced final plan after {iteration + 1} iterations")
                    return parsed

            except json.JSONDecodeError:
                # Response is not JSON - might be raw text plan
                pass

            # If we get here, response is either non-JSON or doesn't have tool_calls
            # Try to parse it as the final response
            self.logger.info(f"Architect produced final response after {iteration + 1} iterations")
            return self._parse_json_response(raw_response)

        # Max iterations reached - return whatever we have
        self.logger.warning(f"Architect reached max iterations ({MAX_REACT_ITERATIONS}), returning last response")
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

    Implements ReAct loop: Builder can read existing files to understand
    interfaces, patterns, and dependencies before generating code.
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

        user_prompt = f"""SPECIFICATION:
{spec_content}

INSTRUCTIONS:
1. If the spec references existing files, base classes, or interfaces, use read_file to examine them FIRST.
2. Use list_directory if you need to understand the project structure.
3. After gathering context, implement the code according to the spec.

CRITICAL: You MUST respond with ONLY valid JSON. No explanations outside JSON.

REQUIRED OUTPUT FORMAT:
```json
{{"content": "<your complete Python code here>", "metadata": {{"language": "python", "file_path": "path/to/file.py"}}}}
```

Implement now and output JSON:"""

        # === ReAct Loop: Iterate until LLM produces final code (no tool calls) ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Builder ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            raw_response = self.gateway.call_model(
                role="BUILDER",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=tools_schema if tools_schema else None
            )

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Execute tools and get results
                    tool_results = await self.execute_tool_calls(parsed)
                    self.logger.info(f"Builder tool calls executed, feeding results back to LLM")

                    # CRITICAL: Append tool results to user_prompt for next iteration
                    user_prompt += f"\n\n[TOOL RESULTS from iteration {iteration + 1}]:\n{tool_results}"

                    # If there's partial content alongside tool calls, note it
                    if parsed.get('content'):
                        user_prompt += f"\n\n[YOUR PREVIOUS THOUGHTS]:\n{parsed['content']}"

                    # Continue loop - let LLM see the tool results
                    continue

                # No tool calls - this is the final response
                if 'content' in parsed:
                    self.logger.info(f"Builder produced final code after {iteration + 1} iterations")
                    result = parsed
                    result['type'] = NodeType.CODE.value
                    result['status'] = NodeStatus.PENDING.value
                    return result

            except json.JSONDecodeError:
                # Response is not JSON - might be raw code
                pass

            # If we get here, response is either non-JSON or doesn't have tool_calls
            self.logger.info(f"Builder produced final response after {iteration + 1} iterations")
            result = self._parse_json_response(raw_response)
            result['type'] = NodeType.CODE.value
            result['status'] = NodeStatus.PENDING.value
            return result

        # Max iterations reached - return whatever we have
        self.logger.warning(f"Builder reached max iterations ({MAX_REACT_ITERATIONS}), returning last response")
        result = self._parse_json_response(raw_response)
        result['type'] = NodeType.CODE.value
        result['status'] = NodeStatus.PENDING.value
        return result


class RealVerifier(BaseAgent):
    """
    Reviews and verifies code nodes.
    Has read-only filesystem access.

    Implements ReAct loop: Verifier can read referenced files to check
    that the code correctly implements interfaces and follows patterns.
    """

    async def process(self, context: Dict) -> Dict:
        code_node = context['nodes'][0]
        system_prompt = self._hydrate_prompt("verifier_core_v1", {
            "builder_id": "Unknown_Builder",
            "spec_id": "Linked_Spec"
        })

        # Get filtered tools for this role
        tools_schema = self.get_tools_schema()

        user_prompt = f"""CODE TO VERIFY:
```python
{code_node['content']}
```

VERIFICATION INSTRUCTIONS:
1. If the code imports from or inherits from other project files, use read_file to examine those files.
2. Verify that method signatures match any base classes or interfaces.
3. Check for common issues: missing imports, incorrect API usage, security vulnerabilities.
4. After thorough review, output your verdict.

CRITICAL: You MUST respond with ONLY valid JSON. No explanations, no questions.
Even if the code is incomplete, respond with a verdict.

REQUIRED OUTPUT FORMAT (JSON only):
```json
{{"verdict": "PASS", "critique": []}}
```
OR
```json
{{"verdict": "FAIL", "critique": ["issue 1", "issue 2"]}}
```

Output your JSON verdict now:"""

        # === ReAct Loop: Iterate until LLM produces final verdict (no tool calls) ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Verifier ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            raw_response = self.gateway.call_model(
                role="VERIFIER",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=tools_schema if tools_schema else None
            )

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Execute tools and get results
                    tool_results = await self.execute_tool_calls(parsed)
                    self.logger.info(f"Verifier tool calls executed, feeding results back to LLM")

                    # CRITICAL: Append tool results to user_prompt for next iteration
                    user_prompt += f"\n\n[TOOL RESULTS from iteration {iteration + 1}]:\n{tool_results}"

                    # If there's partial content alongside tool calls, note it
                    if parsed.get('content'):
                        user_prompt += f"\n\n[YOUR ANALYSIS SO FAR]:\n{parsed['content']}"

                    # Continue loop - let LLM see the tool results
                    continue

                # No tool calls - this is the final verdict
                if 'verdict' in parsed:
                    self.logger.info(f"Verifier produced final verdict after {iteration + 1} iterations: {parsed.get('verdict')}")
                    return parsed

            except json.JSONDecodeError:
                # Response is not JSON
                pass

            # If we get here, response is either non-JSON or doesn't have tool_calls
            self.logger.info(f"Verifier produced final response after {iteration + 1} iterations")
            return self._parse_json_response(raw_response)

        # Max iterations reached - return whatever we have
        self.logger.warning(f"Verifier reached max iterations ({MAX_REACT_ITERATIONS}), returning last response")
        return self._parse_json_response(raw_response)


class RealSocrates(BaseAgent):
    """
    Researches and resolves ambiguity in requirements.
    Has filesystem and web search access.

    Implements ReAct loop: Socrates can research standards, read existing
    code patterns, and gather context to ask better clarifying questions.
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

INSTRUCTIONS:
1. Use read_file to examine any existing code relevant to this requirement.
2. Use search_web or fetch_url to research standards and best practices.
3. After gathering context, identify what is unclear and propose specific questions.
4. Output your final response as JSON with 'questions' (list) and 'research' (summary).

Identify what is unclear and propose specific questions to resolve ambiguity."""

        tools_schema = self.get_tools_schema()
        accumulated_research = []

        # === ReAct Loop: Iterate until LLM produces final questions (no tool calls) ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Socrates ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            raw_response = self.gateway.call_model(
                role="SOCRATES",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=tools_schema if tools_schema else None
            )

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Execute tools and get results
                    tool_results = await self.execute_tool_calls(parsed)
                    accumulated_research.append(tool_results)
                    self.logger.info(f"Socrates tool calls executed, feeding results back to LLM")

                    # CRITICAL: Append tool results to user_prompt for next iteration
                    user_prompt += f"\n\n[RESEARCH RESULTS from iteration {iteration + 1}]:\n{tool_results}"

                    # If there's partial content alongside tool calls, note it
                    if parsed.get('content'):
                        user_prompt += f"\n\n[YOUR THOUGHTS SO FAR]:\n{parsed['content']}"

                    # Continue loop - let LLM see the tool results
                    continue

                # No tool calls - this is the final response
                if 'questions' in parsed:
                    self.logger.info(f"Socrates produced final questions after {iteration + 1} iterations")
                    # Include accumulated research
                    if accumulated_research and not parsed.get('research'):
                        parsed['research'] = "\n".join(accumulated_research)
                    return parsed

            except json.JSONDecodeError:
                # Response is not JSON
                pass

            # If we get here, response is either non-JSON or doesn't have tool_calls
            self.logger.info(f"Socrates produced final response after {iteration + 1} iterations")
            return {
                "questions": raw_response,
                "research": "\n".join(accumulated_research) if accumulated_research else None
            }

        # Max iterations reached - return whatever we have
        self.logger.warning(f"Socrates reached max iterations ({MAX_REACT_ITERATIONS}), returning last response")
        return {
            "questions": raw_response,
            "research": "\n".join(accumulated_research) if accumulated_research else None
        }

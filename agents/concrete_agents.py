"""
CONCRETE AGENTS
Production implementations with optional MCP tool support.
Implements ReAct (Reason + Act) loop for tool-augmented reasoning.

Protocol-Based Output:
    Agents use Pydantic protocols with forced tool_choice to guarantee
    valid structured output, eliminating unreliable regex parsing.
"""
import json
from typing import Dict, Optional
from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus
from core.protocols import (
    ArchitectOutput,
    BuilderOutput,
    VerifierOutput,
    SocratesOutput,
    NodeSpec,
    EdgeSpec,
    get_agent_tools
)

# Maximum iterations for ReAct loop to prevent infinite loops
MAX_REACT_ITERATIONS = 5


class RealArchitect(BaseAgent):
    """
    Decomposes requirements into atomic specs and plans.
    Has read-only filesystem access and web search capability.

    Implements ReAct loop: Agent can call tools (e.g., read_file) and
    receive results back to inform its final plan.

    Output Protocol: ArchitectOutput (guaranteed via forced tool_choice)
    """

    async def process(self, context: Dict) -> Dict:
        req_node = context['nodes'][0]
        req_content = req_node.get('content', 'No requirement content')

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
            user_prompt = f"""REQUIREMENT: {req_content}

INSTRUCTIONS:
1. If the requirement references specific files (e.g., "inherit from X in file Y"), use the read_file tool to examine those files FIRST.
2. After gathering necessary context, decompose into ATOMIC Specs - each SPEC should be implementable by a single Builder.
3. CRITICAL: Identify DEPENDENCIES between specs using DEPENDS_ON edges:
   - If Spec A requires types/functions defined in Spec B, create edge: A --DEPENDS_ON--> B
   - Dependencies determine BUILD ORDER: B will be built before A
   - Use "new_N" notation for edge source/target IDs (e.g., "new_0" for first spec, "new_1" for second)
4. Create a PLAN node summarizing the overall approach.
5. When you have enough information, use the submit_architecture tool to output your final plan.

DEPENDENCY EXAMPLES:
- If you create a "DatabaseConnection" spec (new_0) and a "UserRepository" spec (new_1) that uses the connection:
  new_edges: [{{"source_id": "new_1", "target_id": "new_0", "relation": "DEPENDS_ON"}}]
  This means UserRepository depends on DatabaseConnection, so DatabaseConnection is built first.

Available tools: read_file, list_directory, fetch_url, search_web, submit_architecture"""

        # Get filtered tools for this role + protocol output tool
        tools_schema = self.get_tools_schema()
        protocol_tools = get_agent_tools("ARCHITECT")
        all_tools = tools_schema + protocol_tools if tools_schema else protocol_tools
        self.logger.info(f"Architect processing requirement: {req_content[:100]}...")

        # === ReAct Loop: Iterate until LLM produces final output via protocol ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Architect ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            # Check if this is the final iteration - force protocol output
            if iteration == MAX_REACT_ITERATIONS - 1:
                self.logger.info("Architect final iteration: forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt)

            raw_response = self.gateway.call_model(
                role="ARCHITECT",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=all_tools if all_tools else None
            )

            # Log raw response for debugging
            self.logger.debug(f"Architect raw response length: {len(raw_response)} chars")

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Check if one of the tool calls is submit_architecture (protocol output)
                    for tc in parsed['tool_calls']:
                        tc_name = tc.get('name') or tc.get('function', {}).get('name', '')
                        if tc_name == 'submit_architecture':
                            # Extract and validate protocol output
                            args = tc.get('input') or tc.get('arguments') or tc.get('function', {}).get('arguments', {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            # Parse any nested JSON string fields
                            args = self._parse_nested_json(args)
                            validated = ArchitectOutput.model_validate(args)
                            self.logger.info(f"Architect produced validated plan with {len(validated.new_nodes)} nodes")
                            return validated.model_dump()

                    # Execute other tools and get results
                    tool_names = [tc.get('name', tc.get('function', {}).get('name', '?')) for tc in parsed['tool_calls']]
                    self.logger.info(f"Architect calling tools: {tool_names}")
                    tool_results = await self.execute_tool_calls(parsed)
                    self.logger.debug(f"Tool results: {tool_results[:500]}...")

                    # CRITICAL: Append tool results to user_prompt for next iteration
                    user_prompt += f"\n\n[TOOL RESULTS from iteration {iteration + 1}]:\n{tool_results}"

                    # If there's partial content alongside tool calls, note it
                    if parsed.get('content'):
                        user_prompt += f"\n\n[YOUR PREVIOUS THOUGHTS]:\n{parsed['content']}"

                    # Continue loop - let LLM see the tool results
                    continue

                # No tool calls - force protocol output on next attempt
                self.logger.info("Architect responded without tool calls, forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt)

            except json.JSONDecodeError as e:
                # Response is not JSON - force protocol output
                self.logger.warning(f"Architect response not valid JSON: {e}")
                return await self._produce_final_output(system_prompt, user_prompt)

        # Should not reach here due to final iteration check above
        return await self._produce_final_output(system_prompt, user_prompt)

    async def _produce_final_output(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Force protocol-based output using call_with_protocol.
        Guarantees valid ArchitectOutput structure via forced tool_choice.
        """
        try:
            validated: ArchitectOutput = self.gateway.call_with_protocol(
                role="ARCHITECT",
                system_prompt=system_prompt,
                user_context=user_prompt + "\n\nNow use the submit_architecture tool to output your final plan.",
                output_protocol=ArchitectOutput,
                tool_name="submit_architecture"
            )
            self.logger.info(f"Architect produced validated plan with {len(validated.new_nodes)} nodes")
            return validated.model_dump()
        except Exception as e:
            self.logger.error(f"Architect protocol output failed: {e}")
            # Return minimal valid structure on failure
            return {
                "reasoning": f"Protocol output failed: {e}",
                "new_nodes": [],
                "parse_error": True
            }

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

    Output Protocol: BuilderOutput (guaranteed via forced tool_choice)
    """

    async def process(self, context: Dict) -> Dict:
        spec_content = context.get('nodes', [{}])[0].get('content', "No Spec Found")
        system_prompt = self._hydrate_prompt("builder_core_v1", {
            "target_node_id": "CURRENT_TASK",
            "target_node_spec": spec_content,
            "language": "python",
            "file_path": "generated_module.py"
        })

        # Get filtered tools for this role + protocol output tool
        tools_schema = self.get_tools_schema()
        protocol_tools = get_agent_tools("BUILDER")
        all_tools = tools_schema + protocol_tools if tools_schema else protocol_tools

        user_prompt = f"""SPECIFICATION:
{spec_content}

INSTRUCTIONS:
1. If the spec references existing files, base classes, or interfaces, use read_file to examine them FIRST.
2. Use list_directory if you need to understand the project structure.
3. After gathering context, use the submit_code tool to output your implementation.

IMPORTANT: The submit_code tool requires this exact structure:
{{
  "reasoning": "Brief explanation of your implementation",
  "code": {{
    "file_path": "path/to/file.py",
    "language": "python",
    "content": "your complete source code here"
  }}
}}

Available tools: read_file, list_directory, write_file, submit_code"""

        # === ReAct Loop: Iterate until LLM produces final code via protocol ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Builder ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            # Check if this is the final iteration - force protocol output
            if iteration == MAX_REACT_ITERATIONS - 1:
                self.logger.info("Builder final iteration: forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt)

            raw_response = self.gateway.call_model(
                role="BUILDER",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=all_tools if all_tools else None
            )

            # Log raw response length for debugging
            self.logger.debug(f"Builder raw response length: {len(raw_response)} chars")

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Check if one of the tool calls is submit_code (protocol output)
                    for tc in parsed['tool_calls']:
                        tc_name = tc.get('name') or tc.get('function', {}).get('name', '')
                        if tc_name == 'submit_code':
                            # Extract and validate protocol output
                            args = tc.get('input') or tc.get('arguments') or tc.get('function', {}).get('arguments', {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            # Parse any nested JSON string fields
                            args = self._parse_nested_json(args)
                            validated = BuilderOutput.model_validate(args)
                            result = self._format_builder_output(validated)
                            self.logger.info(f"Builder produced validated code: {validated.code.file_path}")
                            return result

                    # Execute other tools and get results
                    tool_names = [tc.get('name', tc.get('function', {}).get('name', '?')) for tc in parsed['tool_calls']]
                    self.logger.info(f"Builder calling tools: {tool_names}")
                    tool_results = await self.execute_tool_calls(parsed)
                    self.logger.debug(f"Tool results: {tool_results[:500]}...")

                    # CRITICAL: Append tool results to user_prompt for next iteration
                    user_prompt += f"\n\n[TOOL RESULTS from iteration {iteration + 1}]:\n{tool_results}"

                    # If there's partial content alongside tool calls, note it
                    if parsed.get('content'):
                        user_prompt += f"\n\n[YOUR PREVIOUS THOUGHTS]:\n{parsed['content']}"

                    # Continue loop - let LLM see the tool results
                    continue

                # No tool calls - force protocol output on next attempt
                self.logger.info("Builder responded without tool calls, forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt)

            except json.JSONDecodeError as e:
                # Response is not JSON - force protocol output
                self.logger.warning(f"Builder response not valid JSON: {e}")
                return await self._produce_final_output(system_prompt, user_prompt)

        # Should not reach here due to final iteration check above
        return await self._produce_final_output(system_prompt, user_prompt)

    async def _produce_final_output(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Force protocol-based output using call_with_protocol.
        Guarantees valid BuilderOutput structure via forced tool_choice.
        """
        try:
            validated: BuilderOutput = self.gateway.call_with_protocol(
                role="BUILDER",
                system_prompt=system_prompt,
                user_context=user_prompt + "\n\nNow use the submit_code tool to output your implementation.",
                output_protocol=BuilderOutput,
                tool_name="submit_code"
            )
            result = self._format_builder_output(validated)
            self.logger.info(f"Builder produced validated code: {validated.code.file_path}")
            return result
        except Exception as e:
            self.logger.error(f"Builder protocol output failed: {e}")
            # Return minimal valid structure on failure
            return {
                "content": f"# Protocol output failed: {e}",
                "type": NodeType.CODE.value,
                "status": NodeStatus.PENDING.value,
                "metadata": {"language": "python", "file_path": "error.py"},
                "parse_error": True
            }

    def _format_builder_output(self, validated: BuilderOutput) -> Dict:
        """Convert validated BuilderOutput to the expected dict format."""
        return {
            "content": validated.code.content,
            "type": NodeType.CODE.value,
            "status": NodeStatus.PENDING.value,
            "metadata": {
                "language": validated.code.language,
                "file_path": validated.code.file_path,
                "dependencies": validated.code.dependencies,
                "test_hints": validated.code.test_hints
            },
            "reasoning": validated.reasoning,
            "follow_up_specs": [spec.model_dump() for spec in validated.follow_up_specs] if validated.follow_up_specs else None
        }


class RealVerifier(BaseAgent):
    """
    Reviews and verifies code nodes.
    Has read-only filesystem access.

    Implements ReAct loop: Verifier can read referenced files to check
    that the code correctly implements interfaces and follows patterns.

    Output Protocol: VerifierOutput (guaranteed via forced tool_choice)
    """

    async def process(self, context: Dict) -> Dict:
        code_node = context['nodes'][0]
        system_prompt = self._hydrate_prompt("verifier_core_v1", {
            "builder_id": "Unknown_Builder",
            "spec_id": "Linked_Spec"
        })

        # Get filtered tools for this role + protocol output tool
        tools_schema = self.get_tools_schema()
        protocol_tools = get_agent_tools("VERIFIER")
        all_tools = tools_schema + protocol_tools if tools_schema else protocol_tools

        user_prompt = f"""CODE TO VERIFY:
```python
{code_node['content']}
```

VERIFICATION INSTRUCTIONS:
1. If the code imports from or inherits from other project files, use read_file to examine those files.
2. Verify that method signatures match any base classes or interfaces.
3. Check for common issues: missing imports, incorrect API usage, security vulnerabilities.
4. After thorough review, use the submit_verdict tool to output your verdict.

Available tools: read_file, list_directory, submit_verdict"""

        # Log the code being verified (preview)
        code_preview = code_node['content'][:200] if code_node.get('content') else '(no content)'
        self.logger.info(f"Verifier reviewing code: {code_preview}...")

        # === ReAct Loop: Iterate until LLM produces final verdict via protocol ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Verifier ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            # Check if this is the final iteration - force protocol output
            if iteration == MAX_REACT_ITERATIONS - 1:
                self.logger.info("Verifier final iteration: forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt)

            raw_response = self.gateway.call_model(
                role="VERIFIER",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=all_tools if all_tools else None
            )

            # Log raw response for debugging
            self.logger.debug(f"Verifier raw response length: {len(raw_response)} chars")

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Check if one of the tool calls is submit_verdict (protocol output)
                    for tc in parsed['tool_calls']:
                        tc_name = tc.get('name') or tc.get('function', {}).get('name', '')
                        if tc_name == 'submit_verdict':
                            # Extract and validate protocol output
                            args = tc.get('input') or tc.get('arguments') or tc.get('function', {}).get('arguments', {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            # Parse any nested JSON string fields
                            args = self._parse_nested_json(args)
                            validated = VerifierOutput.model_validate(args)
                            result = self._format_verifier_output(validated)
                            self.logger.info(f"Verifier verdict: {validated.verdict}")
                            return result

                    # Execute other tools and get results
                    tool_names = [tc.get('name', tc.get('function', {}).get('name', '?')) for tc in parsed['tool_calls']]
                    self.logger.info(f"Verifier calling tools: {tool_names}")
                    tool_results = await self.execute_tool_calls(parsed)
                    self.logger.debug(f"Tool results: {tool_results[:500]}...")

                    # CRITICAL: Append tool results to user_prompt for next iteration
                    user_prompt += f"\n\n[TOOL RESULTS from iteration {iteration + 1}]:\n{tool_results}"

                    # If there's partial content alongside tool calls, note it
                    if parsed.get('content'):
                        user_prompt += f"\n\n[YOUR ANALYSIS SO FAR]:\n{parsed['content']}"

                    # Continue loop - let LLM see the tool results
                    continue

                # No tool calls - force protocol output on next attempt
                self.logger.info("Verifier responded without tool calls, forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt)

            except json.JSONDecodeError as e:
                # Response is not JSON - force protocol output
                self.logger.warning(f"Verifier response not valid JSON: {e}")
                return await self._produce_final_output(system_prompt, user_prompt)

        # Should not reach here due to final iteration check above
        return await self._produce_final_output(system_prompt, user_prompt)

    async def _produce_final_output(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Force protocol-based output using call_with_protocol.
        Guarantees valid VerifierOutput structure via forced tool_choice.
        """
        try:
            validated: VerifierOutput = self.gateway.call_with_protocol(
                role="VERIFIER",
                system_prompt=system_prompt,
                user_context=user_prompt + "\n\nNow use the submit_verdict tool to output your final verdict.",
                output_protocol=VerifierOutput,
                tool_name="submit_verdict"
            )
            result = self._format_verifier_output(validated)
            self.logger.info(f"Verifier verdict: {validated.verdict}")
            return result
        except Exception as e:
            self.logger.error(f"Verifier protocol output failed: {e}")
            # Return FAIL verdict on protocol failure
            return {
                "verdict": "FAIL",
                "reasoning": f"Protocol output failed: {e}",
                "issues": [{"severity": "error", "category": "logic", "description": str(e)}],
                "parse_error": True
            }

    def _format_verifier_output(self, validated: VerifierOutput) -> Dict:
        """Convert validated VerifierOutput to the expected dict format."""
        result = {
            "verdict": validated.verdict,
            "reasoning": validated.reasoning,
            "verified_aspects": validated.verified_aspects,
            "recommendations": validated.recommendations
        }
        # Convert issues to legacy 'critique' format for backwards compatibility
        if validated.issues:
            result["critique"] = [issue.description for issue in validated.issues]
            result["issues"] = [issue.model_dump() for issue in validated.issues]
        else:
            result["critique"] = []
        return result


class RealSocrates(BaseAgent):
    """
    Researches and resolves ambiguity in requirements.
    Has filesystem and web search access.

    Implements ReAct loop: Socrates can research standards, read existing
    code patterns, and gather context to ask better clarifying questions.

    Output Protocol: SocratesOutput (guaranteed via forced tool_choice)
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
3. After gathering context, use the submit_questions tool to output your clarifying questions.

Available tools: read_file, list_directory, search_web, fetch_url, submit_questions"""

        # Get filtered tools for this role + protocol output tool
        tools_schema = self.get_tools_schema()
        protocol_tools = get_agent_tools("SOCRATES")
        all_tools = tools_schema + protocol_tools if tools_schema else protocol_tools
        accumulated_research = []

        # === ReAct Loop: Iterate until LLM produces final questions via protocol ===
        for iteration in range(MAX_REACT_ITERATIONS):
            self.logger.info(f"Socrates ReAct iteration {iteration + 1}/{MAX_REACT_ITERATIONS}")

            # Check if this is the final iteration - force protocol output
            if iteration == MAX_REACT_ITERATIONS - 1:
                self.logger.info("Socrates final iteration: forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt, accumulated_research)

            raw_response = self.gateway.call_model(
                role="SOCRATES",
                system_prompt=system_prompt,
                user_context=user_prompt,
                tools=all_tools if all_tools else None
            )

            # Try to parse as JSON to check for tool calls
            try:
                parsed = json.loads(raw_response)

                if 'tool_calls' in parsed and parsed['tool_calls']:
                    # Check if one of the tool calls is submit_questions (protocol output)
                    for tc in parsed['tool_calls']:
                        tc_name = tc.get('name') or tc.get('function', {}).get('name', '')
                        if tc_name == 'submit_questions':
                            # Extract and validate protocol output
                            args = tc.get('input') or tc.get('arguments') or tc.get('function', {}).get('arguments', {})
                            if isinstance(args, str):
                                args = json.loads(args)
                            # Parse any nested JSON string fields
                            args = self._parse_nested_json(args)
                            validated = SocratesOutput.model_validate(args)
                            result = self._format_socrates_output(validated, accumulated_research)
                            self.logger.info(f"Socrates produced {len(validated.questions)} clarifying questions")
                            return result

                    # Execute other tools and get results
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

                # No tool calls - force protocol output on next attempt
                self.logger.info("Socrates responded without tool calls, forcing protocol output")
                return await self._produce_final_output(system_prompt, user_prompt, accumulated_research)

            except json.JSONDecodeError:
                # Response is not JSON - force protocol output
                return await self._produce_final_output(system_prompt, user_prompt, accumulated_research)

        # Should not reach here due to final iteration check above
        return await self._produce_final_output(system_prompt, user_prompt, accumulated_research)

    async def _produce_final_output(self, system_prompt: str, user_prompt: str, accumulated_research: list) -> Dict:
        """
        Force protocol-based output using call_with_protocol.
        Guarantees valid SocratesOutput structure via forced tool_choice.
        """
        try:
            validated: SocratesOutput = self.gateway.call_with_protocol(
                role="SOCRATES",
                system_prompt=system_prompt,
                user_context=user_prompt + "\n\nNow use the submit_questions tool to output your clarifying questions.",
                output_protocol=SocratesOutput,
                tool_name="submit_questions"
            )
            result = self._format_socrates_output(validated, accumulated_research)
            self.logger.info(f"Socrates produced {len(validated.questions)} clarifying questions")
            return result
        except Exception as e:
            self.logger.error(f"Socrates protocol output failed: {e}")
            # Return minimal valid structure on failure
            return {
                "ambiguities_found": [f"Protocol output failed: {e}"],
                "questions": [],
                "research": "\n".join(accumulated_research) if accumulated_research else None,
                "parse_error": True
            }

    def _format_socrates_output(self, validated: SocratesOutput, accumulated_research: list) -> Dict:
        """Convert validated SocratesOutput to the expected dict format."""
        result = {
            "ambiguities_found": validated.ambiguities_found,
            "questions": [q.model_dump() for q in validated.questions],
            "assumptions": validated.assumptions,
            "research_summary": validated.research_summary
        }
        # Include accumulated research if not already in the output
        if accumulated_research and not validated.research_summary:
            result["research"] = "\n".join(accumulated_research)
        return result

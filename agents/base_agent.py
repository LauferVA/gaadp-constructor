"""
BASE AGENT (Hardened with MCP & RBAC)
Integrates Runtime Signing, File Locks, LLM Gateway, and Tool Permissions.
"""
import json
import logging
import yaml
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from infrastructure.llm_gateway import LLMGateway
from core.ontology import AgentRole

# Optional MCP import
try:
    from infrastructure.mcp_hub import MCPHub
except ImportError:
    MCPHub = None


class BaseAgent(ABC):
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        graph_db,
        mcp_hub: Optional["MCPHub"] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.graph_db = graph_db
        self.mcp_hub = mcp_hub
        self.gateway = LLMGateway()
        self.logger = logging.getLogger(f"Agent.{role}.{agent_id}")
        self._private_key = ed25519.Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()
        self._save_keys()

        with open(".blueprint/prompt_templates.yaml", "r") as f:
            self.templates = yaml.safe_load(f)
        with open(".blueprint/prime_directives.md", "r") as f:
            self.directives = f.read()

        # Load topology for RBAC
        try:
            with open(".blueprint/topology_config.yaml", "r") as f:
                self.topology = yaml.safe_load(f)
        except FileNotFoundError:
            self.topology = {"tool_permissions": {}}

        # Cache allowed tools for fast lookup
        self.allowed_tools = (
            self.topology
            .get("tool_permissions", {})
            .get(role.value, {})
            .get("allowed_tools", [])
        )

    def _save_keys(self):
        key_dir = ".gaadp/keys"
        os.makedirs(key_dir, exist_ok=True)
        pub_key_path = f"{key_dir}/{self.agent_id}.pub"
        pem = self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        with open(pub_key_path, "wb") as f:
            f.write(pem)

    def acquire_lock(self, node_id: str, timeout: int = 10) -> bool:
        lock_dir = ".gaadp/locks"
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = f"{lock_dir}/{node_id}.lock"
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with open(lock_file, "x") as f:
                    f.write(self.agent_id)
                return True
            except FileExistsError:
                if os.path.getmtime(lock_file) < time.time() - 30:
                    os.remove(lock_file)
                    continue
                time.sleep(0.1)
        return False

    def release_lock(self, node_id: str):
        lock_file = f".gaadp/locks/{node_id}.lock"
        if os.path.exists(lock_file):
            os.remove(lock_file)

    def sign_content(self, content: Any, previous_hash: str = "GENESIS") -> str:
        """
        Cryptographically signs data linked to history.
        Creates a Merkle-like chain of custody.
        """
        payload = {
            "content": content,
            "prev_hash": previous_hash,
            "agent_id": self.agent_id,
            "timestamp": time.time()
        }
        data_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = self._private_key.sign(data_bytes).hex()
        return signature

    def _hydrate_prompt(self, template_id: str, vars: Dict) -> str:
        template = self.templates.get(template_id, {}).get('instruction', "")
        vars['prime_directives_text'] = self.directives
        vars['agent_role'] = self.role.value
        vars['agent_id'] = self.agent_id
        return template.format(**vars)

    def _parse_nested_json(self, obj: Any) -> Any:
        """
        Recursively parse JSON strings within a dict/list structure.
        Some LLM responses return nested JSON as strings that need parsing.

        Args:
            obj: The object to parse (dict, list, or primitive)

        Returns:
            The object with any JSON strings parsed into proper objects
        """
        import re

        if isinstance(obj, str):
            # Try to parse as JSON - handle both raw and escaped strings
            stripped = obj.strip()

            # Strip trailing XML-like tags (common LLM artifact)
            # e.g., "]\n</invoke>" or "}\n</function_call>"
            stripped = re.sub(r'\s*</[a-zA-Z_][a-zA-Z0-9_-]*>\s*$', '', stripped)

            # Check if it looks like JSON (object or array)
            if (stripped.startswith('{') and stripped.endswith('}')) or \
               (stripped.startswith('[') and stripped.endswith(']')):
                # Try multiple parsing strategies
                attempts = [
                    stripped,  # Cleaned version (XML tags removed)
                    obj,  # Original
                    stripped.replace('\\n', '\n').replace('\\t', '\t'),  # Common escapes on cleaned
                ]

                # Handle LLM using Python triple-quotes instead of JSON strings
                # Convert '''content''' or \"\"\"content\"\"\" to proper JSON
                if "'''" in stripped or '"""' in stripped:
                    fixed = stripped
                    # Match '''...''' or """...""" and replace with escaped content
                    for quote in ["'''", '"""']:
                        pattern = re.escape(quote) + r'(.*?)' + re.escape(quote)
                        matches = list(re.finditer(pattern, fixed, re.DOTALL))
                        for match in reversed(matches):  # Reverse to preserve indices
                            content = match.group(1)
                            # Escape for JSON: newlines, quotes, backslashes
                            escaped = content.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\t', '\\t')
                            fixed = fixed[:match.start()] + '"' + escaped + '"' + fixed[match.end():]
                    attempts.append(fixed)

                # Try unicode unescape
                if '\\' in stripped:
                    try:
                        attempts.append(stripped.encode().decode('unicode_escape'))
                    except UnicodeDecodeError:
                        pass

                # Handle unescaped newlines inside JSON string values
                # This is common when LLM outputs multiline code in "content" field
                def fix_unescaped_newlines(text: str) -> str:
                    """Escape newlines that appear inside JSON string values."""
                    result = []
                    in_string = False
                    i = 0
                    while i < len(text):
                        char = text[i]
                        if char == '"' and (i == 0 or text[i-1] != '\\'):
                            in_string = not in_string
                            result.append(char)
                        elif char == '\n' and in_string:
                            result.append('\\n')
                        elif char == '\t' and in_string:
                            result.append('\\t')
                        else:
                            result.append(char)
                        i += 1
                    return ''.join(result)

                attempts.append(fix_unescaped_newlines(stripped))

                for attempt, text in enumerate(attempts):
                    try:
                        parsed = json.loads(text)
                        self.logger.debug(f"Parsed nested JSON on attempt {attempt + 1}")
                        return self._parse_nested_json(parsed)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                # If all parsing failed, return as-is
                self.logger.warning(f"Failed to parse nested JSON string: {obj[:100]}...")
            return obj
        elif isinstance(obj, dict):
            return {k: self._parse_nested_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._parse_nested_json(item) for item in obj]
        else:
            return obj

    def _parse_json_response(self, text: str) -> Dict:
        """
        DEPRECATED: Use protocol-based output via gateway.call_with_protocol() instead.

        Parse LLM response, handling JSON, Markdown blocks, and mixed content.
        This method is kept for backwards compatibility during migration.

        WARNING: This regex-based parsing is unreliable. Agents should use
        Pydantic protocols with forced tool_choice for guaranteed structured output.
        See core/protocols.py and gateway.call_with_protocol().

        Strategies:
        1. Direct JSON parse
        2. Extract from ```json ... ``` blocks
        3. Find first outer brace { ... }
        4. Return error dict

        Args:
            text: Raw LLM response

        Returns:
            Parsed dict

        Note:
            New agents should NOT use this method. Use call_with_protocol() instead.
        """
        import re
        import warnings

        warnings.warn(
            "_parse_json_response is deprecated. Use gateway.call_with_protocol() "
            "with Pydantic protocols for guaranteed structured output.",
            DeprecationWarning,
            stacklevel=2
        )

        if not text or not text.strip():
            self.logger.error("Empty response from LLM")
            return {
                "content": "",
                "verdict": "FAIL",
                "critique": "Empty response from LLM",
                "parse_error": True
            }

        text = text.strip()

        # 1. Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 2. Try extracting from ```json ... ``` blocks
        match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Try finding the first outer brace { ... }
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 4. Return error dict (no longer raises exception)
        self.logger.error(f"JSON PARSE FAILED. Raw Output (first 500 chars): {text[:500]}")
        self.logger.warning("LLM returned non-JSON response, wrapping as content")
        return {
            "content": text,
            "verdict": "FAIL",
            "critique": "LLM failed to produce valid JSON output - response was conversational text",
            "parse_error": True
        }

    def get_tools_schema(self) -> List[Dict]:
        """Get filtered tool schemas for this agent's role."""
        if self.mcp_hub:
            return self.mcp_hub.get_tools_for_role(self.role.value)
        return []

    def check_tool_permission(self, tool_name: str) -> bool:
        """Check if this agent can use a specific tool."""
        return tool_name in self.allowed_tools

    async def execute_tool_calls(self, response: Dict) -> str:
        """
        Execute tool calls from LLM response with permission checking.

        Args:
            response: Parsed LLM response containing tool_calls

        Returns:
            Concatenated results from all tool executions

        Supports multiple tool call formats:
            - OpenAI/Anthropic API format: {"function": {"name": "...", "arguments": "..."}}
            - Simple format: {"name": "...", "input": {...}}
        """
        if not self.mcp_hub:
            return "No MCP Hub configured"

        tool_calls = response.get('tool_calls', [])
        results_log = []

        for call in tool_calls:
            # Support both API format and simple format
            if 'function' in call:
                # OpenAI/Anthropic API format
                func_name = call['function']['name']
                raw_args = call['function'].get('arguments', '{}')
            else:
                # Simple format (e.g., from ManualProvider)
                func_name = call.get('name', 'unknown')
                raw_args = call.get('input', {})

            # Handle arguments: can be string (needs parsing) or dict (use directly)
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    results_log.append(f"Tool '{func_name}' Failed: Invalid JSON arguments: {raw_args}")
                    continue
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                args = {}

            # SECURITY CHECK
            if not self.check_tool_permission(func_name):
                denial_msg = (
                    f"⛔ SECURITY ALERT: Agent '{self.agent_id}' ({self.role.value}) "
                    f"attempted to use forbidden tool '{func_name}'"
                )
                self.logger.warning(denial_msg)
                results_log.append(denial_msg)
                continue

            try:
                self.logger.info(f"🛠️ Executing Tool: {func_name}")
                result = await self.mcp_hub.execute_tool(
                    func_name, args, role_name=self.role.value
                )
                results_log.append(f"Tool '{func_name}' Output: {str(result)}")
            except PermissionError as e:
                results_log.append(f"⛔ Permission Denied: {str(e)}")
            except Exception as e:
                results_log.append(f"Tool '{func_name}' Failed: {str(e)}")

        return "\n".join(results_log)

    @abstractmethod
    async def process(self, context: Dict) -> Dict:
        pass

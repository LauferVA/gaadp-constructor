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

    def _parse_json_response(self, text: str) -> Dict:
        """
        Parse LLM response, handling JSON, Markdown blocks, and mixed content.

        Strategies:
        1. Direct JSON parse
        2. Extract from ```json ... ``` blocks
        3. Find first outer brace { ... }
        4. Raise ValueError with debug output

        Args:
            text: Raw LLM response

        Returns:
            Parsed dict

        Raises:
            ValueError: If no valid JSON can be extracted
        """
        import re

        if not text or not text.strip():
            raise ValueError("Empty response from LLM")

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

        # 4. Debug: Print what failed and return a sensible default
        print(f"❌ JSON PARSE FAILED. Raw Output:\n{text[:500]}...\n")
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

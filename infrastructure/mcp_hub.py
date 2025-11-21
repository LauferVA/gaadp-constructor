"""
MCP HUB - Model Context Protocol Integration
Provides unified tool access with Role-Based Access Control.
"""
import logging
import yaml
import json
import asyncio
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger("MCPHub")


class ToolProvider(ABC):
    """Abstract base class for tool providers."""

    @abstractmethod
    def get_tools(self) -> List[Dict]:
        """Return list of tool definitions."""
        pass

    @abstractmethod
    async def execute(self, tool_name: str, args: Dict) -> Any:
        """Execute a tool with given arguments."""
        pass


class FilesystemProvider(ToolProvider):
    """Provides filesystem tools."""

    def __init__(self, base_path: str = "."):
        self.base_path = base_path

    def get_tools(self) -> List[Dict]:
        return [
            {
                "name": "read_file",
                "description": "Read contents of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to read"}
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "write_file",
                "description": "Write content to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path to write"},
                        "content": {"type": "string", "description": "Content to write"}
                    },
                    "required": ["path", "content"]
                }
            },
            {
                "name": "list_directory",
                "description": "List contents of a directory",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Directory path"}
                    },
                    "required": ["path"]
                }
            }
        ]

    async def execute(self, tool_name: str, args: Dict) -> Any:
        import os

        if tool_name == "read_file":
            path = os.path.join(self.base_path, args["path"])
            with open(path, "r") as f:
                return f.read()

        elif tool_name == "write_file":
            path = os.path.join(self.base_path, args["path"])
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(args["content"])
            return f"Written to {path}"

        elif tool_name == "list_directory":
            path = os.path.join(self.base_path, args.get("path", "."))
            return os.listdir(path)

        raise ValueError(f"Unknown tool: {tool_name}")


class GitProvider(ToolProvider):
    """Provides Git tools."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path

    def get_tools(self) -> List[Dict]:
        return [
            {
                "name": "git_status",
                "description": "Get git repository status",
                "inputSchema": {"type": "object", "properties": {}}
            },
            {
                "name": "git_commit",
                "description": "Commit staged changes",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Commit message"}
                    },
                    "required": ["message"]
                }
            },
            {
                "name": "git_push",
                "description": "Push commits to remote",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]

    async def execute(self, tool_name: str, args: Dict) -> Any:
        import subprocess

        def run_git(cmd):
            result = subprocess.run(
                ["git"] + cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout + result.stderr

        if tool_name == "git_status":
            return run_git(["status"])

        elif tool_name == "git_commit":
            run_git(["add", "."])
            return run_git(["commit", "-m", args["message"]])

        elif tool_name == "git_push":
            return run_git(["push"])

        raise ValueError(f"Unknown tool: {tool_name}")


class MCPHub:
    """
    Central hub for MCP tool management with RBAC.
    """

    def __init__(self, topology_path: str = ".blueprint/topology_config.yaml"):
        self.providers: Dict[str, ToolProvider] = {}
        self.available_tools: List[Dict] = []
        self.topology_path = topology_path

        # Load topology config
        try:
            with open(topology_path, "r") as f:
                self.topology = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Topology config not found: {topology_path}")
            self.topology = {"tool_permissions": {}}

        # Register default providers
        self._register_default_providers()

    def _register_default_providers(self):
        """Register built-in tool providers."""
        self.register_provider("filesystem", FilesystemProvider())
        self.register_provider("git", GitProvider())

    def register_provider(self, name: str, provider: ToolProvider):
        """Register a tool provider."""
        self.providers[name] = provider
        tools = provider.get_tools()
        for tool in tools:
            tool["provider"] = name
        self.available_tools.extend(tools)
        logger.info(f"Registered provider '{name}' with {len(tools)} tools")

    def get_tools_for_role(self, role_name: str) -> List[Dict]:
        """
        Returns only the tool definitions allowed for a specific role.
        Prevents the LLM from seeing tools it cannot use.
        """
        permissions = self.topology.get("tool_permissions", {}).get(role_name, {})
        allowed_tools = permissions.get("allowed_tools", [])

        if not allowed_tools:
            logger.warning(f"No tools defined for role {role_name}")
            return []

        # Filter available tools
        filtered_schemas = []
        for tool in self.available_tools:
            if tool["name"] in allowed_tools:
                filtered_schemas.append({
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("inputSchema", {})
                    }
                })

        return filtered_schemas

    def check_permission(self, role_name: str, tool_name: str) -> bool:
        """Check if a role has permission to use a tool."""
        permissions = self.topology.get("tool_permissions", {}).get(role_name, {})
        allowed_tools = permissions.get("allowed_tools", [])
        return tool_name in allowed_tools

    async def execute_tool(self, tool_name: str, args: Dict, role_name: str = None) -> Any:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments for the tool
            role_name: Optional role for permission checking

        Returns:
            Tool execution result
        """
        # Permission check
        if role_name and not self.check_permission(role_name, tool_name):
            raise PermissionError(
                f"Role '{role_name}' does not have permission to use tool '{tool_name}'"
            )

        # Find the provider for this tool
        for tool in self.available_tools:
            if tool["name"] == tool_name:
                provider_name = tool.get("provider")
                if provider_name and provider_name in self.providers:
                    return await self.providers[provider_name].execute(tool_name, args)

        raise ValueError(f"Tool not found: {tool_name}")

    def list_all_tools(self) -> List[str]:
        """List all available tool names."""
        return [t["name"] for t in self.available_tools]

    def get_role_permissions(self, role_name: str) -> Dict:
        """Get full permission details for a role."""
        return self.topology.get("tool_permissions", {}).get(role_name, {})

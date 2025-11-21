"""
MOCK AGENTS FOR SIMULATION
"""
import uuid
from typing import Dict
from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus

class MockArchitect(BaseAgent):
    async def process(self, context: Dict) -> Dict:
        return {
            "action": "create_graph",
            "new_nodes": [
                {"id": f"spec_{uuid.uuid4().hex[:6]}", "type": NodeType.SPEC, "content": "Mock Spec"},
                {"id": f"plan_{uuid.uuid4().hex[:6]}", "type": NodeType.PLAN, "content": "Mock Plan"}
            ],
            "edges": []
        }

class MockBuilder(BaseAgent):
    async def process(self, context: Dict) -> Dict:
        return {
            "type": NodeType.CODE,
            "status": NodeStatus.PENDING.value,
            "content": "def mock_function(): pass",
            "metadata": {"language": "python", "file_path": "mock_module.py"}
        }

class MockVerifier(BaseAgent):
    async def process(self, context: Dict) -> Dict:
        return {
            "verdict": "PASS",
            "critique": "LGTM",
            "modifications_required": None
        }

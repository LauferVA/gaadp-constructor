"""
CONCRETE AGENTS
"""
from typing import Dict
from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus

class RealArchitect(BaseAgent):
    async def process(self, context: Dict) -> Dict:
        req_node = context['nodes'][0]
        system_prompt = self._hydrate_prompt("architect_core_v1", {})
        user_prompt = f"REQUIREMENT: {req_node['content']}\nDecompose into Atomic Specs and Plans."

        raw_response = self.gateway.call_model(
            role="ARCHITECT",
            system_prompt=system_prompt,
            user_context=user_prompt
        )
        return self._parse_json_response(raw_response)

class RealBuilder(BaseAgent):
    async def process(self, context: Dict) -> Dict:
        spec_content = context.get('nodes', [{}])[0].get('content', "No Spec Found")
        system_prompt = self._hydrate_prompt("builder_core_v1", {
            "target_node_id": "CURRENT_TASK",
            "target_node_spec": spec_content,
            "language": "python",
            "file_path": "generated_module.py"
        })

        raw_response = self.gateway.call_model(
            role="BUILDER",
            system_prompt=system_prompt,
            user_context="Implement the spec now."
        )
        result = self._parse_json_response(raw_response)
        result['type'] = NodeType.CODE.value
        result['status'] = NodeStatus.PENDING.value
        return result

class RealVerifier(BaseAgent):
    async def process(self, context: Dict) -> Dict:
        code_node = context['nodes'][0]
        system_prompt = self._hydrate_prompt("verifier_core_v1", {
            "builder_id": "Unknown_Builder",
            "spec_id": "Linked_Spec"
        })
        user_prompt = f"CODE TO VERIFY:\n{code_node['content']}"

        raw_response = self.gateway.call_model(
            role="VERIFIER",
            system_prompt=system_prompt,
            user_context=user_prompt
        )
        return self._parse_json_response(raw_response)

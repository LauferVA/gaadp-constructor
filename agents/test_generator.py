"""
TEST GENERATOR AGENT
Generates test cases from SPEC and CODE nodes.
"""
import json
import uuid
from typing import Dict, List, Optional

from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus, AgentRole
from infrastructure.graph_db import GraphDB


class RealTestGenerator(BaseAgent):
    """
    Generates pytest test cases from specifications and code.

    Flow:
    1. Reads SPEC node for expected behavior
    2. Reads CODE node for implementation details
    3. Generates comprehensive pytest test cases
    4. Creates TEST node linked to both SPEC and CODE
    """

    async def process(self, context: Dict) -> Dict:
        """
        Generate tests for a SPEC/CODE pair.

        Args:
            context: {
                'spec_node': {'id': str, 'content': str},
                'code_node': {'id': str, 'content': str},  # Optional
                'test_types': ['unit', 'integration', 'edge_cases']  # Optional
            }

        Returns:
            {
                'type': 'TEST',
                'content': str,  # pytest code
                'metadata': {
                    'spec_id': str,
                    'code_id': str,
                    'test_count': int,
                    'file_path': str
                }
            }
        """
        spec_node = context.get('spec_node', context.get('nodes', [{}])[0])
        code_node = context.get('code_node', {})
        test_types = context.get('test_types', ['unit', 'edge_cases'])

        spec_content = spec_node.get('content', '')
        code_content = code_node.get('content', '')

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(spec_content, code_content, test_types)

        # Get tools for file reading if needed
        tools_schema = self.get_tools_schema()

        raw_response = self.gateway.call_model(
            role="BUILDER",  # Use builder model for code generation
            system_prompt=system_prompt,
            user_context=user_prompt,
            tools=tools_schema if tools_schema else None
        )

        # Parse response
        result = self._parse_test_response(raw_response)

        # Add metadata
        result['type'] = NodeType.TEST.value
        result['status'] = NodeStatus.PENDING.value
        result['metadata'] = {
            'spec_id': spec_node.get('id', 'unknown'),
            'code_id': code_node.get('id', 'unknown'),
            'test_types': test_types,
            'file_path': self._generate_test_path(spec_node.get('id', 'test'))
        }

        return result

    def _build_system_prompt(self) -> str:
        """Build system prompt for test generation."""
        return """You are a Test Engineer specializing in pytest test generation.

Your task is to generate comprehensive, high-quality pytest test cases.

Guidelines:
1. Use pytest fixtures for setup/teardown
2. Include docstrings explaining each test
3. Cover: happy path, edge cases, error cases, boundary conditions
4. Use descriptive test names: test_<function>_<scenario>_<expected>
5. Include type hints
6. Use pytest.raises for exception testing
7. Use parametrize for multiple test cases

Output Format:
Return ONLY valid Python code that can be run with pytest.
Include necessary imports at the top.
Do not include any markdown or explanations outside the code.
"""

    def _build_user_prompt(
        self,
        spec_content: str,
        code_content: str,
        test_types: List[str]
    ) -> str:
        """Build user prompt with spec and code."""
        prompt = f"""Generate pytest tests for the following:

SPECIFICATION:
{spec_content}

"""
        if code_content:
            prompt += f"""IMPLEMENTATION:
{code_content}

"""

        prompt += f"""TEST TYPES TO GENERATE: {', '.join(test_types)}

Generate comprehensive pytest test cases covering all specified types.
Return only valid Python code."""

        return prompt

    def _parse_test_response(self, response: str) -> Dict:
        """Parse LLM response to extract test code."""
        # Try to extract code from markdown if present
        if '```python' in response:
            start = response.find('```python') + 9
            end = response.find('```', start)
            if end > start:
                response = response[start:end].strip()
        elif '```' in response:
            start = response.find('```') + 3
            end = response.find('```', start)
            if end > start:
                response = response[start:end].strip()

        return {
            'content': response.strip(),
            'test_count': response.count('def test_')
        }

    def _generate_test_path(self, spec_id: str) -> str:
        """Generate test file path from spec ID."""
        clean_id = spec_id.replace('spec_', '').replace('-', '_')[:20]
        return f"tests/test_{clean_id}.py"


async def generate_tests_for_verified_code(
    db: GraphDB,
    test_generator: RealTestGenerator
) -> List[str]:
    """
    Generate tests for all verified CODE nodes that don't have tests.

    Args:
        db: Graph database
        test_generator: TestGenerator agent instance

    Returns:
        List of created TEST node IDs
    """
    created_tests = []

    for node_id, data in db.graph.nodes(data=True):
        if data.get('type') != NodeType.CODE.value:
            continue
        if data.get('status') != NodeStatus.VERIFIED.value:
            continue

        # Check if this CODE already has tests
        has_tests = any(
            db.graph.nodes[pred].get('type') == NodeType.TEST.value
            for pred in db.graph.predecessors(node_id)
        )
        if has_tests:
            continue

        # Find the SPEC this CODE implements
        spec_id = None
        spec_content = ""
        for _, target, edge_data in db.graph.out_edges(node_id, data=True):
            if edge_data.get('type') == 'IMPLEMENTS':
                spec_id = target
                spec_content = db.graph.nodes[target].get('content', '')
                break

        if not spec_id:
            continue

        # Generate tests
        context = {
            'spec_node': {'id': spec_id, 'content': spec_content},
            'code_node': {'id': node_id, 'content': data.get('content', '')}
        }

        try:
            result = await test_generator.process(context)

            # Create TEST node
            test_id = f"test_{uuid.uuid4().hex[:8]}"
            db.add_node(
                test_id,
                NodeType.TEST,
                result['content'],
                metadata=result['metadata']
            )

            # Link TEST to CODE
            db.add_edge(
                test_id, node_id,
                'VERIFIES',
                signed_by="test_generator",
                signature=f"auto_{uuid.uuid4().hex[:8]}"
            )

            created_tests.append(test_id)

        except Exception as e:
            print(f"Failed to generate tests for {node_id}: {e}")

    return created_tests

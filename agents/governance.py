"""
GOVERNANCE AGENTS
The 'Police' layer of the Swarm.
"""
from typing import Dict, List
import networkx as nx
from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus, EdgeType

class RealTreasurer(BaseAgent):
    """
    Monitors Cost Ledger. Enforces Prime Directive #8.
    """
    async def process(self, context: Dict) -> Dict:
        # 1. Calculate current spend
        current_spend = self.gateway._cost_session
        budget = self.gateway.config['cost_limits']['project_total_limit_usd']

        # 2. Enforce Limits
        if current_spend > budget:
            return {
                "verdict": "HALT",
                "reason": f"Budget exceeded: ${current_spend} > ${budget}"
            }

        # 3. Forecast (Simple Heuristic)
        remaining = budget - current_spend
        return {
            "verdict": "APPROVE",
            "remaining_budget": remaining,
            "status": "SOLVENT"
        }

class RealSentinel(BaseAgent):
    """
    Monitors Data Flow. Enforces Security.
    """
    async def process(self, context: Dict) -> Dict:
        code_content = context.get('nodes', [{}])[0].get('content', "")

        # 1. Static Analysis (Taint Checking stub)
        suspicious_patterns = ["eval(", "exec(", "subprocess.call(shell=True)"]
        issues = [p for p in suspicious_patterns if p in code_content]

        if issues:
            return {
                "verdict": "REJECT",
                "security_issues": issues,
                "action": "BLOCK_NODE"
            }

        return {"verdict": "SAFE", "scan_depth": "static_only"}


class RealCurator(BaseAgent):
    """
    Manages Graph Integrity. Enforces Prime Directive #11-13.
    - Garbage collection of DEAD_END nodes
    - Orphan detection
    - Citation validation
    """
    async def process(self, context: Dict) -> Dict:
        graph = self.graph_db.graph
        issues = []
        actions_taken = []

        # 1. Find orphan nodes (no incoming edges, not REQ type)
        orphans = []
        for node in graph.nodes():
            if graph.in_degree(node) == 0:
                node_type = graph.nodes[node].get('type')
                if node_type != NodeType.REQ.value:
                    orphans.append(node)

        if orphans:
            issues.append(f"Found {len(orphans)} orphan nodes")

        # 2. Find nodes missing TRACES_TO edges
        missing_traceability = []
        for node in graph.nodes():
            has_trace = any(
                d.get('type') == EdgeType.TRACES_TO.value
                for _, _, d in graph.out_edges(node, data=True)
            )
            node_type = graph.nodes[node].get('type')
            if not has_trace and node_type not in [NodeType.REQ.value, NodeType.STATE.value]:
                missing_traceability.append(node)

        if missing_traceability:
            issues.append(f"Found {len(missing_traceability)} nodes missing traceability")

        # 3. Prune DEAD_END nodes
        dead_ends = [
            n for n, d in graph.nodes(data=True)
            if d.get('type') == NodeType.DEAD_END.value
        ]
        if dead_ends:
            self.graph_db.prune_dead_ends()
            actions_taken.append(f"Pruned {len(dead_ends)} dead-end nodes")

        # 4. Check for cycles (should never happen due to add_edge guard)
        try:
            cycles = list(nx.simple_cycles(graph))
            if cycles:
                issues.append(f"CRITICAL: {len(cycles)} cycles detected!")
        except:
            pass

        return {
            "verdict": "HEALTHY" if not issues else "NEEDS_ATTENTION",
            "issues": issues,
            "actions_taken": actions_taken,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges()
        }


class RealLibrarian(BaseAgent):
    """
    Manages System Prompts & Optimization. Enforces prompt quality.
    - Tracks prompt effectiveness
    - Suggests optimizations
    """
    async def process(self, context: Dict) -> Dict:
        # Analyze prompt templates
        templates = self.templates
        analysis = {}

        for template_name, template_data in templates.items():
            if isinstance(template_data, dict) and 'instruction' in template_data:
                instruction = template_data['instruction']
                analysis[template_name] = {
                    "char_count": len(instruction),
                    "estimated_tokens": len(instruction) // 4,
                    "has_output_schema": 'output_schema' in template_data,
                    "placeholder_count": instruction.count('{')
                }

        # Calculate total prompt budget
        total_tokens = sum(a['estimated_tokens'] for a in analysis.values())

        return {
            "verdict": "OPTIMIZED" if total_tokens < 2000 else "REVIEW_NEEDED",
            "template_analysis": analysis,
            "total_estimated_tokens": total_tokens,
            "recommendations": self._generate_recommendations(analysis)
        }

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        recommendations = []
        for name, data in analysis.items():
            if data['estimated_tokens'] > 500:
                recommendations.append(f"Consider shortening {name} (currently ~{data['estimated_tokens']} tokens)")
            if not data['has_output_schema']:
                recommendations.append(f"Add output_schema to {name} for structured responses")
        return recommendations

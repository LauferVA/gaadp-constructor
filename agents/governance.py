"""
GOVERNANCE AGENTS
The 'Police' layer of the Swarm.
"""
from typing import Dict
from agents.base_agent import BaseAgent
from core.ontology import NodeType, NodeStatus

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

#!/usr/bin/env python3
"""
GAADP SIMULATION ENGINE (Phase 6)
"""
import asyncio
import logging
import shutil
import os
from infrastructure.graph_db import GraphDB
from simulation.mock_agents import MockArchitect, MockBuilder, MockVerifier
from core.ontology import AgentRole, NodeType, EdgeType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SIMULATION")

class SimulationEngine:
    def __init__(self):
        if os.path.exists(".gaadp_sim"): shutil.rmtree(".gaadp_sim")
        os.makedirs(".gaadp_sim")
        self.db = GraphDB(persistence_path=".gaadp_sim/graph.pkl")
        self.architect = MockArchitect("arch_sim", AgentRole.ARCHITECT, self.db)
        self.builder = MockBuilder("build_sim", AgentRole.BUILDER, self.db)
        self.verifier = MockVerifier("verif_sim", AgentRole.VERIFIER, self.db)

    async def run_scenario(self):
        logger.info("--- STARTING SIMULATION ---")
        req_id = "req_root_01"
        self.db.add_node(req_id, NodeType.REQ, "System needs Auth")

        arch_result = await self.architect.process({"nodes": [{"id": req_id}]})
        plan_id, spec_id = None, None
        for n in arch_result['new_nodes']:
            self.db.add_node(n['id'], n['type'], n['content'])
            if n['type'] == NodeType.PLAN: plan_id = n['id']
            if n['type'] == NodeType.SPEC: spec_id = n['id']

        self.db.add_edge(plan_id, req_id, EdgeType.TRACES_TO, self.architect.agent_id, self.architect.sign_content(plan_id))
        logger.info(f"   -> Created SPEC: {spec_id}")

        code_id = "code_01"
        self.db.add_node(code_id, NodeType.CODE, "def auth(): pass", {"file_path": "auth.py"})
        self.db.add_edge(code_id, spec_id, EdgeType.IMPLEMENTS, self.builder.agent_id, self.builder.sign_content(code_id))
        logger.info(f"   -> Generated CODE: {code_id}")

        verify_result = await self.verifier.process({"nodes": [{"id": code_id}]})
        if verify_result['verdict'] == "PASS":
            self.db.add_edge(code_id, code_id, EdgeType.VERIFIES, self.verifier.agent_id, self.verifier.sign_content(code_id))
            logger.info("   -> Code VERIFIED and Signed.")
        logger.info("--- SIMULATION COMPLETE ---")

if __name__ == "__main__":
    asyncio.run(SimulationEngine().run_scenario())

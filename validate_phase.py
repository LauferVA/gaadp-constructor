#!/usr/bin/env python3
"""
PHASE VALIDATOR (SMOKE TEST)
"""
import os
import shutil
from core.ontology import NodeType, EdgeType, AgentRole
from infrastructure.graph_db import GraphDB
from agents.base_agent import BaseAgent

class MockBuilder(BaseAgent):
    async def process(self, context): return {"status": "done"}

def test_phase_1_foundation():
    print("\nTEST: Phase 1 (Foundation)")
    try:
        db_path = ".gaadp_test/graph.pkl"
        if os.path.exists(".gaadp_test"): shutil.rmtree(".gaadp_test")
        os.makedirs(".gaadp_test")
        db = GraphDB(persistence_path=db_path)
        db.add_node("test_spec_1", NodeType.SPEC, "Login Requirements")
        db.add_node("test_code_1", NodeType.CODE, "def login(): pass")
        db.add_edge("test_code_1", "test_spec_1", EdgeType.IMPLEMENTS, "mock_agent", "mock_sig_123")
        print("  GraphDB accepted Typed Nodes/Edges")

        try:
            db.add_edge("test_spec_1", "test_code_1", EdgeType.DEPENDS_ON, "mock", "sig")
            db.add_edge("test_code_1", "test_spec_1", EdgeType.DEPENDS_ON, "mock", "sig")
            print("  FAILED: GraphDB allowed a cycle!")
        except ValueError:
            print("  GraphDB prevented a cycle")
    except Exception as e:
        print(f"  CRASH: {e}")
        raise e

def test_phase_2_communication():
    print("\nTEST: Phase 2 (Identity)")
    try:
        db = GraphDB(persistence_path=".gaadp_test/graph.pkl")
        agent = MockBuilder("agent_01", AgentRole.BUILDER, db)
        sig = agent.sign_content({"code": "print('hello')"})
        if len(sig) > 10:
            print(f"  Runtime generated Ed25519 Signature: {sig[:10]}...")
    except Exception as e:
        print(f"  CRASH: {e}")
        raise e

if __name__ == "__main__":
    test_phase_1_foundation()
    test_phase_2_communication()

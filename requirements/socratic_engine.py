"""
SOCRATIC QUESTION ENGINE
"""
import uuid
from typing import Dict, Any
from core.ontology import NodeType, EdgeType, NodeStatus

class SocraticEngine:
    def analyze_gap(self, gap_description: str, parent_req_id: str) -> Dict[str, Any]:
        question_id = f"req_{uuid.uuid4()}"
        source = "USER"
        if any(x in gap_description.lower() for x in ["api", "format", "syntax"]):
            source = "SEARCH"

        node_data = {
            "id": question_id,
            "type": NodeType.REQ.value,
            "status": NodeStatus.PENDING.value,
            "content": gap_description,
            "metadata": {"source_type": source, "ambiguity_score": 1.0, "parent_id": parent_req_id}
        }
        edge_data = {
            "source": question_id,
            "target": parent_req_id,
            "type": EdgeType.TRACES_TO.value
        }
        return {"node": node_data, "edge": edge_data}

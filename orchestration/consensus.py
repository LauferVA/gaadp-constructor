"""
CONSENSUS MANAGER
Implements multi-agent verification with quorum-based consensus.
Satisfies Prime Directive #6: Minimum 2 verifier signatures required.
"""
import asyncio
import logging
import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus
from core.ontology import EdgeType, NodeStatus

logger = logging.getLogger("Consensus")


class ConsensusVerdict(str, Enum):
    """Consensus decision outcomes."""
    UNANIMOUS_PASS = "UNANIMOUS_PASS"      # All verifiers agree: PASS
    QUORUM_PASS = "QUORUM_PASS"            # Quorum agrees: PASS
    QUORUM_FAIL = "QUORUM_FAIL"            # Quorum agrees: FAIL
    NO_CONSENSUS = "NO_CONSENSUS"          # No quorum reached
    UNANIMOUS_FAIL = "UNANIMOUS_FAIL"      # All verifiers agree: FAIL


@dataclass
class VerifierResult:
    """Result from a single verifier."""
    verifier_id: str
    verdict: str  # PASS, FAIL, CONDITIONAL
    critique: str
    signature: str
    timestamp: str


@dataclass
class ConsensusResult:
    """Aggregated consensus from multiple verifiers."""
    consensus_verdict: ConsensusVerdict
    verifier_results: List[VerifierResult]
    pass_count: int
    fail_count: int
    conditional_count: int
    threshold_met: bool
    aggregated_critique: str
    multi_signature: str  # Combined signature


class ConsensusManager:
    """
    Manages multi-agent verification with quorum-based consensus.

    Implements Prime Directive #6:
    "No CODE node reaches VERIFIED status without ≥2 verifier signatures."
    """

    def __init__(
        self,
        db: GraphDB,
        event_bus: EventBus,
        quorum_threshold: float = 0.66,
        minimum_verifiers: int = 2
    ):
        self.db = db
        self.event_bus = event_bus
        self.quorum_threshold = quorum_threshold
        self.minimum_verifiers = minimum_verifiers

    async def achieve_consensus(
        self,
        node_id: str,
        verifiers: List[Any],
        timeout: float = 120
    ) -> ConsensusResult:
        """
        Run multiple verifiers and achieve consensus.

        Args:
            node_id: Node to verify
            verifiers: List of verifier agent instances
            timeout: Timeout per verifier in seconds

        Returns:
            ConsensusResult with aggregated verdict
        """
        if len(verifiers) < self.minimum_verifiers:
            raise ValueError(
                f"Minimum {self.minimum_verifiers} verifiers required, "
                f"got {len(verifiers)}"
            )

        # Get node content
        if node_id not in self.db.graph:
            raise ValueError(f"Node {node_id} not found")

        node_data = self.db.graph.nodes[node_id]
        content = node_data.get('content', '')

        # Run all verifiers in parallel
        tasks = [
            self._run_single_verifier(verifier, node_id, content, timeout)
            for verifier in verifiers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        verifier_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Verifier {i} failed: {result}")
                # Create a FAIL result for failed verifiers
                verifier_results.append(VerifierResult(
                    verifier_id=f"verifier_{i}",
                    verdict="FAIL",
                    critique=f"Verifier crashed: {str(result)}",
                    signature="FAILED",
                    timestamp="",
                ))
            else:
                verifier_results.append(result)

        # Calculate consensus
        consensus = self._calculate_consensus(verifier_results)

        # Publish consensus event
        await self.event_bus.publish(
            topic="consensus",
            message_type="CONSENSUS_ACHIEVED",
            payload={
                "node_id": node_id,
                "verdict": consensus.consensus_verdict.value,
                "pass_count": consensus.pass_count,
                "fail_count": consensus.fail_count,
                "verifier_count": len(verifier_results)
            },
            source_id="consensus_manager"
        )

        return consensus

    async def _run_single_verifier(
        self,
        verifier: Any,
        node_id: str,
        content: str,
        timeout: float
    ) -> VerifierResult:
        """Run a single verifier with timeout."""
        try:
            result = await asyncio.wait_for(
                verifier.process({"nodes": [{"id": node_id, "content": content}]}),
                timeout=timeout
            )

            return VerifierResult(
                verifier_id=verifier.agent_id,
                verdict=result.get('verdict', 'FAIL'),
                critique=result.get('critique', ''),
                signature=verifier.sign_content(node_id),
                timestamp=self.db.graph.nodes[node_id].get('created_at', '')
            )

        except asyncio.TimeoutError:
            return VerifierResult(
                verifier_id=verifier.agent_id,
                verdict="FAIL",
                critique=f"Verifier timed out after {timeout}s",
                signature="TIMEOUT",
                timestamp=""
            )

    def _calculate_consensus(
        self,
        results: List[VerifierResult]
    ) -> ConsensusResult:
        """Calculate consensus from verifier results."""
        pass_count = sum(1 for r in results if r.verdict == "PASS")
        fail_count = sum(1 for r in results if r.verdict == "FAIL")
        conditional_count = sum(1 for r in results if r.verdict == "CONDITIONAL")

        total = len(results)
        pass_ratio = pass_count / total
        fail_ratio = fail_count / total

        # Determine consensus verdict
        if pass_count == total:
            verdict = ConsensusVerdict.UNANIMOUS_PASS
        elif fail_count == total:
            verdict = ConsensusVerdict.UNANIMOUS_FAIL
        elif pass_ratio >= self.quorum_threshold:
            verdict = ConsensusVerdict.QUORUM_PASS
        elif fail_ratio >= self.quorum_threshold:
            verdict = ConsensusVerdict.QUORUM_FAIL
        else:
            verdict = ConsensusVerdict.NO_CONSENSUS

        # Aggregate critiques
        all_critiques = [r.critique for r in results if r.critique]
        aggregated_critique = "\n\n".join([
            f"[{r.verifier_id}]: {r.critique}"
            for r in results if r.critique
        ])

        # Create multi-signature
        signatures = [r.signature for r in results if r.signature != "FAILED"]
        multi_signature = "|".join(signatures)

        threshold_met = (
            verdict in [ConsensusVerdict.UNANIMOUS_PASS, ConsensusVerdict.QUORUM_PASS]
            or verdict == ConsensusVerdict.UNANIMOUS_FAIL
        )

        return ConsensusResult(
            consensus_verdict=verdict,
            verifier_results=results,
            pass_count=pass_count,
            fail_count=fail_count,
            conditional_count=conditional_count,
            threshold_met=threshold_met,
            aggregated_critique=aggregated_critique,
            multi_signature=multi_signature
        )

    def apply_consensus_to_graph(
        self,
        node_id: str,
        consensus: ConsensusResult
    ):
        """
        Apply consensus result to graph with multi-signatures.

        Args:
            node_id: The verified node
            consensus: Consensus result
        """
        if consensus.consensus_verdict in [
            ConsensusVerdict.UNANIMOUS_PASS,
            ConsensusVerdict.QUORUM_PASS
        ]:
            # Mark as VERIFIED
            self.db.set_status(node_id, NodeStatus.VERIFIED, reason="Consensus: PASS")

            # Add VERIFIES edges from each passing verifier
            for result in consensus.verifier_results:
                if result.verdict in ["PASS", "CONDITIONAL"]:
                    edge_id = f"verify_{uuid.uuid4().hex[:8]}"
                    prev_hash = self.db.get_last_node_hash()

                    try:
                        self.db.add_edge(
                            result.verifier_id,
                            node_id,
                            EdgeType.VERIFIES,
                            signed_by=result.verifier_id,
                            signature=result.signature,
                            previous_hash=prev_hash
                        )
                    except Exception as e:
                        logger.error(f"Failed to add verification edge: {e}")

            # Store multi-signature on node
            self.db.graph.nodes[node_id]['multi_signature'] = consensus.multi_signature
            self.db.graph.nodes[node_id]['verifier_count'] = len(consensus.verifier_results)
            self.db._persist()

            logger.info(f"Node {node_id} VERIFIED by consensus ({consensus.pass_count}/{len(consensus.verifier_results)})")

        else:
            # Consensus failed
            self.db.set_status(node_id, NodeStatus.FAILED, reason="Consensus: FAIL")
            self.db.graph.nodes[node_id]['consensus_critique'] = consensus.aggregated_critique
            self.db._persist()

            logger.warning(f"Node {node_id} FAILED consensus ({consensus.fail_count}/{len(consensus.verifier_results)})")

    def get_verification_status(self, node_id: str) -> Optional[Dict]:
        """Get verification status for a node."""
        if node_id not in self.db.graph:
            return None

        node_data = self.db.graph.nodes[node_id]

        # Count VERIFIES edges
        verifier_count = sum(
            1 for _, target, data in self.db.graph.in_edges(node_id, data=True)
            if data.get('type') == EdgeType.VERIFIES.value
        )

        return {
            "node_id": node_id,
            "status": node_data.get('status'),
            "verifier_count": verifier_count,
            "multi_signature": node_data.get('multi_signature'),
            "meets_requirements": verifier_count >= self.minimum_verifiers
        }

"""
HUMAN-IN-THE-LOOP
Pause points, clarification requests, and approval gates.
"""
import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from infrastructure.graph_db import GraphDB
from infrastructure.event_bus import EventBus
from core.ontology import NodeType, NodeStatus

logger = logging.getLogger("HumanLoop")


class InteractionType(str, Enum):
    """Types of human interactions."""
    CLARIFICATION = "CLARIFICATION"      # Socrates asks for clarification
    APPROVAL = "APPROVAL"                 # Approve/reject critical decisions
    REVIEW = "REVIEW"                     # Review generated code/specs
    CHOICE = "CHOICE"                     # Multiple options to choose from
    FREE_INPUT = "FREE_INPUT"             # Open-ended input


@dataclass
class HumanRequest:
    """A request for human input."""
    id: str
    interaction_type: InteractionType
    node_id: str
    question: str
    options: List[str] = field(default_factory=list)  # For CHOICE/APPROVAL
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved: bool = False
    response: Optional[str] = None
    resolved_at: Optional[datetime] = None


class HumanLoopController:
    """
    Manages human-in-the-loop interactions.

    Features:
    - Pause points: Block pipeline until human responds
    - Clarification: Socrates-driven question asking
    - Approval gates: Require sign-off on critical changes
    - Multiple response modes: CLI, callback, or async await
    """

    def __init__(
        self,
        db: GraphDB,
        event_bus: EventBus,
        auto_approve: bool = False,
        timeout_seconds: int = 300
    ):
        self.db = db
        self.event_bus = event_bus
        self.auto_approve = auto_approve  # For testing/CI
        self.timeout_seconds = timeout_seconds

        self._pending_requests: Dict[str, HumanRequest] = {}
        self._response_events: Dict[str, asyncio.Event] = {}
        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable[[HumanRequest], None]):
        """Register a callback for new requests (for UI integration)."""
        self._callbacks.append(callback)

    async def request_clarification(
        self,
        node_id: str,
        question: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Request clarification from human (Socrates integration).

        Args:
            node_id: Related node ID
            question: The clarification question
            context: Additional context

        Returns:
            Human's response
        """
        request = HumanRequest(
            id=f"clarify_{uuid.uuid4().hex[:8]}",
            interaction_type=InteractionType.CLARIFICATION,
            node_id=node_id,
            question=question,
            context=context or {}
        )

        return await self._submit_and_wait(request)

    async def request_approval(
        self,
        node_id: str,
        description: str,
        options: List[str] = None
    ) -> str:
        """
        Request approval for a critical decision.

        Args:
            node_id: Related node ID
            description: What needs approval
            options: Optional list of choices (default: ["Approve", "Reject"])

        Returns:
            Selected option
        """
        request = HumanRequest(
            id=f"approve_{uuid.uuid4().hex[:8]}",
            interaction_type=InteractionType.APPROVAL,
            node_id=node_id,
            question=description,
            options=options or ["Approve", "Reject"]
        )

        if self.auto_approve:
            logger.info(f"Auto-approving: {description[:50]}...")
            return "Approve"

        return await self._submit_and_wait(request)

    async def request_choice(
        self,
        node_id: str,
        question: str,
        options: List[str]
    ) -> str:
        """
        Present multiple options for human to choose.

        Args:
            node_id: Related node ID
            question: The question
            options: List of choices

        Returns:
            Selected option
        """
        request = HumanRequest(
            id=f"choice_{uuid.uuid4().hex[:8]}",
            interaction_type=InteractionType.CHOICE,
            node_id=node_id,
            question=question,
            options=options
        )

        return await self._submit_and_wait(request)

    async def request_review(
        self,
        node_id: str,
        content: str,
        prompt: str = "Review and provide feedback:"
    ) -> str:
        """
        Request human review of content.

        Args:
            node_id: Related node ID
            content: Content to review
            prompt: Review prompt

        Returns:
            Human's feedback
        """
        request = HumanRequest(
            id=f"review_{uuid.uuid4().hex[:8]}",
            interaction_type=InteractionType.REVIEW,
            node_id=node_id,
            question=prompt,
            context={'content': content}
        )

        return await self._submit_and_wait(request)

    async def _submit_and_wait(self, request: HumanRequest) -> str:
        """Submit a request and wait for response."""
        self._pending_requests[request.id] = request
        self._response_events[request.id] = asyncio.Event()

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(request)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # Publish event
        await self.event_bus.publish(
            topic="human_loop",
            message_type="REQUEST_CREATED",
            payload={
                "request_id": request.id,
                "type": request.interaction_type.value,
                "question": request.question,
                "node_id": request.node_id
            },
            source_id="human_loop"
        )

        logger.info(f"Awaiting human input: {request.question[:50]}...")

        # Wait for response with timeout
        try:
            await asyncio.wait_for(
                self._response_events[request.id].wait(),
                timeout=self.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Request {request.id} timed out")
            # Clean up
            del self._pending_requests[request.id]
            del self._response_events[request.id]
            raise TimeoutError(f"Human response timeout after {self.timeout_seconds}s")

        response = request.response

        # Clean up
        del self._pending_requests[request.id]
        del self._response_events[request.id]

        return response

    def respond(self, request_id: str, response: str):
        """
        Provide a response to a pending request.

        Args:
            request_id: The request ID
            response: Human's response
        """
        if request_id not in self._pending_requests:
            raise ValueError(f"No pending request with ID {request_id}")

        request = self._pending_requests[request_id]
        request.response = response
        request.resolved = True
        request.resolved_at = datetime.utcnow()

        # Signal the waiting coroutine
        self._response_events[request_id].set()

        logger.info(f"Request {request_id} resolved: {response[:50]}...")

    def get_pending_requests(self) -> List[HumanRequest]:
        """Get all pending requests."""
        return list(self._pending_requests.values())

    def get_request(self, request_id: str) -> Optional[HumanRequest]:
        """Get a specific request."""
        return self._pending_requests.get(request_id)

    def cli_respond(self):
        """
        Interactive CLI mode for responding to requests.
        Call this in a separate thread/process for interactive use.
        """
        while True:
            pending = self.get_pending_requests()
            if not pending:
                print("No pending requests. Waiting...")
                import time
                time.sleep(2)
                continue

            print("\n" + "=" * 60)
            print("PENDING HUMAN REQUESTS:")
            print("=" * 60)

            for i, req in enumerate(pending):
                print(f"\n[{i+1}] {req.interaction_type.value} (ID: {req.id})")
                print(f"    Node: {req.node_id}")
                print(f"    Question: {req.question}")
                if req.options:
                    print(f"    Options: {', '.join(req.options)}")

            try:
                choice = input("\nEnter request number to respond (or 'q' to quit): ")
                if choice.lower() == 'q':
                    break

                idx = int(choice) - 1
                if 0 <= idx < len(pending):
                    req = pending[idx]

                    if req.options:
                        print(f"\nOptions: {req.options}")
                        response = input("Select option: ")
                    else:
                        response = input("Enter response: ")

                    self.respond(req.id, response)
                    print(f"Response submitted for {req.id}")

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}")
            except KeyboardInterrupt:
                break


class PausePoint:
    """
    A pause point in the pipeline that blocks until released.
    """

    def __init__(self, name: str, human_loop: HumanLoopController):
        self.name = name
        self.human_loop = human_loop
        self._gate = asyncio.Event()
        self._enabled = True

    def enable(self):
        """Enable the pause point."""
        self._enabled = True

    def disable(self):
        """Disable the pause point (auto-pass)."""
        self._enabled = False

    async def wait(self, node_id: str, context: str = "") -> bool:
        """
        Wait at this pause point until human releases.

        Returns:
            True if approved, False if rejected
        """
        if not self._enabled:
            return True

        description = f"Pause Point: {self.name}"
        if context:
            description += f"\n{context}"

        response = await self.human_loop.request_approval(
            node_id=node_id,
            description=description,
            options=["Continue", "Abort"]
        )

        return response == "Continue"


# Pre-defined pause points
class PausePoints:
    """Factory for common pause points."""

    @staticmethod
    def before_implementation(human_loop: HumanLoopController) -> PausePoint:
        """Pause before Builder starts implementing."""
        return PausePoint("Pre-Implementation Review", human_loop)

    @staticmethod
    def before_verification(human_loop: HumanLoopController) -> PausePoint:
        """Pause before Verifier reviews code."""
        return PausePoint("Pre-Verification Check", human_loop)

    @staticmethod
    def before_materialization(human_loop: HumanLoopController) -> PausePoint:
        """Pause before writing to filesystem."""
        return PausePoint("Pre-Materialization Approval", human_loop)

    @staticmethod
    def on_budget_warning(human_loop: HumanLoopController) -> PausePoint:
        """Pause when budget threshold reached."""
        return PausePoint("Budget Warning", human_loop)

"""
ASYNC EVENT BUS
Decoupled messaging system for the Swarm.
"""
import asyncio
import logging
import uuid
import datetime
from typing import Dict, Callable, List, Any

logger = logging.getLogger("EventBus")

class EventBus:
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.queue: asyncio.Queue = asyncio.Queue()
        self.history: List[Dict[str, Any]] = []
        self._running = False

    async def start(self):
        """Run the event loop consumer"""
        logger.info("Event Bus Started")
        self._running = True
        while self._running:
            try:
                event = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                await self._dispatch(event)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue

    def stop(self):
        """Stop the event bus"""
        self._running = False

    def subscribe(self, topic: str, callback: Callable):
        """Subscribe a callback to a topic"""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
        logger.debug(f"Subscribed to {topic}")

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe a callback from a topic"""
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)

    async def publish(self, topic: str, message_type: str, payload: Dict, source_id: str):
        """Publish an event to a topic"""
        event = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "topic": topic,
            "type": message_type,
            "source": source_id,
            "payload": payload
        }
        self.history.append(event)
        await self.queue.put(event)
        logger.debug(f"Published {message_type} to {topic}")

    async def _dispatch(self, event: Dict):
        """Dispatch event to all subscribers"""
        topic = event['topic']
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    asyncio.create_task(callback(event))
                except Exception as e:
                    logger.error(f"Handler failed: {e}")

    def get_history(self, topic: str = None, limit: int = 100) -> List[Dict]:
        """Retrieve event history, optionally filtered by topic"""
        if topic:
            filtered = [e for e in self.history if e['topic'] == topic]
            return filtered[-limit:]
        return self.history[-limit:]


# Message type constants matching event_bus.schema
class MessageType:
    TASK_ASSIGN = "TASK_ASSIGN"
    RESULT_SUBMIT = "RESULT_SUBMIT"
    VERIFY_REQ = "VERIFY_REQ"
    ERROR = "ERROR"
    STOP_SIGNAL = "STOP_SIGNAL"

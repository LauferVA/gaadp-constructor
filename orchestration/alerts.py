"""
ALERT HANDLER
Subscribes to governance alerts and escalates to human when critical.
"""
import asyncio
import logging
from typing import Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime

from infrastructure.event_bus import EventBus

logger = logging.getLogger("AlertHandler")


@dataclass
class Alert:
    """An alert from governance middleware."""
    id: str
    severity: str  # CRITICAL, WARNING, INFO
    source: str    # treasurer, sentinel, curator
    message: str
    action_required: str
    payload: dict
    timestamp: datetime


class AlertHandler:
    """
    Handles governance alerts and escalates critical issues.

    Features:
    - Subscribes to 'alerts' topic on EventBus
    - Maintains alert history
    - Escalates CRITICAL alerts to HumanLoopController
    - Supports custom alert callbacks (webhooks, Slack, etc.)
    """

    def __init__(self, event_bus: EventBus, human_loop: Optional[any] = None):
        self.event_bus = event_bus
        self.human_loop = human_loop
        self.alerts: List[Alert] = []
        self._callbacks: List[Callable] = []
        self._running = False

        # Subscribe to alerts topic
        self.event_bus.subscribe("alerts", self._handle_alert)

    def register_callback(self, callback: Callable[[Alert], None]):
        """Register a callback for alerts (webhooks, Slack, etc.)."""
        self._callbacks.append(callback)

    async def _handle_alert(self, event: dict):
        """Handle incoming alert events."""
        payload = event.get('payload', {})
        source = event.get('source', 'unknown')
        msg_type = event.get('type', '')

        # Parse severity from message type (e.g., "TREASURER_CRITICAL")
        severity = "INFO"
        if "CRITICAL" in msg_type:
            severity = "CRITICAL"
        elif "WARNING" in msg_type:
            severity = "WARNING"

        alert = Alert(
            id=event.get('id', ''),
            severity=severity,
            source=source,
            message=payload.get('message', ''),
            action_required=payload.get('action_required', ''),
            payload=payload,
            timestamp=datetime.utcnow()
        )

        self.alerts.append(alert)
        logger.info(f"[{severity}] {source}: {alert.message}")

        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

        # Escalate critical alerts to human
        if severity == "CRITICAL" and self.human_loop:
            await self._escalate_to_human(alert)

    async def _escalate_to_human(self, alert: Alert):
        """Escalate critical alert to HumanLoopController."""
        try:
            response = await self.human_loop.request_approval(
                node_id=alert.payload.get('node_id', 'system'),
                description=f"""
🚨 CRITICAL ALERT from {alert.source.upper()}

{alert.message}

Action Required: {alert.action_required}

Details:
- Task ID: {alert.payload.get('task_id', 'N/A')}
- Node ID: {alert.payload.get('node_id', 'N/A')}
""",
                options=["Acknowledge & Continue", "Pause Pipeline", "Abort"]
            )

            logger.info(f"Human response to alert {alert.id}: {response}")

            # Handle response
            if response == "Abort":
                # Could trigger engine.stop() via event
                await self.event_bus.publish(
                    topic="control",
                    message_type="ABORT_REQUESTED",
                    payload={"reason": f"Human aborted after alert: {alert.message}"},
                    source_id="alert_handler"
                )
            elif response == "Pause Pipeline":
                await self.event_bus.publish(
                    topic="control",
                    message_type="PAUSE_REQUESTED",
                    payload={"reason": f"Human paused after alert: {alert.message}"},
                    source_id="alert_handler"
                )

        except Exception as e:
            logger.error(f"Failed to escalate alert to human: {e}")

    def get_alerts(self, severity: Optional[str] = None, limit: int = 100) -> List[Alert]:
        """Get alert history, optionally filtered by severity."""
        alerts = self.alerts
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def get_critical_count(self) -> int:
        """Get count of unacknowledged critical alerts."""
        return sum(1 for a in self.alerts if a.severity == "CRITICAL")

    def clear_alerts(self):
        """Clear alert history."""
        self.alerts = []

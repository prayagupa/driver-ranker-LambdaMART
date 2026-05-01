"""Kafka consumer for real-time driver GPS and trip events."""

import json
import logging
from typing import Callable

logger = logging.getLogger(__name__)


class DriverEventConsumer:
    """Consumes driver GPS and status events from Kafka."""

    def __init__(self, bootstrap_servers: str, topic: str = "driver-events", group_id: str = "feature-engine"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.group_id = group_id
        self._consumer = None

    def _create_consumer(self):
        from confluent_kafka import Consumer

        self._consumer = Consumer({
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "auto.offset.reset": "latest",
        })
        self._consumer.subscribe([self.topic])

    def consume(self, handler: Callable[[dict], None], max_messages: int | None = None):
        """Consume messages and pass to handler. Set max_messages for testing."""
        if self._consumer is None:
            self._create_consumer()

        count = 0
        try:
            while max_messages is None or count < max_messages:
                msg = self._consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

                event = json.loads(msg.value().decode("utf-8"))
                handler(event)
                count += 1
        finally:
            if self._consumer:
                self._consumer.close()

    def close(self):
        if self._consumer:
            self._consumer.close()
            self._consumer = None

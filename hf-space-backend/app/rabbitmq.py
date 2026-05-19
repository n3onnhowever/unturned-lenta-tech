from __future__ import annotations

import json
import asyncio
from typing import Any

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message
from aiormq.exceptions import AMQPConnectionError

from .config import get_settings

STAGE_TO_QUEUE = {
    "detect": "pipeline.detect",
    "classify": "pipeline.classify",
    "ocr": "pipeline.ocr",
    "finalize": "pipeline.finalize",
}

STAGE_ORDER = ["detect", "classify", "ocr", "finalize"]


class RabbitMQUnavailable(RuntimeError):
    """Raised when RabbitMQ cannot be reached."""


async def create_channel() -> tuple[aio_pika.abc.AbstractRobustConnection, aio_pika.abc.AbstractChannel]:
    settings = get_settings()
    try:
        connection = await aio_pika.connect_robust(settings.rabbitmq_url, heartbeat=600)
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)
        return connection, channel
    except AMQPConnectionError as exc:
        raise RabbitMQUnavailable(str(exc)) from exc


async def declare_topology(channel: aio_pika.abc.AbstractChannel) -> aio_pika.abc.AbstractExchange:
    settings = get_settings()
    exchange = await channel.declare_exchange(
        settings.exchange_name,
        ExchangeType.DIRECT,
        durable=True,
    )
    for routing_key in STAGE_TO_QUEUE.values():
        queue = await channel.declare_queue(routing_key, durable=True)
        await queue.bind(exchange, routing_key=routing_key)
    return exchange


async def publish_stage(stage: str, payload: dict[str, Any]) -> None:
    connection, channel = await create_channel()
    try:
        exchange = await declare_topology(channel)
        message = Message(
            body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
        )
        await exchange.publish(message, routing_key=STAGE_TO_QUEUE[stage])
    finally:
        await channel.close()
        await connection.close()


async def wait_for_rabbitmq(retry_delay_sec: float = 3.0) -> tuple[
    aio_pika.abc.AbstractRobustConnection,
    aio_pika.abc.AbstractChannel,
]:
    while True:
        try:
            return await create_channel()
        except RabbitMQUnavailable:
            await asyncio.sleep(retry_delay_sec)

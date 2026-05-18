from __future__ import annotations

import argparse
import json
import asyncio
from typing import Any

from aio_pika import IncomingMessage

from .db import get_job, init_db, update_job_state
from .processors import PROCESSORS
from .rabbitmq import STAGE_ORDER, STAGE_TO_QUEUE, declare_topology, publish_stage, wait_for_rabbitmq


def next_stage(current_stage: str) -> str | None:
    index = STAGE_ORDER.index(current_stage)
    return STAGE_ORDER[index + 1] if index + 1 < len(STAGE_ORDER) else None


async def handle_stage(stage: str, payload: dict[str, Any]) -> None:
    job_id = payload["job_id"]
    print(f"[worker:{stage}] started job {job_id}", flush=True)
    update_job_state(job_id, status="processing", current_stage=stage, payload=payload)
    processed = await asyncio.to_thread(PROCESSORS[stage], payload)

    if stage == "finalize":
        update_job_state(
            job_id,
            status="completed",
            current_stage=stage,
            payload=processed,
            result_json_path=processed.get("result_json_path"),
            result_csv_path=processed.get("result_csv_path"),
            error_message=None,
        )
        print(f"[worker:{stage}] completed job {job_id}", flush=True)
        return

    next_stage_name = next_stage(stage)
    if not next_stage_name:
        raise RuntimeError(f"No next stage configured after {stage}")
    update_job_state(job_id, status="queued", current_stage=next_stage_name, payload=processed)
    await publish_stage(next_stage_name, processed)
    print(f"[worker:{stage}] queued job {job_id} for {next_stage_name}", flush=True)


async def on_message(stage: str, message: IncomingMessage) -> None:
    async with message.process(requeue=False):
        payload = json.loads(message.body.decode("utf-8"))
        job = get_job(payload["job_id"])
        if job is None:
            return
        try:
            await handle_stage(stage, payload)
        except Exception as exc:
            print(f"[worker:{stage}] failed job {payload['job_id']}: {exc}", flush=True)
            update_job_state(
                payload["job_id"],
                status="failed",
                current_stage=stage,
                payload=payload,
                error_message=str(exc),
            )
            raise


async def consume_stage(stage: str) -> None:
    init_db()
    connection, channel = await wait_for_rabbitmq()
    try:
        await declare_topology(channel)
        queue = await channel.declare_queue(STAGE_TO_QUEUE[stage], durable=True)
        await queue.consume(lambda message: on_message(stage, message))
        print(f"Worker for stage '{stage}' is waiting for messages.")
        await channel.closed()
    finally:
        await channel.close()
        await connection.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one RabbitMQ stage worker.")
    parser.add_argument("--stage", choices=STAGE_ORDER, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(consume_stage(args.stage))

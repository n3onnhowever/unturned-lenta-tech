from __future__ import annotations

import asyncio
from typing import Any

from .db import update_job_state
from .processors import PROCESSORS
from .rabbitmq import STAGE_ORDER


async def run_inline_pipeline(payload: dict[str, Any]) -> None:
    """Run the same pipeline stages without RabbitMQ for single-service hosts."""
    job_id = payload["job_id"]
    current_payload = payload
    for stage in STAGE_ORDER:
        try:
            update_job_state(job_id, status="processing", current_stage=stage, payload=current_payload)
            current_payload = await asyncio.to_thread(PROCESSORS[stage], current_payload)
            if stage == "finalize":
                update_job_state(
                    job_id,
                    status="completed",
                    current_stage=stage,
                    payload=current_payload,
                    result_json_path=current_payload.get("result_json_path"),
                    result_csv_path=current_payload.get("result_csv_path"),
                    error_message=None,
                )
                return
            next_stage = STAGE_ORDER[STAGE_ORDER.index(stage) + 1]
            update_job_state(job_id, status="queued", current_stage=next_stage, payload=current_payload)
        except Exception as exc:
            update_job_state(
                job_id,
                status="failed",
                current_stage=stage,
                payload=current_payload,
                error_message=str(exc),
            )
            raise

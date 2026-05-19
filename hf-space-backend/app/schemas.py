from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    current_stage: str


class JobStatusResponse(BaseModel):
    job_id: str
    filename: str
    status: str
    current_stage: str
    error_message: str | None
    result_json_url: str | None
    result_csv_url: str | None
    payload: dict[str, Any]
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    rabbitmq_exchange: str


class MLReadinessResponse(BaseModel):
    """Local ML assets expected by the pipeline."""

    bundle_dir: str
    weights_path: str
    weights_present: bool
    upscale_model_path: str | None
    upscale_present: bool
    scripts_present: bool
    ready: bool
    detail: str | None = None

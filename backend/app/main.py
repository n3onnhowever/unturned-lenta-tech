from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import get_settings
from .db import create_job, get_job, init_db, update_job_state
from .ml_bundle import bundle_scripts_dir
from .rabbitmq import RabbitMQUnavailable, publish_stage
from .schemas import HealthResponse, JobCreateResponse, JobStatusResponse, MLReadinessResponse
from .storage import allocate_job_id, save_upload

app = FastAPI(title="Lenta Tech Hackathon BFF", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", rabbitmq_exchange=settings.exchange_name)


@app.get("/health/ml", response_model=MLReadinessResponse)
async def health_ml() -> MLReadinessResponse:
    """Verify YOLO weights, optional upscale model, and bundled scripts exist."""
    settings = get_settings()
    bundle_dir = settings.ml_bundle_dir
    weights = settings.ml_weights_path
    weights_ok = weights.is_file()
    upscale_path = settings.ml_upscale_model_path
    upscale_ok = bool(upscale_path and upscale_path.is_file())
    scripts_dir = bundle_scripts_dir(settings)
    scripts_ok = scripts_dir.is_dir()

    ready = weights_ok and scripts_ok
    detail: str | None = None
    if not bundle_dir.is_dir():
        detail = f"ML bundle directory missing: {bundle_dir}"
        ready = False
    elif not scripts_ok:
        detail = f"ML scripts directory missing: {scripts_dir}"
    elif not weights_ok:
        detail = (
            f"YOLO weights not found: {weights}. "
            "Place price_tag_merged_internal_best.pt under lenta-hackathon-main/weights/ "
            "or set ML_WEIGHTS_PATH."
        )

    return MLReadinessResponse(
        bundle_dir=str(bundle_dir.resolve()),
        weights_path=str(weights.resolve()) if weights.exists() else str(weights),
        weights_present=weights_ok,
        upscale_model_path=str(upscale_path.resolve()) if upscale_path and upscale_path.exists() else None,
        upscale_present=upscale_ok,
        scripts_present=scripts_ok,
        ready=ready,
        detail=detail,
    )


@app.get("/")
async def ui_index() -> FileResponse:
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.post("/jobs/upload", response_model=JobCreateResponse)
async def upload_job(file: UploadFile = File(...)) -> JobCreateResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files are supported")

    job_id = allocate_job_id()
    saved_path = save_upload(job_id, file)
    payload = {
        "job_id": job_id,
        "filename": file.filename,
        "video_path": str(saved_path),
    }
    create_job(job_id, file.filename, str(saved_path), payload)
    try:
        await publish_stage("detect", payload)
    except RabbitMQUnavailable as exc:
        update_job_state(
            job_id,
            status="failed",
            current_stage="detect",
            payload=payload,
            error_message="RabbitMQ is unavailable",
        )
        raise HTTPException(status_code=503, detail="RabbitMQ is unavailable") from exc
    return JobCreateResponse(job_id=job_id, status="queued", current_stage="detect")


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str) -> JobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    result_json_url = f"/jobs/{job_id}/result" if job["result_json_path"] else None
    result_csv_url = f"/jobs/{job_id}/result.csv" if job["result_csv_path"] else None
    return JobStatusResponse(
        job_id=job["id"],
        filename=job["filename"],
        status=job["status"],
        current_stage=job["current_stage"],
        error_message=job["error_message"],
        result_json_url=result_json_url,
        result_csv_url=result_csv_url,
        payload=job["payload_json"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
    )


@app.get("/jobs/{job_id}/result")
async def job_result(job_id: str) -> FileResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job["result_json_path"]:
        raise HTTPException(status_code=409, detail="Result is not ready yet")
    return FileResponse(Path(job["result_json_path"]), media_type="application/json")


@app.get("/jobs/{job_id}/result.csv")
async def job_result_csv(job_id: str) -> FileResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job["result_csv_path"]:
        raise HTTPException(status_code=409, detail="CSV result is not ready yet")
    return FileResponse(Path(job["result_csv_path"]), media_type="text/csv", filename="result.csv")

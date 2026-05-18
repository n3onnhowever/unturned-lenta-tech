from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from .config import get_settings


def allocate_job_id() -> str:
    return uuid4().hex


def job_upload_dir(job_id: str) -> Path:
    settings = get_settings()
    path = settings.uploads_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_result_dir(job_id: str) -> Path:
    settings = get_settings()
    path = settings.results_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_upload(job_id: str, upload: UploadFile) -> Path:
    upload_dir = job_upload_dir(job_id)
    target = upload_dir / upload.filename
    with target.open("wb") as file_obj:
        shutil.copyfileobj(upload.file, file_obj)
    return target

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import get_settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    settings = get_settings()
    settings.app_data_dir.mkdir(parents=True, exist_ok=True)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    with db_cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                current_stage TEXT NOT NULL,
                upload_path TEXT NOT NULL,
                result_json_path TEXT,
                result_csv_path TEXT,
                error_message TEXT,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


@contextmanager
def db_cursor():
    settings = get_settings()
    connection = sqlite3.connect(settings.database_path)
    connection.row_factory = sqlite3.Row
    try:
        cursor = connection.cursor()
        yield cursor
        connection.commit()
    finally:
        connection.close()


def create_job(job_id: str, filename: str, upload_path: str, payload: dict[str, Any]) -> None:
    now = utc_now()
    with db_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO jobs (
                id, filename, status, current_stage, upload_path, result_json_path,
                result_csv_path, error_message, payload_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                filename,
                "queued",
                "detect",
                upload_path,
                None,
                None,
                None,
                json.dumps(payload, ensure_ascii=False),
                now,
                now,
            ),
        )


def get_job(job_id: str) -> dict[str, Any] | None:
    with db_cursor() as cursor:
        row = cursor.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    payload = dict(row)
    payload["payload_json"] = json.loads(payload["payload_json"])
    return payload


def update_job_state(
    job_id: str,
    *,
    status: str | None = None,
    current_stage: str | None = None,
    payload: dict[str, Any] | None = None,
    result_json_path: str | None = None,
    result_csv_path: str | None = None,
    error_message: str | None = None,
) -> None:
    job = get_job(job_id)
    if job is None:
        raise KeyError(f"Job {job_id} not found")

    updated_payload = payload if payload is not None else job["payload_json"]
    with db_cursor() as cursor:
        cursor.execute(
            """
            UPDATE jobs
            SET status = ?, current_stage = ?, payload_json = ?, result_json_path = ?,
                result_csv_path = ?, error_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                status or job["status"],
                current_stage or job["current_stage"],
                json.dumps(updated_payload, ensure_ascii=False),
                result_json_path if result_json_path is not None else job["result_json_path"],
                result_csv_path if result_csv_path is not None else job["result_csv_path"],
                error_message,
                utc_now(),
                job_id,
            ),
        )

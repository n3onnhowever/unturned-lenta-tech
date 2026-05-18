# Architecture

Unturned consists of a React frontend and a Dockerized FastAPI backend.

## Frontend

- React application under `frontend/`.
- One-page operator dashboard.
- Mock mode for offline demo.
- Backend mode through `REACT_APP_API_URL` and `REACT_APP_RECOGNITION_MODE=backend`.
- CSV parser and backend adapter live in frontend source.

## Backend

- FastAPI API service.
- RabbitMQ for staged processing.
- SQLite job storage in `runtime-data/jobs.db`.
- Workers:
  - `worker-detect`
  - `worker-classify`
  - `worker-ocr`
  - `worker-finalize`

## Pipeline

```text
Video upload
  -> API creates job
  -> RabbitMQ detect stage
  -> classify stage
  -> OCR stage
  -> finalize stage
  -> result JSON
  -> result CSV
  -> frontend table and CSV download
```

## Runtime storage

Generated files are written to `runtime-data/` through Docker volumes. This directory is ignored by git.

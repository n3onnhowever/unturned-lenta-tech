---
title: Unturned Lenta Tech Backend
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# Unturned Lenta Tech Backend

Docker Space for the Unturned recognition backend. It exposes the same FastAPI contract used by the frontend:

- `GET /health`
- `GET /health/ml`
- `POST /jobs/upload`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/result.csv`

The Space runs the pipeline in inline mode inside one container and stores demo runtime data under `/tmp/unturned-data`.

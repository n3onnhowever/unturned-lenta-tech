# Lenta Tech Hackathon BFF

`RabbitMQ` here is used as a simple stage-by-stage work queue for video jobs:

`upload -> detect -> classify -> ocr -> finalize`

The stages now call the bundled ML pipeline from `lenta-hackathon-main`.

## What is included

- `FastAPI` BFF for video upload and job polling
- `RabbitMQ` broker
- `SQLite` job state store
- stage workers for `detect`, `classify`, `ocr`, and `finalize`
- ffmpeg-based frame extraction
- YOLO price-tag detection from `lenta-hackathon-main/weights`
- crop export, smart deskew, optional super-resolution upscale, OCR parsing, QR/barcode enrichment
- result export as `result.json` and `result.csv`

**Для Frontend-разработчика:** полная документация endpoints и flow — в файле `API.md`.

## Run with Docker Compose

```bash
docker compose up -d --build
```

Open:

- Web UI: [http://localhost:8000/](http://localhost:8000/)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- RabbitMQ UI: [http://localhost:15672](http://localhost:15672)

RabbitMQ default credentials:

- user: `guest`
- password: `guest`

## Local run without Docker

1. Start RabbitMQ locally.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Start the API:

```bash
uvicorn app.main:app --reload
```

4. Start workers in separate terminals:

```bash
python -m app.worker --stage detect
python -m app.worker --stage classify
python -m app.worker --stage ocr
python -m app.worker --stage finalize
```

## API Reference (для Frontend)

**Base URL:** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs`

### Основной flow

1. Загрузить видео → `POST /jobs/upload`
2. Получить `job_id`
3. Поллить статус → `GET /jobs/{job_id}` (интервал 10 сек)
4. Когда `status === "completed"`:
   - Скачать CSV: `GET /jobs/{job_id}/result.csv`
   - Опционально: полный JSON: `GET /jobs/{job_id}/result`

### Endpoints

#### Health
```http
GET /health
```
```json
{ "status": "ok", "rabbitmq_exchange": "video_pipeline" }
```

#### ML Readiness (проверять перед загрузкой)
```http
GET /health/ml
```
Возвращает наличие YOLO-весов, скриптов и модели upscale. Поле `ready` — основной флаг.

#### Upload видео
```http
POST /jobs/upload
Content-Type: multipart/form-data
```
**Form-поле:** `file`

**Ответ (JobCreateResponse):**
```json
{
  "job_id": "aa82dd40b0854480a3ed3689e5ac8365",
  "status": "queued",
  "current_stage": "detect"
}
```

#### Статус задачи (polling)
```http
GET /jobs/{job_id}
```

**Ответ (JobStatusResponse):**
```json
{
  "job_id": "...",
  "filename": "26_12-20.mp4",
  "status": "processing",
  "current_stage": "ocr",
  "error_message": null,
  "result_json_url": "/jobs/{job_id}/result",
  "result_csv_url": "/jobs/{job_id}/result.csv",
  "payload": { ... },
  "created_at": "...",
  "updated_at": "..."
}
```

**Возможные `status`:** `queued`, `processing`, `completed`, `failed`  
**Возможные `current_stage`:** `detect`, `classify`, `ocr`, `finalize`

#### Результаты
- `GET /jobs/{job_id}/result` → `result.json` (полная мета + `crop_stats` + `dedupe`)
- `GET /jobs/{job_id}/result.csv` → готовый CSV для пользователя (с заголовком `Content-Disposition: attachment`)

### Ключевые поля в `result.json`

```json
{
  "crop_stats": {
    "total_raw_detections": 298690,
    "after_filter": 1520,
    "after_precluster": 125,
    "after_final_dedupe": 34,
    "dedupe_applied": true
  },
  "dedupe": { "enabled": true, "spatial_px": 200.0 }
}
```

### Обработка ошибок (Frontend)

| Код | Ситуация                    | Что показывать                  |
|-----|-----------------------------|---------------------------------|
| 400 | Неверный формат видео       | Только mp4/mov/avi/mkv          |
| 404 | Job не найден               | Задача не найдена               |
| 409 | Результат ещё не готов      | Обработка ещё идёт...           |
| 503 | RabbitMQ недоступен         | Сервис временно недоступен      |

### Полезные советы для Frontend

- Минимальный UI уже есть: `GET /` → `app/static/index.html`
- Диагностические CSV (`per_frame_*_counts.csv`) доступны только внутри контейнера. Для пользователя достаточно `result.csv`.
- Чтобы остановить обработку перед повторной загрузкой: `docker compose stop` или убить worker'ы.

## ML Stage Mapping

The current mapping is:

- `detect`: extract frames with ffmpeg, then run YOLO detection.
- `classify`: filter detections, export crops, smart deskew, and run upscale when a `.pb` model is available.
- `ocr`: run the hybrid OCR parser.
- `finalize`: build the hackathon CSV and result metadata (with deduplication unless `ML_DEDUPE_ENABLED=0`).

Docker Compose uses **`lenta-hackathon-main/weights/FSRCNN_x4.pb`** for super-resolution (lighter). `EDSR_x4.pb` is available for higher quality.

## Debugging crop loss

If fewer tags appear than expected, inspect the per-frame diagnostics written during processing:

- `per_frame_filtered_counts.csv` — raw vs. kept after confidence/size/aspect filtering.
- `per_frame_precluster_counts.csv` — effect of temporal IoU preclustering.
- `*_per_frame_final_counts.csv` — effect of final IoU + spatial deduplication.

These files (and a `crop_stats` summary) are included in `result.json` under `crop_stats` and `bundle_meta`. Example:

```json
{
  "crop_stats": {
    "total_raw_detections": 1234,
    "after_filter": 890,
    "after_precluster": 720,
    "after_final_dedupe": 680,
    "avg_crops_per_frame": 2.8,
    "frames_with_multiple_crops_pct": 0.65,
    "max_crops_on_single_frame": 7
  }
}
```

Quick test: set `ML_DEDUPE_ENABLED=0` or reduce `ML_PRECLUSTER_IOU`/`ML_DEDUPE_SPATIAL_PX` and re-run to see if counts recover.

## Web UI

- [http://localhost:8000/](http://localhost:8000/) — minimal upload, poll status, links to CSV/JSON when complete.
- `GET /health/ml` — checks YOLO weights path and bundled scripts.

## ML weights

Detector weights belong under `lenta-hackathon-main/weights/` (e.g. `price_tag_merged_internal_best.pt`). Override with `ML_WEIGHTS_PATH` if needed.

Useful environment variables:

- `ML_BUNDLE_DIR`
- `ML_WEIGHTS_PATH`
- `ML_UPSCALE_MODEL_PATH`
- `ML_UPSCALE_MODEL_NAME`
- `ML_UPSCALE_SCALE`
- `ML_ENGINE`
- `ML_DEDUPE_SPATIAL_PX`
- `ML_DEDUPE_ENABLED`

## Stopping video processing (re-upload)

If you uploaded videos via Swagger and want to start fresh:

1. Stop the worker containers:
   ```bash
   docker compose stop worker-detect worker-classify worker-ocr worker-finalize
   ```
   Or kill the processes if running locally.

2. Optionally purge RabbitMQ queues (via UI at http://localhost:15672 or `rabbitmqadmin`).

3. Re-upload the videos. Old jobs remain in DB but will not continue processing once workers are stopped.

## Price parsing and dedupe diagnostics

`result.json` now contains explicit `dedupe` section (enabled, spatial_px) and `crop_stats` with `dedupe_applied`.

High prices (1899.99 etc.) are correctly parsed after fixing `_lenta_pair_to_price` and shelf range in parse_ocr_fields.py (previously limited to ~200 RUB and auto-scaled).

# Unturned

## Кратко

Unturned — интерфейс и backend pipeline для автоматического распознавания ценников с видео робота, движущегося вдоль стеллажа. Система находит ценники, извлекает QR/OCR-данные, формирует таблицу результатов и CSV по требованиям Lenta Tech Life Hack.

## Возможности

- загрузка видео через web-интерфейс;
- обработка через FastAPI backend;
- staged pipeline через RabbitMQ workers;
- детекция ценников;
- OCR/QR обработка;
- deduplication;
- result JSON;
- result CSV;
- отображение результата в таблице;
- скачивание CSV;
- mock mode для frontend-demo без backend;
- backend mode для полного e2e.

## Статус проверки

- Docker backend был поднят локально.
- UI-driven e2e был проверен на рабочем проекте.
- Статус: `READY_FOR_DEMO_WITH_BACKEND`.
- Backend health: `GET /health` возвращает `ok`.
- ML health: `GET /health/ml` возвращает `ready: true`, если runtime weights на месте.
- CSV содержит 29 колонок в требуемом порядке.
- Mock mode сохранен и работает без backend.
- Hosted backend на Hugging Face Spaces Docker проверен: upload smoke video -> job completed -> `result.csv`.
- Vercel production frontend собран в backend mode и использует Hugging Face backend.


## Состав проекта

- `frontend/` — React one-page dashboard, mock mode, backend adapter and CSV download UI.
- `backend/` — FastAPI BFF, RabbitMQ worker code, ML/CV pipeline scripts and required runtime weights.
- `scripts/` — PowerShell helpers for local frontend/backend launch and health checks.
- `docker-compose.yml` — local RabbitMQ, API and worker stack.
- `.env.example` — frontend mode and backend URL example.
## Архитектура

Frontend:

- React;
- Vision UI Dashboard React base;
- one-page operator dashboard;
- backend adapter;
- CSV parser;
- mock fallback.

Backend:

- FastAPI BFF;
- RabbitMQ;
- SQLite job storage;
- workers: `detect`, `classify`, `ocr`, `finalize`;
- YOLO / OpenCV / OCR scripts;
- result JSON / result CSV.

Flow:

```text
Видео
  -> POST /jobs/upload
  -> RabbitMQ
  -> detect
  -> classify
  -> ocr
  -> finalize
  -> GET /jobs/{job_id}/result
  -> GET /jobs/{job_id}/result.csv
  -> UI table / CSV download
```

## Требования

- Windows / macOS / Linux
- Node.js 16/18/20
- npm
- Docker Desktop
- Git

Для Windows:

- PowerShell;
- Docker Desktop должен быть запущен.

## Локальный запуск

### 1. Клонировать репозиторий

```powershell
git clone https://github.com/n3onnhowever/unturned-lenta-tech.git
cd unturned-lenta-tech
```

Если репозиторий создан под другим именем, используйте URL из GitHub.

### 2. Установить frontend dependencies

```powershell
cd frontend
npm.cmd install
```

### 3. Запустить frontend в mock mode

```powershell
$env:REACT_APP_RECOGNITION_MODE="mock"
$env:DISABLE_ESLINT_PLUGIN="true"
npm.cmd start
```

Открыть:

```text
http://localhost:3000
```

Или из корня:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-frontend-mock.ps1
```

### 4. Запустить backend

Из корня repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-backend.ps1
```

Проверить:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/health/ml
```

### 5. Запустить frontend в backend mode

В отдельном терминале из корня repo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-frontend-backend.ps1
```

Открыть:

```text
http://localhost:3000
```

### 6. Demo flow

1. Нажать `Выбрать видео`.
2. Выбрать `mp4`, `mov`, `avi` или `mkv`.
3. Нажать `Запустить распознавание`.
4. Дождаться завершения обработки.
5. Проверить таблицу `Распознанные ценники`.
6. Нажать `Скачать CSV`.

## CSV contract

Итоговый CSV содержит 29 колонок строго в таком порядке:

```text
filename
product_name
price_default
price_card
price_discount
barcode
discount_amount
id_sku
print_datetime
code
additional_info
color
special_symbols
frame_timestamp
x_min
y_min
x_max
y_max
qr_code_barcode
price1_qr
price2_qr
price3_qr
price4_qr
wholesale_level_1_count
wholesale_level_1_price
wholesale_level_2_count
wholesale_level_2_price
action_price_qr
action_code_qr
```

Правила:

- если параметра нет на ценнике — `нет`;
- если параметр есть, но не распознан — пустое значение.

## API endpoints

- `GET /health`
- `GET /health/ml`
- `POST /jobs/upload`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/result.csv`

## Режимы frontend

Mock mode:

- работает без backend;
- показывает тестовые данные;
- нужен для быстрого UI-demo.

Backend mode:

- отправляет видео в backend;
- poll-ит job;
- получает backend CSV;
- скачивает backend CSV.

## Что не хранится в репозитории

- видео;
- runtime-data;
- generated frames/crops;
- result.csv/result.json;
- logs;
- Docker cache;
- node_modules;
- build.

## Model weights / artifacts

В репозиторий включены только runtime weights, нужные для локального Docker demo:

- `backend/lenta-hackathon-main/weights/price_tag_merged_internal_best.pt`
- `backend/lenta-hackathon-main/weights/FSRCNN_x4.pb`

Не включены альтернативные/исследовательские веса и generated artifacts. Если `/health/ml` сообщает, что веса отсутствуют, проверьте наличие этих двух файлов.

## Ограничения

- Smoke e2e проверен на коротком фрагменте.
- Качество модели на длинных видео требует отдельной оценки.
- Docker build может быть тяжелым из-за ML/OCR зависимостей.
- Backend deployment требует Docker runtime.

## Хостинг

Production demo:

- Frontend: https://frontend-weld-ten-13.vercel.app
- Backend API: https://n3onn-unturned-lenta-tech-backend.hf.space
- Health: https://n3onn-unturned-lenta-tech-backend.hf.space/health
- ML health: https://n3onn-unturned-lenta-tech-backend.hf.space/health/ml

Текущий hosted status:

- backend mode demo работает через Hugging Face Spaces Docker;
- hosted API smoke e2e прошел: upload -> polling -> result JSON -> result CSV;
- `result.csv` содержит 29 колонок в требуемом порядке;
- Render backend остановлен: free instance не проходил YOLO detect из-за resource limit.

Frontend-only fallback:

- mock mode сохранен для UI-demo без backend;
- для backend-connected demo используйте Vercel frontend выше.

## Репозиторий и команда

Команда: Unturned.

Проект подготовлен для Lenta Tech Life Hack.


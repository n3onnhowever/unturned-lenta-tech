# Unturned

Unturned — web-интерфейс и backend pipeline для распознавания ценников с видео робота, который движется вдоль стеллажа. Результат обработки: таблица найденных ценников и CSV в формате Lenta Tech Life Hack.

## Demo

- Frontend: https://frontend-weld-ten-13.vercel.app
- Backend API: https://n3onn-unturned-lenta-tech-backend.hf.space
- Health: https://n3onn-unturned-lenta-tech-backend.hf.space/health
- ML health: https://n3onn-unturned-lenta-tech-backend.hf.space/health/ml

Hosted backend работает на Hugging Face Spaces Docker.

## Что умеет

- загрузка видео через UI;
- обработка через FastAPI backend;
- pipeline: detect -> classify -> OCR -> finalize;
- polling статуса job;
- просмотр результата в таблице;
- скачивание backend `result.csv`;
- mock mode для frontend без backend;
- backend mode для полного demo flow.

## Архитектура

```text
Frontend React
  -> POST /jobs/upload
  -> FastAPI backend
  -> detect / classify / ocr / finalize
  -> GET /jobs/{job_id}
  -> GET /jobs/{job_id}/result
  -> GET /jobs/{job_id}/result.csv
  -> UI table + CSV download
```

Локально backend запускается через Docker Compose с RabbitMQ и workers. Для Hugging Face Spaces используется Docker backend в inline mode без RabbitMQ, но с тем же HTTP API.

## Структура

- `frontend/` — React dashboard, backend adapter, CSV download.
- `backend/` — локальный FastAPI + RabbitMQ workers + ML scripts.
- `hf-space-backend/` — Docker package для Hugging Face Spaces.
- `scripts/` — PowerShell helpers для локального запуска.
- `docker-compose.yml` — локальный backend stack.

## Быстрый запуск frontend

```powershell
git clone https://github.com/n3onnhowever/unturned-lenta-tech.git
cd unturned-lenta-tech\frontend
npm.cmd install
```

Mock mode:

```powershell
$env:REACT_APP_RECOGNITION_MODE="mock"
$env:DISABLE_ESLINT_PLUGIN="true"
npm.cmd start
```

Backend mode:

```powershell
$env:REACT_APP_RECOGNITION_MODE="backend"
$env:REACT_APP_API_URL="http://localhost:8000"
$env:DISABLE_ESLINT_PLUGIN="true"
npm.cmd start
```

Frontend откроется на http://localhost:3000.

## Локальный backend

Нужен Docker Desktop.

```powershell
cd unturned-lenta-tech
docker compose up -d --build
curl http://localhost:8000/health
curl http://localhost:8000/health/ml
```

Остановить:

```powershell
docker compose down
```

## API

- `GET /health`
- `GET /health/ml`
- `POST /jobs/upload`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`
- `GET /jobs/{job_id}/result.csv`

## CSV contract

CSV содержит 29 колонок строго в таком порядке:

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

- если поля нет на ценнике — `нет`;
- если поле есть, но не распознано — пустое значение.

## Runtime artifacts

В репозитории есть только runtime weights, необходимые для demo:

- `backend/lenta-hackathon-main/weights/price_tag_merged_internal_best.pt`
- `backend/lenta-hackathon-main/weights/FSRCNN_x4.pb`
- такие же веса включены в `hf-space-backend/`.

В репозиторий не добавляются:

- видео;
- `runtime-data/`;
- generated frames/crops;
- `result.csv` / `result.json`;
- logs;
- `node_modules/`;
- `build/`.

## Проверенный статус

- Local backend e2e: passed.
- Hosted HF backend API e2e: passed.
- Vercel frontend собран в backend mode.
- CSV header: 29 columns, exact order.
- Mock mode сохранен.

## Ограничения

- Качество модели на длинных видео требует отдельной оценки.
- Обработка на CPU может идти долго.
- Frontend больше не обрывает polling по таймауту и ждет terminal job status: `completed` или `failed`.

## Команда

- [@n3onnhowever](https://github.com/n3onnhowever)
- [@NikitkaYolo](https://github.com/NikitkaYolo)
- [@Scream-Prox](https://github.com/Scream-Prox)

Проект подготовлен командой Unturned для Lenta Tech Life Hack.

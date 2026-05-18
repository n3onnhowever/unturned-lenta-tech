# Local Run

## Frontend mock mode

```powershell
cd frontend
npm.cmd install
$env:REACT_APP_RECOGNITION_MODE="mock"
$env:DISABLE_ESLINT_PLUGIN="true"
npm.cmd start
```

## Backend

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-backend.ps1
```

Health checks:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/health/ml
```

## Frontend backend mode

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-frontend-backend.ps1
```

## Troubleshooting

- Docker Desktop must be running.
- If ports `5672`, `15672`, or `8000` are busy, stop the conflicting services.
- If `/health/ml` is not ready, verify runtime weights under `backend/lenta-hackathon-main/weights/`.
- If npm scripts are blocked in PowerShell, use `npm.cmd`.

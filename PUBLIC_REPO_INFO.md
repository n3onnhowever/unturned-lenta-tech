# Public Repository Info

## Repository
- URL: https://github.com/n3onnhowever/unturned-lenta-tech
- Visibility: public
- Branch: main
- Initial release commit: bbed71191d9fc460360269d1766d9d482257857c

## Project folder
- Local clean folder: C:\Users\glebs\Documents\unturned-lenta-tech-final

## What is included
- Frontend React source in rontend/.
- Backend FastAPI/RabbitMQ worker source in ackend/.
- Required runtime weights for the verified local smoke path.
- Docker Compose setup.
- PowerShell launch scripts.
- Documentation and CSV contract.

## What is excluded
- 
ode_modules/.
- rontend/build/.
- untime-data/.
- Videos and generated CSV/JSON outputs.
- Logs, uploads, results and local caches.

## Local run commands

### Frontend mock mode
`powershell
cd frontend
npm.cmd install
$env:REACT_APP_RECOGNITION_MODE="mock"
$env:DISABLE_ESLINT_PLUGIN="true"
npm.cmd start
`

### Backend stack
`powershell
powershell -ExecutionPolicy Bypass -File scripts/start-backend.ps1
`

### Frontend backend mode
`powershell
powershell -ExecutionPolicy Bypass -File scripts/start-frontend-backend.ps1
`

## Verification summary
- Frontend install completed.
- Frontend production build passed.
- Docker Compose backend config passed.
- Backend /health returned ok.
- Backend /health/ml returned ready.
- Backend workers ran locally.
- Backend API smoke upload completed.
- Result CSV header matched the required 29-column contract.

## Next step
Deploy frontend in mock mode for a lightweight public demo, then deploy backend to a Docker-capable host and set REACT_APP_API_URL for backend mode.

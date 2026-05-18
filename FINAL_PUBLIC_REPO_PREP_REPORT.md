# Final Public Repo Prep Report

## New repo folder
- `C:\Users\glebs\Documents\unturned-lenta-tech-final`

## What was included
- Clean frontend source in `frontend/`: React/Vision UI application, `src/`, `public/`, package files and config files required for build.
- Clean backend source in `backend/`: FastAPI app, Dockerfile, requirements, worker-compatible source code and runtime scripts/configs.
- Runtime model artifacts required for local `/health/ml` and smoke processing:
  - `backend/lenta-hackathon-main/weights/price_tag_merged_internal_best.pt`
  - `backend/lenta-hackathon-main/weights/FSRCNN_x4.pb`
- Root `docker-compose.yml` for RabbitMQ, API and workers.
- PowerShell helper scripts in `scripts/`.
- Documentation in `docs/`.
- Root `.gitignore`, `.env.example` and detailed `README.md`.

## What was excluded
- `node_modules/` and `frontend/node_modules/`.
- `build/` and `frontend/build/`.
- Runtime `data/`, `artifacts/`, `logs/`, `uploads/`, `results/`.
- Generated `runtime-data/`.
- Video files (`*.mp4`, `*.mov`, `*.avi`, `*.mkv`).
- Generated CSV/JSON outputs.
- Python cache directories and virtual environments.
- Alternative model artifacts that are not required for the verified smoke path.

## Checks
- Frontend dependencies: `npm.cmd install` completed in `frontend/`.
- Frontend build: `npm.cmd run build` passed.
- Docker compose config: passed from the clean repo root.
- Backend API: `/health` returned `status: ok`.
- Backend ML health: `/health/ml` returned `ready: true` with the included runtime weights.
- Backend services: `rabbitmq`, `api`, `worker-detect`, `worker-classify`, `worker-ocr`, `worker-finalize` were running from the clean compose project.
- Backend API smoke upload completed with job `1b20b84736d2493fb93d54e5e6a7fa64` using a local smoke video kept outside the clean repo.
- Backend `result.csv` was available and contained exactly the required 29 columns in the expected order.
- Browser check: clean frontend opened at `http://localhost:3000/`, no critical console errors, and removed legacy UI blocks were not visible.
- UI file picker upload was not repeated in the clean repo because the browser automation surface did not expose a file-picker API here; the already verified project adapter/backend flow was unchanged, and the clean backend API smoke passed.

## Repository size
- The clean source payload before dependency installation was small enough for a normal public repository.
- After local verification, ignored generated folders such as `frontend/node_modules/`, `frontend/build/`, and `runtime-data/` exist locally but must not be staged.
- Staged files were checked before commit for forbidden paths and files larger than 50 MB.

## GitHub readiness
- Target public repository name: `unturned-lenta-tech`.
- README is ready and describes local frontend/backend launch, Docker backend, demo flow and CSV contract.
- `.gitignore` is configured to keep runtime data, outputs, videos, generated CSV/JSON, `node_modules`, and build artifacts out of git.
- No `.env` secrets are included; only `.env.example` is included.
- Runtime weights policy is documented in `docs/artifact-policy.md`.

## Hosting notes
- Frontend can be hosted separately in mock mode for a lightweight demo.
- Backend-connected demo requires Docker-compatible hosting for the FastAPI/RabbitMQ/worker stack.
- Next deployment step: publish frontend, then deploy backend on a Docker host and set `REACT_APP_API_URL` for backend mode.

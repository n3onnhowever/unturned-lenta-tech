#!/usr/bin/env sh
set -e

export PIPELINE_QUEUE_MODE="${PIPELINE_QUEUE_MODE:-inline}"
export APP_DATA_DIR="${APP_DATA_DIR:-/tmp/unturned-data}"
export DATABASE_PATH="${DATABASE_PATH:-/tmp/unturned-data/jobs.db}"
export ML_BUNDLE_DIR="${ML_BUNDLE_DIR:-/app/lenta-hackathon-main}"
export ML_WEIGHTS_PATH="${ML_WEIGHTS_PATH:-/app/lenta-hackathon-main/weights/price_tag_merged_internal_best.pt}"
export ML_UPSCALE_MODEL_PATH="${ML_UPSCALE_MODEL_PATH:-/app/lenta-hackathon-main/weights/FSRCNN_x4.pb}"
export ML_UPSCALE_MODEL_NAME="${ML_UPSCALE_MODEL_NAME:-fsrcnn}"
export ML_UPSCALE_SCALE="${ML_UPSCALE_SCALE:-4}"
export ML_FRAME_STRIDE="${ML_FRAME_STRIDE:-20}"
export ML_IMGSZ="${ML_IMGSZ:-640}"
export ML_MIN_CONF="${ML_MIN_CONF:-0.45}"
export ML_WORKER_THREADS="${ML_WORKER_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OMP_THREAD_LIMIT="${OMP_THREAD_LIMIT:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"
export CORS_ORIGIN_REGEX="${CORS_ORIGIN_REGEX:-https://.*\\.vercel\\.app}"

mkdir -p "$APP_DATA_DIR"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 7860

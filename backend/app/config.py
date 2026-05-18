from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_data_dir: Path
    database_path: Path
    rabbitmq_url: str
    pipeline_queue_mode: str = "rabbitmq"
    cors_origins: tuple[str, ...] = (
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    )
    cors_origin_regex: str | None = r"https://.*\.vercel\.app"
    exchange_name: str = "video_pipeline"
    ml_bundle_dir: Path = Path("lenta-hackathon-main")
    ml_weights_path: Path = Path("lenta-hackathon-main/weights/price_tag_merged_internal_best.pt")
    ml_conf: float = 0.01
    ml_imgsz: int = 960
    ml_frame_stride: int = 10
    ml_min_conf: float = 0.01
    ml_precluster_iou: float = 0.25
    ml_precluster_max_frame_gap: int = 8
    ml_worker_threads: int = 2
    ml_export_padding: float = 0.15
    ml_deskew_pad_ratio: float = 0.28
    ml_engine: str = "paddle"
    ml_dedupe_spatial_px: float = 200.0
    ml_dedupe_enabled: bool = True
    ml_upscale_model_path: Path | None = None
    ml_upscale_model_name: str = "edsr"
    ml_upscale_scale: int = 4

    @property
    def uploads_dir(self) -> Path:
        return self.app_data_dir / "uploads"

    @property
    def results_dir(self) -> Path:
        return self.app_data_dir / "results"


def get_settings() -> Settings:
    data_dir = Path(os.getenv("APP_DATA_DIR", "data")).resolve()
    database_path = Path(os.getenv("DATABASE_PATH", str(data_dir / "jobs.db"))).resolve()
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    cors_origins_raw = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    cors_origins = tuple(origin.strip() for origin in cors_origins_raw.split(",") if origin.strip())
    cors_origin_regex = os.getenv("CORS_ORIGIN_REGEX", r"https://.*\.vercel\.app").strip() or None
    bundle_dir = Path(os.getenv("ML_BUNDLE_DIR", "lenta-hackathon-main")).resolve()
    weights_path = Path(
        os.getenv(
            "ML_WEIGHTS_PATH",
            str(bundle_dir / "weights" / "price_tag_merged_internal_best.pt"),
        )
    ).resolve()
    upscale_model_raw = os.getenv("ML_UPSCALE_MODEL_PATH", "").strip()
    return Settings(
        app_data_dir=data_dir,
        database_path=database_path,
        rabbitmq_url=rabbitmq_url,
        pipeline_queue_mode=os.getenv("PIPELINE_QUEUE_MODE", "rabbitmq").strip().lower(),
        cors_origins=cors_origins,
        cors_origin_regex=cors_origin_regex,
        ml_bundle_dir=bundle_dir,
        ml_weights_path=weights_path,
        ml_conf=float(os.getenv("ML_CONF", "0.01")),
        ml_imgsz=int(os.getenv("ML_IMGSZ", "960")),
        ml_frame_stride=int(os.getenv("ML_FRAME_STRIDE", "10")),
        ml_min_conf=float(os.getenv("ML_MIN_CONF", "0.01")),
        ml_precluster_iou=float(os.getenv("ML_PRECLUSTER_IOU", "0.25")),
        ml_precluster_max_frame_gap=int(os.getenv("ML_PRECLUSTER_MAX_FRAME_GAP", "8")),
        ml_worker_threads=int(os.getenv("ML_WORKER_THREADS", "2")),
        ml_export_padding=float(os.getenv("ML_EXPORT_PADDING", "0.15")),
        ml_deskew_pad_ratio=float(os.getenv("ML_DESKEW_PAD_RATIO", "0.28")),
        ml_engine=os.getenv("ML_ENGINE", "paddle"),
        ml_dedupe_spatial_px=float(os.getenv("ML_DEDUPE_SPATIAL_PX", "200")),
        ml_dedupe_enabled=os.getenv("ML_DEDUPE_ENABLED", "1").strip().lower() not in {"0", "false", "no"},
        ml_upscale_model_path=Path(upscale_model_raw).resolve() if upscale_model_raw else None,
        ml_upscale_model_name=os.getenv("ML_UPSCALE_MODEL_NAME", "edsr"),
        ml_upscale_scale=int(os.getenv("ML_UPSCALE_SCALE", "4")),
    )

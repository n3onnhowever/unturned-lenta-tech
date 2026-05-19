from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from video_preprocess import probe_video, resolve_ffmpeg_exe, run_command

from .config import Settings, get_settings
from .storage import job_result_dir

CROP_MANIFEST_COLUMNS = [
    "source_image",
    "crop_path",
    "rotate",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "class_name",
    "confidence",
]


def bundle_scripts_dir(settings: Settings) -> Path:
    return settings.ml_bundle_dir / "scripts"


def ensure_bundle(settings: Settings) -> None:
    if not settings.ml_bundle_dir.is_dir():
        raise RuntimeError(f"ML bundle directory not found: {settings.ml_bundle_dir}")
    if not bundle_scripts_dir(settings).is_dir():
        raise RuntimeError(f"ML scripts directory not found: {bundle_scripts_dir(settings)}")
    if not settings.ml_weights_path.is_file():
        raise RuntimeError(f"YOLO weights not found: {settings.ml_weights_path}")


def ml_subprocess_env(settings: Settings) -> dict[str, str]:
    env = os.environ.copy()
    threads = str(max(1, settings.ml_worker_threads))
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        env[key] = env.get(key, threads)
    env.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
    return env


def job_work_dir(job_id: str) -> Path:
    path = job_result_dir(job_id) / "work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_bundle_script(settings: Settings, script_name: str, args: list[str]) -> None:
    script = bundle_scripts_dir(settings) / script_name
    if not script.is_file():
        raise RuntimeError(f"ML script not found: {script}")
    command = [sys.executable, str(script), *args]
    print(f"[ml] start {script_name}", flush=True)
    result = subprocess.run(
        command,
        cwd=str(settings.ml_bundle_dir),
        text=True,
        env=ml_subprocess_env(settings),
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with code {result.returncode}")
    print(f"[ml] finished {script_name}", flush=True)


def extract_frames_ffmpeg(video_path: Path, frames_dir: Path, stride: int) -> dict[str, Any]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for frame_path in frames_dir.glob("*.jpg"):
        frame_path.unlink()
    ffmpeg_exe = resolve_ffmpeg_exe()
    print(f"[ml] extracting frames with ffmpeg, stride={stride}", flush=True)
    if stride <= 1:
        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-start_number",
            "0",
            "-q:v",
            "2",
            str(frames_dir / "%06d.jpg"),
        ]
    else:
        command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"select=not(mod(n\\,{stride}))",
            "-vsync",
            "vfr",
            "-start_number",
            "0",
            "-q:v",
            "2",
            str(frames_dir / "%06d.jpg"),
        ]
    run_command(command)
    metadata = probe_video(video_path, ffmpeg_exe)
    frame_count = len(list(frames_dir.glob("*.jpg")))
    print(f"[ml] extracted {frame_count} frames", flush=True)
    return {
        "frames_dir": str(frames_dir),
        "extracted_frames": frame_count,
        "video_metadata": {
            "fps": metadata.fps,
            "frame_count": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
            "duration_sec": metadata.duration_sec,
        },
    }


def count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return max(0, sum(1 for _ in csv.DictReader(csv_file)))


def _write_per_frame_filter_stats(raw_csv: Path, filtered_csv: Path, out_csv: Path) -> None:
    """Write per-frame detection counts before/after filter for diagnostics."""
    from collections import defaultdict

    raw_counts: dict[str, int] = defaultdict(int)
    if raw_csv.is_file():
        with raw_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                img = row.get("image_path", "")
                if img:
                    raw_counts[img] += 1

    kept_counts: dict[str, int] = defaultdict(int)
    if filtered_csv.is_file():
        with filtered_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                img = row.get("image_path", "")
                if img:
                    kept_counts[img] += 1

    frames = sorted(set(raw_counts.keys()) | set(kept_counts.keys()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "raw_count", "kept_count", "lost_in_filter"])
        for img in frames:
            raw = raw_counts.get(img, 0)
            kept = kept_counts.get(img, 0)
            w.writerow([img, raw, kept, raw - kept])


def _write_per_frame_precluster_stats(filtered_csv: Path, preclustered_csv: Path, out_csv: Path) -> None:
    """Write per-frame counts before/after temporal preclustering."""
    from collections import defaultdict

    pre_counts: dict[str, int] = defaultdict(int)
    if filtered_csv.is_file():
        with filtered_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                img = row.get("image_path", "")
                if img:
                    pre_counts[img] += 1

    post_counts: dict[str, int] = defaultdict(int)
    if preclustered_csv.is_file():
        with preclustered_csv.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                img = row.get("image_path", "")
                if img:
                    post_counts[img] += 1

    frames = sorted(set(pre_counts.keys()) | set(post_counts.keys()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "before_precluster", "after_precluster", "lost_in_precluster"])
        for img in frames:
            before = pre_counts.get(img, 0)
            after = post_counts.get(img, 0)
            w.writerow([img, before, after, before - after])


def bbox_iou(left: dict[str, str], right: dict[str, str]) -> float:
    ax1, ay1, ax2, ay2 = (float(left[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    bx1, by1, bx2, by2 = (float(right[key]) for key in ("x_min", "y_min", "x_max", "y_max"))
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def detection_frame_index(row: dict[str, str]) -> int:
    stem = Path(row.get("image_path", "")).stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return int(digits) if digits else 0


def detection_confidence(row: dict[str, str]) -> float:
    try:
        return float(row.get("confidence", 0) or 0)
    except ValueError:
        return 0.0


def precluster_temporal_detections(
    source_csv: Path,
    out_csv: Path,
    *,
    iou_threshold: float,
    max_frame_gap: int,
) -> int:
    """Keep the best crop for repeated detections in nearby frames only."""
    with source_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    clusters: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: (detection_frame_index(item), -detection_confidence(item))):
        frame = detection_frame_index(row)
        matched: dict[str, Any] | None = None
        for cluster in clusters:
            if frame - int(cluster["last_frame"]) > max_frame_gap:
                continue
            if bbox_iou(row, cluster["best_row"]) >= iou_threshold:
                matched = cluster
                break
        if matched is None:
            clusters.append({"last_frame": frame, "best_row": row})
            continue
        matched["last_frame"] = max(int(matched["last_frame"]), frame)
        if detection_confidence(row) > detection_confidence(matched["best_row"]):
            matched["best_row"] = row

    selected = [cluster["best_row"] for cluster in clusters]
    selected.sort(key=lambda item: (detection_frame_index(item), -detection_confidence(item)))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print(
        "[ml] temporal precluster: "
        f"{len(rows)} -> {len(selected)} "
        f"(iou>={iou_threshold}, max_frame_gap={max_frame_gap})",
        flush=True,
    )
    return len(selected)


def write_empty_crop_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CROP_MANIFEST_COLUMNS)
        writer.writeheader()


def find_upscale_model(settings: Settings) -> Path | None:
    if settings.ml_upscale_model_path and settings.ml_upscale_model_path.is_file():
        return settings.ml_upscale_model_path
    candidates = []
    for root_name in ("weights", "models"):
        root = settings.ml_bundle_dir / root_name
        if root.is_dir():
            candidates.extend(sorted(root.rglob("*.pb")))
    candidates.extend(sorted(settings.ml_bundle_dir.glob("*.pb")))
    return candidates[0] if candidates else None


def process_ml_detect(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    ensure_bundle(settings)
    video_path = Path(payload["video_path"]).resolve()
    work = job_work_dir(payload["job_id"])
    frames_dir = work / "frames"
    detect_dir = work / "detect"

    frame_info = extract_frames_ffmpeg(video_path, frames_dir, settings.ml_frame_stride)
    if detect_dir.exists():
        shutil.rmtree(detect_dir)
    run_bundle_script(
        settings,
        "detect_price_tags_trained.py",
        [
            "--weights",
            str(settings.ml_weights_path),
            "--source",
            str(frames_dir),
            "--output",
            str(detect_dir),
            "--conf",
            str(settings.ml_conf),
            "--imgsz",
            str(settings.ml_imgsz),
        ],
    )
    detections_csv = detect_dir / "detections.csv"
    payload["video_metadata"] = frame_info["video_metadata"]
    payload["ml"] = {
        "bundle_dir": str(settings.ml_bundle_dir),
        "work_dir": str(work),
        "frames_dir": str(frames_dir),
        "detections_csv": str(detections_csv),
    }
    payload["detection"] = {
        "status": "completed",
        "frames_extracted": frame_info["extracted_frames"],
        "detections": count_csv_rows(detections_csv),
        "weights": str(settings.ml_weights_path),
    }
    return payload


def process_ml_classify(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    ensure_bundle(settings)
    work = Path(payload["ml"]["work_dir"])
    detections_csv = Path(payload["ml"]["detections_csv"])
    filtered_csv = work / "detections_filtered.csv"
    preclustered_csv = work / "detections_preclustered.csv"
    raw_crops = work / "crops_raw"
    smart_crops = work / "crops_smart"
    upscaled_crops = work / "crops_upscaled"

    for directory in (raw_crops, smart_crops, upscaled_crops):
        if directory.exists():
            shutil.rmtree(directory)

    run_bundle_script(
        settings,
        "filter_detections.py",
        [
            "--detections",
            str(detections_csv),
            "--out",
            str(filtered_csv),
            "--min-conf",
            str(settings.ml_min_conf),
        ],
    )

    per_frame_filtered_csv = work / "per_frame_filtered_counts.csv"
    _write_per_frame_filter_stats(detections_csv, filtered_csv, per_frame_filtered_csv)
    per_frame_precluster_csv = work / "per_frame_precluster_counts.csv"

    filtered_count = count_csv_rows(filtered_csv)
    preclustered_count = 0
    if filtered_count == 0:
        final_manifest = smart_crops / "crops_manifest.csv"
        write_empty_crop_manifest(final_manifest)
        payload["ml"].update(
            {
                "filtered_detections_csv": str(filtered_csv),
                "preclustered_detections_csv": str(preclustered_csv),
                "per_frame_filtered_csv": str(per_frame_filtered_csv),
                "per_frame_precluster_csv": str(per_frame_precluster_csv),
                "raw_crops_dir": str(raw_crops),
                "smart_crops_dir": str(smart_crops),
                "upscaled_crops_dir": "",
                "manifest_csv": str(final_manifest),
            }
        )
        payload["classification"] = {
            "status": "completed_empty",
            "filtered_detections": 0,
            "preclustered_detections": 0,
            "crops": 0,
            "upscale": {"status": "skipped_no_crops"},
        }
        return payload

    preclustered_count = precluster_temporal_detections(
        filtered_csv,
        preclustered_csv,
        iou_threshold=settings.ml_precluster_iou,
        max_frame_gap=settings.ml_precluster_max_frame_gap,
    )

    _write_per_frame_precluster_stats(filtered_csv, preclustered_csv, per_frame_precluster_csv)

    run_bundle_script(
        settings,
        "export_ocr_crops.py",
        [
            "--detections",
            str(preclustered_csv),
            "--output",
            str(raw_crops),
            "--padding",
            str(settings.ml_export_padding),
            "--rotate",
            "ccw90",
        ],
    )
    run_bundle_script(
        settings,
        "process_tags_smart_deskew.py",
        [
            "--input",
            str(raw_crops),
            "--out",
            str(smart_crops),
            "--pad-ratio",
            str(settings.ml_deskew_pad_ratio),
        ],
    )

    upscale_model = find_upscale_model(settings)
    if upscale_model:
        run_bundle_script(
            settings,
            "upscale_crops.py",
            [
                "--input",
                str(smart_crops),
                "--out",
                str(upscaled_crops),
                "--model",
                str(upscale_model),
                "--model-name",
                settings.ml_upscale_model_name,
                "--scale",
                str(settings.ml_upscale_scale),
            ],
        )
        final_manifest = upscaled_crops / "crops_manifest.csv"
        upscale_status = {
            "status": "completed",
            "model": str(upscale_model),
            "model_name": settings.ml_upscale_model_name,
            "scale": settings.ml_upscale_scale,
        }
    else:
        final_manifest = smart_crops / "crops_manifest.csv"
        upscale_status = {
            "status": "skipped_missing_model",
            "message": "Add a .pb super-resolution model or set ML_UPSCALE_MODEL_PATH.",
        }

    payload["ml"].update(
        {
            "filtered_detections_csv": str(filtered_csv),
            "preclustered_detections_csv": str(preclustered_csv),
            "per_frame_filtered_csv": str(per_frame_filtered_csv),
            "per_frame_precluster_csv": str(per_frame_precluster_csv),
            "raw_crops_dir": str(raw_crops),
            "smart_crops_dir": str(smart_crops),
            "upscaled_crops_dir": str(upscaled_crops) if upscale_model else "",
            "manifest_csv": str(final_manifest),
        }
    )
    payload["classification"] = {
        "status": "completed",
        "filtered_detections": count_csv_rows(filtered_csv),
        "preclustered_detections": preclustered_count,
        "crops": count_csv_rows(final_manifest),
        "upscale": upscale_status,
    }
    return payload


def _compute_crop_stats(payload: dict[str, Any], bundle_meta_path: Path | None) -> dict[str, Any] | None:
    """Aggregate per-frame counts into a high-level crop loss summary."""
    from collections import defaultdict

    detection = payload.get("detection", {}) or {}
    classification = payload.get("classification", {}) or {}
    ml = payload.get("ml", {}) or {}

    total_raw = int(detection.get("detections", 0) or 0)
    after_filter = int(classification.get("filtered_detections", 0) or 0)
    after_precluster = int(classification.get("preclustered_detections", 0) or 0)

    # Try to get after_final from bundle_meta if present
    after_final = None
    if bundle_meta_path and bundle_meta_path.is_file():
        try:
            bm = json.loads(bundle_meta_path.read_text(encoding="utf-8"))
            after_final = int(bm.get("rows_after_dedupe", 0) or 0)
        except Exception:
            after_final = None

    # If we have per_frame files, compute more detailed stats
    avg_per_frame = None
    frames_with_multiple = None
    max_on_frame = None

    per_frame_final = ml.get("per_frame_final_counts_csv")
    if per_frame_final:
        p = Path(per_frame_final)
        if p.is_file():
            counts: list[int] = []
            with p.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    try:
                        counts.append(int(row.get("crops_after_dedupe", 0) or 0))
                    except ValueError:
                        pass
            if counts:
                avg_per_frame = round(sum(counts) / len(counts), 2)
                frames_with_multiple = round(sum(1 for c in counts if c > 1) / len(counts), 3)
                max_on_frame = max(counts)

    stats: dict[str, Any] = {
        "total_raw_detections": total_raw,
        "after_filter": after_filter,
        "after_precluster": after_precluster,
    }
    if after_final is not None:
        stats["after_final_dedupe"] = after_final
    if avg_per_frame is not None:
        stats["avg_crops_per_frame"] = avg_per_frame
    if frames_with_multiple is not None:
        stats["frames_with_multiple_crops_pct"] = frames_with_multiple
    if max_on_frame is not None:
        stats["max_crops_on_single_frame"] = max_on_frame

    # Explicit dedupe status for visibility
    dedupe_enabled = payload.get("dedupe", {}).get("enabled")
    if dedupe_enabled is not None:
        stats["dedupe_applied"] = bool(dedupe_enabled)
    else:
        # fallback: if we reached finalize with no --no-dedupe in args
        stats["dedupe_applied"] = True

    return stats if stats else None


def process_ml_ocr(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    ensure_bundle(settings)
    work = Path(payload["ml"]["work_dir"])
    parsed_json = work / "parsed_hybrid.json"
    manifest_csv = Path(payload["ml"]["manifest_csv"])

    run_bundle_script(
        settings,
        "parse_manifest_hybrid.py",
        [
            "--manifest",
            str(manifest_csv),
            "--out",
            str(parsed_json),
            "--mode",
            "hybrid",
            "--engine",
            settings.ml_engine,
        ],
    )
    parsed_count = 0
    if parsed_json.is_file():
        parsed_count = len(json.loads(parsed_json.read_text(encoding="utf-8")))
    payload["ml"]["parsed_json"] = str(parsed_json)
    payload["ocr"] = {
        "status": "completed",
        "engine": settings.ml_engine,
        "parsed_crops": parsed_count,
    }
    return payload


def process_ml_finalize(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    ensure_bundle(settings)
    result_dir = job_result_dir(payload["job_id"])
    work = Path(payload["ml"]["work_dir"])
    result_csv_path = result_dir / "result.csv"
    result_json_path = result_dir / "result.json"
    manifest_csv = Path(payload["ml"]["manifest_csv"])
    parsed_json = Path(payload["ml"]["parsed_json"])

    finalize_args = [
        "--video",
        str(Path(payload["video_path"]).resolve()),
        "--frames",
        payload["ml"]["frames_dir"],
        "--detections",
        payload["ml"]["detections_csv"],
        "--out-csv",
        str(result_csv_path),
        "--work-dir",
        str(work / "finalize"),
        "--use-manifest",
        str(manifest_csv),
        "--parsed-json",
        str(parsed_json),
        "--engine",
        settings.ml_engine,
        "--dedupe-spatial-px",
        str(settings.ml_dedupe_spatial_px),
    ]
    if not settings.ml_dedupe_enabled:
        finalize_args.append("--no-dedupe")

    run_bundle_script(settings, "lenta_hackathon_pipeline.py", finalize_args)

    meta_path = result_csv_path.parent / f"{result_csv_path.stem}_meta.json"
    result = {
        "job_id": payload["job_id"],
        "filename": payload["filename"],
        "pipeline": {
            "detection": payload.get("detection", {}),
            "classification": payload.get("classification", {}),
            "ocr": payload.get("ocr", {}),
        },
        "video_metadata": payload.get("video_metadata", {}),
        "result_csv_path": str(result_csv_path),
        "bundle_meta_path": str(meta_path) if meta_path.is_file() else "",
        "dedupe": {
            "enabled": settings.ml_dedupe_enabled,
            "spatial_px": settings.ml_dedupe_spatial_px,
        },
    }
    if meta_path.is_file():
        result["bundle_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))

    # Compute crop-level diagnostics summary for visibility into detection loss
    crop_stats = _compute_crop_stats(payload, meta_path if meta_path.is_file() else None)
    if crop_stats:
        result["crop_stats"] = crop_stats

    result_json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    payload["result_json_path"] = str(result_json_path)
    payload["result_csv_path"] = str(result_csv_path)
    payload["result_summary"] = result
    return payload

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def resolve_ffmpeg_exe() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise RuntimeError(
            "ffmpeg was not found in PATH and imageio-ffmpeg is not available. "
            "Install imageio-ffmpeg or add ffmpeg to PATH."
        ) from exc


def resolve_ffprobe_exe(ffmpeg_exe: str) -> str | None:
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path

    ffmpeg_path = Path(ffmpeg_exe)
    candidate = ffmpeg_path.with_name(ffmpeg_path.name.replace("ffmpeg", "ffprobe"))
    return str(candidate) if candidate.exists() else None


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_sec: float


@dataclass
class FrameStats:
    frame_name: str
    frame_index: int
    timestamp_ms: int
    width: int
    height: int
    blur_score: float
    brightness_mean: float
    contrast_std: float


def probe_video(video_path: Path, ffmpeg_exe: str) -> VideoMetadata:
    ffprobe_exe = resolve_ffprobe_exe(ffmpeg_exe)
    if ffprobe_exe:
        command = [
            ffprobe_exe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ]
        result = run_command(command)
        payload = json.loads(result.stdout)
        stream = payload["streams"][0]
        fps = parse_fraction(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
        duration_sec = float(stream.get("duration") or 0.0)
        frame_count = int(stream.get("nb_frames") or round(duration_sec * fps))
        return VideoMetadata(
            path=str(video_path),
            width=int(stream["width"]),
            height=int(stream["height"]),
            fps=fps,
            frame_count=frame_count,
            duration_sec=duration_sec,
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration_sec = frame_count / fps if fps else 0.0
    return VideoMetadata(
        path=str(video_path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_sec=duration_sec,
    )


def parse_fraction(value: str) -> float:
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        denominator_value = float(denominator)
        return float(numerator) / denominator_value if denominator_value else 0.0
    return float(value)


def build_spatial_filter(rotation: str, max_width: int | None) -> str:
    filters: list[str] = []
    if rotation == "cw":
        filters.append("transpose=1")
    elif rotation == "ccw":
        filters.append("transpose=2")

    if max_width:
        # This only downsizes wide frames and never upscales them.
        filters.append(f"scale='min(iw,{max_width})':-2")
    return ",".join(filters)


def build_frame_filter(rotation: str, fps: float, max_width: int | None) -> str:
    filters = [item for item in [build_spatial_filter(rotation, max_width), f"fps={fps}"] if item]
    return ",".join(filters)


def normalize_and_extract_frames(
    video_path: Path,
    output_dir: Path,
    ffmpeg_exe: str,
    rotation: str,
    fps: float,
    jpeg_quality: int,
    max_width: int | None,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_video = output_dir / "normalized.mp4"
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "frame_%06d.jpg"

    filter_chain = build_frame_filter(rotation=rotation, fps=fps, max_width=max_width)
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        filter_chain,
        "-q:v",
        str(jpeg_quality),
        "-map",
        "0:v:0",
        str(frame_pattern),
    ]
    run_command(command)

    # Keep a normalized clip for debugging and for downstream teams.
    normalize_filters = build_spatial_filter(rotation=rotation, max_width=max_width)
    if normalize_filters:
        normalize_command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            normalize_filters,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            str(normalized_video),
        ]
    else:
        normalize_command = [
            ffmpeg_exe,
            "-y",
            "-i",
            str(video_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            str(normalized_video),
        ]
    run_command(normalize_command)
    return normalized_video, frames_dir


def analyze_frames(frames_dir: Path, sampled_fps: float) -> list[FrameStats]:
    stats: list[FrameStats] = []
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    for frame_path in frame_paths:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness_mean = float(gray.mean())
        contrast_std = float(gray.std())
        frame_index = int(frame_path.stem.split("_")[-1]) - 1
        timestamp_ms = int(round((frame_index / sampled_fps) * 1000)) if sampled_fps else 0
        stats.append(
            FrameStats(
                frame_name=frame_path.name,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                blur_score=blur_score,
                brightness_mean=brightness_mean,
                contrast_std=contrast_std,
            )
        )
    return stats


def write_manifest(stats: Iterable[FrameStats], manifest_path: Path) -> None:
    rows = [asdict(item) for item in stats]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_summary(
    metadata: VideoMetadata,
    stats: list[FrameStats],
    summary_path: Path,
    rotation: str,
    sampled_fps: float,
) -> None:
    payload = {
        "video": asdict(metadata),
        "rotation": rotation,
        "sampled_fps": sampled_fps,
        "extracted_frames": len(stats),
        "blur_score_mean": round(float(np.mean([row.blur_score for row in stats])) if stats else 0.0, 3),
        "blur_score_median": round(float(np.median([row.blur_score for row in stats])) if stats else 0.0, 3),
        "brightness_mean": round(float(np.mean([row.brightness_mean for row in stats])) if stats else 0.0, 3),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_samples(video_path: Path, output_dir: Path, timestamps_ms: list[int], rotation: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    rotation_map = {
        "none": None,
        "cw": cv2.ROTATE_90_CLOCKWISE,
        "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    rotate_code = rotation_map[rotation]

    for timestamp_ms in timestamps_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
        ok, frame = cap.read()
        if not ok:
            continue
        if rotate_code is not None:
            frame = cv2.rotate(frame, rotate_code)
        cv2.imwrite(str(output_dir / f"sample_{timestamp_ms}.jpg"), frame)
    cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare hackathon shelf videos with ffmpeg + OpenCV."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Show input video metadata.")
    probe_parser.add_argument("--video", required=True, type=Path)

    sample_parser = subparsers.add_parser("sample", help="Save selected timestamps as images.")
    sample_parser.add_argument("--video", required=True, type=Path)
    sample_parser.add_argument("--output-dir", default=Path("artifacts/samples"), type=Path)
    sample_parser.add_argument("--rotation", choices=["none", "cw", "ccw"], default="ccw")
    sample_parser.add_argument("--timestamps-ms", nargs="+", required=True, type=int)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Normalize video, extract frames with ffmpeg, and analyze them with OpenCV.",
    )
    preprocess_parser.add_argument("--video", required=True, type=Path)
    preprocess_parser.add_argument("--output-dir", default=Path("artifacts/preprocess"), type=Path)
    preprocess_parser.add_argument("--rotation", choices=["none", "cw", "ccw"], default="ccw")
    preprocess_parser.add_argument("--fps", type=float, default=2.0)
    preprocess_parser.add_argument("--jpeg-quality", type=int, default=2)
    preprocess_parser.add_argument("--max-width", type=int, default=1920)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg_exe = resolve_ffmpeg_exe()

    if args.command == "probe":
        metadata = probe_video(args.video, ffmpeg_exe)
        print(json.dumps(asdict(metadata), ensure_ascii=False, indent=2))
        return

    if args.command == "sample":
        extract_samples(
            video_path=args.video,
            output_dir=args.output_dir,
            timestamps_ms=args.timestamps_ms,
            rotation=args.rotation,
        )
        print(f"Saved samples to {args.output_dir}")
        return

    if args.command == "preprocess":
        metadata = probe_video(args.video, ffmpeg_exe)
        normalized_video, frames_dir = normalize_and_extract_frames(
            video_path=args.video,
            output_dir=args.output_dir,
            ffmpeg_exe=ffmpeg_exe,
            rotation=args.rotation,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            max_width=args.max_width,
        )
        stats = analyze_frames(frames_dir=frames_dir, sampled_fps=args.fps)
        write_manifest(stats, args.output_dir / "frame_manifest.csv")
        write_summary(
            metadata=metadata,
            stats=stats,
            summary_path=args.output_dir / "summary.json",
            rotation=args.rotation,
            sampled_fps=args.fps,
        )
        print(
            json.dumps(
                {
                    "normalized_video": str(normalized_video),
                    "frames_dir": str(frames_dir),
                    "manifest": str(args.output_dir / "frame_manifest.csv"),
                    "summary": str(args.output_dir / "summary.json"),
                    "frames_extracted": len(stats),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

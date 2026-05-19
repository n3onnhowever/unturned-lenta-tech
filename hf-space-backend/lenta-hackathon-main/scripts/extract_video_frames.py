"""
Извлечение кадров из видео (OpenCV) для последующей детекции в YOLO.

Примеры:
  python scripts/extract_video_frames.py
  python scripts/extract_video_frames.py -i materials/data
  python scripts/extract_video_frames.py -i path/to/video.mp4
  python scripts/extract_video_frames.py -i videos/ -o frames --stride 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})


def _safe_stem(path: Path) -> str:
    stem = path.stem if path.suffix else path.name
    out = "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in stem)
    return out or "video"


def _output_dir_name(video_path: Path) -> str:
    """Имя подпапки для кадров: цепочка родителей + имя файла (уникально для вложенных путей)."""
    stem = _safe_stem(video_path)
    parent = video_path.parent
    if parent == Path(".") or parent == Path(""):
        return stem
    parts = [_safe_stem(Path(p)) for p in parent.parts]
    return "_".join(parts + [stem])


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    stride: int = 1,
    ext: str = ".jpg",
    jpeg_quality: int = 95,
    max_frames: int | None = None,
) -> tuple[int, float, tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target = out_dir / _output_dir_name(video_path)
    target.mkdir(parents=True, exist_ok=True)

    params = []
    if ext.lower() in (".jpg", ".jpeg"):
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    saved = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                name = f"{saved:06d}{ext}"
                out_path = target / name
                if ext.lower() in (".jpg", ".jpeg"):
                    cv2.imwrite(str(out_path), frame, params)
                else:
                    cv2.imwrite(str(out_path), frame)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            frame_index += 1
    finally:
        cap.release()

    return saved, fps, (w, h)


def collect_videos(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in inputs:
        if p.is_file():
            if p.suffix.lower() in VIDEO_SUFFIXES:
                files.append(p)
        elif p.is_dir():
            for item in sorted(p.rglob("*")):
                if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES:
                    files.append(item)
        else:
            print(f"Пропуск (не файл и не папка): {p}", file=sys.stderr)
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(f)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description="Нарезка видео на кадры для YOLO (OpenCV).")
    parser.add_argument(
        "-i",
        "--input",
        nargs="*",
        default=None,
        type=Path,
        metavar="PATH",
        help="Видеофайл(ы) или каталоги (рекурсивно). Если не указано — каталог materials/data при его наличии",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=Path("frames"),
        help="Корневая папка вывода (по умолчанию: ./frames)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Брать каждый N-й кадр (1 = все кадры)",
    )
    parser.add_argument(
        "--ext",
        choices=(".jpg", ".jpeg", ".png"),
        default=".jpg",
        help="Формат сохранения кадров",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="Качество JPEG 1–100 (только для jpg/jpeg)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Ограничение числа сохранённых кадров на одно видео (для тестов)",
    )
    args = parser.parse_args()

    if args.stride < 1:
        print("--stride должен быть >= 1", file=sys.stderr)
        return 2

    inputs: list[Path]
    if args.input:
        inputs = list(args.input)
    else:
        default_dir = Path("materials/data")
        if default_dir.is_dir():
            inputs = [default_dir]
        else:
            print(
                "Укажите вход: -i <видео или папка>. Каталог по умолчанию materials/data не найден.",
                file=sys.stderr,
            )
            return 1

    videos = collect_videos(inputs)
    if not videos:
        print("Не найдено ни одного видео по указанным путям.", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for vp in videos:
        n, fps, (w, h) = extract_frames(
            vp,
            args.out_dir,
            stride=args.stride,
            ext=args.ext,
            jpeg_quality=args.jpeg_quality,
            max_frames=args.max_frames,
        )
        sub = args.out_dir / _output_dir_name(vp)
        print(f"{vp} -> {sub} ({n} кадров, {w}x{h}, fps={fps:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

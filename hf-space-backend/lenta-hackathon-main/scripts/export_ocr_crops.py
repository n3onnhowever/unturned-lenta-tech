"""
Вырезка bbox из detections.csv (YOLO) с поворотом кропа для PaddleOCR.

PaddleOCR обычно лучше работает с горизонтальными строками; для кадров с робота
часто нужен поворот кропа на 90° против часовой стрелки (ccw90) — это значение
по умолчанию. При необходимости смените на cw90 или none.

Пример:
  python scripts/export_ocr_crops.py --detections runs/price_tag_detect/detections.csv --output runs/ocr_crops --rotate ccw90
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

cv2.setNumThreads(max(1, int(os.getenv("ML_WORKER_THREADS", "2"))))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


def crop_xyxy(img, x1: float, y1: float, x2: float, y2: float, padding: float = 0.0):
    h, w = img.shape[:2]
    
    if padding > 0:
        box_w = x2 - x1
        box_h = y2 - y1
        pad_x = box_w * padding
        pad_y = box_h * padding
        x1 -= pad_x
        y1 -= pad_y
        x2 += pad_x
        y2 += pad_y
        
    x1i = max(0, min(w - 1, int(math.floor(x1))))
    y1i = max(0, min(h - 1, int(math.floor(y1))))
    x2i = max(x1i + 1, min(w, int(math.ceil(x2))))
    y2i = max(y1i + 1, min(h, int(math.ceil(y2))))
    return img[y1i:y2i, x1i:x2i].copy()


def rotate_for_ocr(crop, mode: str):
    if mode == "none":
        return crop
    if mode == "cw90":
        return cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    if mode == "ccw90":
        return cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if mode == "180":
        return cv2.rotate(crop, cv2.ROTATE_180)
    raise ValueError(f"Неизвестный режим поворота: {mode}")


def load_detections(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if not row.get("image_path"):
                continue
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Кропы ценников из CSV детекции + поворот для PaddleOCR.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="CSV от detect_price_tags_yolo.py (image_path, x_min, y_min, x_max, y_max, …)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Папка для сохранения кропов (jpg)",
    )
    parser.add_argument(
        "--rotate",
        choices=("none", "cw90", "ccw90", "180"),
        default="ccw90",
        help="Поворот кропа после вырезки: ccw90 = против часовой 90° (по умолчанию, под PaddleOCR)",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=8,
        help="Пропускать кропы, где min(w,h) меньше этого значения",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.0,
        help="Добавить отступы (доля от ширины/высоты)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Опционально: CSV со связью исходный_кадр, bbox, путь_к_кропу (по умолчанию output/crops_manifest.csv)",
    )
    args = parser.parse_args()

    if not args.detections.is_file():
        print(f"Файл не найден: {args.detections}", file=sys.stderr)
        return 1

    rows = load_detections(args.detections)
    if not rows:
        print("В CSV нет строк с image_path.", file=sys.stderr)
        return 1

    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = str(Path(row["image_path"]).resolve())
        by_image[key].append(row)

    args.output.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str]] = []
    saved = 0
    skipped = 0

    items = list(by_image.items())
    if tqdm is not None:
        items = tqdm(items, desc="images", unit="img")

    for img_path_str, img_rows in items:
        p = Path(img_path_str)
        if not p.is_file():
            print(f"Пропуск (нет файла): {p}", file=sys.stderr)
            skipped += len(img_rows)
            continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"Пропуск (не читается): {p}", file=sys.stderr)
            skipped += len(img_rows)
            continue

        stem = p.stem
        parent = p.parent.name
        for idx, row in enumerate(img_rows):
            try:
                x1 = float(row["x_min"])
                y1 = float(row["y_min"])
                x2 = float(row["x_max"])
                y2 = float(row["y_max"])
            except (KeyError, ValueError):
                skipped += 1
                continue

            crop = crop_xyxy(img, x1, y1, x2, y2, args.padding)
            if crop.size == 0:
                skipped += 1
                continue
            ch, cw = crop.shape[:2]
            if min(ch, cw) < args.min_side:
                skipped += 1
                continue

            out = rotate_for_ocr(crop, args.rotate)
            name = f"{parent}_{stem}_{idx:04d}_{args.rotate}.jpg"
            out_path = args.output / name
            cv2.imwrite(str(out_path), out, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            manifest_rows.append(
                {
                    "source_image": str(p.resolve()),
                    "crop_path": str(out_path.resolve()),
                    "rotate": args.rotate,
                    "x_min": row.get("x_min", ""),
                    "y_min": row.get("y_min", ""),
                    "x_max": row.get("x_max", ""),
                    "y_max": row.get("y_max", ""),
                    "class_name": row.get("class_name", ""),
                    "confidence": row.get("confidence", ""),
                }
            )
            saved += 1

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.output / "crops_manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        fieldnames = [
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
        w = csv.DictWriter(mf, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"Сохранено кропов: {saved}, пропущено: {skipped}, манифест: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

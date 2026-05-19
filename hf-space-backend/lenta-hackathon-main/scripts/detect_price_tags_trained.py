"""
Детекция ценников обученной моделью Ultralytics (один класс, например price_tag).

CSV совместим с scripts/export_ocr_crops.py (те же колонки, что у detect_price_tags_yolo.py).

Пример (кадры 43_15 + merged best):
  python scripts/detect_price_tags_trained.py \\
    --weights runs/detect/runs/detect/price_tag_merged/weights/best.pt \\
    --source frames/materials_data_43_15_43_15 \\
    --output runs/detect_merged_43_15 \\
    --conf 0.35 --imgsz 1280
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def configure_runtime_threads() -> None:
    threads = max(1, int(os.getenv("ML_WORKER_THREADS", "2")))
    for key in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(key, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def box_passes_size_filters(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    min_area: float,
    min_short_side: float,
) -> bool:
    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return False
    area = w * h
    short = min(w, h)
    if min_area > 0 and area < min_area:
        return False
    if min_short_side > 0 and short < min_short_side:
        return False
    return True


def collect_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    if not root.is_dir():
        return []
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out.append(p)
    return sorted(out)


def main() -> int:
    configure_runtime_threads()
    ap = argparse.ArgumentParser(description="YOLO (обученная): bbox ценников → detections.csv")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Папка с кадрами или один файл")
    ap.add_argument("--output", type=Path, default=Path("runs/detect_price_tag_trained"))
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-box-area", type=float, default=0.0)
    ap.add_argument("--min-box-short-side", type=float, default=0.0)
    args = ap.parse_args()

    if not args.weights.is_file():
        print(f"Нет весов: {args.weights}", file=sys.stderr)
        return 1

    images = collect_images(args.source)
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]
    if not images:
        print(f"Нет изображений: {args.source}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    device = args.device if args.device else None
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "detections.csv"

    fieldnames = [
        "image_path",
        "class_id",
        "class_name",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]

    n_det = 0
    chunk = max(1, args.batch)
    batch_starts = list(range(0, len(images), chunk))
    if tqdm is not None:
        batch_starts = tqdm(batch_starts, desc="batches", unit="batch")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for start in batch_starts:
            batch_paths = images[start : start + chunk]
            results = model.predict(
                source=[str(p) for p in batch_paths],
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
                save=False,
            )
            for r, src in zip(results, batch_paths):
                path_str = str(src.resolve())
                names = r.names
                if isinstance(names, list):
                    names_map = {i: n for i, n in enumerate(names)}
                else:
                    names_map = dict(names)
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().tolist()
                confs = r.boxes.conf.cpu().tolist()
                clss = r.boxes.cls.cpu().tolist()
                for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clss):
                    if not box_passes_size_filters(
                        x1,
                        y1,
                        x2,
                        y2,
                        min_area=args.min_box_area,
                        min_short_side=args.min_box_short_side,
                    ):
                        continue
                    cid = int(ci)
                    cname = names_map.get(cid, str(cid))
                    w.writerow(
                        {
                            "image_path": path_str,
                            "class_id": cid,
                            "class_name": cname,
                            "confidence": round(float(cf), 6),
                            "x_min": round(float(x1), 2),
                            "y_min": round(float(y1), 2),
                            "x_max": round(float(x2), 2),
                            "y_max": round(float(y2), 2),
                        }
                    )
                    n_det += 1

    print(f"Кадров: {len(images)}, детекций: {n_det} → {csv_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

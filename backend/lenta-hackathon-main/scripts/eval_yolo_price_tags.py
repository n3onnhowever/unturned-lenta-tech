"""
Оценка обученного YOLO на папке с кадрами: число детекций, опционально сохранение разметки.

Пример:
  python scripts/eval_yolo_price_tags.py --weights runs/detect/price_tag_yolov8/weights/best.pt --source frames/materials_data_43_15_43_15 --conf 0.25 --viz --max-images 40
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


def collect_jpg(root: Path, limit: int | None) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in {".jpg", ".jpeg", ".png"} else []
    xs = sorted(root.glob("*.jpg")) + sorted(root.glob("*.jpeg")) + sorted(root.glob("*.png"))
    if limit is not None:
        xs = xs[:limit]
    return xs


def main() -> int:
    ap = argparse.ArgumentParser(description="Статистика + визуализация детекций YOLO (ценник).")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Папка с jpg или один файл")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--viz", action="store_true", help="Сохранить кадры с боксами в --project/--name")
    ap.add_argument("--project", type=Path, default=Path("runs/predict"))
    ap.add_argument("--name", type=str, default="price_tag_eval")
    args = ap.parse_args()

    if not args.weights.is_file():
        print(f"Нет весов: {args.weights}", file=sys.stderr)
        return 1

    imgs = collect_jpg(args.source, args.max_images)
    if not imgs:
        print(f"Нет изображений: {args.source}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(str(args.weights))
    device = args.device if args.device else None

    counts: list[int] = []
    all_conf: list[float] = []

    for p in imgs:
        r = model.predict(
            source=str(p),
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
            save=args.viz,
            project=str(args.project),
            name=args.name,
        )[0]
        n = 0 if r.boxes is None else len(r.boxes)
        counts.append(n)
        if r.boxes is not None and len(r.boxes):
            all_conf.extend(r.boxes.conf.cpu().tolist())

    nz = [c for c in counts if c > 0]
    print(f"weights: {args.weights}")
    print(f"source:  {args.source}  images={len(imgs)}  conf={args.conf}  imgsz={args.imgsz}")
    print(f"frames with >=1 box: {len(nz)} / {len(imgs)}")
    print(f"total boxes: {sum(counts)}  (mean per frame: {statistics.mean(counts):.2f})")
    if all_conf:
        print(f"confidence: min={min(all_conf):.3f} max={max(all_conf):.3f} mean={statistics.mean(all_conf):.3f}")
    if args.viz:
        out = args.project / args.name
        print(f"viz saved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Обучение YOLOv8 на датасете ценников (Ultralytics).

Сначала соберите датасет:
  python scripts/build_yolo_dataset_from_gt_csv.py --materials materials/data --out dataset/price_tags

Затем:
  python scripts/train_price_tag_yolo.py --data dataset/price_tags/data.yaml --epochs 50

На CPU batch лучше 2–4; на GPU можно 8–16.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение YOLOv8 detect для ценника.")
    ap.add_argument("--data", type=Path, default=Path("dataset/price_tags/data.yaml"))
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Базовые веса (n/s/m)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", type=str, default="", help="cuda:0 или пусто=авто")
    ap.add_argument("--project", type=str, default="runs/detect", help="Каталог проекта (weights в project/name/weights)")
    ap.add_argument("--name", type=str, default="price_tag_yolov8")
    ap.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"Нет файла data.yaml: {args.data}", file=sys.stderr)
        print("Сначала: python scripts/build_yolo_dataset_from_gt_csv.py", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    device = args.device if args.device else None
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )
    print(f"Готово. Веса: {Path(args.project).resolve() / args.name / 'weights'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

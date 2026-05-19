"""
Прогон детекции при разных conf: сводка по числу боксов (ищем рабочий порог).

Пример:
  python scripts/sweep_conf_yolo.py --weights runs/detect/runs/detect/price_tag_manual/weights/best.pt --source frames/materials_data_43_15_43_15 --max-images 30
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]


def collect_images(root: Path, limit: int | None) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in {".jpg", ".jpeg", ".png"} else []
    xs = (
        sorted(root.glob("*.jpg"))
        + sorted(root.glob("*.jpeg"))
        + sorted(root.glob("*.png"))
        + sorted(root.glob("*.JPG"))
    )
    if limit is not None:
        xs = xs[:limit]
    return xs


def main() -> int:
    ap = argparse.ArgumentParser(description="Сводка детекций при разных conf.")
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Папка с кадрами или один файл")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--max-images", type=int, default=30)
    ap.add_argument(
        "--confs",
        type=str,
        default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9",
        help="Список conf через запятую",
    )
    ap.add_argument("--device", type=str, default="")
    args = ap.parse_args()

    if YOLO is None:
        print("pip install ultralytics", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"Нет весов: {args.weights}", file=sys.stderr)
        return 1

    imgs = collect_images(args.source, args.max_images)
    if not imgs:
        print(f"Нет изображений: {args.source}", file=sys.stderr)
        return 1

    confs = [float(x.strip()) for x in args.confs.split(",") if x.strip()]
    confs = sorted(set(c for c in confs if 0 < c < 1))

    model = YOLO(str(args.weights))
    device = args.device if args.device else None

    # Кэш «сырых» предсказаний при низком conf — один проход на кадр
    low_conf = 0.001
    raw_per_image: list[list[tuple[float, float, float, float, float]]] = []

    for p in imgs:
        r = model.predict(
            source=str(p),
            conf=low_conf,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
            save=False,
            max_det=500,
        )[0]
        lst: list[tuple[float, float, float, float, float]] = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().tolist()
            cf = r.boxes.conf.cpu().tolist()
            for box, c in zip(xyxy, cf):
                lst.append((float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(c)))
        raw_per_image.append(lst)

    print(f"weights: {args.weights}")
    print(f"source:  {args.source}  images={len(imgs)}  imgsz={args.imgsz}  raw_conf={low_conf}")
    print()
    print("conf  total  mean/frame  median/frame  max/frame")
    for c in confs:
        per_frame_counts: list[int] = []
        tot = 0
        for lst in raw_per_image:
            n = sum(1 for t in lst if t[4] >= c)
            per_frame_counts.append(n)
            tot += n
        med = float(statistics.median(per_frame_counts)) if per_frame_counts else 0.0
        mx = max(per_frame_counts) if per_frame_counts else 0
        mean = tot / len(imgs) if imgs else 0.0
        print(f"{c:>4.2f}  {tot:>5}  {mean:>10.2f}  {med:>12.1f}  {mx:>9}")

    print()
    print(
        "Подсказка: ориентируйтесь на median/mean «боксов на кадр» под вашу полку; "
        "если при низком conf слишком много — поднимайте conf, пока визуально не станет приемлемо "
        "(лучше проверить yolo predict save=True на нескольких кадрах)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

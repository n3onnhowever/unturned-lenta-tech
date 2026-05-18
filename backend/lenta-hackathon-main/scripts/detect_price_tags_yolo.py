"""
Детекция ценников на кадрах через Ultralytics YOLO-World (open vocabulary).

Без собственной разметки используются текстовые классы (на английском —
так стабильнее эмбеддинги). Результат: CSV с bbox + опционально размеченные кадры.

Зависимость CLIP: см. requirements.txt (архив GitHub без git CLI). Распознавание
текста с ценника — отдельный шаг (OCR по crop из bbox), здесь только детекция.
Кропы с поворотом под PaddleOCR: scripts/export_ocr_crops.py (по умолчанию ccw90).

YOLO-World без своей модели путает мелкие наклейки на бутылках с ценниками —
промпты смещены на «ценник на полке», плюс фильтры по площади/стороне bbox
(--min-box-area, --min-box-short-side); при необходимости подстройте под ваш кадр.

Примеры:
  python scripts/detect_price_tags_yolo.py --source frames --output runs/price_tags
  python scripts/detect_price_tags_yolo.py --source frames --min-box-area 40000 --min-box-short-side 160
  python scripts/detect_price_tags_yolo.py --source frames/materials_data_43_15_43_15 --max-images 20 --viz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

# tqdm опционально
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


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
    parser = argparse.ArgumentParser(
        description="YOLO-World: детекция ценников на кадрах (bbox в CSV).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("frames"),
        help="Папка с кадрами (рекурсивно) или один файл изображения",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/price_tag_detect"),
        help="Каталог результатов (CSV, опционально viz/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8s-worldv2.pt",
        help="Веса YOLO-World (скачиваются при первом запуске)",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "supermarket shelf paper price tag with barcode on shelf edge",
            "rectangular store price card mounted on gondola shelf front",
            "printed retail shelf label with prices not on a bottle",
            "flat price ticket on supermarket shelf rail",
        ],
        help="Текстовые классы YOLO-World (англ.); избегайте «sticker» на бутылке",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Порог уверенности",
    )
    parser.add_argument(
        "--min-box-area",
        type=float,
        default=25000.0,
        help="Мин. площадь bbox в пикс² (0 = не фильтровать). Отсекает мелкие этикетки горлышка",
    )
    parser.add_argument(
        "--min-box-short-side",
        type=float,
        default=120.0,
        help="Мин. короткая сторона bbox в px (0 = не фильтровать)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Размер инференса (меньше — быстрее; 4K кадры масштабируются)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda:0 / cpu / пусто = авто",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Ограничить число кадров (отладка)",
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="Сохранять кадры с нарисованными боксами в output/viz/",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=8,
        help="Размер батча для predict (GPU; на CPU можно 1)",
    )
    args = parser.parse_args()

    images = collect_images(args.source)
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]
    if not images:
        print(f"Нет изображений под {args.source}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Установите: pip install ultralytics", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    model.set_classes(list(args.classes))

    args.output.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output / "viz"
    if args.viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.output / "detections.csv"
    device = args.device if args.device else None

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
                kept_xyxy: list[list[float]] = []
                kept_confs: list[float] = []
                kept_clss: list[int] = []
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
                    kept_xyxy.append([x1, y1, x2, y2])
                    kept_confs.append(float(cf))
                    kept_clss.append(cid)

                if args.viz and kept_xyxy:
                    im0 = getattr(r, "orig_img", None)
                    if im0 is None:
                        im0 = cv2.imread(str(src))
                    if im0 is not None:
                        vis = im0.copy()
                        for (x1, y1, x2, y2), cf, cid in zip(kept_xyxy, kept_confs, kept_clss):
                            cname = names_map.get(cid, str(cid))
                            p1 = (int(x1), int(y1))
                            p2 = (int(x2), int(y2))
                            cv2.rectangle(vis, p1, p2, (0, 200, 0), 2)
                            cv2.putText(
                                vis,
                                f"{cname[:28]} {cf:.2f}",
                                (p1[0], max(0, p1[1] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 200, 0),
                                1,
                                cv2.LINE_AA,
                            )
                        out_img = viz_dir / (src.stem + "_det.jpg")
                        cv2.imwrite(str(out_img), vis)

    print(f"Кадров: {len(images)}, детекций: {n_det}, CSV: {csv_path}")
    if args.viz:
        print(f"Визуализация: {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

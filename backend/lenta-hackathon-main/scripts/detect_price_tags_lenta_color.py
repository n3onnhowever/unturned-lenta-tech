"""
Детекция ценников Ленты по цвету/геометрии (без обученного YOLO).

Ориентир на макет: светлый верх (название, QR) + насыщенный оранжевый/красный/жёлтый
низ с крупной ценой и штрихкодом — как на эталонном ценнике. Так отсекаются наклейки
на бутылках и пустые участки полки.

Выход: тот же detections.csv, что и у YOLO-скрипта → дальше export_ocr_crops.py.

Пример:
  python scripts/detect_price_tags_lenta_color.py --source frames/materials_data_43_15_43_15 --output runs/price_tag_color --viz
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def collect_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_EXTS else []
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def lenta_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Оранжевый / красный / жёлтый низ типичного ценника (HSV, OpenCV hue 0..180)."""
    m_orange = cv2.inRange(hsv, np.array([8, 55, 60]), np.array([28, 255, 255]))
    m_red1 = cv2.inRange(hsv, np.array([0, 55, 50]), np.array([12, 255, 255]))
    m_red2 = cv2.inRange(hsv, np.array([168, 55, 50]), np.array([180, 255, 255]))
    m_yellow = cv2.inRange(hsv, np.array([18, 40, 100]), np.array([45, 255, 255]))
    return cv2.bitwise_or(m_orange, cv2.bitwise_or(m_red1, cv2.bitwise_or(m_red2, m_yellow)))


def nms_xyxy(boxes: list[list[float]], scores: list[float], iou_thr: float) -> list[int]:
    if not boxes:
        return []
    B = np.array(boxes, dtype=np.float32)
    x1, y1, x2, y2 = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = np.argsort(-np.array(scores))
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter + 1e-6
        iou = inter / union
        order = rest[iou < iou_thr]
    return keep


def tag_layout_ok(sat: np.ndarray, x1: int, y1: int, x2: int, y2: int, min_delta: float) -> bool:
    """Нижняя часть bbox по насыщенности выше верхней (типичный ценник: цветной низ)."""
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(sat.shape[1], x2)
    y2 = min(sat.shape[0], y2)
    if x2 <= x1 + 4 or y2 <= y1 + 4:
        return False
    roi = sat[y1:y2, x1:x2]
    h = roi.shape[0]
    split = max(1, int(h * 0.52))
    top = roi[:split, :]
    bot = roi[split:, :]
    return float(np.mean(bot)) >= float(np.mean(top)) + min_delta


def expand_tag_bbox(
    gray: np.ndarray,
    sat: np.ndarray,
    h0: int,
    w0: int,
    x: int,
    y: int,
    wc: int,
    hc: int,
    *,
    extend_frac: float,
    margin_side_frac: float,
    white_gray_min: float,
    color_sat_min: float,
) -> tuple[int, int, int, int] | None:
    """Ищет светлую полосу (верх макета) с одной из сторон цветного blob и объединяет bbox."""
    col_roi = sat[y : y + hc, x : x + wc]
    if col_roi.size == 0 or float(np.mean(col_roi)) < color_sat_min:
        return None

    ext_h = max(12, int(hc * extend_frac))
    ext_w = max(12, int(wc * extend_frac))

    best: tuple[float, int, int, int, int] | None = None  # mean_gray, x1,y1,x2,y2

    def consider(mean_g: float, xa: int, ya: int, xb: int, yb: int) -> None:
        nonlocal best
        if mean_g < white_gray_min:
            return
        if best is None or mean_g > best[0]:
            best = (mean_g, xa, ya, xb, yb)

    # полоса СЕВЕР (классический макет: белый верх)
    y1n = max(0, y - ext_h)
    roi = gray[y1n:y, x : x + wc]
    if roi.size:
        consider(float(np.mean(roi)), x, y1n, x + wc, y + hc)

    # ЗАПАД / ВОСТОК / ЮГ — для «бокового» кадра с полки
    x1w = max(0, x - ext_w)
    roi = gray[y : y + hc, x1w:x]
    if roi.size:
        consider(float(np.mean(roi)), x1w, y, x + wc, y + hc)

    x2e = min(w0, x + wc + ext_w)
    roi = gray[y : y + hc, x + wc : x2e]
    if roi.size:
        consider(float(np.mean(roi)), x, y, x2e, y + hc)

    y2s = min(h0, y + hc + ext_h)
    roi = gray[y + hc : y2s, x : x + wc]
    if roi.size:
        consider(float(np.mean(roi)), x, y, x + wc, y2s)

    if best is None:
        return None

    _, xa, ya, xb, yb = best
    x1 = max(0, int(min(xa, x) - wc * margin_side_frac))
    y1 = max(0, int(min(ya, y)))
    x2 = min(w0, int(max(xb, x + wc) + wc * margin_side_frac))
    y2 = min(h0, int(max(yb, y + hc)))
    return x1, y1, x2, y2


def detect_on_image(
    bgr: np.ndarray,
    *,
    morph_ksize: int,
    min_color_area: int,
    extend_top_frac: float,
    margin_side_frac: float,
    white_gray_min: float,
    color_sat_min: float,
    min_tag_short: int,
    max_ar: float,
    min_ar: float,
    max_box_frac: float,
    max_per_image: int,
    layout_sat_delta: float,
) -> tuple[list[list[float]], list[float]]:
    h0, w0 = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = lenta_color_mask(hsv)
    k = max(3, morph_ksize | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand: list[list[float]] = []
    scores: list[float] = []

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sat = hsv[:, :, 1].astype(np.float32)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_color_area:
            continue
        x, y, wc, hc = cv2.boundingRect(c)
        if wc < 8 or hc < 8:
            continue

        ex = expand_tag_bbox(
            gray,
            sat,
            h0,
            w0,
            x,
            y,
            wc,
            hc,
            extend_frac=extend_top_frac,
            margin_side_frac=margin_side_frac,
            white_gray_min=white_gray_min,
            color_sat_min=color_sat_min,
        )
        if ex is None:
            continue
        x1, y1, x2, y2 = ex
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) < min_tag_short:
            continue
        ar = max(bw, bh) / max(1.0, min(bw, bh))
        if ar < min_ar or ar > max_ar:
            continue
        if (bw * bh) > max_box_frac * w0 * h0:
            continue

        if not tag_layout_ok(sat, int(x1), int(y1), int(x2), int(y2), layout_sat_delta):
            continue

        cand.append([float(x1), float(y1), float(x2), float(y2)])
        scores.append(float(bw * bh))

    if not cand:
        return [], []

    keep_idx = nms_xyxy(cand, scores, iou_thr=0.42)
    cand = [cand[i] for i in keep_idx]
    scores = [scores[i] for i in keep_idx]
    order = sorted(range(len(cand)), key=lambda i: scores[i], reverse=True)
    order = order[: max(1, max_per_image)]
    return [cand[i] for i in order], [scores[i] for i in order]


def main() -> int:
    p = argparse.ArgumentParser(description="Ценники Ленты: детекция по цвету/форме.")
    p.add_argument("--source", type=Path, default=Path("frames"), help="Кадры (рекурсивно) или один файл")
    p.add_argument("--output", type=Path, default=Path("runs/price_tag_lenta_color"))
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--viz", action="store_true")
    p.add_argument("--morph", type=int, default=11, help="Размер ядра morphology (нечётное)")
    p.add_argument("--min-color-area", type=int, default=6500)
    p.add_argument("--extend-top-frac", type=float, default=1.0, help="Глубина поиска светлой полосы относительно размера цветного блока")
    p.add_argument("--margin-side-frac", type=float, default=0.08)
    p.add_argument("--white-gray-min", type=float, default=108.0, help="Средняя яркость светлой полосы (0-255)")
    p.add_argument("--color-sat-min", type=float, default=58.0, help="Средняя насыщенность цветной полосы")
    p.add_argument("--min-tag-short", type=int, default=72, help="Мин. короткая сторона итогового bbox (px)")
    p.add_argument("--min-ar", type=float, default=1.15)
    p.add_argument("--max-ar", type=float, default=5.2)
    p.add_argument("--max-box-frac", type=float, default=0.06)
    p.add_argument(
        "--max-tags-per-image",
        type=int,
        default=10,
        help="Макс. число bbox на кадр после фильтров (по площади)",
    )
    p.add_argument(
        "--layout-sat-delta",
        type=float,
        default=18.0,
        help="Мин. разница mean(S) нижней половины − верхней внутри bbox",
    )
    args = p.parse_args()

    images = collect_images(args.source)
    if args.max_images is not None:
        images = images[: max(0, args.max_images)]
    if not images:
        print(f"Нет изображений: {args.source}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output / "viz"
    if args.viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

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

    n = 0
    it = images
    if tqdm is not None:
        it = tqdm(images, desc="frames", unit="img")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for src in it:
            bgr = cv2.imread(str(src))
            if bgr is None:
                continue
            boxes, scores = detect_on_image(
                bgr,
                morph_ksize=args.morph,
                min_color_area=args.min_color_area,
                extend_top_frac=args.extend_top_frac,
                margin_side_frac=args.margin_side_frac,
                white_gray_min=args.white_gray_min,
                color_sat_min=args.color_sat_min,
                min_tag_short=args.min_tag_short,
                max_ar=args.max_ar,
                min_ar=args.min_ar,
                max_box_frac=args.max_box_frac,
                max_per_image=args.max_tags_per_image,
                layout_sat_delta=args.layout_sat_delta,
            )
            path_str = str(src.resolve())
            if args.viz:
                vis = bgr.copy()
            for bi, (x1, y1, x2, y2) in enumerate(boxes):
                sc = scores[bi] if bi < len(scores) else 1.0
                conf = min(0.99, 0.35 + 0.25 * math.log1p(sc / (bgr.shape[0] * bgr.shape[1] + 1.0)))
                w.writerow(
                    {
                        "image_path": path_str,
                        "class_id": 0,
                        "class_name": "lenta_price_tag_color",
                        "confidence": round(conf, 6),
                        "x_min": round(x1, 2),
                        "y_min": round(y1, 2),
                        "x_max": round(x2, 2),
                        "y_max": round(y2, 2),
                    }
                )
                n += 1
                if args.viz:
                    p1 = (int(x1), int(y1))
                    p2 = (int(x2), int(y2))
                    cv2.rectangle(vis, p1, p2, (0, 200, 0), 3)
                    cv2.putText(
                        vis,
                        f"tag {conf:.2f}",
                        (p1[0], max(0, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )
            if args.viz:
                out_v = viz_dir / (src.stem + "_color.jpg")
                cv2.imwrite(str(out_v), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    print(f"Кадров: {len(images)}, детекций: {n}, CSV: {csv_path}")
    if args.viz:
        print(f"viz: {viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

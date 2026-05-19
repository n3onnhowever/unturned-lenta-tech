"""Filter YOLO detections: confidence, size, aspect, per-image NMS."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--min-short-side", type=float, default=40.0)
    ap.add_argument("--min-aspect", type=float, default=0.2)
    ap.add_argument("--max-aspect", type=float, default=5.0)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.detections.open(encoding="utf-8")))
    kept: list[dict] = []
    for row in rows:
        try:
            conf = float(row.get("confidence", 0))
            x1, y1, x2, y2 = (
                float(row["x_min"]),
                float(row["y_min"]),
                float(row["x_max"]),
                float(row["y_max"]),
            )
        except (KeyError, ValueError):
            continue
        w, h = x2 - x1, y2 - y1
        if conf < args.min_conf or w <= 1 or h <= 1:
            continue
        aspect = w / h
        if aspect < args.min_aspect or aspect > args.max_aspect:
            continue
        if min(w, h) < args.min_short_side:
            continue
        kept.append(row)

    by_image: dict[str, list[dict]] = {}
    for row in kept:
        by_image.setdefault(row["image_path"], []).append(row)

    final: list[dict] = []
    for _img, group in by_image.items():
        group.sort(key=lambda r: float(r["confidence"]), reverse=True)
        selected: list[dict] = []
        for row in group:
            box = (
                float(row["x_min"]),
                float(row["y_min"]),
                float(row["x_max"]),
                float(row["y_max"]),
            )
            if any(
                iou(box, (float(s["x_min"]), float(s["y_min"]), float(s["x_max"]), float(s["y_max"])))
                >= args.nms_iou
                for s in selected
            ):
                continue
            selected.append(row)
        final.extend(selected)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(final[0].keys()) if final else list(rows[0].keys()) if rows else [
        "image_path",
        "class_id",
        "class_name",
        "confidence",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final)
    print(
        f"filter: {len(rows)} -> {len(kept)} (rules) -> {len(final)} (nms), conf>={args.min_conf}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

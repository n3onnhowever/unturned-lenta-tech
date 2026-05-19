"""Field ROIs and red-panel detection (from NIK price_tag_hybrid_baseline)."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def crop_rel(image: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = image.shape[:2]
    xa = max(0, min(w, int(round(x1 * w))))
    xb = max(0, min(w, int(round(x2 * w))))
    ya = max(0, min(h, int(round(y1 * h))))
    yb = max(0, min(h, int(round(y2 * h))))
    if xb <= xa or yb <= ya:
        return image[0:0, 0:0]
    return image[ya:yb, xa:xb]


def detect_red_panel(image: np.ndarray) -> dict[str, float] | None:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 50), (15, 255, 255))
    red2 = cv2.inRange(hsv, (165, 70, 50), (180, 255, 255))
    mask = ((red1 > 0) | (red2 > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    h, w = image.shape[:2]
    best = None
    for idx in range(1, count):
        x, y, ww, hh, area = [int(v) for v in stats[idx]]
        area_ratio = area / max(h * w, 1)
        if area_ratio < 0.04:
            continue
        candidate = {
            "x1": x / max(w, 1),
            "y1": y / max(h, 1),
            "x2": (x + ww) / max(w, 1),
            "y2": (y + hh) / max(h, 1),
            "area_ratio": area_ratio,
            "aspect": ww / max(hh, 1),
            "cy": (y + hh / 2.0) / max(h, 1),
        }
        if best is None or candidate["area_ratio"] > best["area_ratio"]:
            best = candidate
    return best


def infer_tag_color(image: np.ndarray) -> str:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    yellow = cv2.inRange(hsv, (15, 80, 80), (40, 255, 255))
    red_ratio = float(((red1 > 0) | (red2 > 0)).mean())
    yellow_ratio = float((yellow > 0).mean())
    if red_ratio >= yellow_ratio and red_ratio > 0.12:
        return "red"
    if yellow_ratio > 0.08:
        return "yellow"
    return ""


def field_rois(
    image: np.ndarray,
    red_box: dict[str, float] | None,
) -> dict[str, np.ndarray]:
    if red_box is not None:
        rx1, ry1, rx2, ry2 = red_box["x1"], red_box["y1"], red_box["x2"], red_box["y2"]
        red_w = max(rx2 - rx1, 0.01)
        red_h = max(ry2 - ry1, 0.01)
        return {
            "product_name": crop_rel(image, 0.03, 0.02, 0.97, max(0.16, ry1 - 0.03)),
            "price_default": crop_rel(
                image,
                min(0.95, rx1 + 0.36 * red_w),
                max(0.0, ry1 - 0.06 * red_h),
                min(0.98, rx1 + 0.88 * red_w),
                min(0.98, ry1 + 0.52 * red_h),
            ),
            "discount_amount": crop_rel(
                image,
                max(0.0, rx1),
                max(0.0, ry1),
                min(0.98, rx1 + 0.25 * red_w),
                min(0.98, ry1 + 0.50 * red_h),
            ),
            "price_card": crop_rel(
                image,
                max(0.0, rx1 + 0.03 * red_w),
                max(0.0, ry1 + 0.08 * red_h),
                min(0.98, rx1 + 0.42 * red_w),
                min(0.99, ry2),
            ),
            "barcode_text": crop_rel(
                image,
                max(0.0, rx1 + 0.48 * red_w),
                max(0.0, ry1 + 0.52 * red_h),
                min(0.99, rx2),
                min(0.99, ry2),
            ),
        }
    return {
        "product_name": crop_rel(image, 0.04, 0.02, 0.60, 0.42),
        "price_default": crop_rel(image, 0.58, 0.26, 0.98, 0.62),
        "discount_amount": crop_rel(image, 0.00, 0.50, 0.34, 0.90),
        "price_card": crop_rel(image, 0.28, 0.44, 0.90, 0.95),
        "barcode_text": crop_rel(image, 0.32, 0.78, 0.98, 1.00),
    }

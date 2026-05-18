"""
Hybrid Lenta price-tag parser: smart deskew + NIK field ROIs + Paddle/Rapid OCR + parse_ocr_fields.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lenta_barcode import is_valid_ean13, read_barcode_from_tag
from lenta_field_rois import detect_red_panel, field_rois, infer_tag_color
from lenta_price_normalize import normalize_price_display, parse_price_float
from parse_ocr_fields import (
    extract_discount_percent,
    extract_discount_percent_from_zone,
    extract_lenta_prices,
    format_price_value,
    infer_price_default_from_card,
    normalize_ocr_text,
    refine_prices_with_discount,
)
import logging
logger = logging.getLogger(__name__)
from process_tags_smart_deskew import smart_deskew

DISCOUNT_INFER_MIN = 5
DISCOUNT_INFER_MAX = 40


def _ocr_roi_text(roi: np.ndarray, engine: str = "paddle") -> str:
    if roi is None or roi.size == 0:
        return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants = [roi, clahe, cv2.resize(clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)]
    best_text = ""
    best_score = -1.0
    for variant in variants:
        if variant.ndim == 2:
            img = cv2.cvtColor(variant, cv2.COLOR_GRAY2BGR)
        else:
            img = variant
        text = _run_ocr_engine(img, engine)
        score = len(text) + 0.3 * sum(c.isdigit() for c in text)
        if score > best_score:
            best_score = score
            best_text = text
    return best_text


def _run_ocr_engine(img_bgr: np.ndarray, engine: str) -> str:
    if engine == "rapidocr":
        try:
            from ocr_engines import run_engine

            return run_engine("rapidocr", img_bgr).full_text.replace("\n", " ")
        except Exception:
            engine = "paddle"
    from ocr_engines import run_engine

    return run_engine(engine, img_bgr).full_text.replace("\n", " ")


def _clamp_discount_pct(pct: int | None) -> int | None:
    if pct is None:
        return None
    if DISCOUNT_INFER_MIN <= pct <= DISCOUNT_INFER_MAX:
        return pct
    return None


def _price_str_from_roi_text(text: str, prefer: str = "low") -> str:
    prices = extract_lenta_prices(text)
    if not prices:
        return ""
    val = prices[0] if prefer == "high" else prices[-1]
    if val < 50:
        val = round(val * 100, 2)
    return format_price_value(val) if val < 50 else normalize_price_display(val)


def _price_candidates_from_zone_text(text: str) -> list[float]:
    """Weak OCR fallback for large shelf-card prices like `234` -> `234.99`."""
    candidates: list[float] = []
    norm = normalize_ocr_text(text or "")
    tokens = [
        m.group(0)
        for m in re.finditer(r"(?<![A-Za-zА-Яа-я0-9])\d{2,4}(?![A-Za-zА-Яа-я0-9])", norm)
    ]
    for token in tokens:
        # Relaxed SKU prefix filter: previously blocked 500/700/900 entirely,
        # which also blocked legitimate high prices like 599.99 or 799.00.
        # Now we allow them and rely on downstream validation.
        val = int(token)
        if 50 <= val <= 9999:
            candidates.append(float(val) + 0.99)
            candidates.append(float(val))
        if len(token) == 4:
            rub = int(token[:3])
            cents = int(token[3:])
            if 50 <= rub <= 999 and cents <= 99:
                candidates.append(rub + cents / 100.0)

    nums = [
        m.group(0)
        for m in re.finditer(r"(?<![A-Za-zА-Яа-я0-9])\d+(?![A-Za-zА-Яа-я0-9])", norm)
    ]
    for i, rub_s in enumerate(nums[:-1]):
        cents_s = nums[i + 1]
        if 2 <= len(rub_s) <= 3 and len(cents_s) == 2:
            rub = int(rub_s)
            cents = int(cents_s)
            if 50 <= rub <= 999 and cents <= 99:
                candidates.append(rub + cents / 100.0)

    out: list[float] = []
    for value in candidates:
        if 50 <= value <= 9999 and all(abs(value - old) > 0.01 for old in out):
            out.append(round(value, 2))
    return out


def _price_zone_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if image is None or image.size == 0:
        return []
    h, w = image.shape[:2]
    variants: list[tuple[str, np.ndarray]] = []
    if h > 4 and w > 4:
        variants.append(("bottom60_x3", image[int(h * 0.40) : h, :]))
        variants.append(("right65_x3", image[int(h * 0.35) : h, int(w * 0.35) : w]))

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 35, 40), (15, 255, 255))
    mask2 = cv2.inRange(hsv, (165, 35, 40), (179, 255, 255))
    mask = cv2.morphologyEx(mask1 | mask2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, ww, hh = cv2.boundingRect(c)
        pad = 12
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + ww + pad)
        y2 = min(h, y + hh + pad)
        if (x2 - x1) * (y2 - y1) > 100:
            variants.append(("redmask_x4", image[y1:y2, x1:x2]))
    return [(name, crop) for name, crop in variants if crop is not None and crop.size > 0]


def _price_zone_fallback(image: np.ndarray, engine: str = "rapidocr") -> dict[str, Any]:
    best: dict[str, Any] = {
        "price_card": "",
        "price_default": "",
        "discount_pct": None,
        "text": "",
        "variant": "",
        "engine": engine,
        "candidates": [],
    }
    for name, crop in _price_zone_variants(image):
        scale = 4.0 if name.startswith("redmask") else 3.0
        up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        text = _run_ocr_engine(up, engine)
        candidates = _price_candidates_from_zone_text(text)
        discount_pct = _clamp_discount_pct(extract_discount_percent(text))
        if discount_pct is None:
            discount_pct = _clamp_discount_pct(extract_discount_percent_from_zone(text))
        card = candidates[0] if candidates else None
        default = infer_price_default_from_card(card, discount_pct) if card and discount_pct else None
        best = {
            "price_card": format_price_value(card) if card is not None else "",
            "price_default": format_price_value(default) if default is not None else "",
            "discount_pct": discount_pct,
            "text": text,
            "variant": name,
            "engine": engine,
            "candidates": candidates,
        }
        if candidates and discount_pct is not None:
            return best
        if candidates and not best["price_card"]:
            return best
        if candidates and best["price_card"]:
            return best
    return best


def _name_from_roi_text(text: str) -> str:
    lines = [ln.strip() for ln in re.split(r"[|\n]", text) if ln.strip()]
    best = ""
    best_score = -1.0
    for line in lines:
        norm = normalize_ocr_text(line)
        if len(line) < 3:
            continue
        digits = sum(c.isdigit() for c in norm)
        if digits / max(len(line), 1) >= 0.5:
            continue
        low = line.lower()
        if any(token in low for token in ("руб", "шт", "скид", "цена")):
            continue
        letters = sum(ch.isalpha() for ch in line)
        score = letters + 0.25 * len(line) - 0.75 * digits
        if score > best_score:
            best_score = score
            best = line
    return best


def parse_tag_image(
    image_bgr: np.ndarray,
    *,
    deskew: bool = True,
    deskew_pad_ratio: float = 0.25,
    engine: str = "paddle",
) -> dict[str, Any]:
    fallback = parse_baseline_fullcrop(
        image_bgr, deskew=deskew, engine=engine, deskew_pad_ratio=deskew_pad_ratio
    )

    image = image_bgr
    if deskew:
        image, _ok = smart_deskew(image_bgr, pad_ratio=deskew_pad_ratio)
    red_box = detect_red_panel(image)
    rois = field_rois(image, red_box)

    roi_text = {k: _ocr_roi_text(v, engine) for k, v in rois.items()}

    product_name = _name_from_roi_text(roi_text.get("product_name", ""))
    price_card_s = _price_str_from_roi_text(roi_text.get("price_card", ""), prefer="low")
    price_default_s = _price_str_from_roi_text(roi_text.get("price_default", ""), prefer="high")

    disc_raw = extract_discount_percent(roi_text.get("discount_amount", ""))
    if disc_raw is None:
        disc_raw = extract_discount_percent_from_zone(roi_text.get("discount_amount", ""))
    disc_pct = _clamp_discount_pct(disc_raw)

    parsed: dict[str, Any] = {
        "product_name": product_name or fallback.get("product_name", ""),
        "price_default": price_default_s or fallback.get("price_default", ""),
        # Do NOT fallback price_card to price_default here.
        # Doing so breaks discount inference (infer_price_default_from_card)
        # because card would equal default.
        "price_card": price_card_s or fallback.get("price_card", ""),
        "discount_amount": f"-{disc_pct}%" if disc_pct else (f"-{disc_raw}%" if disc_raw else ""),
        "price_default_inferred": False,
        "barcode": "",
        "template": "red_panel" if red_box else "fallback",
    }
    price_zone = _price_zone_fallback(image_bgr, engine="rapidocr")
    if not parsed["price_card"] and price_zone.get("price_card"):
        parsed["price_card"] = price_zone["price_card"]
        parsed["template"] = f"{parsed['template']}+price_zone"
    if not parsed["discount_amount"] and price_zone.get("discount_pct"):
        parsed["discount_amount"] = f"-{price_zone['discount_pct']}%"
        disc_pct = int(price_zone["discount_pct"])
    if not parsed["price_default"] and price_zone.get("price_default"):
        parsed["price_default"] = price_zone["price_default"]
        parsed["price_default_inferred"] = True
    if not parsed["discount_amount"] and fallback.get("discount_amount"):
        parsed["discount_amount"] = fallback["discount_amount"]

    card_f = parse_price_float(parsed["price_card"])
    if disc_pct and card_f > 0:
        inferred = infer_price_default_from_card(
            card_f if card_f >= 50 else card_f * 100, disc_pct
        )
        if inferred is None and card_f < 50:
            inferred = infer_price_default_from_card(card_f, disc_pct)
            if inferred and inferred < 50:
                inferred = round(inferred * 100, 2)
        if inferred is not None:
            parsed = refine_prices_with_discount(
                parsed,
                discount_pct=disc_pct,
                card_price=card_f if card_f >= 50 else card_f * 100,
            )
            if parsed.get("price_default_inferred") and inferred >= 50:
                parsed["price_default"] = format_price_value(inferred)

    parsed["color"] = infer_tag_color(image)
    ocr_blob = " ".join(roi_text.values())
    parsed["barcode"] = read_barcode_from_tag(
        image,
        ocr_texts=[
            ocr_blob,
            fallback.get("_ocr_full_text", ""),
            _barcode_from_texts(list(roi_text.values()) + [fallback.get("_ocr_full_text", "")]),
        ],
    ) or fallback.get("barcode", "") or _barcode_from_texts(list(roi_text.values()))
    parsed["_roi_text"] = roi_text
    parsed["_fallback"] = fallback
    parsed["_price_zone_fallback"] = price_zone
    return parsed


def _barcode_from_texts(texts: list[str]) -> str:
    candidates: list[str] = []
    for text in texts:
        for m in re.findall(r"\d{8,14}", text or ""):
            if len(m) >= 8:
                candidates.append(m)
    valid = [item for item in candidates if is_valid_ean13(item)]
    if valid:
        valid.sort(key=lambda item: (item.startswith("46"), len(item)), reverse=True)
        return valid[0]
    # Do not return an invalid barcode just because it is the longest.
    # Return empty so downstream can decide (or leave blank).
    return ""


def parse_baseline_fullcrop(
    image_bgr: np.ndarray,
    *,
    deskew: bool = True,
    deskew_pad_ratio: float = 0.25,
    engine: str = "paddle",
) -> dict[str, Any]:
    """Legacy: one OCR pass on full deskewed crop + parse_ocr_text."""
    from parse_ocr_fields import parse_ocr_text

    image = image_bgr
    if deskew:
        image, _ = smart_deskew(image_bgr, pad_ratio=deskew_pad_ratio)
    text = _run_ocr_engine(image, engine)
    parsed = parse_ocr_text(text.replace("\n", " | "))
    parsed.setdefault("price_default_inferred", False)
    parsed.setdefault("barcode", _barcode_from_texts([text]))
    parsed["template"] = "fullcrop"
    parsed["_ocr_full_text"] = text
    return parsed

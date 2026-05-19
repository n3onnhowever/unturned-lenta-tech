"""Barcode decode for price-tag crops (pyzbar + OCR fallback)."""
from __future__ import annotations

import re
import json
from urllib.parse import parse_qsl, urlsplit
from typing import Iterable

import cv2
import numpy as np

from lenta_field_rois import detect_red_panel, field_rois

_OPENCV_BARCODE_DETECTOR = None
_OPENCV_QR_DETECTOR = None


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def ean13_check_digit(value: str) -> int:
    digits = [int(ch) for ch in value[:12]]
    odd = sum(digits[::2])
    even = sum(digits[1::2])
    return (10 - ((odd + 3 * even) % 10)) % 10


def is_valid_ean13(value: str) -> bool:
    text = _digits(value)
    if len(text) != 13:
        return False
    try:
        return ean13_check_digit(text) == int(text[-1])
    except ValueError:
        return False


def _pick_best_barcode(candidates: Iterable[str]) -> str:
    uniq = list(dict.fromkeys(_digits(c) for c in candidates if len(_digits(c)) >= 8))
    valid = [c for c in uniq if is_valid_ean13(c)]
    if valid:
        valid.sort(key=lambda x: (x.startswith("46"), len(x)), reverse=True)
        return valid[0]
    if not uniq:
        return ""
    uniq.sort(key=len, reverse=True)
    return uniq[0]


def _decoded_to_strings(decoded: object) -> list[str]:
    if decoded is None:
        return []
    if isinstance(decoded, bytes):
        text = decoded.decode("utf-8", errors="ignore")
        return [text] if text else []
    if isinstance(decoded, str):
        return [decoded] if decoded else []
    if isinstance(decoded, np.ndarray):
        return _decoded_to_strings(decoded.tolist())
    if isinstance(decoded, (list, tuple, set)):
        values: list[str] = []
        for item in decoded:
            values.extend(_decoded_to_strings(item))
        return values
    text = str(decoded)
    return [text] if text else []


def decode_barcodes_pyzbar(image: np.ndarray) -> list[str]:
    try:
        from pyzbar.pyzbar import decode as zbar_decode
    except ImportError:
        return []

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    hits: list[str] = []
    bases = (gray, clahe, otsu, adaptive)
    for base in bases:
        for fx in (1.0, 2.0, 3.0, 4.0):
            variant = base
            if fx != 1.0:
                variant = cv2.resize(
                    base, None, fx=fx, fy=fx, interpolation=cv2.INTER_LANCZOS4
                )
            try:
                for code in zbar_decode(variant):
                    text = _digits(code.data.decode("utf-8", errors="ignore"))
                    if len(text) >= 8:
                        hits.append(text)
            except Exception:
                continue
    return hits


def _barcode_decode_images(image: np.ndarray, *, exhaustive: bool = False) -> list[np.ndarray]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bases = [gray, clahe] if not exhaustive else [image, gray, clahe, otsu]
    variants: list[np.ndarray] = []
    for base in bases:
        rotations = [base]
        if exhaustive:
            rotations.extend(
                [
                    cv2.rotate(base, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(base, cv2.ROTATE_90_COUNTERCLOCKWISE),
                    cv2.rotate(base, cv2.ROTATE_180),
                ]
            )
        for rot in rotations:
            for fx in ((1.0, 2.0) if not exhaustive else (1.0, 2.0, 3.0)):
                variants.append(
                    rot
                    if fx == 1.0
                    else cv2.resize(rot, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
                )
    return variants


def decode_barcodes_opencv(image: np.ndarray, *, exhaustive: bool = False) -> list[str]:
    global _OPENCV_BARCODE_DETECTOR
    if not hasattr(cv2, "barcode_BarcodeDetector"):
        return []
    try:
        if _OPENCV_BARCODE_DETECTOR is None:
            _OPENCV_BARCODE_DETECTOR = cv2.barcode_BarcodeDetector()
        detector = _OPENCV_BARCODE_DETECTOR
    except Exception:
        return []

    hits: list[str] = []
    for variant in _barcode_decode_images(image, exhaustive=exhaustive):
        try:
            _ok, decoded, _points = detector.detectAndDecode(variant)
        except Exception:
            continue
        values = _decoded_to_strings(decoded)
        for value in values:
            text = _digits(value)
            if len(text) >= 8:
                hits.append(text)
    return hits


def decode_qr_barcodes_opencv(image: np.ndarray, *, exhaustive: bool = False) -> list[str]:
    return [
        m
        for text in decode_qr_payloads_opencv(image, exhaustive=exhaustive)
        for m in re.findall(r"\d{8,14}", text or "")
    ]


def decode_qr_payloads_opencv(image: np.ndarray, *, exhaustive: bool = False) -> list[str]:
    global _OPENCV_QR_DETECTOR
    if _OPENCV_QR_DETECTOR is None:
        _OPENCV_QR_DETECTOR = cv2.QRCodeDetector()
    detector = _OPENCV_QR_DETECTOR
    hits: list[str] = []
    for variant in _barcode_decode_images(image, exhaustive=exhaustive):
        texts: list[str] = []
        try:
            text, _points, _straight = detector.detectAndDecode(variant)
            if text:
                texts.append(text)
        except Exception:
            pass
        try:
            ok, decoded, _points, _straights = detector.detectAndDecodeMulti(variant)
            if ok:
                texts.extend(_decoded_to_strings(decoded))
        except Exception:
            pass
        for text in texts:
            if text:
                hits.append(text)
    return list(dict.fromkeys(hits))


_QR_FIELD_ALIASES = {
    "qr_code_barcode": ("barcode", "b"),
    "price1_qr": ("price1", "p1"),
    "price2_qr": ("price2", "p2"),
    "price3_qr": ("price3", "p3"),
    "price4_qr": ("price4", "p4"),
    "wholesale_level_1_count": ("wholesaleLevel1Count", "wL1C"),
    "wholesale_level_1_price": ("wholesaleLevel1Price", "wL1P"),
    "wholesale_level_2_count": ("wholesaleLevel2Count", "wL2C"),
    "wholesale_level_2_price": ("wholesaleLevel2Price", "wL2P"),
    "action_price_qr": ("actionPrice", "aP"),
    "action_code_qr": ("actionCode", "aC"),
}


def _parse_qr_payload(text: str) -> dict[str, str]:
    raw = text or ""
    pairs: dict[str, str] = {}
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, dict):
            pairs.update({str(k): str(v) for k, v in decoded.items() if v is not None})
    except Exception:
        pass
    query = urlsplit(raw).query
    if not pairs:
        for key, value in parse_qsl(query or raw, keep_blank_values=True):
            if key:
                pairs[key] = value
    if not pairs:
        for key, value in re.findall(r"([A-Za-z][A-Za-z0-9_]*)\s*[:=]\s*([^;&|,\s]+)", raw):
            pairs[key] = value

    out: dict[str, str] = {}
    for target, aliases in _QR_FIELD_ALIASES.items():
        for alias in aliases:
            if alias in pairs and str(pairs[alias]).strip():
                out[target] = str(pairs[alias]).strip()
                break
    return out


def read_qr_fields_from_tag(image_bgr: np.ndarray, *, exhaustive: bool = False) -> dict[str, str]:
    merged: dict[str, str] = {}
    for payload in decode_qr_payloads_opencv(image_bgr, exhaustive=exhaustive):
        for key, value in _parse_qr_payload(payload).items():
            merged.setdefault(key, value)

    red = detect_red_panel(image_bgr)
    rois = field_rois(image_bgr, red)
    for name in ("barcode_text", "qr"):
        roi = rois.get(name)
        if roi is None or roi.size == 0:
            continue
        for payload in decode_qr_payloads_opencv(roi, exhaustive=exhaustive):
            for key, value in _parse_qr_payload(payload).items():
                merged.setdefault(key, value)
    return merged


def read_barcode_from_tag(
    image_bgr: np.ndarray,
    ocr_texts: list[str] | None = None,
    *,
    include_qr: bool = False,
    exhaustive: bool = False,
) -> str:
    """Full crop + barcode ROI + optional OCR strings."""
    candidates: list[str] = []
    candidates.extend(decode_barcodes_pyzbar(image_bgr))
    candidates.extend(decode_barcodes_opencv(image_bgr, exhaustive=exhaustive))
    if include_qr:
        candidates.extend(decode_qr_barcodes_opencv(image_bgr, exhaustive=exhaustive))

    red = detect_red_panel(image_bgr)
    rois = field_rois(image_bgr, red)
    bc_roi = rois.get("barcode_text")
    if bc_roi is not None and bc_roi.size > 0:
        candidates.extend(decode_barcodes_pyzbar(bc_roi))
        candidates.extend(decode_barcodes_opencv(bc_roi, exhaustive=exhaustive))
        if include_qr:
            candidates.extend(decode_qr_barcodes_opencv(bc_roi, exhaustive=exhaustive))

    for text in ocr_texts or []:
        for m in re.findall(r"\d{8,14}", text or ""):
            candidates.append(m)

    return _pick_best_barcode(candidates)

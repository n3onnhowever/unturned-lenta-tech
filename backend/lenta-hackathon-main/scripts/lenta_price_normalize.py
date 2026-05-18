"""Normalize Lenta shelf/catalog prices for fair comparison (3.05 vs 305.99)."""
from __future__ import annotations

import re
from typing import Iterable


def parse_price_float(s: str) -> float:
    s = (s or "").strip().replace(" ", "").replace(",", ".")
    if not s or s.lower() in {"нет", "nan"}:
        return -1.0
    try:
        return float(s)
    except ValueError:
        return -1.0


def price_candidates_rub(value: float) -> list[float]:
    """OCR may return rubles (305.99) or mis-scaled display (3.05)."""
    if value < 0:
        return []
    out: list[float] = [round(value, 2)]
    if value < 50:
        out.append(round(value * 100, 2))
        out.append(round(value * 100 + 0.99 - (value * 100) % 1, 2))
    if value > 50:
        out.append(round(value / 100.0, 2))
    # de-dup close values
    uniq: list[float] = []
    for v in out:
        if not any(abs(v - u) < 0.02 for u in uniq):
            uniq.append(v)
    return uniq


def best_price_match(pred: str, gt: str, tol: float = 2.0) -> bool:
    gp = parse_price_float(pred)
    gg = parse_price_float(gt)
    if gp < 0 and gg < 0:
        return True
    if gp < 0 or gg < 0:
        return False
    for pc in price_candidates_rub(gp):
        for gc in price_candidates_rub(gg):
            if abs(pc - gc) <= tol:
                return True
    return False


def normalize_price_display(value: float) -> str:
    """Prefer catalog-scale rubles in outputs when value looks like OCR cents block."""
    if value < 0:
        return ""
    if 0 < value < 50:
        value = round(value * 100, 2)
    return f"{value:.2f}".rstrip("0").rstrip(".") + (
        ".0" if "." not in f"{value:.2f}".rstrip("0").rstrip(".") else ""
    )


def discount_percent_value(s: str) -> int | None:
    m = re.search(r"(\d{1,2})", (s or "").replace(" ", ""))
    if not m:
        return None
    return int(m.group(1))

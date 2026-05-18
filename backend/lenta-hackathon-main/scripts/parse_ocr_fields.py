import re
import json
import csv
import argparse
from pathlib import Path
from typing import Optional

# OCR confusions on price tags (Cyrillic / Latin -> digit)
_CHAR_TO_DIGIT = str.maketrans({
    "О": "0", "о": "0", "O": "0", "o": "0", "Q": "0", "D": "0",
    "З": "3", "з": "3",
    "Б": "6", "б": "6",
    "В": "8", "в": "8", "B": "8",
    "І": "1", "і": "1", "I": "1", "l": "1", "|": "1",
    "Г": "4", "г": "4",
    "Ч": "4", "ч": "4",
    "S": "5", "s": "5",
    "G": "6", "g": "9",
    "T": "7", "t": "7",
    "З": "3",
})

_PRICE_PAIR = re.compile(r"\b(\d{2,4})\s*[^\d]{0,4}(\d{2})\b")
_PRICE_DECIMAL = re.compile(r"\b(\d{1,4})[.,](\d{2})\b")
_DIGIT_TOKEN = re.compile(r"\b(\d{2,4})\b")
_DISCOUNT_PCT = re.compile(r"[-−]?\s*(\d{1,2})\s*[%°‰¢]")
_DISCOUNT_MINUS_NUM = re.compile(r"[-−]\s*(\d{1,2})(?:\D|$)")
_LONG_DIGIT_RUN = re.compile(r"\b\d{6,14}\b")
_DATE_LIKE = re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b")


def normalize_ocr_text(text: str) -> str:
    if not text:
        return ""
    return text.translate(_CHAR_TO_DIGIT)


def _price_in_shelf_range(val: float, max_rub: float = 9999.99) -> bool:
    return 0.5 <= val <= max_rub


def _price_in_catalog_range(val: float) -> bool:
    return 0.5 <= val <= 9999.99


def _lenta_pair_to_price(main: str, cents: str) -> Optional[float]:
    try:
        m, c = int(main), int(cents)
    except ValueError:
        return None
    if c < 0 or c > 99:
        return None
    # Always treat as rubles.cents (e.g. 1899 99 -> 1899.99).
    # Previously, m>=100 was divided by 100, breaking high prices like 1899.99.
    val = m + c / 100.0
    return val if _price_in_shelf_range(val) else None


def _remove_non_price_number_context(text: str) -> str:
    """Drop numeric contexts that are much more likely barcode/SKU/date than price."""
    text = _LONG_DIGIT_RUN.sub(" ", text)
    text = _DATE_LIKE.sub(" ", text)
    return text


def extract_lenta_prices(text: str) -> list[float]:
    """Prices on Lenta shelf tags: 368+99 -> 3.68, 305|99 -> 3.05, also 12.99."""
    norm = _remove_non_price_number_context(normalize_ocr_text(text))
    found: list[float] = []

    for m, c in _PRICE_PAIR.findall(norm):
        val = _lenta_pair_to_price(m, c)
        if val is not None:
            found.append(val)

    for m, c in _PRICE_DECIMAL.findall(norm):
        val = _lenta_pair_to_price(m, c)
        if val is not None:
            found.append(val)

    # Lone 3-4 digit fragments often cents-less ruble display (368 -> 3.68).
    # Treat them as weak candidates only after removing long SKU/barcode runs.
    tokens = _DIGIT_TOKEN.findall(norm)
    for i, tok in enumerate(tokens):
        if len(tok) >= 3:
            val = int(tok) / 100.0
            if _price_in_shelf_range(val):
                found.append(val)
        if i + 1 < len(tokens) and len(tokens[i + 1]) == 2:
            val = _lenta_pair_to_price(tok, tokens[i + 1])
            if val is not None:
                found.append(val)

    return sorted(set(round(p, 2) for p in found), reverse=True)


def extract_discount_percent(text: str) -> Optional[int]:
    """Lenta tags: discount like -23% (OCR may read % as °)."""
    norm = normalize_ocr_text(text)
    for pattern in (_DISCOUNT_PCT,):
        for m in pattern.finditer(norm):
            pct = int(m.group(1))
            if DISCOUNT_INFER_MIN <= pct <= 80:
                return pct
    for m in _DISCOUNT_MINUS_NUM.finditer(norm):
        pct = int(m.group(1))
        if DISCOUNT_INFER_MIN <= pct <= 80:
            return pct
    return None


def extract_discount_percent_from_zone(text: str) -> Optional[int]:
    """
    Left strip of the price block: often only «-23» without a readable % sign.
    """
    explicit = extract_discount_percent(text)
    if explicit is not None:
        return explicit
    norm = normalize_ocr_text(text)
    candidates: list[int] = []
    for m in re.finditer(r"\b(\d{1,2})\b", norm):
        v = int(m.group(1))
        if 5 <= v <= 80:
            candidates.append(v)
    if not candidates:
        return None
    # Prefer typical Lenta discount steps when several two-digit numbers appear.
    for preferred in (36, 30, 25, 24, 23, 21, 19, 16, 15, 14, 13, 11):
        if preferred in candidates:
            return preferred
    return max(candidates)


DISCOUNT_INFER_MIN = 5
DISCOUNT_INFER_MAX = 40


def infer_price_default_from_card(card: float, discount_pct: int) -> Optional[float]:
    if not (DISCOUNT_INFER_MIN <= discount_pct <= DISCOUNT_INFER_MAX):
        return None
    denom = 1.0 - discount_pct / 100.0
    if denom <= 0.05:
        return None
    val = round(card / denom, 2)
    ok = _price_in_catalog_range if card >= 50 else _price_in_shelf_range
    return val if ok(val) else None


def _discount_matches_prices(card: float, default: float, discount_pct: int, tol: float = 0.12) -> bool:
    expected = infer_price_default_from_card(card, discount_pct)
    if expected is None:
        return False
    return abs(default - expected) / max(expected, 0.01) <= tol


def refine_prices_with_discount(
    parsed: dict,
    discount_pct: Optional[int] = None,
    card_price: Optional[float] = None,
) -> dict:
    """Apply card + % -> default when zone OCR gives clearer signals than full crop."""
    out = dict(parsed)
    pct = discount_pct if discount_pct is not None else extract_discount_percent(
        out.get("discount_amount", "")
    )
    if pct is None:
        return out

    out["discount_amount"] = f"-{pct}%"
    card = card_price
    if card is None and out.get("price_card"):
        try:
            card = float(str(out["price_card"]).replace(",", "."))
        except ValueError:
            card = None

    default = None
    if out.get("price_default"):
        try:
            default = float(str(out["price_default"]).replace(",", "."))
        except ValueError:
            default = None

    if card is None:
        return out

    out["price_card"] = format_price_value(card)
    inferred = infer_price_default_from_card(card, pct)
    if inferred is None:
        return out

    use_inferred = default is None or default <= card + 0.01
    if default is not None and not use_inferred:
        use_inferred = not _discount_matches_prices(card, default, pct)
    if use_inferred:
        out["price_default"] = format_price_value(inferred)
        out["price_default_inferred"] = True
    elif default is not None:
        out["price_default"] = format_price_value(default)

    return out


def format_price_value(val: float) -> str:
    s = f"{val:.2f}".rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def price_fields_to_display_line(
    price_default: str, price_card: str, raw_text: str = ""
) -> str:
    """Approximate price-zone text for rec training (368 99 style)."""
    tokens: list[str] = []
    for p in (price_default, price_card):
        p = (p or "").strip().replace(",", ".")
        if not p:
            continue
        try:
            v = float(p)
        except ValueError:
            continue
        rub = int(round(v * 100))
        if rub >= 100:
            tokens.append(str(rub))
        elif v >= 1:
            tokens.append(str(int(v)))
    if tokens:
        return " ".join(dict.fromkeys(tokens))
    norm = normalize_ocr_text(raw_text or "")
    nums = re.findall(r"\b\d{2,4}\b", norm)
    return " ".join(nums[:6]) if nums else ""


def is_tag_readable(parsed: dict, raw_text: str = "") -> bool:
    """False => export crop coordinates for manual review."""
    name = (parsed.get("product_name") or "").strip()
    pd = (parsed.get("price_default") or "").strip()
    pc = (parsed.get("price_card") or "").strip()
    if not raw_text or not raw_text.strip():
        return False
    has_name = len(name) >= 3 and sum(c.isdigit() for c in name) / max(len(name), 1) < 0.55
    has_price = bool(pd or pc)
    return has_name or has_price


def parse_ocr_text(text: str) -> dict:
    result = {
        "product_name": "",
        "price_default": "",
        "price_card": "",
        "discount_amount": "",
        "price_default_inferred": False,
    }
    if not text:
        return result

    lines = [line.strip() for line in text.split("|") if line.strip()]
    discount_pct = extract_discount_percent(text)
    if discount_pct is not None:
        result["discount_amount"] = f"-{discount_pct}%"

    prices = extract_lenta_prices(text)
    card: Optional[float] = None
    default: Optional[float] = None

    if len(prices) >= 2:
        default, card = prices[0], prices[-1]
    elif len(prices) == 1:
        card = prices[0]

    if discount_pct is not None and card is not None:
        inferred = infer_price_default_from_card(card, discount_pct)
        if inferred is not None:
            use_inferred = default is None or default <= card + 0.01
            if default is not None and not use_inferred:
                use_inferred = not _discount_matches_prices(card, default, discount_pct)
            if use_inferred:
                default = inferred
                result["price_default_inferred"] = True

    if card is not None:
        result["price_card"] = format_price_value(card)
    if default is not None:
        result["price_default"] = format_price_value(default)
    elif card is not None and not result["price_default"]:
        result["price_default"] = result["price_card"]

    name_lines = []
    for line in lines:
        line_norm = normalize_ocr_text(line)
        digits = sum(c.isdigit() for c in line_norm)
        if len(line) > 3 and digits / len(line) < 0.5:
            if not re.search(r"\d{2}\.\d{2}\.\d{4}", line_norm):
                low = line.lower()
                if "шт" not in low and not re.fullmatch(r"[\d\s\.,\-+%]+", line_norm):
                    name_lines.append(line.strip())

    if name_lines:
        name_lines.sort(key=len, reverse=True)
        result["product_name"] = name_lines[0]

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse fields from OCR CSV output.")
    parser.add_argument("--ocr-csv", type=str, required=True, help="Path to OCR results CSV")
    parser.add_argument("--out", type=str, required=True, help="Output JSON with parsed fields")
    args = parser.parse_args()

    ocr_path = Path(args.ocr_csv)
    out_path = Path(args.out)

    if not ocr_path.exists():
        print(f"File not found: {ocr_path}")
        return

    parsed_results = []

    with open(ocr_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("full_text", "")
            parsed = parse_ocr_text(text)

            parsed_results.append({
                "image_path": row.get("image_path"),
                "engine": row.get("engine"),
                "raw_text": text,
                "parsed": parsed,
            })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(parsed_results)} OCR results. Saved to {out_path}")


if __name__ == "__main__":
    main()

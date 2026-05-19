"""Hackathon CSV field formatting (Lenta comma decimals, "нет")."""
from __future__ import annotations

NO_VALUE = "нет"

HACKATHON_COLUMNS = [
    "filename",
    "product_name",
    "price_default",
    "price_card",
    "price_discount",
    "barcode",
    "discount_amount",
    "id_sku",
    "print_datetime",
    "code",
    "additional_info",
    "color",
    "special_symbols",
    "frame_timestamp",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "qr_code_barcode",
    "price1_qr",
    "price2_qr",
    "price3_qr",
    "price4_qr",
    "wholesale_level_1_count",
    "wholesale_level_1_price",
    "wholesale_level_2_count",
    "wholesale_level_2_price",
    "action_price_qr",
    "action_code_qr",
]


def format_price_csv(value: str) -> str:
    s = (value or "").strip().replace(" ", "")
    if not s:
        return ""
    s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return s
    if v < 50 and v > 0:
        v = round(v * 100, 2)
    text = f"{v:.2f}".replace(".", ",")
    return text


def format_qr_price_csv(value: str) -> str:
    s = (value or "").strip().replace(" ", "")
    if not s:
        return ""
    s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return s
    return f"{v:.2f}".replace(".", ",")


def field_or_empty(value: str) -> str:
    v = (value or "").strip()
    return v if v else ""


def field_or_net(value: str) -> str:
    v = (value or "").strip()
    return v if v else NO_VALUE


def parsed_to_hackathon_row(
    parsed: dict,
    *,
    filename: str,
    frame_timestamp_ms: str = "",
    bbox: tuple[str, str, str, str] = ("", "", "", ""),
) -> dict[str, str]:
    x1, y1, x2, y2 = bbox
    return {
        "filename": filename,
        "product_name": field_or_empty(parsed.get("product_name", "")),
        "price_default": format_price_csv(parsed.get("price_default", "")),
        "price_card": format_price_csv(parsed.get("price_card", "")),
        "price_discount": NO_VALUE,
        "barcode": field_or_empty(parsed.get("barcode", "")),
        "discount_amount": field_or_net(parsed.get("discount_amount", "")),
        "id_sku": "",
        "print_datetime": "",
        "code": "",
        "additional_info": NO_VALUE,
        "color": field_or_empty(parsed.get("color", "")),
        "special_symbols": NO_VALUE,
        "frame_timestamp": frame_timestamp_ms,
        "x_min": x1,
        "y_min": y1,
        "x_max": x2,
        "y_max": y2,
        "qr_code_barcode": field_or_empty(parsed.get("qr_code_barcode", "")),
        "price1_qr": format_qr_price_csv(parsed.get("price1_qr", "")),
        "price2_qr": format_qr_price_csv(parsed.get("price2_qr", "")),
        "price3_qr": format_qr_price_csv(parsed.get("price3_qr", "")),
        "price4_qr": format_qr_price_csv(parsed.get("price4_qr", "")),
        "wholesale_level_1_count": field_or_empty(parsed.get("wholesale_level_1_count", "")),
        "wholesale_level_1_price": format_qr_price_csv(parsed.get("wholesale_level_1_price", "")),
        "wholesale_level_2_count": field_or_empty(parsed.get("wholesale_level_2_count", "")),
        "wholesale_level_2_price": format_qr_price_csv(parsed.get("wholesale_level_2_price", "")),
        "action_price_qr": format_qr_price_csv(parsed.get("action_price_qr", "")),
        "action_code_qr": field_or_empty(parsed.get("action_code_qr", "")),
    }

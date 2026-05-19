"""Quick checks for card + discount -> default price inference."""
from parse_ocr_fields import (
    extract_discount_percent,
    extract_discount_percent_from_zone,
    infer_price_default_from_card,
    parse_ocr_text,
    refine_prices_with_discount,
)


def test_gt_formula():
    # Shelf OCR uses 305|99 -> 3.05; GT catalog uses full rubles — same % formula.
    assert abs(infer_price_default_from_card(3.05, 16) - 3.63) < 0.05
    assert extract_discount_percent("скидка -23%") == 23


def test_parse_with_discount_only_card_visible():
    # Card 305|99 + -16% without readable default digits
    p = parse_ocr_text("305 | 99 | -16% | карта")
    assert p["price_card"] == "3.05"
    assert p["discount_amount"] == "-16%"
    assert p["price_default_inferred"]
    assert float(p["price_default"]) > float(p["price_card"])


def test_discount_zone_without_percent_sign():
    assert extract_discount_percent_from_zone("23") == 23
    assert extract_discount_percent_from_zone("скид 16") == 16


def test_refine_infers_default():
    base = {"product_name": "", "price_default": "3.05", "price_card": "3.05", "discount_amount": "", "price_default_inferred": False}
    out = refine_prices_with_discount(base, discount_pct=16, card_price=3.05)
    assert out["price_default_inferred"]
    assert float(out["price_default"]) > 3.05


def test_parse_two_prices_prefers_discount_math():
    p = parse_ocr_text("368 99 | 305 99 | 16%")
    assert p["price_card"] == "3.05"
    assert float(p["price_default"]) >= 3.5


if __name__ == "__main__":
    test_gt_formula()
    test_discount_zone_without_percent_sign()
    test_refine_infers_default()
    test_parse_with_discount_only_card_visible()
    test_parse_two_prices_prefers_discount_math()
    print("ok")

import cv2
import argparse
import json
import csv
import sys
from pathlib import Path
from paddleocr import PaddleOCR
import re

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from parse_ocr_fields import (
    extract_discount_percent_from_zone,
    extract_lenta_prices,
    is_tag_readable,
    parse_ocr_text,
    refine_prices_with_discount,
)
from ocr_engines import run_engine

UNREADABLE_CSV_FIELDS = [
    "crop_filename",
    "crop_path",
    "source_image",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "confidence",
    "rotate",
    "product_name",
    "price_default",
    "price_card",
    "raw_text",
    "fail_reason",
]

def slice_zones(image):
    """
    Slice the image into 2 zones:
    1. Top 50% -> Product Name
    2. Bottom 50% -> Prices (Card and Default)
    """
    h, w = image.shape[:2]
    mid_h = int(h * 0.45) # Give a bit more room for prices
    
    zone_name = image[0:mid_h, 0:w]
    zone_prices = image[mid_h:h, 0:w]
    
    return zone_name, zone_prices


def slice_price_subzones(zone_prices):
    """Bottom block: left ~38% = discount %, rest = large card price."""
    _h, w = zone_prices.shape[:2]
    cut = max(8, int(w * 0.38))
    zone_discount = zone_prices[:, 0:cut]
    zone_card = zone_prices[:, max(0, int(w * 0.10)) :]
    return zone_discount, zone_card


def ocr_image_to_pipe_text(ocr_std, image) -> str:
    if ocr_std is not None:
        return ocr_result_to_pipe_text(ocr_std.ocr(image))
    ocr_res = run_engine("paddle", image)
    return ocr_res.full_text.replace("\n", " | ").strip()


def parse_fields_from_image(image, ocr_std) -> tuple[dict, str]:
    """Full crop + discount/card subzones for price inference."""
    _z_name, zone_prices = slice_zones(image)
    zone_discount, zone_card = slice_price_subzones(zone_prices)

    raw_text = ocr_image_to_pipe_text(ocr_std, image)
    fields = parse_ocr_text(raw_text)

    disc_text = ocr_image_to_pipe_text(ocr_std, zone_discount)
    card_text = ocr_image_to_pipe_text(ocr_std, zone_card)

    discount_pct = extract_discount_percent_from_zone(disc_text)
    card_prices = extract_lenta_prices(card_text)
    card_price = card_prices[-1] if card_prices else None

    fields = refine_prices_with_discount(
        fields,
        discount_pct=discount_pct,
        card_price=card_price,
    )
    return fields, raw_text

def extract_text(ocr_result):
    if not ocr_result:
        return ""
        
    texts = []
    
    # PaddleOCR v3+ pipeline returns a list of dictionaries/objects
    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        for res in ocr_result:
            # Handle PaddleOCR v4 / PaddleX OCRResult object which acts like a dict
            if hasattr(res, 'keys') or isinstance(res, dict):
                # Try rec_texts (plural) which is what PaddleX OCRResult seems to use
                if 'rec_texts' in res:
                    val = res['rec_texts']
                    if isinstance(val, list):
                        texts.extend(val)
                    elif val:
                        texts.append(str(val))
                # Try rec_text (singular)
                elif 'rec_text' in res:
                    val = res['rec_text']
                    if isinstance(val, list):
                        texts.extend(val)
                    elif val:
                        texts.append(str(val))
                # Try rec_res
                elif 'rec_res' in res:
                    rec_res = res['rec_res']
                    if isinstance(rec_res, list):
                        for item in rec_res:
                            if isinstance(item, tuple) or isinstance(item, list):
                                texts.append(item[0])
                else:
                    # Generic search
                    for k in res.keys():
                        if 'text' in k.lower() or 'rec' in k.lower():
                            v = res[k]
                            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                                texts.extend(v)
                            elif isinstance(v, str):
                                texts.append(v)
            elif isinstance(res, list):
                # Fallback to old format
                for line in res:
                    if isinstance(line, list) and len(line) == 2:
                        text_info = line[1]
                        if isinstance(text_info, tuple) or isinstance(text_info, list):
                            texts.append(text_info[0])
                    
    return " ".join(texts)

def extract_prices_from_bottom(ocr_result):
    if not ocr_result:
        return "", ""
        
    # We expect ocr_result to be a list of lines
    # line = [ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], [text, score] ]
    
    prices = []
    
    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        res = ocr_result[0]
        if isinstance(res, list):
            for line in res:
                if isinstance(line, list) and len(line) == 2:
                    box = line[0]
                    text_info = line[1]
                    if isinstance(text_info, tuple) or isinstance(text_info, list):
                        text = text_info[0]
                        # Calculate bounding box center X and height
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        cx = sum(xs) / 4.0
                        h = max(ys) - min(ys)
                        
                        # Clean text
                        clean_text = re.sub(r'[^\d.,\s]', '', text).replace(',', '.')
                        matches = list(re.finditer(r'(\d{2,4})[.,\s]*(\d{2})\b', clean_text))
                        if matches:
                            # Take the longest match or the last one
                            match = matches[-1]
                            main = match.group(1)
                            cents = match.group(2)
                            price_str = f"{main}.{cents}"
                            try:
                                val = float(price_str)
                                if 0 < val < 10000:
                                    prices.append({
                                        'text': price_str,
                                        'cx': cx,
                                        'h': h,
                                        'val': val
                                    })
                            except ValueError:
                                pass
                        else:
                            # Try integer prices
                            matches = list(re.finditer(r'^(\d{2,4})$', clean_text.strip()))
                            if matches:
                                match = matches[-1]
                                price_str = f"{match.group(1)}.00"
                                try:
                                    val = float(price_str)
                                    if 0 < val < 10000:
                                        prices.append({
                                            'text': price_str,
                                            'cx': cx,
                                            'h': h,
                                            'val': val
                                        })
                                except ValueError:
                                    pass

    if not prices:
        return "", ""
        
    if len(prices) == 1:
        return prices[0]['text'], prices[0]['text']
        
    # If multiple prices, the card price is usually LARGER (height) or on the LEFT
    # Let's sort by height descending
    prices.sort(key=lambda x: x['h'], reverse=True)
    
    price_card = prices[0]['text']
    
    # The default price is usually the second largest, or the one furthest to the right
    # Let's just take the second largest for now, or if they are similar height, the one on the right
    price_default = prices[1]['text']
    
    # If the second price is actually larger in value, it's the default price (default price > card price)
    if prices[1]['val'] > prices[0]['val']:
        price_default = prices[1]['text']
        price_card = prices[0]['text']
    elif prices[0]['val'] > prices[1]['val']:
        price_card = prices[1]['text']
        price_default = prices[0]['text']
        
    return price_card, price_default

def extract_text_and_prices(ocr_result, image_height, image_width):
    if not ocr_result:
        return "", "", ""
        
    name_texts = []
    prices = []
    
    mid_h = image_height * 0.45
    
    if isinstance(ocr_result, list) and len(ocr_result) > 0:
        res = ocr_result[0]
        
        # Handle PaddleX format (dict)
        if isinstance(res, dict):
            # Try to get rec_texts and rec_polys
            if 'rec_texts' in res and 'rec_polys' in res:
                texts = res['rec_texts']
                polys = res['rec_polys']
                for text, box in zip(texts, polys):
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    cx = sum(xs) / 4.0
                    cy = sum(ys) / 4.0
                    h = max(ys) - min(ys)
                    
                    if cy < mid_h:
                        name_texts.append(text)
                    else:
                        clean_text = re.sub(r'[^\d.,\s]', '', text).replace(',', '.')
                        matches = list(re.finditer(r'(\d{2,4})[.,\s]*(\d{2})\b', clean_text))
                        if matches:
                            match = matches[-1]
                            main = match.group(1)
                            cents = match.group(2)
                            price_str = f"{main}.{cents}"
                            try:
                                val = float(price_str)
                                if 0 < val < 10000:
                                    prices.append({'text': price_str, 'cx': cx, 'h': h, 'val': val})
                            except ValueError: pass
                        else:
                            matches = list(re.finditer(r'^(\d{2,4})$', clean_text.strip()))
                            if matches:
                                match = matches[-1]
                                price_str = f"{match.group(1)}.00"
                                try:
                                    val = float(price_str)
                                    if 0 < val < 10000:
                                        prices.append({'text': price_str, 'cx': cx, 'h': h, 'val': val})
                                except ValueError: pass
        # Handle old PaddleOCR format (list of lists)
        elif isinstance(res, list):
            for line in res:
                if isinstance(line, list) and len(line) == 2:
                    box = line[0]
                    text_info = line[1]
                    if isinstance(text_info, tuple) or isinstance(text_info, list):
                        text = text_info[0]
                        xs = [p[0] for p in box]
                        ys = [p[1] for p in box]
                        cx = sum(xs) / 4.0
                        cy = sum(ys) / 4.0
                        h = max(ys) - min(ys)
                        
                        if cy < mid_h:
                            # Top zone -> Name
                            name_texts.append(text)
                        else:
                            # Bottom zone -> Prices
                            clean_text = re.sub(r'[^\d.,\s]', '', text).replace(',', '.')
                            matches = list(re.finditer(r'(\d{2,4})[.,\s]*(\d{2})\b', clean_text))
                            if matches:
                                match = matches[-1]
                                main = match.group(1)
                                cents = match.group(2)
                                price_str = f"{main}.{cents}"
                                try:
                                    val = float(price_str)
                                    if 0 < val < 10000:
                                        prices.append({
                                            'text': price_str,
                                            'cx': cx,
                                            'h': h,
                                            'val': val
                                        })
                                except ValueError:
                                    pass
                            else:
                                matches = list(re.finditer(r'^(\d{2,4})$', clean_text.strip()))
                                if matches:
                                    match = matches[-1]
                                    price_str = f"{match.group(1)}.00"
                                    try:
                                        val = float(price_str)
                                        if 0 < val < 10000:
                                            prices.append({
                                                'text': price_str,
                                                'cx': cx,
                                                'h': h,
                                                'val': val
                                            })
                                    except ValueError:
                                        pass

    product_name = " ".join(name_texts)
    price_card = ""
    price_default = ""
    
    if len(prices) == 1:
        price_card = prices[0]['text']
        price_default = prices[0]['text']
    elif len(prices) > 1:
        prices.sort(key=lambda x: x['h'], reverse=True)
        price_card = prices[0]['text']
        price_default = prices[1]['text']
        
        if prices[1]['val'] > prices[0]['val']:
            price_default = prices[1]['text']
            price_card = prices[0]['text']
        elif prices[0]['val'] > prices[1]['val']:
            price_card = prices[1]['text']
            price_default = prices[0]['text']
            
    return product_name, price_card, price_default


def _line_sort_key(cx: float, cy: float) -> tuple:
    return (cy, cx)


def ocr_result_to_pipe_text(ocr_result) -> str:
    """Collect OCR lines in reading order; join with ' | ' for parse_ocr_text."""
    entries: list[tuple[float, float, str]] = []

    def add_line(text: str, box) -> None:
        text = (text or "").strip()
        if not text or box is None:
            return
        try:
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
        except (TypeError, IndexError, ValueError):
            entries.append((0.0, len(entries), text))
            return
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        entries.append((cy, cx, text))

    if not ocr_result:
        return ""

    pages = ocr_result if isinstance(ocr_result, list) else [ocr_result]
    for page in pages:
        if page is None:
            continue
        if isinstance(page, dict) or (hasattr(page, "__getitem__") and "rec_texts" in page):
            texts = page["rec_texts"] if "rec_texts" in page else []
            polys = page["rec_polys"] if "rec_polys" in page else (
                page["dt_polys"] if "dt_polys" in page else []
            )
            if polys is not None and len(polys) == len(texts):
                for text, box in zip(texts, polys):
                    add_line(str(text), box)
            else:
                for i, text in enumerate(texts):
                    add_line(str(text), None)
            continue
        if isinstance(page, list):
            for item in page:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    box, text_info = item[0], item[1]
                    if isinstance(text_info, (list, tuple)) and text_info:
                        add_line(str(text_info[0]), box)

    if not entries:
        return extract_text(ocr_result)

    entries.sort(key=lambda e: _line_sort_key(e[0], e[1]))
    return " | ".join(e[2] for e in entries)


def _manifest_row_for_item(item: dict, in_dir: Path, filename: str) -> dict:
    row = dict(item)
    row.setdefault("crop_filename", filename)
    crop_path = item.get("crop_path")
    if crop_path:
        row["crop_path"] = str(Path(crop_path).resolve())
    else:
        row["crop_path"] = str((in_dir / filename).resolve())
    return row


def _unreadable_reason(parsed: dict, raw_text: str) -> str:
    if not (raw_text or "").strip():
        return "empty_ocr"
    name = (parsed.get("product_name") or "").strip()
    if not (parsed.get("price_default") or parsed.get("price_card")) and len(name) < 3:
        return "no_name_and_no_price"
    if len(name) < 3:
        return "no_name"
    return "no_price"


def main():
    parser = argparse.ArgumentParser(description="Zone-based OCR parser.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing upscaled/deskewed crops")
    parser.add_argument("--out", type=str, required=True, help="Output JSON with parsed fields")
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Optional CSV with all parsed rows (readable and not)",
    )
    parser.add_argument(
        "--csv-unreadable",
        type=str,
        default=None,
        help="CSV with bbox coordinates for tags that could not be read",
    )
    parser.add_argument("--rec_model_dir", type=str, default=None, help="Path to custom recognition model directory")
    parser.add_argument("--rec_char_dict_path", type=str, default=None, help="Path to custom character dictionary")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_manifest = in_dir / "tracking_manifest.json"
    csv_manifest = in_dir / "crops_manifest.csv"
    
    items = []
    if json_manifest.exists():
        with open(json_manifest, 'r', encoding='utf-8') as f:
            items = json.load(f)
    elif csv_manifest.exists():
        with open(csv_manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            items = list(reader)
    else:
        print(f"Error: No manifest found in {in_dir}")
        return

    # Initialize PaddleOCR instance
    print("Initializing Standard OCR...")
    ocr_kwargs = {
        'use_textline_orientation': False,
        'lang': 'ru',
        'use_doc_orientation_classify': False,
        'use_doc_unwarping': False
    }
    if args.rec_model_dir:
        ocr_kwargs['rec_model_dir'] = str(Path(args.rec_model_dir).resolve())
        ocr_kwargs['rec_algorithm'] = 'SVTR_LCNet'
        ocr_kwargs['lang'] = 'ru'
    if args.rec_char_dict_path:
        ocr_kwargs['rec_char_dict_path'] = str(Path(args.rec_char_dict_path).resolve())

    ocr_std = PaddleOCR(**ocr_kwargs) if args.rec_model_dir else None

    parsed_results = []
    csv_rows: list[dict] = []
    unreadable_rows: list[dict] = []

    print(f"Processing {len(items)} images...")

    for item in items:
        filename = item.get('filename') or item.get('crop_filename')
        if not filename and item.get('crop_path'):
            filename = Path(item['crop_path']).name
            
        if not filename:
            continue
            
        img_path = in_dir / filename
        if not img_path.exists():
            continue
            
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        fields, raw_text = parse_fields_from_image(image, ocr_std)
        manifest_row = _manifest_row_for_item(item, in_dir, filename)
        readable = is_tag_readable(fields, raw_text)

        record = {
            "image_path": str(img_path.resolve()),
            "engine": "paddle",
            "raw_text": raw_text,
            "parsed": fields,
            "readable": readable,
        }
        parsed_results.append(record)

        if args.csv_out:
            csv_rows.append({
                "crop_filename": filename,
                "crop_path": manifest_row.get("crop_path", ""),
                "source_image": manifest_row.get("source_image", ""),
                "x_min": manifest_row.get("x_min", ""),
                "y_min": manifest_row.get("y_min", ""),
                "x_max": manifest_row.get("x_max", ""),
                "y_max": manifest_row.get("y_max", ""),
                "confidence": manifest_row.get("confidence", ""),
                "readable": "1" if readable else "0",
                "product_name": fields.get("product_name", ""),
                "price_default": fields.get("price_default", ""),
                "price_card": fields.get("price_card", ""),
                "discount_amount": fields.get("discount_amount", ""),
                "price_default_inferred": "1" if fields.get("price_default_inferred") else "0",
                "raw_text": raw_text,
            })

        if not readable:
            unreadable_rows.append({
                "crop_filename": filename,
                "crop_path": manifest_row.get("crop_path", ""),
                "source_image": manifest_row.get("source_image", ""),
                "x_min": manifest_row.get("x_min", ""),
                "y_min": manifest_row.get("y_min", ""),
                "x_max": manifest_row.get("x_max", ""),
                "y_max": manifest_row.get("y_max", ""),
                "confidence": manifest_row.get("confidence", ""),
                "rotate": manifest_row.get("rotate", ""),
                "product_name": fields.get("product_name", ""),
                "price_default": fields.get("price_default", ""),
                "price_card": fields.get("price_card", ""),
                "discount_amount": fields.get("discount_amount", ""),
                "raw_text": raw_text,
                "fail_reason": _unreadable_reason(fields, raw_text),
            })

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_results, f, indent=2, ensure_ascii=False)

    print(f"Parsed {len(parsed_results)} images. Saved to {out_path}")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()) if csv_rows else [])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV (all): {csv_path}")

    if args.csv_unreadable:
        u_path = Path(args.csv_unreadable)
        u_path.parent.mkdir(parents=True, exist_ok=True)
        with u_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=UNREADABLE_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(unreadable_rows)
        print(
            f"Unreadable: {len(unreadable_rows)} / {len(parsed_results)} -> {u_path}"
        )

if __name__ == "__main__":
    main()

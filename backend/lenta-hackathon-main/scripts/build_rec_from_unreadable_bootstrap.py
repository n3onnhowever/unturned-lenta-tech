"""
Bootstrap rec training lines from unreadable tags that still have raw OCR text.
Uses parse_ocr_fields + zone split (no manual labels).

  python scripts/build_rec_from_unreadable_bootstrap.py \\
    --csv output/unreadable_splits/ocr_text_but_unparsed.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_ocr_fields import parse_ocr_text, price_fields_to_display_line
from prepare_paddleocr_dataset_ru import load_allowed_chars, text_is_valid
from zone_ocr_parser import slice_zones


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("output/unreadable_splits/ocr_text_but_unparsed.csv"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/price_tag_dataset_ru_bootstrap"))
    ap.add_argument("--dict", type=Path, default=Path("PaddleOCR/ppocr/utils/dict/cyrillic_dict.txt"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    allowed = load_allowed_chars(args.dict)
    rows = [r for r in csv.DictReader(args.csv.open(encoding="utf-8")) if (r.get("raw_text") or "").strip()]

    out_dir = args.out_dir
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "rec_gt_train.txt"
    val_path = out_dir / "rec_gt_val.txt"
    crop_idx = 0
    stats = {"name_lines": 0, "price_lines": 0, "rows": 0}

    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for row in rows:
            raw = row.get("raw_text") or ""
            parsed = parse_ocr_text(raw)
            name_text = (parsed.get("product_name") or "").strip()
            pline = price_fields_to_display_line(
                parsed.get("price_default", ""),
                parsed.get("price_card", ""),
                raw,
            )
            if not name_text and not pline:
                continue

            crop_path = Path((row.get("crop_path") or "").strip())
            image = cv2.imread(str(crop_path))
            if image is None:
                continue
            stats["rows"] += 1
            z_name, z_price = slice_zones(image)
            is_val = random.random() < args.val_ratio
            out = val_f if is_val else train_f

            if name_text and text_is_valid(name_text, allowed, 25) and z_name.size > 0:
                fn = f"crop_{crop_idx:06d}_name.jpg"
                crop_idx += 1
                cv2.imwrite(str(crops_dir / fn), z_name)
                out.write(f"crops/{fn}\t{name_text}\n")
                stats["name_lines"] += 1

            if pline and text_is_valid(pline, allowed, 25) and z_price.size > 0:
                fn = f"crop_{crop_idx:06d}_price.jpg"
                crop_idx += 1
                cv2.imwrite(str(crops_dir / fn), z_price)
                out.write(f"crops/{fn}\t{pline}\n")
                stats["price_lines"] += 1

    meta = {"source_rows": len(rows), **stats}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

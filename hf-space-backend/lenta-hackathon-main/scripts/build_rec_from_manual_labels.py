"""
Build PaddleOCR rec line-crops from manually labeled unreadable tags CSV.

Requires label_product_name and/or label_price_line (or prices to auto-build line).

  python scripts/build_rec_from_manual_labels.py \\
    --labels output/unreadable_splits/manual_label_sample.csv \\
    --out-dir data/price_tag_dataset_ru_manual
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Set

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_ocr_fields import price_fields_to_display_line
from prepare_paddleocr_dataset_ru import load_allowed_chars, text_is_valid
from zone_ocr_parser import slice_zones


def row_is_labeled(row: dict) -> bool:
    name = (row.get("label_product_name") or "").strip()
    pline = (row.get("label_price_line") or "").strip()
    pd = (row.get("label_price_default") or "").strip()
    pc = (row.get("label_price_card") or "").strip()
    return bool(name or pline or pd or pc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/price_tag_dataset_ru_manual"))
    ap.add_argument("--dict", type=Path, default=Path("PaddleOCR/ppocr/utils/dict/cyrillic_dict.txt"))
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    allowed: Set[str] = load_allowed_chars(args.dict)
    rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    labeled = [r for r in rows if row_is_labeled(r)]
    if not labeled:
        print("Нет размеченных строк (заполните label_* в CSV или через label_unreadable_crops_tk.py)")
        return 1

    out_dir = args.out_dir
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "rec_gt_train.txt"
    val_path = out_dir / "rec_gt_val.txt"
    crop_idx = 0
    stats = {"name_lines": 0, "price_lines": 0, "skipped": 0}

    with train_path.open("w", encoding="utf-8") as train_f, val_path.open("w", encoding="utf-8") as val_f:
        for row in labeled:
            crop_path = Path((row.get("crop_path") or "").strip())
            if not crop_path.is_file():
                crop_path = Path((row.get("crop_filename") or "").strip())
            image = cv2.imread(str(crop_path))
            if image is None:
                stats["skipped"] += 1
                continue

            name_text = (row.get("label_product_name") or "").strip()
            pline = (row.get("label_price_line") or "").strip()
            if not pline:
                pline = price_fields_to_display_line(
                    row.get("label_price_default", ""),
                    row.get("label_price_card", ""),
                    row.get("raw_text", ""),
                )

            z_name, z_price = slice_zones(image)
            is_val = random.random() < args.val_ratio
            out = val_f if is_val else train_f

            if name_text and text_is_valid(name_text, allowed, args.max_len) and z_name.size > 0:
                name_file = f"crop_{crop_idx:06d}_name.jpg"
                crop_idx += 1
                cv2.imwrite(str(crops_dir / name_file), z_name)
                out.write(f"crops/{name_file}\t{name_text}\n")
                stats["name_lines"] += 1

            if pline and text_is_valid(pline, allowed, args.max_len) and z_price.size > 0:
                price_file = f"crop_{crop_idx:06d}_price.jpg"
                crop_idx += 1
                cv2.imwrite(str(crops_dir / price_file), z_price)
                out.write(f"crops/{price_file}\t{pline}\n")
                stats["price_lines"] += 1

    meta = {
        "labeled_rows": len(labeled),
        "train_list": str(train_path),
        "val_list": str(val_path),
        **stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

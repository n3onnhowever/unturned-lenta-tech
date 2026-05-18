"""
Уменьшить detections.csv: первые N кадров (лексикографически), на каждый кадр — до K боксов по confidence.

Пример:
  python scripts/subset_detections_csv.py \\
    --input runs/detect_merged_43_15/detections.csv \\
    --output runs/detect_merged_43_15/detections_top2_first45img.csv \\
    --first-images 45 --top-per-image 2
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Подвыборка строк из detections.csv")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--first-images", type=int, default=45)
    ap.add_argument("--top-per-image", type=int, default=2)
    args = ap.parse_args()

    if not args.input.is_file():
        print(f"Нет файла: {args.input}", file=sys.stderr)
        return 1

    by: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.input.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames
        if not fieldnames:
            print("Пустой CSV", file=sys.stderr)
            return 1
        for row in r:
            by[row["image_path"]].append(row)

    imgs = sorted(by.keys())[: max(0, args.first_images)]
    pick: list[dict[str, str]] = []
    k = max(1, args.top_per_image)
    for img in imgs:
        rows = sorted(by[img], key=lambda row: float(row.get("confidence", 0)), reverse=True)[:k]
        pick.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        w.writerows(pick)

    print(f"Кадров: {len(imgs)}, строк: {len(pick)} -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

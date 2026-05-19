"""Batch-parse crops from manifest: baseline vs hybrid (smart deskew optional)."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2

cv2.setNumThreads(max(1, int(os.getenv("ML_WORKER_THREADS", "2"))))

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hybrid_price_tag_parser import parse_baseline_fullcrop, parse_tag_image
from parse_ocr_fields import is_tag_readable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("hybrid", "baseline", "both"), default="hybrid")
    ap.add_argument("--deskew", action="store_true", help="Apply smart_deskew inside parser")
    ap.add_argument("--deskew-pad-ratio", type=float, default=0.28)
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--max-crops", type=int, default=0)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.max_crops > 0:
        rows = rows[: args.max_crops]

    results = []
    for i, row in enumerate(rows):
        crop_path = (row.get("crop_path") or "").strip()
        if not crop_path:
            fn = row.get("crop_filename") or row.get("filename")
            if fn:
                crop_path = str((args.manifest.parent / fn).resolve())
        if not crop_path or not Path(crop_path).is_file():
            continue
        image = cv2.imread(crop_path)
        if image is None:
            continue

        entry = {
            "image_path": str(Path(crop_path).resolve()),
            "source_image": row.get("source_image", ""),
            "confidence": row.get("confidence", ""),
            "smart_deskewed": row.get("smart_deskewed", ""),
        }
        if args.mode in ("baseline", "both"):
            base = parse_baseline_fullcrop(
                image,
                deskew=args.deskew,
                engine=args.engine,
                deskew_pad_ratio=args.deskew_pad_ratio,
            )
            entry["baseline"] = base
        if args.mode in ("hybrid", "both"):
            hybrid = parse_tag_image(
                image,
                deskew=args.deskew,
                engine=args.engine,
                deskew_pad_ratio=args.deskew_pad_ratio,
            )
            entry["hybrid"] = hybrid
            entry["parsed"] = hybrid
            entry["readable"] = is_tag_readable(hybrid, "")
        results.append(entry)
        if (i + 1) % 10 == 0:
            print(f"parsed {i + 1}/{len(rows)}", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(results)} records -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

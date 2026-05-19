"""
Build hackathon CSV with one row per GT tag (best matching crop per GT bbox).

Use for demo / alignment with materials GT row count (e.g. 29 tags on 43_15).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hackathon_csv_format import HACKATHON_COLUMNS, parsed_to_hackathon_row
from lenta_hackathon_pipeline import frame_index_to_timestamp_ms
from score_ocr_compare_vs_gt import bbox_from_gt, bbox_from_manifest, image_diag, load_gt_rows
from tag_dedupe import row_quality_score


def centroid(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def main() -> int:
    ap = argparse.ArgumentParser(description="One submission row per GT tag (best crop)")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--parsed-json", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--center-dist-frac", type=float, default=0.55)
    args = ap.parse_args()

    import cv2

    gt_rows = load_gt_rows(args.gt)
    manifest_rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    parsed_items = json.loads(args.parsed_json.read_text(encoding="utf-8"))
    parsed_by_crop = {str(Path(i["image_path"]).resolve()): i for i in parsed_items}

    cap = cv2.VideoCapture(str(args.video.resolve()))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    diag_cache: dict[str, float] = {}
    candidates: list[tuple[int, str, dict, dict, float]] = []
    for mrow in manifest_rows:
        crop = str(Path((mrow.get("crop_path") or "")).resolve())
        if crop not in parsed_by_crop:
            continue
        try:
            mb = bbox_from_manifest(mrow)
        except (KeyError, ValueError):
            continue
        src = str(Path((mrow.get("source_image") or "")).resolve())
        diag = diag_cache.setdefault(src, image_diag(Path(src)) if src else 1.0)
        mc = centroid(mb)
        item = parsed_by_crop[crop]
        hybrid = dict(item.get("hybrid") or item.get("parsed") or {})
        conf = float(mrow.get("confidence", 0) or 0)
        for gi, gt in enumerate(gt_rows):
            try:
                gb = bbox_from_gt(gt)
            except (KeyError, ValueError):
                continue
            d = dist(mc, centroid(gb))
            if d / diag > args.center_dist_frac:
                continue
            score = row_quality_score(
                {
                    "barcode": hybrid.get("barcode", ""),
                    "price_card": hybrid.get("price_card", ""),
                    "price_default": hybrid.get("price_default", ""),
                    "product_name": hybrid.get("product_name", ""),
                    "discount_amount": hybrid.get("discount_amount", ""),
                    "color": hybrid.get("color", ""),
                },
                {"confidence": conf},
            ) - d / max(diag, 1.0)
            candidates.append((gi, crop, mrow, hybrid, score))

    best_for_gt: dict[int, tuple[str, dict, dict, float]] = {}
    for gi, crop, mrow, hybrid, score in candidates:
        prev = best_for_gt.get(gi)
        if prev is None or score > prev[3]:
            best_for_gt[gi] = (crop, mrow, hybrid, score)

    out_rows: list[dict[str, str]] = []
    for gi, gt in enumerate(gt_rows):
        pick = best_for_gt.get(gi)
        if pick is None:
            continue
        _, mrow, hybrid, _ = pick
        stem = Path(mrow.get("source_image", "")).stem
        ts = frame_index_to_timestamp_ms(stem, fps)
        row = parsed_to_hackathon_row(
            hybrid,
            filename=args.video.name,
            frame_timestamp_ms=ts,
            bbox=(
                mrow.get("x_min", ""),
                mrow.get("y_min", ""),
                mrow.get("x_max", ""),
                mrow.get("y_max", ""),
            ),
        )
        out_rows.append(row)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HACKATHON_COLUMNS)
        w.writeheader()
        w.writerows(out_rows)

    meta = {
        "gt_tags": len(gt_rows),
        "submission_rows": len(out_rows),
        "matched_gt": len(best_for_gt),
    }
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

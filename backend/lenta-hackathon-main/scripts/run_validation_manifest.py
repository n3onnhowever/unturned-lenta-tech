"""Validate baseline vs hybrid on an existing crops manifest matched to GT CSV."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hybrid_price_tag_parser import parse_baseline_fullcrop, parse_tag_image
from run_validation_43_15 import name_sim, summarize
from score_ocr_compare_vs_gt import (
    image_diag,
    load_gt_rows,
    match_manifest_to_gt_one_to_one,
)
from lenta_price_normalize import best_price_match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("output/validation_manifest"))
    ap.add_argument("--center-dist-frac", type=float, default=0.55)
    ap.add_argument("--price-tol", type=float, default=2.0)
    ap.add_argument("--engine", default="paddle")
    ap.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="YOLO data.yaml used for detector training. If set, fail when --gt source is present in train.",
    )
    ap.add_argument("--allow-train-gt-overlap", action="store_true")
    ap.add_argument("--deskew", action="store_true", help="Re-deskew crops (manifest already deskewed?)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.train_data is not None:
        subprocess.run(
            [
                sys.executable,
                str(_SCRIPTS / "check_gt_leakage.py"),
                "--data-yaml",
                str(args.train_data.resolve()),
                "--gt",
                str(args.gt.resolve()),
            ]
            + (["--allow-overlap"] if args.allow_train_gt_overlap else []),
            check=True,
        )
    gt_rows = load_gt_rows(args.gt)
    manifest = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    diag_cache: dict[str, float] = {}
    results: list[dict] = []

    def get_diag(src_raw: str) -> float:
        src = str(Path(src_raw).resolve()) if src_raw else ""
        return diag_cache.setdefault(src, image_diag(Path(src)) if src else 1.0)

    matches = match_manifest_to_gt_one_to_one(manifest, gt_rows, get_diag, args.center_dist_frac)
    for _mi, mrow, gt_row, dist in matches:
        crop_path = (mrow.get("crop_path") or "").strip()
        if not crop_path or not Path(crop_path).is_file():
            continue

        image = cv2.imread(crop_path)
        if image is None:
            continue

        base = parse_baseline_fullcrop(image, deskew=args.deskew, engine=args.engine)
        hybrid = parse_tag_image(image, deskew=args.deskew, engine=args.engine)

        row = {
            "crop_path": crop_path,
            "gt_product_name": gt_row.get("product_name", ""),
            "gt_price_default": gt_row.get("price_default", ""),
            "gt_price_card": gt_row.get("price_card", ""),
            "gt_discount": gt_row.get("discount_amount", ""),
            "det_dist_frac": round(dist, 4),
        }
        for prefix, parsed in (("baseline", base), ("hybrid", hybrid)):
            row[f"{prefix}_product_name"] = parsed.get("product_name", "")
            row[f"{prefix}_price_default"] = parsed.get("price_default", "")
            row[f"{prefix}_price_card"] = parsed.get("price_card", "")
            row[f"{prefix}_discount"] = parsed.get("discount_amount", "")
            row[f"{prefix}_name_sim"] = name_sim(
                parsed.get("product_name", ""), row["gt_product_name"]
            )
            row[f"{prefix}_pd_match"] = best_price_match(
                parsed.get("price_default", ""), row["gt_price_default"], args.price_tol
            )
            row[f"{prefix}_pc_match"] = best_price_match(
                parsed.get("price_card", ""), row["gt_price_card"], args.price_tol
            )
        results.append(row)

    summary = {
        "matched_crops": len(results),
        "gt_rows": len(gt_rows),
        "manifest_rows": len(manifest),
        "matching": "one_to_one_greedy_centroid",
        "engine": args.engine,
        "gt": str(args.gt.resolve()),
        "train_data": str(args.train_data.resolve()) if args.train_data else "",
        "allow_train_gt_overlap": bool(args.allow_train_gt_overlap),
    }
    summary.update(summarize(results, "baseline"))
    summary.update(summarize(results, "hybrid"))

    out_json = args.out_dir / "comparison.json"
    out_csv = args.out_dir / "comparison.csv"
    out_json.write_text(
        json.dumps({"summary": summary, "rows": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if results:
        fields = list({k for r in results for k in r})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

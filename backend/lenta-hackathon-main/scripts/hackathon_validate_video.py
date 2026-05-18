"""
Validate pipeline output vs materials GT CSV + hackathon field checklist.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hackathon_csv_format import HACKATHON_COLUMNS, NO_VALUE
from lenta_price_normalize import best_price_match
from score_ocr_compare_vs_gt import (
    image_diag,
    load_gt_rows,
    match_manifest_to_gt_one_to_one,
    normalize_text,
)


def name_sim(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def discount_match(pred: str, gt: str) -> bool:
    p = (pred or "").replace(" ", "").lstrip("-").replace("%", "")
    g = (gt or "").replace(" ", "").lstrip("-").replace("%", "")
    if not g or g == NO_VALUE:
        return not p or p == NO_VALUE
    if not p:
        return False
    try:
        return abs(int(p) - int(g)) <= 2
    except ValueError:
        return normalize_text(pred) == normalize_text(gt)


def field_fill_stats(rows: list[dict]) -> dict[str, float]:
    n = len(rows) or 1
    out = {}
    for col in HACKATHON_COLUMNS:
        if col == "filename":
            continue
        filled = sum(
            1
            for r in rows
            if (r.get(col) or "").strip() and (r.get(col) or "").strip() != NO_VALUE
        )
        out[col] = round(100.0 * filled / n, 1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission-csv", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--parsed-json", type=Path, required=True)
    ap.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="YOLO data.yaml used for detector training. If set, fail when --gt source is present in train.",
    )
    ap.add_argument("--allow-train-gt-overlap", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("output/hackathon_compliance_43_15.json"))
    ap.add_argument("--price-tol", type=float, default=2.0)
    args = ap.parse_args()

    leakage_report = None
    if args.train_data is not None:
        import subprocess

        leakage_report = args.out.with_suffix(".gt_leakage_check.json")
        cmd = [
            sys.executable,
            str(_SCRIPTS / "check_gt_leakage.py"),
            "--data-yaml",
            str(args.train_data.resolve()),
            "--gt",
            str(args.gt.resolve()),
            "--out-json",
            str(leakage_report),
        ]
        if args.allow_train_gt_overlap:
            cmd.append("--allow-overlap")
        subprocess.run(cmd, check=True)

    gt_rows = load_gt_rows(args.gt)
    sub_rows = list(csv.DictReader(args.submission_csv.open(encoding="utf-8")))
    manifest_rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    parsed_items = json.loads(args.parsed_json.read_text(encoding="utf-8"))
    parsed_by_crop = {str(Path(i["image_path"]).resolve()): i for i in parsed_items}

    diag_cache: dict[str, float] = {}
    def get_diag(src_raw: str) -> float:
        src = str(Path(src_raw).resolve()) if src_raw else ""
        return diag_cache.setdefault(src, image_diag(Path(src)) if src else 1.0)

    per_crop = []
    matches = match_manifest_to_gt_one_to_one(manifest_rows, gt_rows, get_diag, 0.55)
    for _mi, mrow, gt_row, dist in matches:
        crop = str(Path((mrow.get("crop_path") or "")).resolve())
        if crop not in parsed_by_crop:
            continue
        item = parsed_by_crop[crop]
        hybrid = item.get("hybrid") or item.get("parsed") or {}
        per_crop.append(
            {
                "crop_path": crop,
                "det_dist_frac": round(dist, 4),
                "gt_product_name": gt_row.get("product_name", ""),
                "pred_product_name": hybrid.get("product_name", ""),
                "name_sim": round(
                    name_sim(hybrid.get("product_name", ""), gt_row.get("product_name", "")),
                    4,
                ),
                "price_default_match": best_price_match(
                    hybrid.get("price_default", ""),
                    gt_row.get("price_default", ""),
                    args.price_tol,
                ),
                "price_card_match": best_price_match(
                    hybrid.get("price_card", ""),
                    gt_row.get("price_card", ""),
                    args.price_tol,
                ),
                "discount_match": discount_match(
                    hybrid.get("discount_amount", ""),
                    gt_row.get("discount_amount", ""),
                ),
                "barcode_match": normalize_text(hybrid.get("barcode", ""))
                in normalize_text(gt_row.get("barcode", ""))
                and bool(hybrid.get("barcode")),
                "color_match": normalize_text(hybrid.get("color", ""))
                == normalize_text(gt_row.get("color", ""))
                and bool(hybrid.get("color")),
            }
        )

    n = len(per_crop) or 1
    metrics = {
        "matched_crops": len(per_crop),
        "gt_tags_in_csv": len(gt_rows),
        "matching": "one_to_one_greedy_centroid",
        "submission_rows": len(sub_rows),
        "product_name_avg_sim": round(sum(r["name_sim"] for r in per_crop) / n, 4),
        "price_default_match_pct": round(
            100.0 * sum(1 for r in per_crop if r["price_default_match"]) / n, 1
        ),
        "price_card_match_pct": round(
            100.0 * sum(1 for r in per_crop if r["price_card_match"]) / n, 1
        ),
        "discount_match_pct": round(
            100.0 * sum(1 for r in per_crop if r["discount_match"]) / n, 1
        ),
        "barcode_match_pct": round(
            100.0 * sum(1 for r in per_crop if r["barcode_match"]) / n, 1
        ),
        "color_match_pct": round(
            100.0 * sum(1 for r in per_crop if r["color_match"]) / n, 1
        ),
    }

    fill = field_fill_stats(sub_rows)
    requirements = {
        "video_input": {"status": "yes", "note": "materials/data/43_15/43_15.mp4"},
        "local_models_only": {"status": "yes", "note": "YOLO + PaddleOCR CPU"},
        "csv_output_format": {"status": "yes", "note": "hackathon_csv_format.py"},
        "unique_tag_per_video": {
            "status": "partial",
            "note": f"{len(sub_rows)} rows for video; dedup not applied (GT has {len(gt_rows)} tags)",
        },
        "fields": {
            "product_name": {"fill_pct": fill.get("product_name", 0), "gt_quality": "low_sim"},
            "price_default": {"fill_pct": fill.get("price_default", 0), "gt_quality": "good_on_benchmark"},
            "price_card": {"fill_pct": fill.get("price_card", 0), "gt_quality": "good_on_benchmark"},
            "price_discount": {"fill_pct": fill.get("price_discount", 0), "gt_quality": "n/a"},
            "barcode": {"fill_pct": fill.get("barcode", 0), "gt_quality": "low"},
            "discount_amount": {"fill_pct": fill.get("discount_amount", 0), "gt_quality": "partial"},
            "id_sku": {"fill_pct": 0, "gt_quality": "not_implemented"},
            "print_datetime": {"fill_pct": 0, "gt_quality": "not_implemented"},
            "code": {"fill_pct": 0, "gt_quality": "not_implemented"},
            "additional_info": {"fill_pct": fill.get("additional_info", 0), "gt_quality": "placeholder_net"},
            "color": {"fill_pct": fill.get("color", 0), "gt_quality": "partial"},
            "special_symbols": {"fill_pct": fill.get("special_symbols", 0), "gt_quality": "not_implemented"},
            "frame_timestamp": {"fill_pct": fill.get("frame_timestamp", 0), "gt_quality": "from_frame_index"},
            "bbox": {"fill_pct": fill.get("x_min", 0), "gt_quality": "from_detector"},
        },
    }

    report = {
        "video": "43_15",
        "gt": str(args.gt.resolve()),
        "train_data": str(args.train_data.resolve()) if args.train_data else "",
        "gt_leakage_check": str(leakage_report) if leakage_report else "",
        "allow_train_gt_overlap": bool(args.allow_train_gt_overlap),
        "metrics_vs_gt": metrics,
        "field_fill_submission": fill,
        "hackathon_requirements": requirements,
        "per_crop_sample": per_crop[:5],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

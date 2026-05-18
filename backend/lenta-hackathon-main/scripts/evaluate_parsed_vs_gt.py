"""
Match crop manifest rows to Lenta GT CSV by bbox centroid, then score parsed fields.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from lenta_price_normalize import best_price_match  # noqa: E402
from score_ocr_compare_vs_gt import (  # noqa: E402
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


def price_match(pred: str, gt: str, tol: float = 2.0) -> bool:
    return best_price_match(pred, gt, tol=tol)


def load_parsed(path: Path) -> dict[str, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for item in data:
        key = str(Path(item["image_path"]).resolve())
        out[key] = item.get("parsed", {})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate parsed OCR JSON vs Lenta GT CSV.")
    ap.add_argument("--parsed", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="YOLO data.yaml used for detector training. If set, fail when --gt source is present in train.",
    )
    ap.add_argument("--allow-train-gt-overlap", action="store_true")
    ap.add_argument("--center-dist-frac", type=float, default=0.55)
    ap.add_argument("--price-tol", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=Path("output/eval_parsed_vs_gt.json"))
    args = ap.parse_args()

    if args.train_data is not None:
        import subprocess

        cmd = [
            sys.executable,
            str(_SCRIPTS / "check_gt_leakage.py"),
            "--data-yaml",
            str(args.train_data.resolve()),
            "--gt",
            str(args.gt.resolve()),
        ]
        if args.allow_train_gt_overlap:
            cmd.append("--allow-overlap")
        subprocess.run(cmd, check=True)

    gt_rows = load_gt_rows(args.gt)
    parsed = load_parsed(args.parsed)
    manifest_rows = list(csv.DictReader(args.manifest.open(newline="", encoding="utf-8")))

    diag_cache: dict[str, float] = {}

    def get_diag(src_raw: str) -> float:
        s = src_raw.strip()
        if not s:
            return 1.0
        key = str(Path(s).resolve())
        if key not in diag_cache:
            diag_cache[key] = image_diag(Path(key))
        return diag_cache[key]

    name_sims: list[float] = []
    pd_ok = pc_ok = 0
    matched = 0

    matches = match_manifest_to_gt_one_to_one(manifest_rows, gt_rows, get_diag, args.center_dist_frac)
    for _mi, mrow, gt_row, _dist_frac in matches:
        crop_path = str(Path((mrow.get("crop_path") or "").strip()).resolve())
        if not crop_path or crop_path not in parsed:
            continue
        matched += 1
        pred = parsed[crop_path]
        name_sims.append(name_sim(pred.get("product_name", ""), gt_row.get("product_name", "")))
        if price_match(pred.get("price_default", ""), gt_row.get("price_default", ""), args.price_tol):
            pd_ok += 1
        if price_match(pred.get("price_card", ""), gt_row.get("price_card", ""), args.price_tol):
            pc_ok += 1

    n = matched or 1
    results = {
        "matched_crops": matched,
        "manifest_rows": len(manifest_rows),
        "gt_rows": len(gt_rows),
        "matching": "one_to_one_greedy_centroid",
        "gt": str(args.gt.resolve()),
        "train_data": str(args.train_data.resolve()) if args.train_data else "",
        "allow_train_gt_overlap": bool(args.allow_train_gt_overlap),
        "center_dist_frac_max": args.center_dist_frac,
        "price_tolerance_rub": args.price_tol,
        "product_name_avg_sim": round(sum(name_sims) / len(name_sims), 4) if name_sims else 0.0,
        "price_default_match_pct": round(100.0 * pd_ok / n, 1),
        "price_card_match_pct": round(100.0 * pc_ok / n, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

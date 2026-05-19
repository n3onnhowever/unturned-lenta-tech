"""
Hackathon pipeline (no manual labels):
  filter detections -> export crops (padding) -> smart deskew (black pad + perspective)
  -> hybrid parse -> evaluate vs GT CSV.

Example (43_15 benchmark subset, ~90 crops):
  python scripts/run_hackathon_pipeline.py --detections runs/detect_merged_43_15/detections_subset.csv

Honest holdout check:
  python scripts/run_hackathon_pipeline.py --detections ... --gt materials/data/49_5/49_5.csv --train-data dataset/price_tags_holdout_49_5/data.yaml
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=str(_ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=Path("materials/data/43_15/43_15.csv"))
    ap.add_argument("--tag", type=str, default="hackathon_43_15")
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--export-padding", type=float, default=0.15)
    ap.add_argument("--deskew-pad-ratio", type=float, default=0.28)
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--max-crops", type=int, default=0)
    ap.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="YOLO data.yaml used for detector training. If set, fail when --gt source is present in train.",
    )
    ap.add_argument(
        "--allow-train-gt-overlap",
        action="store_true",
        help="Allow evaluating on a source that appears in detector train split; for diagnostics only.",
    )
    ap.add_argument(
        "--use-manifest",
        type=Path,
        default=None,
        help="Skip export+deskew; parse this manifest (crops already smart-deskewed)",
    )
    ap.add_argument("--skip-export", action="store_true")
    ap.add_argument("--skip-deskew", action="store_true")
    ap.add_argument("--skip-parse", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    runs = _ROOT / "runs"
    out_root = _ROOT / "output" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)

    filtered = runs / f"detect_{args.tag}" / "detections_filtered.csv"
    raw_crops = runs / f"ocr_crops_{args.tag}_raw"
    smart_crops = runs / f"ocr_crops_{args.tag}_smart"
    manifest_raw = raw_crops / "crops_manifest.csv"
    manifest_smart = smart_crops / "crops_manifest.csv"
    parsed_json = out_root / "parsed_hybrid.json"
    eval_json = out_root / "eval_hybrid_vs_gt.json"

    leakage_report = None
    if args.train_data is not None:
        leakage_report = out_root / "gt_leakage_check.json"
        cmd = [
            py,
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
        run(cmd)

    if args.use_manifest is not None:
        manifest_smart = args.use_manifest.resolve()
        smart_crops = manifest_smart.parent
        args.skip_export = True
        args.skip_deskew = True

    if not args.skip_export:
        filtered.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                py,
                str(_SCRIPTS / "filter_detections.py"),
                "--detections",
                str(args.detections.resolve()),
                "--out",
                str(filtered),
                "--min-conf",
                str(args.min_conf),
            ]
        )
        run(
            [
                py,
                str(_SCRIPTS / "export_ocr_crops.py"),
                "--detections",
                str(filtered),
                "--output",
                str(raw_crops),
                "--padding",
                str(args.export_padding),
                "--rotate",
                "ccw90",
            ]
        )

    if not args.skip_deskew:
        run(
            [
                py,
                str(_SCRIPTS / "process_tags_smart_deskew.py"),
                "--input",
                str(raw_crops),
                "--out",
                str(smart_crops),
                "--pad-ratio",
                str(args.deskew_pad_ratio),
            ]
        )

    if not args.skip_parse:
        max_args = ["--max-crops", str(args.max_crops)] if args.max_crops > 0 else []
        run(
            [
                py,
                str(_SCRIPTS / "parse_manifest_hybrid.py"),
                "--manifest",
                str(manifest_smart),
                "--out",
                str(parsed_json),
                "--mode",
                "hybrid",
                "--engine",
                args.engine,
            ]
            + max_args
        )

        # Convert parsed json for evaluate_parsed_vs_gt (uses hybrid.parsed shape)
        data = json.loads(parsed_json.read_text(encoding="utf-8"))
        eval_rows = []
        for item in data:
            h = item.get("hybrid") or item.get("parsed") or {}
            eval_rows.append(
                {
                    "image_path": item["image_path"],
                    "parsed": h,
                    "readable": item.get("readable", False),
                }
            )
        eval_in = out_root / "parsed_for_eval.json"
        eval_in.write_text(json.dumps(eval_rows, indent=2, ensure_ascii=False), encoding="utf-8")

        run(
            [
                py,
                str(_SCRIPTS / "evaluate_parsed_vs_gt.py"),
                "--parsed",
                str(eval_in),
                "--manifest",
                str(manifest_smart),
                "--gt",
                str(args.gt.resolve()),
                "--out",
                str(eval_json),
                "--price-tol",
                "2",
            ]
            + (["--train-data", str(args.train_data.resolve())] if args.train_data else [])
            + (["--allow-train-gt-overlap"] if args.allow_train_gt_overlap else [])
        )

        # Baseline eval
        base_rows = [{"image_path": i["image_path"], "parsed": i.get("baseline", {})} for i in data]
        base_in = out_root / "parsed_baseline_eval.json"
        base_in.write_text(json.dumps(base_rows, indent=2, ensure_ascii=False), encoding="utf-8")
        base_eval = out_root / "eval_baseline_vs_gt.json"
        run(
            [
                py,
                str(_SCRIPTS / "evaluate_parsed_vs_gt.py"),
                "--parsed",
                str(base_in),
                "--manifest",
                str(manifest_smart),
                "--gt",
                str(args.gt.resolve()),
                "--out",
                str(base_eval),
                "--price-tol",
                "2",
            ]
            + (["--train-data", str(args.train_data.resolve())] if args.train_data else [])
            + (["--allow-train-gt-overlap"] if args.allow_train_gt_overlap else [])
        )

        summary = {
            "tag": args.tag,
            "manifest": str(manifest_smart),
            "export_padding": args.export_padding,
            "deskew_pad_ratio": args.deskew_pad_ratio,
            "gt": str(args.gt.resolve()),
            "train_data": str(args.train_data.resolve()) if args.train_data else "",
            "gt_leakage_check": str(leakage_report) if leakage_report else "",
            "allow_train_gt_overlap": bool(args.allow_train_gt_overlap),
            "hybrid_eval": json.loads(eval_json.read_text(encoding="utf-8")),
            "baseline_eval": json.loads(base_eval.read_text(encoding="utf-8")),
        }
        summary_path = out_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

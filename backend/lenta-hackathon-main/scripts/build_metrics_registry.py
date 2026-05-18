"""Build a machine-readable registry of known benchmark artifacts.

The registry separates:
- honest: source-holdout or explicitly non-overlapping protocols;
- contaminated: GT source appears in the detector train split;
- diagnostic: useful for debugging, but not an accuracy claim.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from check_gt_leakage import dataset_split_sources, gt_sources


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _load_yolo_tail(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[-1]
    return {
        "epoch": row.get("epoch", ""),
        "precision": _float(row.get("metrics/precision(B)", "")),
        "recall": _float(row.get("metrics/recall(B)", "")),
        "mAP50": _float(row.get("metrics/mAP50(B)", "")),
        "mAP50_95": _float(row.get("metrics/mAP50-95(B)", "")),
    }


def _count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)


def _float(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _leakage(data_yaml: Path, gt: Path) -> dict[str, Any]:
    if not data_yaml.exists() or not gt.exists():
        return {
            "checked": False,
            "reason": "missing data_yaml or gt",
            "data_yaml": str(data_yaml),
            "gt": str(gt),
        }
    train = dataset_split_sources(data_yaml, "train")
    val = dataset_split_sources(data_yaml, "val")
    gt_src = gt_sources(gt)
    overlap = sorted(gt_src & set(train))
    return {
        "checked": True,
        "data_yaml": str(data_yaml),
        "gt": str(gt),
        "gt_sources": sorted(gt_src),
        "train_sources": train,
        "val_sources": val,
        "train_gt_overlap": overlap,
        "leakage_detected": bool(overlap),
    }


def root_43_15_artifacts(root: Path) -> list[dict[str, Any]]:
    data_yaml = root / "dataset" / "price_tags_merged" / "data.yaml"
    gt = root / "materials" / "data" / "43_15" / "43_15.csv"
    leak = _leakage(data_yaml, gt)
    return [
        {
            "id": "root_final_smart_v2",
            "path": "output/final_smart_v2/summary.json",
            "gt_source": "43_15",
            "trust_level": "contaminated",
            "leakage": leak,
            "metrics": _load_json(root / "output" / "final_smart_v2" / "summary.json"),
            "notes": "Crop/OCR internal benchmark; detector train split contains 43_15.",
        },
        {
            "id": "root_hackathon_compliance_final",
            "path": "output/hackathon_compliance_final.json",
            "gt_source": "43_15",
            "trust_level": "contaminated",
            "leakage": leak,
            "metrics": _load_json(root / "output" / "hackathon_compliance_final.json").get("metrics_vs_gt", {}),
            "notes": "CSV/field coverage QA; repeats contaminated 43_15 price metrics.",
        },
        {
            "id": "root_price_tag_merged_yolo_val",
            "path": "runs/detect/runs/detect/price_tag_merged/results.csv",
            "gt_source": "mixed_val_split",
            "trust_level": "contaminated",
            "leakage": leak,
            "metrics": _load_yolo_tail(root / "runs" / "detect" / "runs" / "detect" / "price_tag_merged" / "results.csv"),
            "notes": "Ultralytics validation split mixes sources; not a source-holdout metric.",
        },
    ]


def diagnostic_artifacts(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "id": "root_validation_43_15_detector_crop",
            "path": "output/validation_43_15_det/comparison.json",
            "gt_source": "43_15",
            "trust_level": "diagnostic",
            "metrics": _load_json(root / "output" / "validation_43_15_det" / "comparison.json").get("summary", {}),
            "notes": "Full-frame detector coupling diagnostic; current detector crops yielded 0% price matches.",
        },
        {
            "id": "root_validation_43_15_gt_oracle",
            "path": "output/validation_43_15_gt_oracle/comparison.json",
            "gt_source": "43_15",
            "trust_level": "diagnostic",
            "metrics": _load_json(root / "output" / "validation_43_15_gt_oracle" / "comparison.json").get("summary", {}),
            "notes": "GT-bbox OCR diagnostic; shows bbox alone is not enough without robust crop preparation.",
        },
        {
            "id": "root_hackathon_ideal_tol1",
            "path": "output/hackathon_ideal/summary.json",
            "gt_source": "43_15",
            "trust_level": "diagnostic",
            "metrics": _load_json(root / "output" / "hackathon_ideal" / "summary.json"),
            "notes": "Tolerance sensitivity check at 1 ruble; not holdout.",
        },
        {
            "id": "root_smoke_one_to_one_eval",
            "path": "output/smoke_one_to_one_eval.json",
            "gt_source": "43_15",
            "trust_level": "diagnostic",
            "metrics": _load_json(root / "output" / "smoke_one_to_one_eval.json"),
            "notes": "One-to-one matching smoke test; still contaminated when run against price_tags_merged.",
        },
        {
            "id": "root_hackathon_compliance_one_to_one",
            "path": "output/hackathon_compliance_one_to_one.json",
            "gt_source": "43_15",
            "trust_level": "diagnostic",
            "metrics": _load_json(root / "output" / "hackathon_compliance_one_to_one.json").get("metrics_vs_gt", {}),
            "notes": "Submission compliance re-scored with one-to-one matching; still contaminated by 43_15 train overlap.",
        },
    ]


def honest_known_artifacts(root: Path) -> list[dict[str, Any]]:
    holdout_43_15_data = root / "dataset" / "price_tags_holdout_43_15" / "data.yaml"
    holdout_43_15_gt = root / "materials" / "data" / "43_15" / "43_15.csv"
    holdout_43_15_aug_data = root / "dataset" / "price_tags_holdout_43_15_aug" / "data.yaml"
    holdout_43_15_neg_data = root / "dataset" / "price_tags_holdout_43_15_neg" / "data.yaml"
    return [
        {
            "id": "root_holdout_43_15_neg_detector_e12",
            "path": "runs/detect/runs/detect/price_tag_holdout_43_15_neg_e12/results.csv",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_neg_data, holdout_43_15_gt),
            "metrics": {
                **_load_yolo_tail(
                    root
                    / "runs"
                    / "detect"
                    / "runs"
                    / "detect"
                    / "price_tag_holdout_43_15_neg_e12"
                    / "results.csv"
                ),
                "full_video_detections_conf_0_01": _count_csv_rows(
                    root / "output" / "holdout_43_15_neg_detect_conf001" / "detections.csv"
                ),
                "full_video_detections_conf_0_03": _count_csv_rows(
                    root / "output" / "holdout_43_15_neg_detect_conf003" / "detections.csv"
                ),
                "full_video_detections_conf_0_05": _count_csv_rows(
                    root / "output" / "holdout_43_15_neg_detect_conf005" / "detections.csv"
                ),
                "full_video_detections_conf_0_10": _count_csv_rows(
                    root / "output" / "holdout_43_15_neg_detect_conf010" / "detections.csv"
                ),
                "full_video_detections_conf_0_15": _count_csv_rows(
                    root / "output" / "holdout_43_15_neg_detect_conf015" / "detections.csv"
                ),
                "full_video_detections_conf_0_25": _count_csv_rows(
                    root / "output" / "holdout_43_15_neg_detect_conf025" / "detections.csv"
                ),
            },
            "notes": (
                "Honest detector trained with 163 negative/background frames. It strongly reduced full-video "
                "candidate count, but val mAP is lower than the augmented-positive detector."
            ),
        },
        {
            "id": "root_holdout_43_15_neg_e2e_conf001",
            "path": "output/holdout_43_15_neg_conf001/summary.json",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_neg_data, holdout_43_15_gt),
            "metrics": _load_json(root / "output" / "holdout_43_15_neg_conf001" / "summary.json").get(
                "hybrid_eval", {}
            ),
            "notes": (
                "Honest E2E with negative-trained detector. Matching improved to all 29 GT rows with only "
                "304 candidates, but OCR price metrics remain 0%, indicating crop/orientation/OCR quality is now "
                "the main bottleneck."
            ),
        },
        {
            "id": "root_holdout_43_15_neg_pricezone_raw",
            "path": "output/holdout_43_15_neg_pricezone_raw/compliance.json",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_neg_data, holdout_43_15_gt),
            "metrics": _load_json(
                root / "output" / "holdout_43_15_neg_pricezone_raw" / "compliance.json"
            ).get("metrics_vs_gt", {}),
            "notes": (
                "Best fast final honest E2E run: negative-trained detector plus raw-crop RapidOCR price-zone "
                "fallback. Leakage gate is enabled; card-price accuracy improved from 0% to a useful level."
            ),
        },
        {
            "id": "root_holdout_43_15_neg_pricezone_discount_raw",
            "path": "output/holdout_43_15_neg_pricezone_discount_raw/compliance.json",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_neg_data, holdout_43_15_gt),
            "metrics": _load_json(
                root / "output" / "holdout_43_15_neg_pricezone_discount_raw" / "compliance.json"
            ).get("metrics_vs_gt", {}),
            "notes": (
                "Best current honest E2E run: negative-trained detector plus raw-crop RapidOCR price-zone "
                "fallback with discount extraction and default-price inference from card price."
            ),
        },
        {
            "id": "root_holdout_43_15_neg_pricezone_discount_barcode_raw",
            "path": "output/holdout_43_15_neg_pricezone_discount_barcode_raw/compliance.json",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_neg_data, holdout_43_15_gt),
            "metrics": _load_json(
                root / "output" / "holdout_43_15_neg_pricezone_discount_barcode_raw" / "compliance.json"
            ).get("metrics_vs_gt", {}),
            "notes": (
                "Best current honest E2E artifact with barcode scan enabled. Barcode remains 0% on 43_15, "
                "but the enabled OpenCV fallback does not regress price, discount, or color metrics."
            ),
        },
        {
            "id": "root_holdout_43_15_aug_detector_e12",
            "path": "runs/detect/runs/detect/price_tag_holdout_43_15_aug_e12/results.csv",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_aug_data, holdout_43_15_gt),
            "metrics": {
                **_load_yolo_tail(
                    root
                    / "runs"
                    / "detect"
                    / "runs"
                    / "detect"
                    / "price_tag_holdout_43_15_aug_e12"
                    / "results.csv"
                ),
                "full_video_detections_conf_0_01": _count_csv_rows(
                    root / "output" / "holdout_43_15_aug_detect_conf001" / "detections.csv"
                ),
                "full_video_detections_conf_0_03": _count_csv_rows(
                    root / "output" / "holdout_43_15_aug_detect_conf003" / "detections.csv"
                ),
                "full_video_detections_conf_0_05": _count_csv_rows(
                    root / "output" / "holdout_43_15_aug_detect_conf005" / "detections.csv"
                ),
                "full_video_detections_conf_0_10": _count_csv_rows(
                    root / "output" / "holdout_43_15_aug_detect_conf010" / "detections.csv"
                ),
                "full_video_detections_conf_0_15": _count_csv_rows(
                    root / "output" / "holdout_43_15_aug_detect_conf015" / "detections.csv"
                ),
                "full_video_detections_conf_0_25": _count_csv_rows(
                    root / "output" / "holdout_43_15_aug_detect_conf025" / "detections.csv"
                ),
            },
            "notes": (
                "Honest augmented 43_15 source-holdout detector. Adjacent-frame train expansion improved "
                "YOLO val quality, but full-video candidate volume remains high and needs better post-filtering."
            ),
        },
        {
            "id": "root_holdout_43_15_aug_e2e_conf025_max200",
            "path": "output/holdout_43_15_aug_conf025_max200/summary.json",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_aug_data, holdout_43_15_gt),
            "metrics": _load_json(root / "output" / "holdout_43_15_aug_conf025_max200" / "summary.json").get(
                "hybrid_eval", {}
            ),
            "notes": (
                "Honest E2E OCR validation with leakage gate enabled. The detector produced many candidates "
                "and only 6 one-to-one GT matches, so field metrics are currently 0%; this is a detector/crop "
                "quality failure, not a leakage issue."
            ),
        },
        {
            "id": "root_holdout_43_15_smoke_detector_e3",
            "path": "runs/detect/runs/detect/price_tag_holdout_43_15_smoke_e3/results.csv",
            "gt_source": "43_15",
            "train_sources": ["25_12-20", "26_12-20", "49_5"],
            "trust_level": "honest",
            "leakage": _leakage(holdout_43_15_data, holdout_43_15_gt),
            "metrics": {
                **_load_yolo_tail(
                    root
                    / "runs"
                    / "detect"
                    / "runs"
                    / "detect"
                    / "price_tag_holdout_43_15_smoke_e3"
                    / "results.csv"
                ),
                "full_video_detections_conf_0_25": _count_csv_rows(
                    root / "output" / "holdout_43_15_smoke_detect" / "detections.csv"
                ),
                "full_video_detections_conf_0_01": _count_csv_rows(
                    root / "output" / "holdout_43_15_smoke_detect_conf001" / "detections.csv"
                ),
            },
            "notes": (
                "Honest 43_15 source-holdout smoke train, only 3 CPU epochs. "
                "Use as protocol proof, not final detector quality; conf=0.25 produced no full-video detections."
            ),
        },
        {
            "id": "nik_holdout_gt_bbox_recognition",
            "path": "NIK/lenta_tech_ml/qr_research/holdout_validation_findings.md",
            "gt_source": "25_2-10,49_5",
            "train_sources": ["25_12-20", "26_12-20", "43_15"],
            "trust_level": "honest",
            "metrics": {
                "raw_mean_field_accuracy": 0.121,
                "catalog_matched_mean_field_accuracy": 0.142,
                "row_80pct_fields": 0.0,
            },
            "notes": "GT boxes on held-out sources; evaluates recognition/layout without detector quality.",
        },
        {
            "id": "nik_tiled_detector_newval",
            "path": "NIK/lenta_tech_ml/qr_research/validation_detector_findings.md",
            "gt_source": "25_2-10,49_5",
            "train_sources": ["25_12-20", "26_12-20", "43_15"],
            "trust_level": "honest",
            "metrics": {
                "precision_iou03": 0.325,
                "recall_iou03": 0.221,
                "f1_iou03": 0.263,
            },
            "notes": "Source-holdout detector estimate; 49_5 was effectively failed.",
        },
    ]


def build_registry(root: Path) -> dict[str, Any]:
    artifacts = root_43_15_artifacts(root) + diagnostic_artifacts(root) + honest_known_artifacts(root)
    return {
        "schema_version": 1,
        "generated_by": "scripts/build_metrics_registry.py",
        "policy": {
            "honest": "source-holdout or no train/GT source overlap",
            "contaminated": "GT source appears in train split or protocol uses GT-aligned/demo data",
            "diagnostic": "useful for debugging, not an accuracy claim",
        },
        "artifacts": artifacts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build metrics registry with leakage labels.")
    ap.add_argument("--out", type=Path, default=Path("output/metrics_registry.json"))
    args = ap.parse_args()
    payload = build_registry(_ROOT)
    out = args.out if args.out.is_absolute() else _ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(out), "artifacts": len(payload["artifacts"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Validate pipeline on materials/data/43_15 (GT CSV + video frames).

Per GT row: nearest detection on mapped frame -> crop -> deskew -> baseline vs hybrid parse.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from difflib import SequenceMatcher
from pathlib import Path

import cv2

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from export_ocr_crops import crop_xyxy, rotate_for_ocr
from filter_detections import iou
from hybrid_price_tag_parser import parse_baseline_fullcrop, parse_tag_image
from lenta_price_normalize import best_price_match, parse_price_float
from score_ocr_compare_vs_gt import bbox_from_gt, load_gt_rows, normalize_text


def parse_float_csv(s: str) -> float:
    return float((s or "").strip().replace(",", ".").replace(" ", ""))


def bbox_centroid(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def centroid_dist_frac(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    diag: float,
) -> float:
    ac = bbox_centroid(a)
    bc = bbox_centroid(b)
    return ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5 / max(diag, 1.0)


def name_sim(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()
    return fps if fps > 1 else 25.0


def frame_path_for_timestamp(
    frames_dir: Path, ts_ms: float, fps: float, max_index: int
) -> Path:
    idx = int(round(ts_ms / 1000.0 * fps))
    idx = max(0, min(max_index, idx))
    return frames_dir / f"{idx:06d}.jpg"


def load_detections(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def image_diag(path: Path) -> float:
    img = cv2.imread(str(path))
    if img is None:
        return 1.0
    h, w = img.shape[:2]
    return (w * w + h * h) ** 0.5


def summarize(rows: list[dict], prefix: str) -> dict:
    n = len(rows) or 1
    return {
        f"{prefix}_name_sim_avg": round(
            sum(r[f"{prefix}_name_sim"] for r in rows) / n, 4
        ),
        f"{prefix}_price_default_match_pct": round(
            100.0 * sum(1 for r in rows if r[f"{prefix}_pd_match"]) / n, 1
        ),
        f"{prefix}_price_card_match_pct": round(
            100.0 * sum(1 for r in rows if r[f"{prefix}_pc_match"]) / n, 1
        ),
        f"{prefix}_discount_match_pct": round(
            100.0
            * sum(1 for r in rows if r.get(f"{prefix}_disc_match"))
            / n,
            1,
        ),
        f"{prefix}_barcode_match_pct": round(
            100.0 * sum(1 for r in rows if r.get(f"{prefix}_bc_match")) / n, 1
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, default=Path("materials/data/43_15/43_15.csv"))
    ap.add_argument("--video", type=Path, default=Path("materials/data/43_15/43_15.mp4"))
    ap.add_argument("--frames", type=Path, default=Path("frames/materials_data_43_15_43_15"))
    ap.add_argument(
        "--detections",
        type=Path,
        default=Path("runs/detect_merged_43_15/detections.csv"),
    )
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--center-dist-frac", type=float, default=0.55)
    ap.add_argument("--price-tol", type=float, default=2.0)
    ap.add_argument("--engine", default="paddle", choices=("paddle", "rapidocr"))
    ap.add_argument("--deskew-pad-ratio", type=float, default=0.28)
    ap.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="YOLO data.yaml used for detector training. If set, fail when --gt source is present in train.",
    )
    ap.add_argument("--allow-train-gt-overlap", action="store_true")
    ap.add_argument(
        "--use-gt-bbox",
        action="store_true",
        help="Crop GT bbox (oracle) instead of detector box — tests OCR/parser only",
    )
    ap.add_argument("--crop-padding", type=float, default=0.15)
    ap.add_argument("--out-dir", type=Path, default=Path("output/validation_43_15"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    filtered_det = args.out_dir / "detections_filtered.csv"

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

    subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS / "filter_detections.py"),
            "--detections",
            str(args.detections),
            "--out",
            str(filtered_det),
            "--min-conf",
            str(args.min_conf),
        ],
        check=True,
    )

    gt_rows = load_gt_rows(args.gt)
    dets = load_detections(filtered_det)
    dets_by_image: dict[str, list[dict]] = {}
    for d in dets:
        dets_by_image.setdefault(d["image_path"], []).append(d)

    frame_files = sorted(args.frames.glob("*.jpg"))
    max_idx = len(frame_files) - 1
    fps = video_fps(args.video)

    results: list[dict] = []
    for gi, gt in enumerate(gt_rows):
        try:
            gt_box = bbox_from_gt(gt)
        except (KeyError, ValueError):
            continue
        ts = parse_float_csv(gt.get("frame_timestamp", "0"))
        frame_path = frame_path_for_timestamp(args.frames, ts, fps, max_idx)
        if not frame_path.is_file():
            continue
        diag = image_diag(frame_path)
        fp = str(frame_path.resolve())
        candidates = dets_by_image.get(fp, [])
        if not candidates:
            for alt in (frame_files[max(0, int(ts / 1000 * fps) - 2) : int(ts / 1000 * fps) + 3]):
                candidates.extend(dets_by_image.get(str(alt.resolve()), []))

        best_det = None
        best_dist = 1e9
        for d in candidates:
            box = (
                float(d["x_min"]),
                float(d["y_min"]),
                float(d["x_max"]),
                float(d["y_max"]),
            )
            dist = centroid_dist_frac(box, gt_box, diag)
            if dist < best_dist:
                best_dist = dist
                best_det = d

        row: dict = {
            "gt_index": gi,
            "product_name_gt": gt.get("product_name", ""),
            "price_default_gt": gt.get("price_default", ""),
            "price_card_gt": gt.get("price_card", ""),
            "discount_gt": gt.get("discount_amount", ""),
            "barcode_gt": gt.get("barcode", ""),
            "frame_timestamp_ms": ts,
            "frame_path": fp,
            "det_matched": best_det is not None and best_dist <= args.center_dist_frac,
            "det_dist_frac": round(best_dist, 4),
        }
        use_gt = args.use_gt_bbox
        if not use_gt and (not row["det_matched"] or best_det is None):
            results.append(row)
            continue

        img = cv2.imread(fp)
        if img is None:
            results.append(row)
            continue
        if use_gt:
            x1, y1, x2, y2 = gt_box
        else:
            x1, y1, x2, y2 = (
                float(best_det["x_min"]),
                float(best_det["y_min"]),
                float(best_det["x_max"]),
                float(best_det["y_max"]),
            )
        crop = crop_xyxy(img, x1, y1, x2, y2, padding=args.crop_padding)
        crop = rotate_for_ocr(crop, "ccw90")

        base = parse_baseline_fullcrop(
            crop,
            deskew=True,
            engine=args.engine,
            deskew_pad_ratio=args.deskew_pad_ratio,
        )
        hybrid = parse_tag_image(
            crop,
            deskew=True,
            engine=args.engine,
            deskew_pad_ratio=args.deskew_pad_ratio,
        )

        gt_name = gt.get("product_name", "")
        gt_pd = gt.get("price_default", "")
        gt_pc = gt.get("price_card", "")
        gt_disc = gt.get("discount_amount", "")
        gt_bc = gt.get("barcode", "")

        for prefix, parsed in (("baseline", base), ("hybrid", hybrid)):
            row[f"{prefix}_product_name"] = parsed.get("product_name", "")
            row[f"{prefix}_price_default"] = parsed.get("price_default", "")
            row[f"{prefix}_price_card"] = parsed.get("price_card", "")
            row[f"{prefix}_discount"] = parsed.get("discount_amount", "")
            row[f"{prefix}_barcode"] = parsed.get("barcode", "")
            row[f"{prefix}_template"] = parsed.get("template", "")
            row[f"{prefix}_name_sim"] = name_sim(parsed.get("product_name", ""), gt_name)
            row[f"{prefix}_pd_match"] = best_price_match(
                parsed.get("price_default", ""), gt_pd, args.price_tol
            )
            row[f"{prefix}_pc_match"] = best_price_match(
                parsed.get("price_card", ""), gt_pc, args.price_tol
            )
            disc_pred = (parsed.get("discount_amount", "") or "").replace(" ", "")
            disc_gt = (gt_disc or "").replace(" ", "")
            row[f"{prefix}_disc_match"] = (
                bool(disc_pred)
                and bool(disc_gt)
                and disc_pred.lstrip("-") == disc_gt.lstrip("-")
            ) or (
                parse_price_float(disc_pred.replace("%", "")) > 0
                and parse_price_float(disc_gt.replace("%", "").replace("-", "")) > 0
                and abs(
                    parse_price_float(disc_pred.replace("%", "").replace("-", ""))
                    - parse_price_float(disc_gt.replace("%", "").replace("-", ""))
                )
                < 1
            )
            row[f"{prefix}_bc_match"] = normalize_text(parsed.get("barcode", "")) in normalize_text(
                gt_bc
            ) and bool(parsed.get("barcode"))

        results.append(row)

    matched = [r for r in results if r.get("det_matched")]
    summary = {
        "gt_rows": len(gt_rows),
        "detection_matched": len(matched),
        "min_conf": args.min_conf,
        "price_tolerance_rub": args.price_tol,
        "crop_padding": args.crop_padding,
        "deskew_pad_ratio": args.deskew_pad_ratio,
        "engine": args.engine,
        "gt": str(args.gt.resolve()),
        "train_data": str(args.train_data.resolve()) if args.train_data else "",
        "allow_train_gt_overlap": bool(args.allow_train_gt_overlap),
    }
    summary.update(summarize(matched, "baseline"))
    summary.update(summarize(matched, "hybrid"))

    out_json = args.out_dir / "comparison.json"
    out_csv = args.out_dir / "comparison.csv"
    out_json.write_text(
        json.dumps({"summary": summary, "rows": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if results:
        fields = list(results[0].keys())
        for r in results[1:]:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(results)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out_json} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

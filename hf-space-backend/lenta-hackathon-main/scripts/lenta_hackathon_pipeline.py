"""
Unified Lenta hackathon pipeline (best-known stack).

Video / frames + detections -> filter -> export -> smart deskew -> hybrid OCR
-> barcode (pyzbar) -> dedupe -> CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from hackathon_csv_format import HACKATHON_COLUMNS, parsed_to_hackathon_row
from lenta_barcode import read_barcode_from_tag, read_qr_fields_from_tag
from tag_dedupe import dedupe_tag_rows, row_quality_score


def run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    subprocess.run(cmd, check=True, cwd=str(_ROOT))


def frame_index_to_timestamp_ms(frame_stem: str, fps: float) -> str:
    try:
        idx = int(frame_stem)
    except ValueError:
        return ""
    return str(int(round(idx / max(fps, 1.0) * 1000.0)))


def field_fill_profile(row: dict[str, str]) -> dict[str, str]:
    key_fields = [
        "product_name",
        "price_default",
        "price_card",
        "barcode",
        "discount_amount",
        "color",
    ]
    filled = [
        name
        for name in key_fields
        if (row.get(name) or "").strip() and (row.get(name) or "").strip().lower() != "нет"
    ]
    return {
        "filled_fields": "|".join(filled),
        "filled_field_count": str(len(filled)),
        "has_price_pair": str(bool(row.get("price_default") and row.get("price_card"))),
        "has_barcode": str(bool((row.get("barcode") or "").strip())),
        "has_discount": str(bool((row.get("discount_amount") or "").strip() not in {"", "нет"})),
    }


def build_rows_from_parsed(
    parsed_items: list[dict],
    manifest: dict[str, dict],
    video_name: str,
    fps: float,
    *,
    enrich_barcode: bool = True,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    import cv2

    rows: list[dict[str, str]] = []
    metas: list[dict[str, Any]] = []
    for item in parsed_items:
        crop_key = str(Path(item["image_path"]).resolve())
        m = manifest.get(crop_key, {})
        stem = Path(m.get("source_image", "")).stem
        ts = frame_index_to_timestamp_ms(stem, fps)
        hybrid = dict(item.get("hybrid") or item.get("parsed") or {})

        img = cv2.imread(crop_key) if enrich_barcode else None
        if img is not None:
            if not hybrid.get("color"):
                from lenta_field_rois import infer_tag_color

                hybrid["color"] = infer_tag_color(img)
            qr_fields = read_qr_fields_from_tag(img)
            for key, value in qr_fields.items():
                hybrid.setdefault(key, value)
            if not hybrid.get("barcode"):
                ocr_hints = [
                    hybrid.get("product_name", ""),
                    item.get("baseline", {}).get("product_name", ""),
                ]
                hybrid["barcode"] = read_barcode_from_tag(
                    img,
                    ocr_texts=ocr_hints,
                    include_qr=True,
                )

        row = parsed_to_hackathon_row(
            hybrid,
            filename=video_name,
            frame_timestamp_ms=ts,
            bbox=(
                m.get("x_min", ""),
                m.get("y_min", ""),
                m.get("x_max", ""),
                m.get("y_max", ""),
            ),
        )
        profile = field_fill_profile(row)
        profile.update(
            {
                "crop_path": crop_key,
                "source_image": m.get("source_image", ""),
                "detector_confidence": str(m.get("confidence", "")),
                "template": str(hybrid.get("template", "")),
                "price_default_inferred": str(bool(hybrid.get("price_default_inferred"))),
                "smart_deskewed": str(m.get("smart_deskewed", "")),
                "row_quality_score": f"{row_quality_score(row, {'confidence': m.get('confidence', 0) or 0}):.3f}",
            }
        )
        rows.append(row)
        metas.append(
            {
                "confidence": float(m.get("confidence", 0) or 0),
                "crop_path": crop_key,
                "diagnostics": profile,
            }
        )
    return rows, metas


def write_diagnostics(path: Path, rows: list[dict[str, str]], metas: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    diag_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        meta = metas[idx] if idx < len(metas) else {}
        diag = dict(meta.get("diagnostics") or {})
        diag.update(
            {
                "row_index": str(idx),
                "filename": row.get("filename", ""),
                "frame_timestamp": row.get("frame_timestamp", ""),
                "x_min": row.get("x_min", ""),
                "y_min": row.get("y_min", ""),
                "x_max": row.get("x_max", ""),
                "y_max": row.get("y_max", ""),
                "product_name": row.get("product_name", ""),
                "price_default": row.get("price_default", ""),
                "price_card": row.get("price_card", ""),
                "barcode": row.get("barcode", ""),
                "discount_amount": row.get("discount_amount", ""),
                "color": row.get("color", ""),
            }
        )
        diag_rows.append(diag)
    if not diag_rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for diag in diag_rows:
        for key in diag:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(diag_rows)


def row_identity(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return tuple(str(row.get(k, "")) for k in ("frame_timestamp", "x_min", "y_min", "x_max", "y_max"))  # type: ignore[return-value]


def align_metas_after_dedupe(
    selected_rows: list[dict[str, str]],
    original_rows: list[dict[str, str]],
    original_metas: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_identity: dict[tuple[str, str, str, str, str], list[int]] = {}
    for idx, row in enumerate(original_rows):
        by_identity.setdefault(row_identity(row), []).append(idx)

    used: set[int] = set()
    aligned: list[dict[str, Any]] = []
    for row in selected_rows:
        candidates = [idx for idx in by_identity.get(row_identity(row), []) if idx not in used]
        if not candidates:
            aligned.append({})
            continue
        best_idx = max(candidates, key=lambda idx: row_quality_score(original_rows[idx], original_metas[idx]))
        used.add(best_idx)
        aligned.append(original_metas[best_idx])
    return aligned


def _write_per_frame_final_stats(
    original_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
    out_csv: Path,
) -> None:
    """Write per-frame crop counts before/after final deduplication."""
    from collections import defaultdict

    before: dict[str, int] = defaultdict(int)
    for r in original_rows:
        ts = (r.get("frame_timestamp") or "").strip() or "0"
        before[ts] += 1

    after: dict[str, int] = defaultdict(int)
    for r in final_rows:
        ts = (r.get("frame_timestamp") or "").strip() or "0"
        after[ts] += 1

    frames = sorted(set(before.keys()) | set(after.keys()))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_timestamp", "crops_before_dedupe", "crops_after_dedupe", "lost"])
        for ts in frames:
            b = before.get(ts, 0)
            a = after.get(ts, 0)
            w.writerow([ts, b, a, b - a])


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified Lenta price-tag hackathon pipeline")
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path("runs/pipeline_work"))
    ap.add_argument("--use-manifest", type=Path, default=None, help="Skip export+deskew")
    ap.add_argument("--parsed-json", type=Path, default=None, help="Skip OCR parse step")
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--export-padding", type=float, default=0.15)
    ap.add_argument("--deskew-pad-ratio", type=float, default=0.28)
    ap.add_argument("--engine", default="paddle")
    ap.add_argument("--max-crops", type=int, default=0)
    ap.add_argument("--upscale-model", type=Path, default=None)
    ap.add_argument("--upscale-model-name", default="edsr")
    ap.add_argument("--upscale-scale", type=int, default=4)
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--no-barcode-scan", action="store_true")
    ap.add_argument("--dedupe-iou", type=float, default=0.45)
    ap.add_argument(
        "--dedupe-spatial-px",
        type=float,
        default=0.0,
        help="Merge same shelf slot across frames (e.g. 300 for robot revisits)",
    )
    args = ap.parse_args()

    py = sys.executable
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)

    filtered = work / "detections_filtered.csv"
    raw = work / "crops_raw"
    smart = work / "crops_smart"
    upscaled = work / "crops_upscaled"
    manifest_smart = smart / "crops_manifest.csv"
    parsed_json = args.parsed_json or (work / "parsed_hybrid.json")

    if args.use_manifest is not None:
        manifest_smart = args.use_manifest.resolve()
        smart = manifest_smart.parent
    elif not args.parsed_json:
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
                str(raw),
                "--padding",
                str(args.export_padding),
                "--rotate",
                "ccw90",
            ]
        )
        run(
            [
                py,
                str(_SCRIPTS / "process_tags_smart_deskew.py"),
                "--input",
                str(raw),
                "--out",
                str(smart),
                "--pad-ratio",
                str(args.deskew_pad_ratio),
            ]
        )
        if args.upscale_model is not None:
            run(
                [
                    py,
                    str(_SCRIPTS / "upscale_crops.py"),
                    "--input",
                    str(smart),
                    "--out",
                    str(upscaled),
                    "--model",
                    str(args.upscale_model.resolve()),
                    "--model-name",
                    args.upscale_model_name,
                    "--scale",
                    str(args.upscale_scale),
                ]
            )
            manifest_smart = upscaled / "crops_manifest.csv"

    if args.parsed_json is None:
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

    import cv2

    cap = cv2.VideoCapture(str(args.video.resolve()))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    cap.release()

    parsed_items = json.loads(Path(parsed_json).read_text(encoding="utf-8"))
    manifest = {
        str(Path(r["crop_path"]).resolve()): r
        for r in csv.DictReader(manifest_smart.open(encoding="utf-8"))
    }

    rows, metas = build_rows_from_parsed(
        parsed_items,
        manifest,
        args.video.name,
        fps,
        enrich_barcode=not args.no_barcode_scan,
    )
    n_before = len(rows)
    original_rows = list(rows)
    original_metas = list(metas)
    diagnostics_before = args.out_csv.parent / f"{args.out_csv.stem}_diagnostics_before_dedupe.csv"
    write_diagnostics(diagnostics_before, rows, metas)
    if not args.no_dedupe:
        rows = dedupe_tag_rows(
            rows,
            meta_by_index=metas,
            iou_threshold=args.dedupe_iou,
            dedupe_spatial_px=args.dedupe_spatial_px,
        )
        metas = align_metas_after_dedupe(rows, original_rows, original_metas)

    per_frame_final_csv = args.out_csv.parent / f"{args.out_csv.stem}_per_frame_final_counts.csv"
    _write_per_frame_final_stats(original_rows, rows, per_frame_final_csv)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=HACKATHON_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    diagnostics_after = args.out_csv.parent / f"{args.out_csv.stem}_diagnostics.csv"
    write_diagnostics(diagnostics_after, rows, metas)

    meta = {
        "video": str(args.video.resolve()),
        "rows_before_dedupe": n_before,
        "rows_after_dedupe": len(rows),
        "work_dir": str(work),
        "manifest": str(manifest_smart),
        "parsed_json": str(parsed_json),
        "diagnostics_csv": str(diagnostics_after),
        "diagnostics_before_dedupe_csv": str(diagnostics_before),
        "per_frame_final_counts_csv": str(per_frame_final_csv),
        "settings": {
            "min_conf": args.min_conf,
            "export_padding": args.export_padding,
            "deskew_pad_ratio": args.deskew_pad_ratio,
            "dedupe_iou": args.dedupe_iou,
            "dedupe_spatial_px": args.dedupe_spatial_px,
        },
    }
    meta_path = args.out_csv.parent / f"{args.out_csv.stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

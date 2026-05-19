"""
Оценка OCR по эталонному CSV Ленты: для каждого кропа из manifest находится ближайшая строка GT
по расстоянию между центрами bbox (детектор часто шире эталона — IoU не используем).

Пример:
  python scripts/score_ocr_compare_vs_gt.py \\
    --compare runs/ocr_engine_compare.csv \\
    --manifest runs/ocr_crops_latest/crops_manifest.csv \\
    --gt materials/data/43_15/43_15.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path


def parse_float_csv(s: str) -> float:
    s = (s or "").strip().strip('"').replace(" ", "")
    s = s.replace(",", ".")
    return float(s)


def bbox_from_manifest(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        parse_float_csv(row["x_min"]),
        parse_float_csv(row["y_min"]),
        parse_float_csv(row["x_max"]),
        parse_float_csv(row["y_max"]),
    )


def bbox_from_gt(row: dict[str, str]) -> tuple[float, float, float, float]:
    return (
        parse_float_csv(row["x_min"]),
        parse_float_csv(row["y_min"]),
        parse_float_csv(row["x_max"]),
        parse_float_csv(row["y_max"]),
    )


def gt_reference_text(row: dict[str, str]) -> str:
    parts = [
        row.get("product_name", ""),
        row.get("price_default", ""),
        row.get("price_card", ""),
        row.get("price_discount", ""),
        row.get("barcode", ""),
        row.get("discount_amount", ""),
        row.get("id_sku", ""),
        row.get("code", ""),
        row.get("additional_info", ""),
        row.get("special_symbols", ""),
    ]
    return " ".join(p.strip() for p in parts if p and str(p).strip())


_WS = re.compile(r"\s+")


def normalize_digits(s: str) -> str:
    return re.sub(r"\D+", "", s or "")


def barcode_in_ocr(gt_row: dict[str, str], ocr_text: str) -> bool:
    bc = normalize_digits(gt_row.get("barcode", ""))
    if len(bc) < 8:
        return False
    ocr_d = normalize_digits(ocr_text)
    return bc in ocr_d


def normalize_text(s: str) -> str:
    s = _WS.sub(" ", (s or "").strip()).lower()
    # запятые в числах → точки для совпадения с OCR вариантами
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)
    return s


def similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def load_gt_rows(gt_path: Path) -> list[dict[str, str]]:
    with gt_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def best_gt_match_centroid(
    mb: tuple[float, float, float, float],
    gt_rows: Iterable[dict[str, str]],
    image_diag: float,
    max_dist_frac: float,
) -> tuple[dict[str, str] | None, float]:
    """Подбираем строку GT с минимальной дистанцией между центрами bbox (в пикселях)."""
    cx_m = (mb[0] + mb[2]) / 2.0
    cy_m = (mb[1] + mb[3]) / 2.0
    best: dict[str, str] | None = None
    best_d = float("inf")
    for row in gt_rows:
        try:
            gb = bbox_from_gt(row)
        except (KeyError, ValueError):
            continue
        cx_g = (gb[0] + gb[2]) / 2.0
        cy_g = (gb[1] + gb[3]) / 2.0
        d = ((cx_m - cx_g) ** 2 + (cy_m - cy_g) ** 2) ** 0.5
        if d < best_d:
            best_d = d
            best = row
    if best is None or image_diag <= 0:
        return None, best_d
    if best_d / image_diag > max_dist_frac:
        return None, best_d
    return best, best_d


def match_manifest_to_gt_one_to_one(
    manifest_rows: list[dict[str, str]],
    gt_rows: list[dict[str, str]],
    get_image_diag,
    max_dist_frac: float,
) -> list[tuple[int, dict[str, str], dict[str, str], float]]:
    """Greedy one-to-one crop/GT matching by centroid distance.

    Old evaluations matched each crop independently, so many crops could score
    against the same GT row. This helper prevents that optimistic many-to-one
    accounting while keeping the same centroid-distance matching criterion.
    """
    candidates: list[tuple[float, int, int, dict[str, str], dict[str, str]]] = []
    for mi, mrow in enumerate(manifest_rows):
        try:
            mb = bbox_from_manifest(mrow)
        except (KeyError, ValueError):
            continue
        src = str(Path((mrow.get("source_image") or "").strip()).resolve())
        diag = get_image_diag(src) if src else 1.0
        for gi, gt_row in enumerate(gt_rows):
            try:
                gb = bbox_from_gt(gt_row)
            except (KeyError, ValueError):
                continue
            cx_m = (mb[0] + mb[2]) / 2.0
            cy_m = (mb[1] + mb[3]) / 2.0
            cx_g = (gb[0] + gb[2]) / 2.0
            cy_g = (gb[1] + gb[3]) / 2.0
            dist = ((cx_m - cx_g) ** 2 + (cy_m - cy_g) ** 2) ** 0.5
            if diag > 0 and dist / diag <= max_dist_frac:
                candidates.append((dist, mi, gi, mrow, gt_row))

    candidates.sort(key=lambda item: item[0])
    used_manifest: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[int, dict[str, str], dict[str, str], float]] = []
    for dist, mi, gi, mrow, gt_row in candidates:
        if mi in used_manifest or gi in used_gt:
            continue
        used_manifest.add(mi)
        used_gt.add(gi)
        src = str(Path((mrow.get("source_image") or "").strip()).resolve())
        diag = get_image_diag(src) if src else 1.0
        matches.append((mi, mrow, gt_row, dist / max(diag, 1.0)))
    matches.sort(key=lambda item: item[0])
    return matches


def image_diag(path: Path) -> float:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        buf = path.read_bytes()
        import numpy as np

        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return 1.0
    h, w = img.shape[:2]
    return float((w * w + h * h) ** 0.5)


def main() -> int:
    ap = argparse.ArgumentParser(description="Оценка OCR vs эталон CSV (центры bbox + текст).")
    ap.add_argument("--compare", type=Path, required=True, help="CSV из compare_ocr_on_images.py")
    ap.add_argument("--manifest", type=Path, required=True, help="crops_manifest.csv")
    ap.add_argument("--gt", type=Path, required=True, help="Эталон, например materials/data/43_15/43_15.csv")
    ap.add_argument(
        "--train-data",
        type=Path,
        default=None,
        help="YOLO data.yaml used for detector training. If set, fail when --gt source is present in train.",
    )
    ap.add_argument("--allow-train-gt-overlap", action="store_true")
    ap.add_argument(
        "--center-dist-frac",
        type=float,
        default=0.55,
        help="Макс. расстояние центров det/GT как доля диагонали кадра (det часто шире эталона — IoU бессмысленен)",
    )
    ap.add_argument("--out-json", type=Path, default=Path("runs/ocr_engine_scores.json"))
    args = ap.parse_args()

    if args.train_data is not None:
        import subprocess

        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "check_gt_leakage.py"),
            "--data-yaml",
            str(args.train_data.resolve()),
            "--gt",
            str(args.gt.resolve()),
        ]
        if args.allow_train_gt_overlap:
            cmd.append("--allow-overlap")
        subprocess.run(cmd, check=True)

    gt_rows = load_gt_rows(args.gt)
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

    compare_by_crop_engine: dict[tuple[str, str], str] = {}
    with args.compare.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            crop = str(Path(row["image_path"].strip()).resolve())
            eng = row["engine"].strip().lower()
            if row.get("error"):
                compare_by_crop_engine[(crop, eng)] = ""
            else:
                compare_by_crop_engine[(crop, eng)] = row.get("full_text", "") or ""

    engines = sorted({e for (_, e) in compare_by_crop_engine.keys()})

    per_engine_scores: dict[str, list[float]] = defaultdict(list)
    per_engine_barcode_hits: dict[str, list[int]] = defaultdict(list)
    matched_crops = 0

    matches = match_manifest_to_gt_one_to_one(manifest_rows, gt_rows, get_diag, args.center_dist_frac)
    for _mi, mrow, gt_row, _dist_frac in matches:
        crop_path = str(Path((mrow.get("crop_path") or "").strip()).resolve())
        if not crop_path:
            continue
        matched_crops += 1
        ref = gt_reference_text(gt_row)
        for eng in engines:
            ocr = compare_by_crop_engine.get((crop_path, eng), "")
            per_engine_scores[eng].append(similarity(ref, ocr))
            per_engine_barcode_hits[eng].append(1 if barcode_in_ocr(gt_row, ocr) else 0)

    summary = []
    for eng in engines:
        xs = per_engine_scores.get(eng, [])
        bh = per_engine_barcode_hits.get(eng, [])
        summary.append(
            {
                "engine": eng,
                "samples": len(xs),
                "mean_similarity_full_gt": float(sum(xs) / len(xs)) if xs else 0.0,
                "mean_barcode_in_ocr": float(sum(bh) / len(bh)) if bh else 0.0,
            }
        )
    summary.sort(key=lambda x: (-x["mean_barcode_in_ocr"], -x["mean_similarity_full_gt"], x["engine"]))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "center_dist_frac_max": args.center_dist_frac,
        "matched_crops_in_manifest": matched_crops,
        "matching": "one_to_one_greedy_centroid",
        "manifest_rows": len(manifest_rows),
        "gt": str(args.gt.resolve()),
        "train_data": str(args.train_data.resolve()) if args.train_data else "",
        "allow_train_gt_overlap": bool(args.allow_train_gt_overlap),
        "engines_ranked": summary,
    }
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nСохранено: {args.out_json.resolve()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

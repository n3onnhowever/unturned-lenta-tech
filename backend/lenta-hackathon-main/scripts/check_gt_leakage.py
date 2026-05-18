"""
Check whether an evaluation GT source appears in a YOLO train split.

This is a guard against reporting metrics on videos/sources that were also used
to train the detector. It works with this repo's generated label names, including
merged datasets with prefixes like `price_tags_manual__...`.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


def _read_data_yaml_path(data_yaml: Path) -> Path:
    root = data_yaml.parent
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("path:"):
            continue
        raw = stripped.split(":", 1)[1].strip().strip("'\"")
        if raw:
            return Path(raw)
    return root


def _read_split(data_yaml: Path, split: str) -> str:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{split}:"):
            return stripped.split(":", 1)[1].strip().strip("'\"")
    return f"images/{split}"


def _normalize_source(value: str) -> str:
    text = value.replace("\\", "/").strip()
    if not text:
        return ""
    text = Path(text).stem if "/" in text or "." in Path(text).name else text
    match = re.search(r"(\d{2}_\d{1,2}(?:-\d{1,2})?)", text)
    if match:
        return match.group(1)
    return text


def _source_from_label_stem(stem: str) -> str:
    if "__" in stem:
        stem = stem.split("__", 1)[1]
    if stem.startswith("materials_data_"):
        stem = stem[len("materials_data_") :]
    parts = stem.rsplit("_", 2)
    raw = parts[0] if len(parts) >= 3 else stem
    # Manual datasets sometimes use materials_data_43_15_43_15_<frame>_<hash>.
    chunks = raw.split("_")
    if len(chunks) >= 4 and chunks[0] == chunks[2] and chunks[1] == chunks[3]:
        raw = "_".join(chunks[:2])
    return _normalize_source(raw)


def dataset_split_sources(data_yaml: Path, split: str) -> dict[str, int]:
    root = _read_data_yaml_path(data_yaml)
    split_path = _read_split(data_yaml, split)
    image_dir = (root / split_path).resolve()
    label_dir = Path(str(image_dir).replace(f"{Path('images')}", f"{Path('labels')}"))
    if not label_dir.is_dir():
        label_dir = root / "labels" / split
    counts: dict[str, int] = {}
    for label in sorted(label_dir.glob("*.txt")):
        src = _source_from_label_stem(label.stem)
        counts[src] = counts.get(src, 0) + 1
    return counts


def gt_sources(gt_csv: Path) -> set[str]:
    if gt_csv.is_dir():
        csv_paths = sorted(gt_csv.rglob("*.csv"))
    else:
        csv_paths = [gt_csv]
    sources: set[str] = set()
    for path in csv_paths:
        source = _normalize_source(path.parent.name)
        try:
            with path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError:
            continue
        for row in rows:
            filename = row.get("filename", "")
            sources.add(_normalize_source(filename) or source)
        if not rows:
            sources.add(source)
    return {src for src in sources if src}


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect GT/source overlap with YOLO train split.")
    ap.add_argument("--data-yaml", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True, help="GT CSV or directory with GT CSV files")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--allow-overlap", action="store_true")
    args = ap.parse_args()

    train = dataset_split_sources(args.data_yaml, "train")
    val = dataset_split_sources(args.data_yaml, "val")
    gt = gt_sources(args.gt)
    overlap = sorted(gt & set(train))
    payload: dict[str, Any] = {
        "data_yaml": str(args.data_yaml.resolve()),
        "gt": str(args.gt.resolve()),
        "gt_sources": sorted(gt),
        "train_sources": train,
        "val_sources": val,
        "train_gt_overlap": overlap,
        "leakage_detected": bool(overlap),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text, encoding="utf-8")
    print(text)
    if overlap and not args.allow_overlap:
        print(
            "GT source is present in the train split. Rebuild the dataset with --holdout-sources "
            + " ".join(overlap),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

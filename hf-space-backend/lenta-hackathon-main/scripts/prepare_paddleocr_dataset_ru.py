"""
Build PaddleOCR recognition train/val lists from Lenta price-tag crops (Russian pseudo-labels).
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

import cv2
from paddleocr import PaddleOCR

os.environ.setdefault("GLOG_minloglevel", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zone_ocr_parser import slice_zones


def load_allowed_chars(dict_path: Path) -> Set[str]:
    chars: Set[str] = set()
    with open(dict_path, "rb") as fin:
        for line in fin:
            chars.update(line.decode("utf-8").strip("\n\r"))
    chars.add(" ")
    return chars


def text_is_valid(text: str, allowed: Set[str], max_len: int) -> bool:
    text = text.strip()
    if not text or len(text) > max_len:
        return False
    return all(ch in allowed for ch in text)


def frame_id_from_name(name: str) -> str:
    m = re.search(r"_(\d{6})_", name)
    return m.group(1) if m else name


def parse_ocr_lines(ocr_result) -> List[Tuple]:
    """Return list of (box, text, score)."""
    lines = []
    if not ocr_result:
        return lines

    payload = ocr_result[0] if isinstance(ocr_result, list) and ocr_result else ocr_result
    if isinstance(payload, dict):
        polys = payload.get("dt_polys") or payload.get("rec_polys")
        texts = payload.get("rec_texts", [])
        scores = payload.get("rec_scores", [])
        if polys is not None and texts:
            for box, text, score in zip(polys, texts, scores):
                lines.append((box, str(text), float(score)))
        return lines

    if isinstance(payload, list):
        for line in payload:
            if isinstance(line, list) and len(line) == 2:
                box, info = line
                if isinstance(info, (list, tuple)) and len(info) >= 2:
                    lines.append((box, str(info[0]), float(info[1])))
    return lines


def crop_from_box(image, box) -> Optional[Tuple[int, int, int, int]]:
    xs = [int(p[0]) for p in box]
    ys = [int(p[1]) for p in box]
    x1, x2 = max(0, min(xs)), min(image.shape[1], max(xs))
    y1, y2 = max(0, min(ys)), min(image.shape[0], max(ys))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def process_image(
    image_path: Path,
    ocr,
    allowed: Set[str],
    max_len: int,
    min_score: float,
    crops_dir: Path,
    train_out,
    val_out,
    val_ratio: float,
    zone_split: bool,
    crop_counter: list[int],
    stats: dict,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        stats["read_fail"] += 1
        return

    zones = [image]
    if zone_split:
        z_name, z_prices = slice_zones(image)
        zones = [z_name, z_prices]

    fid = frame_id_from_name(image_path.name)
    is_val = (int(fid) % max(1, int(1 / val_ratio))) == 0 if fid.isdigit() else random.random() < val_ratio
    out_file = val_out if is_val else train_out

    for zone in zones:
        result = ocr.ocr(zone)
        for box, text, score in parse_ocr_lines(result):
            if score < min_score:
                stats["low_score"] += 1
                continue
            text = text.strip()
            if not text_is_valid(text, allowed, max_len):
                stats["invalid_text"] += 1
                continue
            bounds = crop_from_box(zone, box)
            if bounds is None:
                continue
            x1, y1, x2, y2 = bounds
            crop = zone[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"crop_{crop_counter[0]:06d}.jpg"
            crop_counter[0] += 1
            cv2.imwrite(str(crops_dir / crop_name), crop)
            out_file.write(f"crops/{crop_name}\t{text}\n")
            stats["crops"] += 1


def load_processed(progress_path: Path) -> Set[str]:
    if not progress_path.exists():
        return set()
    with open(progress_path, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def next_crop_index(crops_dir: Path) -> int:
    max_idx = -1
    for p in crops_dir.glob("crop_*.jpg"):
        try:
            max_idx = max(max_idx, int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max_idx + 1


def collect_images(input_dirs: List[Path], max_images: Optional[int]) -> List[Path]:
    seen = set()
    paths: List[Path] = []
    for d in input_dirs:
        if not d.exists():
            print(f"Skip missing dir: {d}")
            continue
        for p in sorted(d.glob("*.jpg")):
            if p.name in seen:
                continue
            seen.add(p.name)
            paths.append(p)
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def main():
    parser = argparse.ArgumentParser(description="Prepare Russian PaddleOCR rec dataset.")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["runs/ocr_crops_padded_deskewed_43_15"],
        help="Directories with price-tag crop images",
    )
    parser.add_argument(
        "--out-dir",
        default="data/price_tag_dataset_ru/paddle_rec_data",
        help="Output root (crops + label lists)",
    )
    parser.add_argument(
        "--dict",
        default="PaddleOCR/ppocr/utils/dict/cyrillic_dict.txt",
        help="Character dictionary (must match pretrained rec model)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument("--max-len", type=int, default=25)
    parser.add_argument("--max-images", type=int, default=None, help="Limit tag images (debug)")
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="OCR full tag crop (default: split name/price zones)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from existing crops/labels (skip images in progress file)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip first N images in sorted list (one-time recovery if no progress file)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "processed_images.txt"

    dict_path = Path(args.dict)
    allowed = load_allowed_chars(dict_path)
    images = collect_images([Path(p) for p in args.input], args.max_images)
    print(f"Found {len(images)} tag images")

    processed = load_processed(progress_path) if args.resume else set()
    if args.resume and args.start_index > 0 and not processed:
        # Recovery after interrupted run without progress file
        skip_names = {p.name for p in images[: args.start_index]}
        processed |= skip_names
        print(f"Resume: skipping first {args.start_index} images (recovery)")

    pending = [p for p in images if p.name not in processed]
    print(f"Pending: {len(pending)} images ({len(processed)} already done)")

    if not pending:
        print("Nothing to do.")
        return

    print("Initializing PaddleOCR (lang=ru)...")
    ocr = PaddleOCR(
        use_textline_orientation=False,
        lang="ru",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        show_log=False,
    )

    train_path = out_dir / "rec_gt_train.txt"
    val_path = out_dir / "rec_gt_val.txt"
    stats = {"crops": 0, "low_score": 0, "invalid_text": 0, "read_fail": 0, "skipped": len(processed)}
    crop_counter = [next_crop_index(crops_dir) if args.resume else 0]
    train_mode = "a" if args.resume and train_path.exists() else "w"
    val_mode = "a" if args.resume and val_path.exists() else "w"

    with open(train_path, train_mode, encoding="utf-8") as train_out, open(
        val_path, val_mode, encoding="utf-8"
    ) as val_out, open(progress_path, "a", encoding="utf-8") as progress_out:
        total = len(images)
        for i, img_path in enumerate(pending):
            done = len(processed) + i + 1
            if done % 50 == 0:
                print(f"  {done}/{total} images, +{stats['crops']} new line crops...")
            process_image(
                img_path,
                ocr,
                allowed,
                args.max_len,
                args.min_score,
                crops_dir,
                train_out,
                val_out,
                args.val_ratio,
                not args.full_image,
                crop_counter,
                stats,
            )
            progress_out.write(f"{img_path.name}\n")
            progress_out.flush()
            train_out.flush()
            val_out.flush()

    train_lines = sum(1 for _ in open(train_path, encoding="utf-8"))
    val_lines = sum(1 for _ in open(val_path, encoding="utf-8")) if val_path.exists() else 0
    meta = {
        "images": len(images),
        "images_processed": len(processed) + len(pending),
        "crops_new": stats["crops"],
        "crops_total": train_lines + val_lines,
        "train_lines": train_lines,
        "val_lines": val_lines,
        "train_list": str(train_path),
        "val_list": str(val_path),
        "dict": str(dict_path),
        **{k: stats[k] for k in stats if k not in ("crops",)},
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Done. New line crops: {stats['crops']}, total labels: {train_lines + val_lines}")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    print(f"  meta:  {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()

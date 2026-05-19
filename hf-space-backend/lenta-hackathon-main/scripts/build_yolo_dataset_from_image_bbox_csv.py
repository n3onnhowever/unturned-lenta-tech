"""
Сбор датасета YOLO из CSV: готовые картинки + bbox в пикселях (без видео).

Ожидаемые колонки (заголовок обязателен):
  image_path — путь к .jpg/.png относительно корня репозитория или абсолютный
  x_min, y_min, x_max, y_max — числа; допускается запятая как десятичный разделитель

Несколько строк с одним и тем же image_path = несколько ценников на кадре.

Пример:
  python scripts/build_yolo_dataset_from_image_bbox_csv.py --csv my_tags.csv --out dataset/price_tags_from_images

Шаблон строк: см. templates/bboxes_from_images.example.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2


def parse_float(s: str) -> float | None:
    s = (s or "").strip().strip('"').replace(" ", "")
    if not s or s.lower() in ("нет", "nan", "-"):
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def xyxy_to_yolo_line(
    x1: float, y1: float, x2: float, y2: float, iw: int, ih: int
) -> str | None:
    w = x2 - x1
    h = y2 - y1
    if w < 2 or h < 2:
        return None
    xc = (x1 + x2) / 2.0 / iw
    yc = (y1 + y2) / 2.0 / ih
    nw = w / iw
    nh = h / ih
    if nw <= 0 or nh <= 0:
        return None
    xc = min(1.0, max(0.0, xc))
    yc = min(1.0, max(0.0, yc))
    nw = min(1.0, max(1e-6, nw))
    nh = min(1.0, max(1e-6, nh))
    return f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="YOLO-датасет из CSV (картинка + bbox в пикселях).",
    )
    ap.add_argument("--csv", type=Path, required=True, help="CSV с колонками image_path,x_min,y_min,x_max,y_max")
    ap.add_argument("--root", type=Path, default=Path("."), help="Корень для относительных путей в CSV")
    ap.add_argument("--out", type=Path, default=Path("dataset/price_tags_from_images"))
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.csv.is_file():
        print(f"Нет файла: {args.csv}", file=sys.stderr)
        return 1

    groups: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            raw = (row.get("image_path") or row.get("filepath") or "").strip()
            if not raw:
                continue
            p = Path(raw)
            if not p.is_absolute():
                p = (args.root / p).resolve()
            x1 = parse_float(row.get("x_min", ""))
            y1 = parse_float(row.get("y_min", ""))
            x2 = parse_float(row.get("x_max", ""))
            y2 = parse_float(row.get("y_max", ""))
            if None in (x1, y1, x2, y2) or x2 <= x1 + 1 or y2 <= y1 + 1:
                continue
            key = str(p)
            groups[key].append((x1, y1, x2, y2))

    if not groups:
        print("Нет валидных строк (путь + bbox).", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    keys = list(groups.keys())
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * args.val_ratio)) if len(keys) > 1 else 0
    val_set = set(keys[:n_val]) if n_val else set()

    img_train = args.out / "images" / "train"
    lbl_train = args.out / "labels" / "train"
    img_val = args.out / "images" / "val"
    lbl_val = args.out / "labels" / "val"
    for d in (img_train, lbl_train, img_val, lbl_val):
        d.mkdir(parents=True, exist_ok=True)

    ok_train = ok_val = skip = 0

    for key in keys:
        src = Path(key)
        if not src.is_file():
            print(f"Пропуск (нет файла): {src}", file=sys.stderr)
            skip += 1
            continue
        img = cv2.imread(str(src))
        if img is None:
            print(f"Пропуск (не читается): {src}", file=sys.stderr)
            skip += 1
            continue
        ih, iw = img.shape[:2]
        lines: list[str] = []
        for x1, y1, x2, y2 in groups[key]:
            x1c = max(0.0, min(float(iw - 1), x1))
            y1c = max(0.0, min(float(ih - 1), y1))
            x2c = max(0.0, min(float(iw), x2))
            y2c = max(0.0, min(float(ih), y2))
            ln = xyxy_to_yolo_line(x1c, y1c, x2c, y2c, iw, ih)
            if ln:
                lines.append(ln)
        if not lines:
            skip += 1
            continue

        h8 = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
        flat = f"{src.parent.name}_{src.stem}_{h8}.jpg"
        is_val = key in val_set
        idir = img_val if is_val else img_train
        ldir = lbl_val if is_val else lbl_train
        dst_img = idir / flat
        dst_lbl = ldir / f"{Path(flat).stem}.txt"

        cv2.imwrite(str(dst_img), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        with dst_lbl.open("w", encoding="utf-8") as tf:
            tf.writelines(lines)

        if is_val:
            ok_val += 1
        else:
            ok_train += 1

    yaml_path = args.out / "data.yaml"
    root_abs = args.out.resolve()
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root_abs.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names:",
                "  0: price_tag",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"train={ok_train} val={ok_val} пропусков={skip} yaml={yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

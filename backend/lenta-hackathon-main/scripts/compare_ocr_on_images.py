"""
Сравнение нескольких OCR-движков на одних и тех же изображениях (кропы ценников).

Движки: paddle, easyocr, tesseract, rapidocr, doctr.

Примеры:
  python scripts/compare_ocr_on_images.py --input runs/ocr_crops --out runs/ocr_compare.csv
  python scripts/compare_ocr_on_images.py --engines paddle,easyocr --list-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from ocr_engines import detect_installed_engines, run_engine


def read_img(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def iter_images(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Сравнение OCR на папке кропов.")
    ap.add_argument("--input", type=Path, required=False, help="Файл или папка с изображениями")
    ap.add_argument("--out", type=Path, default=Path("runs/ocr_engine_compare.csv"))
    ap.add_argument(
        "--engines",
        type=str,
        default="paddle,easyocr,tesseract,rapidocr,doctr",
        help="Через запятую",
    )
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Обработать только первые N файлов (по сортировке путей); 0 = все",
    )
    args = ap.parse_args()

    installed = detect_installed_engines()
    print("Обнаружены зависимости для движков:", ", ".join(installed) if installed else "(нет)")
    if args.list_only:
        return 0

    if args.input is None:
        print("Нужен --input (или только --list-only)", file=sys.stderr)
        return 2

    requested = [x.strip().lower() for x in args.engines.split(",") if x.strip()]
    engines = [e for e in requested if e in installed]
    skip = [e for e in requested if e not in installed]
    if skip:
        print("Пропуск (нет пакета или tesseract в PATH):", ", ".join(skip), file=sys.stderr)
    if not engines:
        print(
            "Нет ни одного доступного движка из запрошенных. Установите пакеты из requirements-ocr.txt",
            file=sys.stderr,
        )
        return 2

    paths = iter_images(args.input)
    if not paths:
        print(f"Нет изображений: {args.input}", file=sys.stderr)
        return 1
    if args.max_images > 0:
        paths = paths[: args.max_images]

    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "engine", "elapsed_sec", "full_text", "lines_json", "error"])
        for p in paths:
            img = read_img(p)
            if img is None:
                print(f"Не прочитать: {p}", file=sys.stderr)
                continue
            rel = str(p)
            for eng in engines:
                res = run_engine(eng, img)
                w.writerow(
                    [
                        rel,
                        res.engine,
                        f"{res.elapsed_sec:.4f}",
                        res.full_text.replace("\r", " ").replace("\n", " | "),
                        res.to_json_lines(),
                        res.error or "",
                    ]
                )

    print(f"Готово: {len(paths)} изображений, движков {len(engines)} -> {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

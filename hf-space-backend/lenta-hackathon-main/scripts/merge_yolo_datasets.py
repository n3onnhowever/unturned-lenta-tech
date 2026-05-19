"""
Объединение двух YOLO-датасетов (у каждого свой data.yaml, images/, labels/) в один.

Имена файлов при коллизии получают префикс из имени исходного датасета.

Пример:
  python scripts/merge_yolo_datasets.py --a dataset/price_tags --b dataset/price_tags_manual --out dataset/price_tags_merged
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def copy_split(
    src_root: Path,
    tag: str,
    split: str,
    dst_img: Path,
    dst_lbl: Path,
) -> tuple[int, int]:
    """Копирует images/{split} и labels/{split}. Возвращает (число картинок, число лейблов)."""
    si = src_root / "images" / split
    sl = src_root / "labels" / split
    if not si.is_dir() or not sl.is_dir():
        return 0, 0
    n_img = 0
    n_lbl = 0
    for img in sorted(si.iterdir()):
        if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        stem = img.stem
        lbl = sl / f"{stem}.txt"
        new_stem = f"{tag}__{stem}"
        ext = img.suffix
        dst_i = dst_img / f"{new_stem}{ext}"
        shutil.copy2(img, dst_i)
        n_img += 1
        if lbl.is_file():
            shutil.copy2(lbl, dst_lbl / f"{new_stem}.txt")
        else:
            (dst_lbl / f"{new_stem}.txt").write_text("", encoding="utf-8")
        n_lbl += 1
    return n_img, n_lbl


def read_nc(data_yaml: Path) -> int:
    text = data_yaml.read_text(encoding="utf-8")
    nc = 1
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("nc:"):
            try:
                nc = int(s.split(":", 1)[1].strip())
            except ValueError:
                pass
    return nc


def main() -> int:
    ap = argparse.ArgumentParser(description="Слияние двух YOLO-датасетов.")
    ap.add_argument("--a", type=Path, required=True, help="Первый датасет (корень с data.yaml)")
    ap.add_argument("--b", type=Path, required=True, help="Второй датасет")
    ap.add_argument("--out", type=Path, required=True, help="Куда положить объединённый датасет")
    ap.add_argument("--tag-a", type=str, default=None, help="Префикс имён из A (по умолчанию имя папки A)")
    ap.add_argument("--tag-b", type=str, default=None, help="Префикс имён из B")
    args = ap.parse_args()

    ya = args.a / "data.yaml"
    yb = args.b / "data.yaml"
    if not ya.is_file() or not yb.is_file():
        print("У каждого датасета должен быть data.yaml в корне.", file=sys.stderr)
        return 1

    tag_a = args.tag_a or args.a.name.replace(" ", "_")
    tag_b = args.tag_b or args.b.name.replace(" ", "_")

    out = args.out
    if out.exists():
        print(f"Папка уже существует: {out} — удалите вручную или укажите другой --out.", file=sys.stderr)
        return 1

    for sp in ("train", "val"):
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)

    ta = copy_split(args.a, tag_a, "train", out / "images" / "train", out / "labels" / "train")
    va = copy_split(args.a, tag_a, "val", out / "images" / "val", out / "labels" / "val")
    tb = copy_split(args.b, tag_b, "train", out / "images" / "train", out / "labels" / "train")
    vb = copy_split(args.b, tag_b, "val", out / "images" / "val", out / "labels" / "val")

    nc_a, nc_b = read_nc(ya), read_nc(yb)
    if nc_a != nc_b:
        print(f"Предупреждение: nc различается ({nc_a} vs {nc_b}), в merged выставлено max.", file=sys.stderr)
    nc = max(nc_a, nc_b)

    yaml_path = out / "data.yaml"
    root_abs = out.resolve()
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root_abs.as_posix()}",
                "train: images/train",
                "val: images/val",
                f"nc: {nc}",
                "names:",
                "  0: price_tag",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Merged -> {yaml_path}")
    print(f"  from A ({tag_a}): train img/lbl={ta}, val img/lbl={va}")
    print(f"  from B ({tag_b}): train img/lbl={tb}, val img/lbl={vb}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

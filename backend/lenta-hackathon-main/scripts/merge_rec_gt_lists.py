"""
Merge several PaddleOCR rec_gt train/val lists into one directory.

  python scripts/merge_rec_gt_lists.py \\
    --sources data/price_tag_dataset_ru/paddle_rec_data \\
              data/price_tag_dataset_ru_bootstrap \\
              data/price_tag_dataset_ru_manual \\
    --out-dir data/price_tag_dataset_ru_merged
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def append_list(src_list: Path, src_crops: Path, dst_list, dst_crops: Path, prefix: str, counter: list[int]) -> int:
    n = 0
    if not src_list.is_file():
        return 0
    with src_list.open(encoding="utf-8") as fin, dst_list.open("a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line or "\t" not in line:
                continue
            rel, label = line.split("\t", 1)
            src_img = src_crops / Path(rel).name
            if not src_img.is_file():
                src_img = src_crops / rel.replace("crops/", "")
            if not src_img.is_file():
                continue
            new_name = f"{prefix}_{counter[0]:06d}.jpg"
            counter[0] += 1
            shutil.copy2(src_img, dst_crops / new_name)
            fout.write(f"crops/{new_name}\t{label}\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.out_dir
    crops = out / "crops"
    crops.mkdir(parents=True, exist_ok=True)
    train = out / "rec_gt_train.txt"
    val = out / "rec_gt_val.txt"
    train.write_text("", encoding="utf-8")
    val.write_text("", encoding="utf-8")
    counter = [0]
    totals = {"train": 0, "val": 0}

    for i, src in enumerate(args.sources):
        prefix = f"s{i}"
        totals["train"] += append_list(
            src / "rec_gt_train.txt", src / "crops", train, crops, prefix, counter
        )
        totals["val"] += append_list(
            src / "rec_gt_val.txt", src / "crops", val, crops, prefix, counter
        )

    meta = {"train_lines": totals["train"], "val_lines": totals["val"], "sources": [str(s) for s in args.sources]}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Split unreadable price tags into actionable CSV subsets for labeling / det / deskew / OCR.

Example:
  python scripts/filter_unreadable_tags.py \\
    --unreadable output/unreadable_tags_padded_deskewed_43_15.csv \\
    --out-dir output/unreadable_splits \\
    --label-sample 200
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_conf(row: dict) -> float:
    try:
        return float((row.get("confidence") or "0").strip())
    except ValueError:
        return 0.0


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def stratified_sample(
    buckets: dict[str, list[dict]], total: int, rng: random.Random
) -> list[dict]:
    """Pick ~equal count from each bucket; remainder from largest buckets."""
    if total <= 0:
        return []
    keys = [k for k, v in buckets.items() if v]
    if not keys:
        return []
    per = max(1, total // len(keys))
    picked: list[dict] = []
    for key in keys:
        pool = buckets[key][:]
        rng.shuffle(pool)
        picked.extend(pool[:per])
    if len(picked) < total:
        rest = [r for k in keys for r in buckets[k] if r not in picked]
        rng.shuffle(rest)
        picked.extend(rest[: total - len(picked)])
    return picked[:total]


def main() -> int:
    ap = argparse.ArgumentParser(description="Filter unreadable tags into work queues.")
    ap.add_argument(
        "--unreadable",
        type=Path,
        default=Path("output/unreadable_tags_padded_deskewed_43_15.csv"),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("output/unreadable_splits"))
    ap.add_argument(
        "--conf-threshold",
        type=float,
        default=0.7,
        help="Tags below this detector confidence -> low_confidence_det.csv",
    )
    ap.add_argument(
        "--label-sample",
        type=int,
        default=200,
        help="Stratified sample for manual labeling (0 = skip)",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.unreadable.open(encoding="utf-8")))
    if not rows:
        print("No rows in unreadable CSV")
        return 1

    fieldnames = list(rows[0].keys())
    rng = random.Random(args.seed)

    by_reason: dict[str, list[dict]] = defaultdict(list)
    low_conf: list[dict] = []
    has_text_no_fields: list[dict] = []
    empty_ocr: list[dict] = []

    for row in rows:
        reason = (row.get("fail_reason") or "").strip()
        by_reason[reason].append(row)
        conf = parse_conf(row)
        raw = (row.get("raw_text") or "").strip()

        if conf < args.conf_threshold:
            low_conf.append(row)
        if reason == "empty_ocr":
            empty_ocr.append(row)
        elif reason == "no_name_and_no_price" and raw:
            has_text_no_fields.append(row)

    # OCR ran but parser found nothing useful — best candidates to re-label for rec fine-tune
    ocr_hard = [r for r in rows if (r.get("raw_text") or "").strip()]

    out = args.out_dir
    write_csv(out / "all_unreadable.csv", rows, fieldnames)
    write_csv(out / "empty_ocr_deskew.csv", empty_ocr, fieldnames)
    write_csv(out / "low_confidence_det.csv", low_conf, fieldnames)
    write_csv(out / "ocr_text_but_unparsed.csv", has_text_no_fields, fieldnames)
    write_csv(out / "ocr_any_text.csv", ocr_hard, fieldnames)

    for reason, subset in sorted(by_reason.items()):
        safe = reason.replace(" ", "_") or "unknown"
        write_csv(out / f"reason_{safe}.csv", subset, fieldnames)

    label_rows: list[dict] = []
    if args.label_sample > 0:
        buckets = {
            "empty_ocr": empty_ocr,
            "no_name_and_no_price": by_reason.get("no_name_and_no_price", []),
            "low_conf": low_conf[: max(1, len(low_conf) // 2)],
        }
        label_rows = stratified_sample(buckets, args.label_sample, rng)
        for r in label_rows:
            r["label_product_name"] = ""
            r["label_price_default"] = ""
            r["label_price_card"] = ""
            r["label_notes"] = ""
        label_fields = fieldnames + [
            "label_product_name",
            "label_price_default",
            "label_price_card",
            "label_price_line",
            "label_notes",
        ]
        write_csv(out / "manual_label_sample.csv", label_rows, label_fields)

    conf_values = [parse_conf(r) for r in rows]
    summary = {
        "source": str(args.unreadable.resolve()),
        "total_unreadable": len(rows),
        "fail_reason_counts": dict(Counter(r.get("fail_reason", "") for r in rows)),
        "confidence_threshold": args.conf_threshold,
        "below_conf_threshold": len(low_conf),
        "empty_ocr_count": len(empty_ocr),
        "ocr_text_but_unparsed_count": len(has_text_no_fields),
        "confidence_stats": {
            "min": round(min(conf_values), 4),
            "max": round(max(conf_values), 4),
            "avg": round(sum(conf_values) / len(conf_values), 4),
        },
        "outputs": {
            "all": str((out / "all_unreadable.csv").resolve()),
            "empty_ocr_deskew": str((out / "empty_ocr_deskew.csv").resolve()),
            "low_confidence_det": str((out / "low_confidence_det.csv").resolve()),
            "ocr_text_but_unparsed": str((out / "ocr_text_but_unparsed.csv").resolve()),
            "manual_label_sample": str((out / "manual_label_sample.csv").resolve())
            if label_rows
            else None,
        },
        "recommended_next_steps": [
            {
                "queue": "manual_label_sample.csv",
                "action": "Ручная разметка 200 кропов - эталон для метрик и fine-tune",
                "count": len(label_rows),
            },
            {
                "queue": "ocr_text_but_unparsed.csv",
                "action": "OCR что-то видит, парсер нет - приоритет для улучшения parse_ocr_fields или rec fine-tune",
                "count": len(has_text_no_fields),
            },
            {
                "queue": "empty_ocr_deskew.csv",
                "action": "Пустой OCR - проверить deskew/качество кропа, переснять det",
                "count": len(empty_ocr),
            },
            {
                "queue": "low_confidence_det.csv",
                "action": f"confidence < {args.conf_threshold} - дообучить YOLO / поднять порог det",
                "count": len(low_conf),
            },
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

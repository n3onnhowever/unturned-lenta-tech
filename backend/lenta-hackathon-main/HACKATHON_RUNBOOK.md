# Lenta Hackathon Runbook

This repository now separates honest evaluation from the strongest submission path.

## 1. Metric Policy

- `honest`: the GT source is not present in the detector train split.
- `contaminated`: the GT source appears in train, or the run is GT-aligned/demo-only.
- `diagnostic`: useful for debugging, but not an accuracy claim.

Build the registry:

```powershell
python scripts/build_metrics_registry.py --out output/metrics_registry.json
```

Check a suspected leak:

```powershell
python scripts/check_gt_leakage.py --data-yaml dataset/price_tags_merged/data.yaml --gt materials/data/43_15/43_15.csv
```

The command above should fail because `43_15` is in the merged train split.

## 2. Honest Source-Holdout Benchmark

Build a holdout dataset:

```powershell
python scripts/build_yolo_dataset_from_gt_csv.py --materials materials/data --out dataset/price_tags_holdout_43_15 --holdout-sources 43_15
```

Train detector on that split:

```powershell
python scripts/train_price_tag_yolo.py --data dataset/price_tags_holdout_43_15/data.yaml --name price_tag_holdout_43_15 --epochs 50 --imgsz 1280
```

Run validation with the leakage gate enabled:

```powershell
python scripts/run_hackathon_pipeline.py --detections runs/detect_merged_43_15/detections.csv --gt materials/data/43_15/43_15.csv --train-data dataset/price_tags_holdout_43_15/data.yaml --tag holdout_43_15
```

If the GT source appears in train, the command fails unless `--allow-train-gt-overlap` is explicitly supplied.

## 3. Internal Benchmark

Use the old `43_15` numbers only as an internal crop/OCR regression test:

```powershell
python scripts/run_hackathon_pipeline.py --detections runs/detect_merged_43_15/detections_subset.csv --gt materials/data/43_15/43_15.csv --train-data dataset/price_tags_merged/data.yaml --allow-train-gt-overlap --tag internal_43_15
```

Do not present this as holdout quality.

## 4. Submission Path

For the strongest hackathon submission, train on all available labeled data, then run:

```powershell
python scripts/lenta_hackathon_pipeline.py --video materials/data/43_15/43_15.mp4 --frames frames/materials_data_43_15_43_15 --detections runs/detect_merged_43_15/detections.csv --out-csv output/43_15_submission_full.csv --work-dir runs/pipeline_43_15_full --dedupe-spatial-px 300
```

This mode is optimized for output quality. It is not an honest holdout metric if the same source was used for training.

## 5. Current Known Facts

- `output/holdout_43_15_neg_pricezone_discount_barcode_raw/compliance.json`: best current honest end-to-end benchmark with barcode scan enabled. Leakage check passes (`leakage_detected=false`), one-to-one matching finds `29/29` GT tags, `price_card_match_pct=75.9%`, `price_default_match_pct=17.2%`, `discount_match_pct=10.3%`, `barcode_match_pct=0.0%`, `color_match_pct=62.1%`.
- `output/holdout_43_15_neg_pricezone_discount_barcode_raw/submission.csv`: best honest CSV artifact from the negative-frame detector plus raw-crop RapidOCR price-zone fallback, discount extraction, default-price inference from card price, and enabled OpenCV barcode fallback. It is the current reference for source-holdout reporting.
- `output/holdout_43_15_neg_pricezone_discount_raw/compliance.json`: same price/discount settings before enabling barcode scan; metrics are unchanged except barcode scan was skipped.
- `output/holdout_43_15_neg_pricezone_raw/compliance.json`: previous price-zone fallback run before discount/default inference; `price_card_match_pct=72.4%`, `price_default_match_pct=0.0%`.
- `output/holdout_43_15_neg_conf001/summary.json`: previous honest run before price-zone fallback; detector matched `29/29`, but price OCR remained `0%`.
- `output/final_smart_v2/summary.json`: contaminated internal benchmark, `43_15`, price metrics about `84.4% / 90.0%` at `2` rubles tolerance.
- `output/validation_43_15_det/comparison.json`: diagnostic full-frame coupling, old detector crops gave `0%` price matches.
- `NIK/lenta_tech_ml/qr_research/holdout_validation_findings.md`: honest recognition estimate on held-out sources, much lower accuracy.
- `NIK/lenta_tech_ml/qr_research/validation_detector_findings.md`: honest detector estimate on held-out sources, recall around `0.22` at loose IoU.

Current bottleneck: the fast fallback recovers card prices well and now provides some inferred default prices, but product names, discounts, and barcodes are still weak on honest full-video crops. Treat the contaminated `final_smart_v2` numbers as an upper-bound/internal regression test, not as holdout quality.

Barcode status: `pyzbar` is not available in the current environment, so the old barcode path was effectively inactive. A fast OpenCV barcode fallback was added in `scripts/lenta_barcode.py`; QR/OpenCV smoke checks on matched crops and expanded source-frame crops still produced `0%` barcode matches on `43_15`, so barcode remains an image quality / crop content bottleneck rather than a disabled-code issue.

## 6. Quality Gates

Before reporting numbers:

```powershell
python -m compileall scripts NIK/lenta_tech_ml/qr_research
python scripts/build_metrics_registry.py --out output/metrics_registry.json
```

For any GT-scored command, pass `--train-data` unless the run is explicitly diagnostic.

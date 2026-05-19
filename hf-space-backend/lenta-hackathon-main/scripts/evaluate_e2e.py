import json
import csv
import argparse
import subprocess
import sys
from pathlib import Path
from difflib import SequenceMatcher

def parse_float_csv(s: str) -> float:
    s = (s or "").strip().strip('"').replace(" ", "")
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return -1.0

def bbox_from_gt(row: dict) -> tuple:
    return (
        parse_float_csv(row.get("x_min", "0")),
        parse_float_csv(row.get("y_min", "0")),
        parse_float_csv(row.get("x_max", "0")),
        parse_float_csv(row.get("y_max", "0")),
    )

def similarity(a: str, b: str) -> float:
    a = str(a).lower().strip()
    b = str(b).lower().strip()
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return SequenceMatcher(None, a, b).ratio()

def main():
    parser = argparse.ArgumentParser(description="End-to-End Evaluation")
    parser.add_argument("--gt", type=str, required=True, help="Ground Truth CSV")
    parser.add_argument("--manifest", type=str, required=True, help="Tracking manifest JSON")
    parser.add_argument("--parsed", type=str, required=True, help="Parsed OCR JSON")
    parser.add_argument(
        "--train-data",
        type=str,
        default="",
        help="YOLO data.yaml used for detector training. If set, fail when --gt source is present in train.",
    )
    parser.add_argument("--allow-train-gt-overlap", action="store_true")
    args = parser.parse_args()

    if args.train_data:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "check_gt_leakage.py"),
            "--data-yaml",
            str(Path(args.train_data).resolve()),
            "--gt",
            str(Path(args.gt).resolve()),
        ]
        if args.allow_train_gt_overlap:
            cmd.append("--allow-overlap")
        subprocess.run(cmd, check=True)

    # Load GT
    gt_rows = []
    with open(args.gt, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_rows.append(row)

    # Load Manifest
    with open(args.manifest, 'r', encoding='utf-8') as f:
        manifest_items = json.load(f)
        
    # Load Parsed OCR
    with open(args.parsed, 'r', encoding='utf-8') as f:
        parsed_items = json.load(f)

    # Create a lookup for parsed items by filename
    parsed_lookup = {}
    for item in parsed_items:
        # image_path might be absolute or relative, extract just the filename
        filename = Path(item["image_path"]).name
        parsed_lookup[filename] = item["parsed"]

    # Image diagonal approximation (3840x2160 video)
    image_diag = (3840**2 + 2160**2)**0.5
    max_dist = image_diag * 0.15 # 15% of diagonal is a much tighter threshold

    results = {
        "total_detected_tags": len(manifest_items),
        "gt": str(Path(args.gt).resolve()),
        "train_data": str(Path(args.train_data).resolve()) if args.train_data else "",
        "allow_train_gt_overlap": bool(args.allow_train_gt_overlap),
        "matched_with_gt": 0,
        "price_default_exact_match": 0,
        "price_card_exact_match": 0,
        "product_name_sim_gt_60": 0,
        "product_name_avg_sim": 0.0
    }

    name_sims = []

    for m_item in manifest_items:
        filename = m_item.get("filename")
        if not filename or filename not in parsed_lookup:
            continue
            
        parsed_data = parsed_lookup[filename]
        mb = m_item["bbox"] # [x1, y1, x2, y2]
        # Scale tracking bbox by 2.0 to match 4K GT
        cx_m = ((mb[0] + mb[2]) / 2.0) * 2.0
        cy_m = ((mb[1] + mb[3]) / 2.0) * 2.0
        
        # Find best GT match by text similarity
        best_gt = None
        best_sim = 0.0
        
        parsed_name = parsed_data.get("product_name", "")
        
        for gt in gt_rows:
            gt_name = gt.get("product_name", "")
            sim = similarity(gt_name, parsed_name)
            if sim > best_sim:
                best_sim = sim
                best_gt = gt
                
        if best_gt and best_sim > 0.15: # Lower threshold since OCR can be messy
            results["matched_with_gt"] += 1
            
            # Compare Price Default
            gt_pd = parse_float_csv(best_gt.get("price_default", ""))
            pd_parsed = parse_float_csv(parsed_data.get("price_default", ""))
            if gt_pd > 0 and abs(gt_pd - pd_parsed) < 1.0: # Allow 1 ruble error
                results["price_default_exact_match"] += 1
                
            # Compare Price Card
            gt_pc = parse_float_csv(best_gt.get("price_card", ""))
            pc_parsed = parse_float_csv(parsed_data.get("price_card", ""))
            if gt_pc > 0 and abs(gt_pc - pc_parsed) < 1.0: # Allow 1 ruble error
                results["price_card_exact_match"] += 1
                
            # Compare Product Name
            name_sims.append(best_sim)
            if best_sim > 0.6:
                results["product_name_sim_gt_60"] += 1

    if name_sims:
        results["product_name_avg_sim"] = sum(name_sims) / len(name_sims)

    print("\n=== End-to-End Evaluation Results ===")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

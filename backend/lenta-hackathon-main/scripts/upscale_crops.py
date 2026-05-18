import cv2
import argparse
import os
from pathlib import Path
import json
import csv

def main():
    cv2.setNumThreads(max(1, int(os.getenv("ML_WORKER_THREADS", "2"))))
    parser = argparse.ArgumentParser(description="Upscale crops using OpenCV dnn_superres.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing crops and manifest")
    parser.add_argument("--out", type=str, required=True, help="Output directory for upscaled crops")
    parser.add_argument("--model", type=str, required=True, help="Path to super resolution model (e.g. EDSR_x4.pb)")
    parser.add_argument("--model-name", type=str, default="edsr", help="Model name (edsr, espcn, fsrcnn, lapsrn)")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor (2, 3, 4, 8)")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Super Resolution
    print(f"Loading {args.model_name} model from {args.model}...")
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(args.model)
    sr.setModel(args.model_name, args.scale)

    json_manifest = in_dir / "tracking_manifest.json"
    csv_manifest = in_dir / "crops_manifest.csv"
    
    items = []
    is_csv = False
    
    if json_manifest.exists():
        print(f"Loading JSON manifest: {json_manifest}")
        with open(json_manifest, 'r', encoding='utf-8') as f:
            items = json.load(f)
    elif csv_manifest.exists():
        print(f"Loading CSV manifest: {csv_manifest}")
        is_csv = True
        with open(csv_manifest, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            items = list(reader)
    else:
        print(f"Error: No manifest found in {in_dir}")
        return

    print(f"Processing {len(items)} images...")
    
    upscaled_count = 0
    
    for item in items:
        filename = item.get('filename') or item.get('crop_filename')
        if not filename and item.get('crop_path'):
            filename = Path(item['crop_path']).name
            
        if not filename:
            continue
            
        img_path = in_dir / filename
        if not img_path.exists():
            print(f"Warning: Image not found {img_path}")
            continue
            
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        # Upscale
        try:
            upscaled = sr.upsample(image)
            upscaled_count += 1
            item['upscaled'] = True
        except Exception as e:
            print(f"Error upscaling {filename}: {e}")
            upscaled = image
            item['upscaled'] = False
        
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), upscaled)
        
        if 'crop_path' in item:
            item['crop_path'] = str(out_path.resolve())
        elif 'filename' in item:
            item['filename'] = filename

    print(f"\nResults:")
    print(f"Total processed: {len(items)}")
    print(f"Successfully upscaled: {upscaled_count}")

    if is_csv:
        out_manifest = out_dir / "crops_manifest.csv"
        if items:
            fieldnames = list(items[0].keys())
            with open(out_manifest, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(items)
    else:
        out_manifest = out_dir / "tracking_manifest.json"
        with open(out_manifest, 'w', encoding='utf-8') as f:
            json.dump(items, f, indent=2)
            
    print(f"Saved updated manifest to {out_manifest}")

if __name__ == "__main__":
    main()

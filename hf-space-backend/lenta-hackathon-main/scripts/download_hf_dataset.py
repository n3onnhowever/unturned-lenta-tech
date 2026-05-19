import argparse
import os
from pathlib import Path
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Download sample from HF dataset")
    parser.add_argument("--dataset", type=str, default="openfoodfacts/price-tag-extraction", help="Dataset name")
    parser.add_argument("--out", type=str, default="data/price_tag_dataset", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to download")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset {args.dataset}...")
    try:
        # Load dataset in streaming mode to avoid downloading everything if it's huge
        dataset = load_dataset(args.dataset, split="train", streaming=True)
        
        print(f"Downloading {args.num_samples} samples to {out_dir}...")
        
        # We will save images and a simple jsonl file
        img_dir = out_dir / "images"
        img_dir.mkdir(exist_ok=True)
        
        import json
        
        count = 0
        with open(out_dir / "samples.jsonl", "w", encoding="utf-8") as f:
            for item in dataset:
                if count >= args.num_samples:
                    break
                    
                # Item structure depends on the dataset. Usually contains 'image' and some annotations
                if 'image' in item:
                    img = item['image']
                    img_filename = f"sample_{count:05d}.jpg"
                    img_path = img_dir / img_filename
                    img.save(img_path)
                    
                    # Create a metadata dict without the PIL image object
                    meta = {k: v for k, v in item.items() if k != 'image'}
                    meta['image_filename'] = img_filename
                    
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Downloaded {count} samples...")
                else:
                    print(f"Warning: No 'image' key in item {count}")
                    
        print(f"Successfully downloaded {count} samples.")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

if __name__ == "__main__":
    main()

import json
import cv2
import os
from pathlib import Path
from paddleocr import PaddleOCR
import re

def main():
    dataset_dir = Path("data/price_tag_dataset")
    images_dir = dataset_dir / "images"
    jsonl_path = dataset_dir / "samples.jsonl"
    
    out_dir = dataset_dir / "paddle_rec_data"
    out_crops_dir = out_dir / "crops"
    out_crops_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = out_dir / "rec_gt_train.txt"
    
    print("Initializing PaddleOCR for pseudo-labeling...")
    ocr = PaddleOCR(use_textline_orientation=False, lang='en', use_doc_orientation_classify=False, use_doc_unwarping=False)
    
    crop_count = 0
    
    with open(jsonl_path, 'r', encoding='utf-8') as f_in, open(train_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            img_filename = data['image_filename']
            img_path = images_dir / img_filename
            
            if not img_path.exists():
                continue
                
            # Parse ground truth
            try:
                output_data = json.loads(data['output'])
            except:
                continue
                
            gt_texts = []
            if 'product_name' in output_data and output_data['product_name']:
                # Split product name into words for easier matching, or keep as is
                gt_texts.append(str(output_data['product_name']).upper())
                
            if 'prices' in output_data:
                for p in output_data['prices']:
                    if 'price' in p:
                        gt_texts.append(str(p['price']))
            
            if not gt_texts:
                continue
                
            # Run OCR to get bounding boxes
            image = cv2.imread(str(img_path))
            if image is None:
                continue
                
            result = ocr.ocr(image)
            if not result or not isinstance(result, list) or len(result) == 0:
                continue
                
            res_list = result[0]
            if isinstance(res_list, dict):
                # PaddleX format
                if 'dt_polys' in res_list and 'rec_texts' in res_list and 'rec_scores' in res_list:
                    polys = res_list['dt_polys']
                    texts = res_list['rec_texts']
                    scores = res_list['rec_scores']
                    
                    for box, text, score in zip(polys, texts, scores):
                        text_upper = str(text).upper()
                        is_valid = False
                        for gt in gt_texts:
                            if text_upper in gt or gt in text_upper:
                                is_valid = True
                                break
                                
                        if is_valid and score > 0.8:
                            xs = [int(p[0]) for p in box]
                            ys = [int(p[1]) for p in box]
                            x1, x2 = max(0, min(xs)), min(image.shape[1], max(xs))
                            y1, y2 = max(0, min(ys)), min(image.shape[0], max(ys))
                            
                            if x2 <= x1 or y2 <= y1: continue
                            
                            crop = image[y1:y2, x1:x2]
                            crop_filename = f"crop_{crop_count:06d}.jpg"
                            cv2.imwrite(str(out_crops_dir / crop_filename), crop)
                            f_out.write(f"crops/{crop_filename}\t{text}\n")
                            crop_count += 1
                    
    print(f"Generated {crop_count} pseudo-labeled crops for training.")

if __name__ == "__main__":
    main()

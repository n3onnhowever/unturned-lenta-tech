import cv2
import numpy as np
import argparse
import os
from pathlib import Path
import json
import csv

cv2.setNumThreads(max(1, int(os.getenv("ML_WORKER_THREADS", "2"))))

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def smart_deskew(image, pad_ratio=0.2):
    """
    Add artificial padding, use morphological operations to find the tag mask,
    and apply perspective transform.
    """
    h, w = image.shape[:2]
    pad_y = int(h * pad_ratio)
    pad_x = int(w * pad_ratio)
    
    # Add black padding
    padded = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    # Convert to grayscale
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    
    # Binarization (Otsu)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological closing to merge text and background into a solid white mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours on the mask
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, False
        
    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        
        if len(approx) == 4:
            # Check if area is reasonable
            img_area = padded.shape[0] * padded.shape[1]
            if cv2.contourArea(approx) > 0.1 * img_area:
                screenCnt = approx
                break
                
    if screenCnt is None:
        return image, False
        
    # Order points
    pts = screenCnt.reshape(4, 2)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    # Compute height
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
        
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(padded, M, (maxWidth, maxHeight))
    
    return warped, True

def main():
    parser = argparse.ArgumentParser(description="Smart deskew with artificial padding and morphology.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing crops and manifest")
    parser.add_argument("--out", type=str, required=True, help="Output directory for deskewed crops")
    parser.add_argument("--pad-ratio", type=float, default=0.2, help="Padding ratio")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    
    deskewed_count = 0
    
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
            
        warped, deskewed = smart_deskew(image, args.pad_ratio)
        if deskewed:
            deskewed_count += 1
            
        item['smart_deskewed'] = deskewed
        
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), warped)
        
        if 'crop_path' in item:
            item['crop_path'] = str(out_path.resolve())
        elif 'filename' in item:
            item['filename'] = filename

    print(f"\nResults:")
    print(f"Total processed: {len(items)}")
    print(f"Successfully deskewed: {deskewed_count}")

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

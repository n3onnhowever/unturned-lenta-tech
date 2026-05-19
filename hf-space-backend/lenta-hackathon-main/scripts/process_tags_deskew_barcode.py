import cv2
import numpy as np
import argparse
from pathlib import Path
import json
import csv
from pyzbar.pyzbar import decode

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

def deskew_image(image):
    """Attempt to find a quadrilateral and apply perspective transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use adaptive thresholding instead of just Canny
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, False
        
    # Sort contours by area, keep largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    screenCnt = None
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True) # Increased epsilon for more robust quadrilateral finding
        
        # If our approximated contour has four points, we can assume we found the tag
        if len(approx) == 4:
            # Check if the contour area is reasonably large (at least 10% of the image)
            img_area = image.shape[0] * image.shape[1]
            if cv2.contourArea(approx) > 0.1 * img_area:
                screenCnt = approx
                break
            
    if screenCnt is None:
        # Fallback to Canny if adaptive thresholding failed
        edged = cv2.Canny(blur, 50, 150)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.05 * peri, True)
                if len(approx) == 4:
                    img_area = image.shape[0] * image.shape[1]
                    if cv2.contourArea(approx) > 0.1 * img_area:
                        screenCnt = approx
                        break
                        
    if screenCnt is None:
        return image, False
        
    # Order points and apply perspective transform
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped, True

def read_barcode(image):
    """Attempt to read barcode from image using pyzbar with multiple fallbacks."""
    # Try directly
    barcodes = decode(image)
    if barcodes:
        return barcodes[0].data.decode('utf-8')
        
    # Try grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = decode(gray)
    if barcodes:
        return barcodes[0].data.decode('utf-8')
        
    # Try thresholding
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    barcodes = decode(thresh)
    if barcodes:
        return barcodes[0].data.decode('utf-8')
        
    # Try CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray)
    barcodes = decode(cl1)
    if barcodes:
        return barcodes[0].data.decode('utf-8')
        
    return None

def main():
    parser = argparse.ArgumentParser(description="Deskew price tag crops and read barcodes.")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing crops and manifest")
    parser.add_argument("--out", type=str, required=True, help="Output directory for deskewed crops")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect manifest type
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
    barcode_count = 0
    
    for item in items:
        # Get filename depending on manifest format
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
            
        # 1. Deskew
        warped, deskewed = deskew_image(image)
        if deskewed:
            deskewed_count += 1
            
        # 2. Read Barcode on the warped (or original if deskew failed) image
        barcode = read_barcode(warped)
        if barcode:
            barcode_count += 1
            
        # Update item metadata
        item['deskewed'] = deskewed
        item['barcode_pyzbar'] = barcode
        
        # Save output image
        out_path = out_dir / filename
        cv2.imwrite(str(out_path), warped)
        
        # Update path in manifest so scoring scripts can match it
        if 'crop_path' in item:
            item['crop_path'] = str(out_path.resolve())
        elif 'filename' in item:
            item['filename'] = filename

    print(f"\nResults:")
    print(f"Total processed: {len(items)}")
    print(f"Successfully deskewed: {deskewed_count}")
    print(f"Barcodes found: {barcode_count}")

    # Save updated manifest
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

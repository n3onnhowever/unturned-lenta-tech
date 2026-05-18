import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import json

def calculate_sharpness(image):
    """Calculate the sharpness of an image using the variance of the Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    parser = argparse.ArgumentParser(description="Track price tags and extract the best crop for each unique tag.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, required=True, help="Path to YOLO model weights")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for best crops")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker type (bytetrack.yaml or botsort.yaml)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detection")
    parser.add_argument("--padding", type=float, default=0.15, help="Padding to add around the bounding box (fraction of width/height)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)
    
    print(f"Opening video {args.video}...")
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    # Dictionary to store the best observation for each track_id
    # track_id -> {'score': float, 'crop': np.ndarray, 'frame_idx': int, 'bbox': list, 'sharpness': float, 'area': int}
    best_tags = {} 

    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames to process: {total_frames}")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run tracking on the current frame
        # persist=True is crucial for tracking across frames
        results = model.track(frame, persist=True, tracker=args.tracker, conf=args.conf, verbose=False)
        
        # Check if there are any detections and if they have track IDs assigned
        if len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, conf in zip(boxes, track_ids, confs):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Apply padding
                box_w = x2 - x1
                box_h = y2 - y1
                pad_x = int(box_w * args.padding)
                pad_y = int(box_h * args.padding)
                
                x1 -= pad_x
                y1 -= pad_y
                x2 += pad_x
                y2 += pad_y
                
                # Ensure coordinates are within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = frame[y1:y2, x1:x2]
                area = (x2 - x1) * (y2 - y1)
                
                # Skip ridiculously small crops that can't possibly contain readable text
                if area < 1000: 
                    continue
                    
                sharpness = calculate_sharpness(crop)
                
                # Heuristic score: balance size (area) and clarity (sharpness)
                # Using sqrt(area) prevents huge but blurry crops from dominating
                score = sharpness * np.sqrt(area)
                
                if track_id not in best_tags or score > best_tags[track_id]['score']:
                    best_tags[track_id] = {
                        'score': score,
                        'crop': crop.copy(),
                        'frame_idx': frame_idx,
                        'bbox': [x1, y1, x2, y2],
                        'sharpness': sharpness,
                        'area': area,
                        'conf': float(conf)
                    }
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames. Unique tags found so far: {len(best_tags)}")

    cap.release()
    print("Video processing complete.")

    # Save the best crops and metadata
    print(f"\nFound {len(best_tags)} unique price tags. Saving best crops to {out_dir}...")
    
    manifest = []
    for track_id, data in best_tags.items():
        filename = f"tag_{track_id:04d}_frame_{data['frame_idx']:04d}.jpg"
        crop_path = out_dir / filename
        cv2.imwrite(str(crop_path), data['crop'])
        
        manifest.append({
            'track_id': track_id,
            'filename': filename,
            'frame_idx': data['frame_idx'],
            'bbox': data['bbox'],
            'conf': data['conf'],
            'area': data['area'],
            'sharpness': data['sharpness'],
            'score': data['score']
        })
        
    # Save manifest for later analysis
    manifest_path = out_dir / "tracking_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Saved {len(manifest)} crops and manifest to {out_dir}")

if __name__ == "__main__":
    main()

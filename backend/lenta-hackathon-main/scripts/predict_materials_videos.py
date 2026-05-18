"""
Предсказание YOLO по всем .mp4 в materials/data (рекурсивно).

Сохраняет размеченные видео в project/name (Ultralytics).

Пример:
  python scripts/predict_materials_videos.py --weights runs/detect/runs/detect/price_tag_merged/weights/best.pt --conf 0.5 --vid-stride 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLO predict по всем видео в materials/data.")
    ap.add_argument(
        "--materials",
        type=Path,
        default=Path("materials/data"),
        help="Корень с видео",
    )
    ap.add_argument(
        "--weights",
        type=Path,
        default=Path("runs/detect/runs/detect/price_tag_merged/weights/best.pt"),
    )
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--vid-stride", type=int, default=4, help="Каждый N-й кадр видео (быстрее)")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--project", type=Path, default=Path("runs/predict_materials_merged"))
    args = ap.parse_args()

    if not args.weights.is_file():
        print(f"Нет весов: {args.weights}", file=sys.stderr)
        return 1
    if not args.materials.is_dir():
        print(f"Нет папки: {args.materials}", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("pip install ultralytics", file=sys.stderr)
        return 2

    vids = sorted(args.materials.rglob("*.mp4")) + sorted(args.materials.rglob("*.MP4"))
    vids = list(dict.fromkeys(vids))
    if not vids:
        print("Видео .mp4 не найдены.", file=sys.stderr)
        return 1

    model = YOLO(str(args.weights))
    device = args.device if args.device else None

    print(f"Видео: {len(vids)}, weights={args.weights}, conf={args.conf}, imgsz={args.imgsz}, vid_stride={args.vid_stride}")
    for vid in vids:
        name = f"{vid.parent.name}__{vid.stem}"
        print(f"  -> {vid}  (run name: {name})")
        model.predict(
            source=str(vid),
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            save=True,
            project=str(args.project),
            name=name,
            exist_ok=True,
            vid_stride=args.vid_stride,
            stream=True,
            verbose=False,
        )

    print("Готово. Типичный путь к результатам: runs/detect/runs/predict_materials_merged/<имя_папки>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
